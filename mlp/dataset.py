import os
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, TensorDataset, RandomSampler
import pytorch_lightning as pl
from string import digits
from typing import Optional
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers import normalizers
from tokenizers.normalizers import NFD, StripAccents

# see https://github.com/huggingface/transformers/issues/5486 for context
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# language-specific tokenizers, as well as a universal tokenizer that incorporates
# characters and phonemes across all languages used in experiment 
# (English, Spanish, Italian, Finnish, Dutch, Croatian)
TOKENIZER_FILES = {
    'english': ["/token_encodings/word_tokenizer-eng.json", "/token_encodings/phoneme_tokenizer-eng.json"],
    'spanish': ["/token_encodings/word_tokenizer-spanish.json", "/token_encodings/phoneme_tokenizer-spanish.json"],
    'universal': ["/token_encodings/word_tokenizer-universal.json", "/token_encodings/phoneme_tokenizer-universal.json"]
}

class DataModule(pl.LightningDataModule):
    def __init__(self,  batch_size: int, datafile, seed=100, tokenizer_lang='english', root='/model_ready/csv/', num_data_workers: int=4):
        super().__init__()
        self.seed = seed
        self.root = root
        self.datafile = [file.strip() for file in datafile.split(',')]
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_data_workers = num_data_workers
        self.word_tokenizer = Tokenizer.from_file(TOKENIZER_FILES[tokenizer_lang][0])
        self.phoneme_tokenizer = Tokenizer.from_file(TOKENIZER_FILES[tokenizer_lang][1])
        self.character_vocab_size = self.word_tokenizer.get_vocab_size()
        self.phoneme_vocab_size = self.phoneme_tokenizer.get_vocab_size()
        self.character_padding_idx = self.word_tokenizer.get_vocab()['PAD']
        self.phoneme_padding_idx = self.phoneme_tokenizer.get_vocab()['PAD']

    def get_splits(self, df, seed=None):
        """
        helper function for randomly shuffling and splitting data.
        :param df: DataFrame object with raw data
        :param seed: int for determining random seed to use.
        """
        np.random.seed(seed)
        shuffled_idx = np.random.permutation(df.index)
        # determine indices for partitioning df
        rows = len(df.index)
        # 0.8 corresponds to 80% allocated for training set
        train_idx = int(0.8 * rows)
        # 0.1 corresponds to 10% allocated for validation. 
        # remainder is for test set
        validate_idx = int(0.1 * rows) + train_idx
        train = df.loc[shuffled_idx[:train_idx]]
        validate = df.loc[shuffled_idx[train_idx:validate_idx]]
        test = df.loc[shuffled_idx[validate_idx:]]
        return train, validate, test

    def tokenize_data(self, df):
        # minor pre-processing done since we are no longer using BPE
        df['word'] = df['word'].map(lambda row: ' '.join(list(row.strip())))

        # set fixed padding length to avoid variable sequence lengths from batch-to-batch
        # ensures we can set a fixed input dim for our first nn.Linear
        self.word_tokenizer.enable_padding(direction='right', pad_token='PAD',
                                           length=45)
        self.phoneme_tokenizer.enable_padding(direction='right', pad_token='PAD',
                                           length=45)
        words = self.word_tokenizer.encode_batch(list(df['word'].values))
        phonemes = self.phoneme_tokenizer.encode_batch(list(df['phonemes'].values))

        # not the most DRY code but after we've fixed padding for a specific length
        # for our sequence of phonemes we revert to padding to max length,
        #  i.e. length-1 for labels.
        self.phoneme_tokenizer.enable_padding(direction='right', pad_token='PAD')
        labels = self.phoneme_tokenizer.encode_batch(list(df['label'].values))

        word_ids = torch.tensor([x.ids for x in words], dtype=torch.long)
        phoneme_ids = torch.tensor([x.ids for x in phonemes], dtype=torch.long)
        label_ids = torch.tensor([x.ids for x in labels], dtype=torch.long)
        return TensorDataset(word_ids, phoneme_ids, label_ids)


    def setup(self, stage: Optional[str]=None) -> None:
        # specifying more than a single dataset defaults to bilingual training
        # both datasets are mixed before downsampling the randomized mixture
        if len(self.datafile) > 1:
            df = [pd.read_csv(self.root + file) for file in self.datafile]
            df = pd.concat(df)
            vocab = df.word.unique()
            # hard-coded min vocab size 14563 -- corresponds to Italian, which
            # is smallest dataset
            mask = np.random.choice(len(vocab), 14563, replace=False)
            vocab = vocab[mask]
            df = df[df.word.isin(vocab)]
        else:
            df = pd.read_csv(self.root + self.datafile[0])
        df.dropna(inplace=True)
        train, val, test = self.get_splits(df, seed=self.seed)
        self.train_dataset = self.tokenize_data(train)
        self.val_dataset = self.tokenize_data(val)
        self.test_dataset = self.tokenize_data(test)


    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        return DataLoader(
            dataset=self.train_dataset,
            sampler=RandomSampler(self.train_dataset),
            batch_size=self.batch_size,
            num_workers=self.num_data_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        return DataLoader(
            dataset=self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_data_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the test set. """
        return DataLoader(
            dataset=self.test_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_data_workers,
            )
