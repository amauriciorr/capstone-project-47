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

class DataModule(pl.LightningDataModule):
    def __init__(self,  batch_size: int, datafile, seed=100, root='/model_ready/csv/', num_data_workers: int=4):
        super().__init__()
        self.seed = seed
        self.root = root
        self.datafile = self.root + datafile
        self.batch_size = batch_size
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None
        self.num_data_workers = num_data_workers
        self.word_tokenizer = Tokenizer.from_file("/token_encodings/word_tokenizer-eng.json")
        self.phoneme_tokenizer = Tokenizer.from_file("/token_encodings/phoneme_tokenizer-eng.json")
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

        words = self.word_tokenizer.encode_batch(list(df['word'].values))
        phonemes = self.phoneme_tokenizer.encode_batch(list(df['phonemes'].values))
        labels = self.phoneme_tokenizer.encode_batch(list(df['label'].values))

        word_ids = torch.tensor([x.ids for x in words], dtype=torch.long)
        phoneme_ids = torch.tensor([x.ids for x in phonemes], dtype=torch.long)
        label_ids = torch.tensor([x.ids for x in labels], dtype=torch.long)
        return TensorDataset(word_ids, phoneme_ids, label_ids)


    def setup(self, stage: Optional[str]=None) -> None:

        df = pd.read_csv(self.datafile)
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
