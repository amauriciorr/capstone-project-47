import pandas as pd
import numpy as np
word_tokenizer = Tokenizer.from_file("../data/token_encodings/tokenizer-wiki.json")   
phoneme_tokenizer = Tokenizer.from_file("../data/token_encodingstokenizer-wiki.json") 

class DataModule(pl.LightningDataModule):
    def __init__(self, classifier_instance):
        super().__init__()
        # self.hparams = classifier_instance.hparams
        # self.classifier = classifier_instance
        self.batch_size = batch_size
        self.root = root
        self.train_ds = None
        self.val_ds = None
        self.num_data_workers = num_data_workers


        # Label Encoder
        self.label_encoder = LabelEncoder(
            pd.read_csv(self.hparams.train_csv).label.astype(str).unique().tolist(), 
            reserved_labels=[]
        )
        self.label_encoder.unknown_index = None

    def read_csv(self, path: str) -> list:
            """ Reads a comma separated value file.
            :param path: path to a csv file.
            
            :return: List of records as dictionaries
            """
            df = pd.read_csv(path)
            df = df.dropna()
            words = word_tokenizer.encode_batch(list(cro_df['word'].values))
            phonemes = phoneme_tokenizer.encode_batch(list(cro_df['phonemes'].values))
            labels = phoneme_tokenizer.encode_batch(list(cro_df['label'].values))
            word_ids = torch.tensor(np.array([x.ids for x in words]))
            phoneme_ids = torch.tensor(np.array([x.ids for x in phonemes]))
            label_ids = torch.tensor(np.array([x.ids for x in labels]))
            return word_ids, phoneme_ids, label_ids  


    def train_dataloader(self) -> DataLoader:
        """ Function that loads the train set. """
        self._train_dataset = self.≈(self.hparams.train_csv)
        return DataLoader(
            dataset=self._train_dataset,
            sampler=RandomSampler(self._train_dataset),
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.loader_workers,
        )

    def val_dataloader(self) -> DataLoader:
        """ Function that loads the validation set. """
        self._val_dataset = self.read_csv(self.hparams.val_csv)
        return DataLoader(
            dataset=self._val_dataset,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.loader_workers,
        )

    def test_dataloader(self) -> DataLoader:
        """ Function that loads the test set. """
        self._test_dataset = self.read_csv(self.hparams.test_csv)
        return DataLoader(
            dataset=self._test_dataset,
            batch_size=self.hparams.batch_size,
            collate_fn=self.classifier.prepare_sample,
            num_workers=self.hparams.loader_workers,
            )