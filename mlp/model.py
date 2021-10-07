import dataclasses
from typing import Any, Dict, Optional

import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics


@dataclasses.dataclass
class MLPModelConfig:
    embedding_size: int = 128
    character_size: Optional[int] = None
    character_padding_idx: Optional[int] = None
    phoneme_size: Optional[int] = None
    phoneme_padding_idx: Optional[int] = None

@dataclasses.dataclass
class MLPOptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: Optional[float] = None


@dataclasses.dataclass
class MLPDataConfig:
    # base case will always start with english 
    # then we will seek to either leverage transfer learning
    # with same task on another language or return to english.
    datafile: str = 'processed_english.csv'
    dataset_size: Optional[int] = None
    num_workers: int = 4


@dataclasses.dataclass
class MLPTrainingConfig:
    data: MLPDataConfig = MLPDataConfig()
    model: MLPModelConfig = MLPModelConfig()
    optim: MLPOptimConfig = MLPOptimConfig()
    lightning: Dict[str, Any] = dataclasses.field(default_factory=dict)
    batch_size: int = 256
    max_epochs: int = 30
    gpus: int = 2


class MLP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        if not isinstance(config, omegaconf.DictConfig):
            config = omegaconf.OmegaConf.structured(config)

        # note: below function call makes hparams available in other functions
        # see configure_optimizer for our current usage
        # see https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html#lightningmodule-hyperparameters
        # for more information.
        self.save_hyperparameters(config)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy_top1 = torchmetrics.Accuracy(num_classes=config.model.phoneme_size)
        # embedding_dim need to be same across both character and phoneme embeddings 
        # since we're looking to concatenate them before doing forward-pass. note: although embeddings
        # are the same size, they are not learned together, i.e. phoneme embeddings are learned
        # independently of character embeddings.
        self.character_embedding = nn.Embedding(config.model.character_size, config.model.embedding_size,
                                                padding_idx=config.model.character_padding_idx)
        self.phoneme_embedding = nn.Embedding(config.model.phoneme_size, config.model.embedding_size,
                                              padding_idx=config.model.phoneme_padding_idx)
        # consider making a wider model!
        self.layers = nn.Sequential(
            nn.Linear(config.model.embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.model.phoneme_size)
        )


    def forward(self, characters, phoneme):
        # WIP
        # idea is to scale / normalize embedding vectors by word- or phoneme-length (respectively)
        # then concat two vectors before doing forward-pass through linear layers
        word_lengths = torch.sum(characters != self.hparams.model.character_padding_idx, dim=1)
        word_representation = self.character_embedding(characters)
        word_representation = torch.sum(word_representation, dim=1)

        phone_lengths = torch.sum(phoneme != self.hparams.model.phoneme_padding_idx, dim=1)
        phone_representation = self.phoneme_embedding(phoneme)
        phone_representation = torch.sum(phone_representation, dim=1)
        word = torch.cat([word_representation, phone_representation])
        return self.layers(word)

    def _compute_loss(self, batch):
        word, phoneme, label = batch
        logits = self(word, phoneme)
        loss = self.criterion(logits, label)
        accuracy = self.accuracy_top1(logits, label)
        return loss, accuracy

    def training_step(self, batch, *_):
        loss, accuracy = self._compute_loss(batch)
        self.log('accuracy', accuracy, prog_bar=True)
        return loss

    def validation_step(self, batch, *_):
        loss, _ = self._compute_loss(batch)
        self.log('val/loss', loss)

    def configure_optimizers(self):
        base_lr = self.hparams.optim.learning_rate / 256 * self.hparams.batch_size
        optimizer = torch.optim.Adam(self.parameters(), lr=base_lr, weight_decay=self.hparams.optim.weight_decay)
        steps_per_epoch = (self.hparams.data.dataset_size + self.hparams.batch_size - 1) // self.hparams.batch_size
        lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=base_lr * 10,
            epochs=self.hparams.max_epochs,
            steps_per_epoch=steps_per_epoch)

        scheduler_config = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [optimizer], [scheduler_config]

