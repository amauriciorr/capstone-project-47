import dataclasses
from typing import Any, Dict, Optional

import omegaconf
import pytorch_lightning as pl
import torchvision
import torch
from torch import nn
import torchmetrics


@dataclasses.dataclass
class MLPModelConfig:
    embedding_size: int = 128
    # assuming we're starting with english
    character_size: int = 26
    #need to change to correct phoneme size / # of distinct phonemes
    phoneme_size: int = 39

@dataclasses.dataclass
class MLPOptimConfig:
    # learning_rate: float = 1e-2
    # weight_decay: float = 1e-5
    learning_rate: float = 1e-3
    weight_decay: float = 0.0
    grad_clip_norm: Optional[float] = None


@dataclasses.dataclass
class MLPDataConfig:
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

        self.save_hyperparameters(config)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.accuracy_top1 = torchmetrics.Accuracy(num_classes=config.model.phoneme_size)
        # need to add config or hparams specifying vocab_size, embedding_dim, padding_idx
        # make sure to make embedding_dim same across both!
        self.character_embedding = nn.Embedding(config.model.character_size, config.model.embedding_size)
        self.phoneme_embedding = nn.Embedding(config.model.phoneme_size, config.model.embedding_size)
        self.layers = nn.Sequential(
            nn.Linear(config.model.embedding_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, config.model.phoneme_size)
        )


    def forward(self, characters, phoneme):
        word_representation = self.character_embedding(characters)
        phone_representation = self.phoneme_embedding(phoneme)
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
        optimizer = torch.optim.Adam(self.parameters(), lr=base_lr, weight_decay=hparams.optim.weight_decay)
        steps_per_epoch = (self.hparams.data.dataset_size + self.hparams.batch_size - 1) // self.hparams.batch_size
         lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
            opt,
            max_lr=base_lr * 10,
            epochs=self.hparams.max_epochs,
            steps_per_epoch=steps_per_epoch)

        scheduler_config = {
            'scheduler': lr_scheduler,
            'interval': 'step',
            'frequency': 1
        }

        return [opt], [scheduler_config]
