import dataclasses
from typing import Any, Dict, Optional

import omegaconf
import pytorch_lightning as pl
import torch
from torch import nn
import torchmetrics


@dataclasses.dataclass
class MLPModelConfig:
    embedding_size: int = 256
    character_size: Optional[int] = None
    character_padding_idx: Optional[int] = None
    phoneme_size: Optional[int] = None
    phoneme_padding_idx: Optional[int] = None
    patience: int = 10

@dataclasses.dataclass
class MLPOptimConfig:
    learning_rate: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: Optional[float] = None


@dataclasses.dataclass
class MLPDataConfig:
    # by default, we use universal tokenizer
    # as this allows for the most flexibility in terms of language choice.
    datafile: Optional[str] = None
    dataset_size: Optional[int] = None
    tokenizer_lang: str = 'universal'
    num_workers: int = 4

@dataclasses.dataclass
class MLPDirectoryConfig:
    save_dir: Optional[str] = None
    load_path: Optional[str] = None

@dataclasses.dataclass
class MLPTrainingConfig:
    data: MLPDataConfig = MLPDataConfig()
    model: MLPModelConfig = MLPModelConfig()
    optim: MLPOptimConfig = MLPOptimConfig()
    dir: MLPDirectoryConfig = MLPDirectoryConfig()
    lightning: Dict[str, Any] = dataclasses.field(default_factory=dict)
    batch_size: int = 256
    max_epochs: int = 30
    gpus: int = 2


class MLP(pl.LightningModule):
    def __init__(self, config):
        super().__init__()

        if not isinstance(config, omegaconf.DictConfig):
            config = omegaconf.OmegaConf.structured(config)
        """
        save_hyperparameters() call makes hparams available in other functions
        see https://pytorch-lightning.readthedocs.io/en/latest/common/hyperparameters.html#lightningmodule-hyperparameters
        for more information.
        """
        self.save_hyperparameters(config)
        self.criterion = nn.CrossEntropyLoss()
        self.softmax = nn.Softmax(dim=1)
        self.accuracy_top1 = torchmetrics.Accuracy(num_classes=config.model.phoneme_size)
        # shared size embedding dim is arbitrary, though is also a choice based off convenience.
        self.character_embedding = nn.Embedding(config.model.character_size, config.model.embedding_size,
                                                padding_idx=config.model.character_padding_idx)
        self.phoneme_embedding = nn.Embedding(config.model.phoneme_size, config.model.embedding_size,
                                              padding_idx=config.model.phoneme_padding_idx)
        # class variables for saving test predictions vs labels for generating heatmaps later.
        self.y_preds = []
        self.y_labels = []
        """
        since both phoneme and word inputs are padded to length 45 we have a 45 x emb_dim output 
        from nn.Embedding for each, which we change into a concatenated 1-dim vector. multiply by 2 to account 
        for both character and phoneme sequences.
        """
        self.layers = nn.Sequential(
            nn.Linear(config.model.embedding_size * 45 * 2, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, config.model.phoneme_size)
        )


    def forward(self, word, phoneme):
        """
        idea is to concatenate the embedding representations for each character in a sequence 
        of characters end-to-end, do the same for the sequence of phonemes. once this is 
        accomplished for both, then concatenate both these long embedding sequences together
        NOTE: each input is batch_size x sequence_length, after applying embedding function we have
        batch_size x sequence_length x embedding_dim
        """
        word_representation = self.character_embedding(word)
        word_dims = word_representation.size()
        word_representation = word_representation.view(word_dims[0], word_dims[1] * word_dims[2])

        phone_representation = self.phoneme_embedding(phoneme)
        phone_dims = phone_representation.size()
        phone_representation = phone_representation.view(phone_dims[0], phone_dims[1] * phone_dims[2])

        word = torch.cat([word_representation, phone_representation], dim=1)
        return self.layers(word)

    def _compute_loss(self, batch):
        # TO-DO: incorporate penalty based on phoneme distance measure
        word, phoneme, label = batch
        logits = self(word, phoneme)
        loss = self.criterion(logits, label.squeeze())
        predictions = torch.argmax(self.softmax(logits), dim=1)
        accuracy = self.accuracy_top1(predictions, label.squeeze())
        return loss, accuracy, predictions, label.squeeze()

    def training_step(self, batch, *_):
        loss, accuracy, _, _ = self._compute_loss(batch)
        self.log('train/accuracy', accuracy, prog_bar=True)
        self.log('train/loss', loss)
        # log both step- and epoch-level performance.
        self.logger.experiment.add_scalar("train/loss-epoch", loss, self.current_epoch)
        self.logger.experiment.add_scalar("train/accuracy-epoch", accuracy, self.current_epoch)
        return loss


    def validation_step(self, batch, *_):
        loss, accuracy, _, _ = self._compute_loss(batch)
        self.log('val/loss', loss)
        self.log('val/accuracy', accuracy)
        # log both step- and epoch-level performance.
        self.logger.experiment.add_scalar("val/loss-epoch", loss, self.current_epoch)
        self.logger.experiment.add_scalar("val/accuracy-epoch", accuracy, self.current_epoch)

    def test_step(self, batch, *_):
        loss, accuracy, predictions, labels = self._compute_loss(batch)
        self.log('test/loss', loss)
        self.log('test/accuracy', accuracy)
        self.y_labels.extend(labels.cpu().numpy())
        self.y_preds.extend(predictions.cpu().numpy())

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

