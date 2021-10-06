import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
import pytorch_lightning.callbacks
from . import dataset, model

@hydra.main(config_name='conf', config_path=None)
def main(config):
    callbacks = [
        pytorch_lightning.callbacks.GPUStatsMonitor(),
        pytorch_lightning.callbacks.LearningRateMonitor(log_momentum=True),
    ]

    trainer_kwargs = { **config.lightning }
    trainer_kwargs['gpus'] = config.gpus
    trainer_kwargs['callbacks'] = callbacks
    trainer_kwargs['max_epochs'] = config.max_epochs

    if config.gpus > 1:
        trainer_kwargs['accelerator'] = 'ddp'

    trainer = pytorch_lightning.Trainer(**trainer_kwargs)
    dm = dataset.DataModule(config.batch_size // config.gpus,
                            config.data.datafile)
    dm.setup()
    config.data.dataset_size = len(dm.train_dataset)
    config.model.character_size = dm.character_vocab_size
    config.model.phoneme_size = dm.phoneme_vocab_size

    word_2_phone_model = model.MLP(config)
    trainer.fit(word_2_phone_model, datamodule=dm)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=model.MLPTrainingConfig)
    main()
