import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
import pytorch_lightning.callbacks
from . import model

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

    """
    need to add portion of code to setup data and loaders, etc
    """

    config.data.dataset_size = len()
    model = model.MLP(config)
    # remember to add necessary data module here.
    trainer.fit(model, datamodule=)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=model.MLPTrainingConfig)
    main()
