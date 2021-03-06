import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
from . import dataset, model
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
import torch

@hydra.main(config_name='conf', config_path=None)
def main(config):
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
                            monitor="val/loss",
                            filename="capstone-{epoch:02d}-{val/loss:.2f}",
                            save_top_k=1,
                            mode="min"
                          )

    if config.dir.save_dir:
        checkpoint_callback.dirpath = config.dir.save_dir

    callbacks = [
        pl.callbacks.GPUStatsMonitor(),
        pl.callbacks.LearningRateMonitor(log_momentum=True),
        checkpoint_callback,
        EarlyStopping(monitor="val/loss", patience=config.model.patience)
    ]

    trainer_kwargs = { **config.lightning }
    trainer_kwargs['gpus'] = config.gpus
    trainer_kwargs['callbacks'] = callbacks
    trainer_kwargs['max_epochs'] = config.max_epochs

    if config.gpus > 1:
        trainer_kwargs['accelerator'] = 'ddp'

    trainer = pl.Trainer(**trainer_kwargs)
    dm = dataset.DataModule(config.batch_size // config.gpus,
                            config.data.datafile,
                            tokenizer_lang=config.data.tokenizer_lang)
    dm.setup()
    config.data.dataset_size = len(dm.train_dataset)
    config.model.character_size = dm.character_vocab_size
    config.model.character_padding_idx = dm.character_padding_idx
    config.model.phoneme_size = dm.phoneme_vocab_size
    config.model.phoneme_padding_idx = dm.phoneme_padding_idx


    word_2_phone_model = model.MLP(config)

    if config.dir.load_path:
        checkpoint = torch.load(config.dir.load_path)
        word_2_phone_model.load_state_dict(checkpoint['state_dict'])
        
    trainer.fit(word_2_phone_model, datamodule=dm)
    print('Best checkpoint saved at: {}'.format(checkpoint_callback.best_model_path))

if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=model.MLPTrainingConfig)
    main()
