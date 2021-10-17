import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
import pytorch_lightning.callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
from . import dataset, model

@hydra.main(config_name='conf', config_path=None)
def main(config):


    dm = dataset.DataModule(config.batch_size // config.gpus,
                            config.data.datafile)
        
    dm.setup()

    config.data.dataset_size = len(dm.train_dataset)
    config.model.character_size = dm.character_vocab_size
    config.model.character_padding_idx = dm.character_padding_idx
    config.model.phoneme_size = dm.phoneme_vocab_size
    config.model.phoneme_padding_idx = dm.phoneme_padding_idx


    word_2_phone_model = model.MLP(config)
    #model_test = word_2_phone_model.load_from_checkpoint(checkpoint_path="/home/sk7327/cds-bootcamp/lecture2/capstone-project-47/capstone-project-47/outputs/2021-10-16/16-30-46/lightning_logs/version_0/checkpoints/capstone-spanish-epoch=02-val/loss=0.17.ckpt")
    model_test = word_2_phone_model.load_from_checkpoint(checkpoint_path=config.dir.load_path)


    trainer = pl.Trainer()  
    trainer.test(model_test, datamodule=dm, verbose=True)


if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=model.MLPTrainingConfig)
    main()
