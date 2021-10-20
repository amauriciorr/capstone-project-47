import hydra
from hydra.core.config_store import ConfigStore
import pytorch_lightning as pl
import pytorch_lightning.callbacks
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from . import dataset, model

@hydra.main(config_name='conf', config_path=None)
def main(config):


    dm = dataset.DataModule(config.batch_size // config.gpus,
                            config.data.datafile,
                            tokenizer_lang=config.data.tokenizer_lang)
        
    dm.setup()

    config.data.dataset_size = len(dm.train_dataset)
    config.model.character_size = dm.character_vocab_size
    config.model.character_padding_idx = dm.character_padding_idx
    config.model.phoneme_size = dm.phoneme_vocab_size
    config.model.phoneme_padding_idx = dm.phoneme_padding_idx

    trainer_kwargs = { **config.lightning }
    trainer_kwargs['gpus'] = config.gpus
    if config.gpus > 1:
        trainer_kwargs['accelerator'] = 'ddp'

    word_2_phone_model = model.MLP(config)

    checkpoint = torch.load(config.dir.load_path)
    word_2_phone_model.load_state_dict(checkpoint['state_dict'])
    trainer = pl.Trainer()  
    trainer.test(model_test, datamodule=dm, verbose=True)

    checkpoint = torch.load(config.dir.load_path)
    word_2_phone_model.load_state_dict(checkpoint['state_dict'])

    trainer = pl.Trainer(**trainer_kwargs)
    trainer.test(word_2_phone_model, datamodule=dm, verbose=True)



if __name__ == '__main__':
    cs = ConfigStore()
    cs.store('conf', node=model.MLPTrainingConfig)
    main()
