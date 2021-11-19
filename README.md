# capstone-project-47

## Background
Catastrophic forgetting in the context of language acquisition: It has been observed that while neural nets perform well on tasks they have been trained on, when they keep being trained on slightly different tasks, they do not perform as well on the initial tasks any more. This phenomenon has been called "catastrophic forgetting" and has received a lot of attention. At the same time, psychologists and neuroscientists observe similar types of amnesia when studying infants that learn and then forget their first languages, as is the case of cross-border adoptees (see [here](https://www.sciencedirect.com/science/article/pii/S0010027721002079) for a recent study). Moreover, it is observed that in children who seemingly have forgotten their native language, *accelerated learning* takes place when exposed to the language again. Neuroimaging research suggests are large degree in neural overlap between areas activated when processing subsequently acquired languages. In this project we would like to mimic some of these studies with neural architectures. In particular we want to see if we observe "accelerated" learning of forgotten information, study aspects of transfer learning and try to identify subarchitectures that might be responsible for these phenomena.

## Directory structure
* `/notebooks` -  contains exploratory, preliminary data processing, or scratch work notebooks. 
* `/data` - contains original / raw data as well as minorly pre-processed data
	* `/cmudict` - contains original CMUDICT data
	* `/words_to_phones` - contains words and their corresponding ARPAbet pronunciations/spellings in csv format for various languages.
	* `token_encodings/` - includes the word and phoneme tokenizers pairs, e.g. English specific. We've trained "universal" tokenizers that takes words and phonemes from each language.
	*`model_ready/` - processed pronunciation data in the form of csv and dict files
* `/mlp` - contains implementation for multilayer perceptron baseline model
	* `dataset.py` - lightning data module for tokenizing data and preparing dataloaders
	* `model.py` - defines mlp model (forward-pass, training + validation steps, loss function, etc.) along with various config params
	* `train.py` - for initializing mlp model and training.
	* `test.py` - for testing already-trained mlp model.

## How to run experiments
### Training
To start training model, you can run `python mlp.train gpus=1 data.datafile='processed_english.csv'` from the root directory. We provide data files for Spanish, Finnish, Italian, Dutch, and Croatian as well. By default, the model is set to train for at most 30 epochs, you can adjust this by adding the `max_epochs` command-line argument, e.g. `max_epochs=10`. Additionally, we have a default patience of 10, which monitors changes in validation loss; this can also be adjusted with the `model.patience` command-line argument.

For tokenizers, we have three options:
* English - `'english'`
* Spanish - `'spanish'`
* Universal - `'universal'`
The first two are self-explanatory, namely they are tokenizers trained on the respective langauges. The "universal" tokenizer, in turn, is trained on all six languages used in our experiments, i.e. English, Spanish, Italian, Finnish, Dutch, Croatian. By default, the tokenizer is set to universal. This can be adjusted by the command-line argument `data.tokenizer_lang`, e.g. `data.tokenizer_lang='spanish'`.

For transfer learning, you can use the `dir.load_path` argument, specifying a checkpoint from an earlier run. Alternatively, this can also be used to continue training from an already trained model.


Lastly, we also provide the ability to do "bilingual learning". You can specify more than one datafile in the `data.datafile` argument, e.g. `data.datafile='processed_english.csv, processed_spanish.csv'`. Both files will be read, combined, shuffled, then broken into train-validation-test sets in a 80:10:10 ratio. It is important to note that for bilingual training we downsample the dataset to match the vocabulary size of our smallest dataset, which is Italian in this case. 

Other command-line flags worth noting:
* `batch_size` - by default is set to 256
* `model.embedding_size` - by default is set to 256
* `optim.learning_rate` - by default is set to 1e-3
* `optim.weight_decay` -  by default is set to 1e-5

### Testing
To test your model, you can run `python mlp.test gpus=1 dir.load_path='path/to/your/model/ data.tokenizer_lang='universal' data.datafile='processed_english.csv` for example. 