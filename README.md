# capstone-project-47

## Background
Catastrophic forgetting in the context of language acquisition: It has been observed that while neural nets perform well on tasks they have been trained on, when they keep being trained on slightly different tasks, they do not perform as well on the initial tasks any more. This phenomenon has been called "catastrophic forgetting" and has received a lot of attention. At the same time, psychologists and neuroscientists observe similar types of amnesia when studying infants that learn and then forget their first languages, as is the case of cross-border adoptees (see [here](https://www.sciencedirect.com/science/article/pii/S0010027721002079) for a recent study). Moreover, it is observed that in children who seemingly have forgotten their native language, *accelerated learning* takes place when exposed to the language again. Neuroimaging research suggests are large degree in neural overlap between areas activated when processing subsequently acquired languages. In this project we would like to mimic some of these studies with neural architectures. In particular we want to see if we observe "accelerated" learning of forgotten information, study aspects of transfer learning and try to identify subarchitectures that might be responsible for these phenomena.

## Directory structure
* `/notebooks` -  contains exploratory, preliminary data processing, or scratch work notebooks. 
* `/data` - contains original / raw data as well as minorly pre-processed data
	* `/cmudict` - contains original CMUDICT data
	* `/words_to_phones` - contains words and their corresponding ARPAbet pronunciations/spellings in csv format for various languages.
* **[WIP]** `/mlp` - contains implementation for baseline model