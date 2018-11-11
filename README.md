# POS Tagging for Twitter

TensorFlow implementation of [Transferring from Formal Newswire Domain with Hypernet for Twitter POS Tagging](http://aclweb.org/anthology/D18-1275). 

![compgraph](./img/model.pdf)

The code is partially referred to https://github.com/guitaowufeng/TPANN and https://github.com/hardmaru/supercell/blob/master/supercell.py

## Requirements

- Python 2.7 or higher
- Numpy 
- Tensorflow 1.0 
- Gensim

In addition, anyone who want to run these codes should download the word embedding from https://github.com/guitaowufeng/TPANN. The files 'word2vec_200dim.model...
' should be placed at 'dada/'.

## Usage

1.Reproducing the results of paper:
```
$ python hyper_train.py
```
which will reproduce the results of the paper. The code will save the best model for future use. 

2.retraining the model 

You can change the hyperparameters to observe the change of results. Please run
```
$ python hyper_train.py --## ##
```

The parameters in this model are: 

| params          | meaning                                           | default     |
|:---------------:|:-------------------------------------------------:|:-----------:|
| rnn_size        | size of LSTM internal state                       | 250         |
| kernels         | CNN kernel widths                                 | [1,2,3,4,5,6]|
| kernel_features | number of features in the CNN kernel              | [50,50,100,100,200,200]|
| hyper_input_size | window of context | 7         |
| char_embed_size | dimensionality of character embeddings            | 25          |
| word_embed_size | dimensionality of word embeddings                 | 200         |
| max_word_length | maximum word length                               | 35          |
| param_init      | initialize parameters at                          | 0.05        |
| batch_size      | number of sequences to train on in parallel       | 20          |
| max_epochs      | number of full passes through the training data   | 100         |
and so on.


## Reference

[Transferring from Formal Newswire Domain with Hypernet for Twitter POS Tagging](http://aclweb.org/anthology/D18-1275)







