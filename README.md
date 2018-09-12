# Quora Question Duplicates

Quora question duplicate detection in Keras based off off the paper ["Siamese Recurrent Architectures for Learning Sentence Similarity".](http://www.mit.edu/~jonasm/info/MuellerThyagarajan_AAAI16.pdf)

## Prerequisites
- Python 3
- Install necessary packages (gensim, tensorflow, pandas, numpy)
- Download [word2vec](https://code.google.com/archive/p/word2vec/) model from Google
- Download [Quora datasets](https://www.kaggle.com/c/quora-question-pairs/data)
- Place the downloads in a folder called input

## Todo
Pre-processing:
- augment data with spell-checking
- augment data with thesauraus 

Embeddings:
- train embeddings on questions

Model:
- pretrain LSTM to choose better weights
