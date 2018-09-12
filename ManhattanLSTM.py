from time import time
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import gensim
import re
import numpy as np
import itertools  
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.preprocessing.sequence import pad_sequences
from tensorflow.python.keras.models import Model, Sequential
from tensorflow.python.keras.layers import Layer, Input, Embedding, LSTM, Dense, Dropout

# File paths
TRAIN_CSV = 'input/features_train.csv'
TEST_CSV = 'input/features_test.csv'
EMBEDDING_FILE = 'input/GoogleNews-vectors-negative300.bin.gz'
MODEL_SAVING_DIR = 'input/'

# Load training and test set
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
stops = set(stopwords.words('english'))

# Setup pre-trained word2vec model
model = gensim.models.KeyedVectors.load_word2vec_format('input/GoogleNews-vectors-negative300.bin.gz', binary=True, limit=500000)
model.save('word2vec')


# Create embedding matrix
def text_to_word_list(text):
    text = str(text)
    text = text.lower()

    # Clean the text
    text = re.sub(r"[^A-Za-z0-9^,!.\/'+-=]", " ", text)
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r",", " ", text)
    text = re.sub(r"\.", " ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\/", " ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"\-", " - ", text)
    text = re.sub(r"\=", " = ", text)
    text = re.sub(r"'", " ", text)
    text = re.sub(r"(\d+)(k)", r"\g<1>000", text)
    text = re.sub(r":", " : ", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" b g ", " bg ", text)
    text = re.sub(r" u s ", " american ", text)
    text = re.sub(r"\0s", "0", text)
    text = re.sub(r" 9 11 ", "911", text)
    text = re.sub(r"e - mail", "email", text)
    text = re.sub(r"j k", "jk", text)
    text = re.sub(r"\s{2,}", " ", text)

    text = text.split()

    return text

# Prepare embedding
vocabulary = dict()
inverse_vocabulary = ['<na>']  #placeholder for the [0, 0, ....0] embedding
questions_cols = ['question1', 'question2']

# Iterate through the text of both questions in each dataset, converting string to vector representation
for dataset in [train_df, test_df]:
    for index, row in dataset.iterrows():
        for question in questions_cols:
            question_to_vec = []  

            for word in text_to_word_list(row[question]):
                if word in stops and word not in word2vec.vocab:
                    continue

                if word not in vocabulary:
                    vocabulary[word] = len(inverse_vocabulary)
                    question_to_vec.append(len(inverse_vocabulary))
                    inverse_vocabulary.append(word)

                else:
                    question_to_vec.append(vocabulary[word])

            # Replace questions as with the new vector representation
            dataset.set_value(index, question, question_to_vec)
            
embedding_dim = 300
embeddings = 1 * np.random.randn(len(vocabulary) + 1, embedding_dim)  # embedding matrix
embeddings[0] = 0  # ignore padding

# Build the embedding matrix
for word, index in vocabulary.items():
    if word in word2vec.vocab:
        embeddings[index] = word2vec.word_vec(word)

del word2vec


# Prepare training and validation data

max_seq_length = max(train_df.question1.map(lambda x: len(x)).max(),
                     train_df.question2.map(lambda x: len(x)).max(),
                     test_df.question1.map(lambda x: len(x)).max(),
                     test_df.question2.map(lambda x: len(x)).max())

# Split to train validation
validation_size = 40000
training_size = len(train_df) - validation_size

X = train_df[questions_cols]
Y = train_df['is_duplicate']

X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y, test_size=validation_size)

# Add in features
train_nlp_features = pd.read_csv("input/nlp_features_train.csv")
test_nlp_features = pd.read_csv("input/nlp_features_test.csv")

# Split to LR dictionaries
X_train = {'left': X_train.question1, 'right': X_train.question2}
X_validation = {'left': X_validation.question1, 'right': X_validation.question2}
X_test = {'left': test_df.question1, 'right': test_df.question2}

# Convert labels to their numpy representations
Y_train = Y_train.values
Y_validation = Y_validation.values

# Zero padding
for dataset, side in itertools.product([X_train, X_validation], ['left', 'right']):
    dataset[side] = pad_sequences(dataset[side], maxlen=max_seq_length)

# Ensure all steps applied properly
assert X_train['left'].shape == X_train['right'].shape
assert len(X_train['left']) == len(Y_train)


# Build the model

# Custom Keras layer for calculating Manhattan Distance
class ManhattanDist(Layer):
    """
    Custom Keras layer to calculate Manhattan Distance.
    """

    def __init__(self, **kwargs):
        self.result = None
        super(ManhattanDist, self).__init__(**kwargs)

    # Automatically collect input shapes to build layer
    def build(self, input_shape):
        super(ManhattanDist, self).build(input_shape)

    # Calculate Manhattan distance
    def call(self, x, **kwargs):
        self.result = K.exp(-K.sum(K.abs(x[0] - x[1]), axis=1, keepdims=True))
        return self.result

    # Return output shape
    def compute_output_shape(self, input_shape):
        return K.int_shape(self.result)

# Model variables
batch_size = 1024 
n_epoch = 50
n_hidden = 50

# Define the shared model
x = Sequential()
x.add(Embedding(len(embeddings), embedding_dim, weights=[embeddings], input_shape=(max_seq_length,), trainable=False))

x.add(LSTM(n_hidden))

shared_model = x

# The visible layer
left_input = Input(shape=(max_seq_length,), dtype='int32')
right_input = Input(shape=(max_seq_length,), dtype='int32')

# Use Manhattan Distance model
malstm_distance = ManhattanDist()([shared_model(left_input), shared_model(right_input)])
model = Model(inputs=[left_input, right_input], outputs=[malstm_distance])

model.compile(loss='mean_squared_error', optimizer=tf.keras.optimizers.Adam(), metrics=['accuracy'])
model.summary()
shared_model.summary()

# Start training
malstm_trained = model.fit([X_train['left'], X_train['right']], Y_train,
                           batch_size=batch_size, epochs=n_epoch,
                           validation_data=([X_validation['left'], X_validation['right']], Y_validation))

model.save('input/SiameseLSTM.h5')
