import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from transformers import BertTokenizer
from plots import *
import torch


def tensorize_data(tokenized_X, wv_model, word_vector_size, longest_comment_size):
    """
    Tensorize data - change each token to the vector.

    :param tokenized_X: data as a list of tokens
    :param wv_model: trained word2vec model
    :param word_vector_size: int - size of the output word vector
    :param longest_comment_size: int - size of the longest comment in tokenized_X
    :return: tensor of shape (#samples, sequence length, #features)
    """
    X_vectorized = []
    for text in tokenized_X:
        x_vectorized = []
        for index, s in enumerate(text):
            if index < 256:
                if s in wv_model.wv.key_to_index:
                    x_vectorized.extend(wv_model.wv[s])
                else:   # if word key is not in a dictionary add zeros
                    x_vectorized.extend([0] * word_vector_size)

        # pad or truncate comments
        pad_n = (word_vector_size * longest_comment_size) - len(x_vectorized)
        if pad_n >= 0:  # pad with zeros
            x_vectorized.extend([0] * pad_n)
        else:   # truncate
            x_vectorized = x_vectorized[:word_vector_size * longest_comment_size]
        X_vectorized.append(x_vectorized)

    X = np.reshape(np.array(X_vectorized), newshape=(len(X_vectorized), longest_comment_size, word_vector_size))
    return torch.tensor(X)


def prepare_key_padding_mask(tokenized_X, comment_lengths, longest_comment_size):
    """
    Create key padding mask, which is an input to Transformer model.

    :param tokenized_X: tokenized input samples
    :param comment_lengths: list of lengths of all samples after tokenization
    :param longest_comment_size: chosen longest size of the sample (samples ar padded or truncated to that size)
    :return: key padding mask as byte tensor
    """
    key_padding_mask = np.ones(shape=(len(tokenized_X), longest_comment_size))
    for i, length in enumerate(comment_lengths):
        if length > longest_comment_size:
            key_padding_mask[i] = np.zeros(longest_comment_size)
        else:
            key_padding_mask[i, :length] = np.zeros(length)
    key_padding_mask = torch.tensor(key_padding_mask).byte()
    return key_padding_mask


def preprocess_hate_speech(filepath):
    """
    Preprocess hate speech data from CSV and split them to train and test dataset.
    1 - hate speech, 0 – no hate speech.

    :param filepath: string path to the file with data
    :return: data in the shape of (X, y, mask)
    """
    data = pd.read_csv(filepath)

    X = data["tweet"]
    y = data["class"]

    # make target label binary
    X = list(map(lambda s: str(s), X))
    y = list(map(lambda c: [1, 0] if c == 2 else [0, 1], y))  # one-hot encode labels

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_X = [bert_tokenizer.tokenize(x) for x in X]

    comment_lengths = [len(comment) for comment in tokenized_X]
    # plot_histogram(comment_lengths, 'length', 'count', 30)  # plot histogram of comment lengths
    longest_comment_size = 256   # not actually the longest, chosen by hand, to encompass most of samples (equal to sequence size)

    # min_count: ignores all words with total absolute frequency lower than this
    # window: maximum distance between the current and predicted word within a sentence
    # vector_size: dimensionality of the word output vectors
    word_vector_size = 32
    model = Word2Vec(min_count=5, vector_size=word_vector_size, window=5)
    model.build_vocab(tokenized_X)
    model.train(tokenized_X, total_examples=model.corpus_count, epochs=1000)  # train word vectors

    X_tensorized = tensorize_data(tokenized_X, model, word_vector_size, longest_comment_size)
    y_tensorized = torch.tensor(y)

    # prepare key padding mask
    key_padding_mask = prepare_key_padding_mask(tokenized_X, comment_lengths, longest_comment_size)

    return X_tensorized, y_tensorized, key_padding_mask


def preprocess_IMDB_reviews(filepath):
    """
    Preprocess hate speech data from CSV and split them to train and test dataset.
    1 - positive sentiment, 0 – negative sentiment.

    :param filepath: string path to the file with data
    :return: data in the shape of (X_train, y_train, X_test, y_test)
    """
    data = pd.read_csv(filepath)

    X = data["review"]
    y = data["sentiment"]

    # make target label binary integer
    X = list(map(lambda s: str(s), X))
    y = list(map(lambda sentiment: [1, 0] if sentiment == 'negative' else [0, 1], y))    # one-hot encode labels

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_X = [bert_tokenizer.tokenize(x) for x in X]

    comment_lengths = [len(comment) for comment in tokenized_X]
    # plot_histogram(comment_lengths, 'length', 'count', 30)  # plot histogram of comment lengths
    longest_comment_size = 256  # not actually the longest, chosen by hand, to encompass most of samples (equal to sequence size)

    # min_count: ignores all words with total absolute frequency lower than this
    # window: maximum distance between the current and predicted word within a sentence
    # vector_size: dimensionality of the word output vectors
    word_vector_size = 32
    model = Word2Vec(min_count=5, vector_size=word_vector_size, window=5)
    model.build_vocab(tokenized_X)
    model.train(tokenized_X, total_examples=model.corpus_count, epochs=1000)  # train word vectors

    X_tensorized = tensorize_data(tokenized_X, model, word_vector_size, longest_comment_size)
    y_tensorized = torch.tensor(y)

    # prepare key padding mask
    key_padding_mask = prepare_key_padding_mask(tokenized_X, comment_lengths, longest_comment_size)

    return X_tensorized, y_tensorized, key_padding_mask


def preprocess_SMS_spam(filepath):
    """
    Preprocess SMS spam data from CSV and split them to train and test dataset.
    1 - spam, 0 – not spam.

    :param filepath: string path to the file with data
    :return: data in the shape of (X_train, y_train, X_test, y_test)
    """
    data = pd.read_csv(filepath, encoding='latin-1')

    X = data["v2"]
    y = data["v1"]

    # make target label binary integer
    X = list(map(lambda s: str(s), X))
    y = list(map(lambda c: [1, 0] if c == 'ham' else [0, 1], y))   # one-hot encode labels

    bert_tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    tokenized_X = [bert_tokenizer.tokenize(x) for x in X]

    comment_lengths = [len(comment) for comment in tokenized_X]
    # plot_histogram(comment_lengths, 'length', 'count', 30)  # plot histogram of comment lengths
    longest_comment_size = 256  # not actually the longest, chosen by hand, to encompass most of samples (equal to sequence size)

    # min_count: ignores all words with total absolute frequency lower than this
    # window: maximum distance between the current and predicted word within a sentence
    # vector_size: dimensionality of the word output vectors
    word_vector_size = 32
    model = Word2Vec(min_count=5, vector_size=word_vector_size, window=5)
    model.build_vocab(tokenized_X)
    model.train(tokenized_X, total_examples=model.corpus_count, epochs=1000)  # train word vectors

    X_tensorized = tensorize_data(tokenized_X, model, word_vector_size, longest_comment_size)
    y_tensorized = torch.tensor(y)

    # prepare key padding mask
    key_padding_mask = prepare_key_padding_mask(tokenized_X, comment_lengths, longest_comment_size)

    return X_tensorized, y_tensorized, key_padding_mask



