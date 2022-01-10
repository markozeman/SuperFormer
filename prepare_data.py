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
    Preprocess hate speech data from CSV.
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
    Preprocess hate speech data from CSV.
    1 - positive sentiment, 0 – negative sentiment.

    :param filepath: string path to the file with data
    :return: data in the shape of (X, y, mask)
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
    Preprocess SMS spam data from CSV.
    1 - spam, 0 – not spam.

    :param filepath: string path to the file with data
    :return: data in the shape of (X, y, mask)
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


def preprocess_sentiment_analysis(filepath):
    """
    Preprocess sentiment analysis data from two text files.
    1 - positive, 0 – negative.

    :param filepath: string path to the folder with data
    :return: data in the shape of (X, y, mask)
    """
    X = []
    y = []

    # read Amazon data
    with open(filepath + 'amazon_cells_labelled.txt') as f:
        lines = f.readlines()
        for line in lines:
            txt, label = line.split('\t')
            X.append(txt.strip())
            y.append(label.strip())

    # read Yelp data
    with open(filepath + 'yelp_labelled.txt') as f:
        lines = f.readlines()
        for line in lines:
            txt, label = line.split('\t')
            X.append(txt.strip())
            y.append(label.strip())

    # make target label binary integer
    y = list(map(lambda c: [1, 0] if c == '0' else [0, 1], y))  # one-hot encode labels

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


def preprocess_clickbait(filepath):
    """
    Preprocess clickbait data from two text files.
    1 - clickbait, 0 – not clickbait.

    :param filepath: string path to the folder with data
    :return: data in the shape of (X, y, mask)
    """
    X = []
    y = []

    # read clickbait data
    with open(filepath + 'clickbait_data') as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                X.append(line.strip())
                y.append(1)

    # read non-clickbait data
    with open(filepath + 'non_clickbait_data', encoding="utf8") as f:
        lines = f.readlines()
        for line in lines:
            if line != '\n':
                X.append(line.strip())
                y.append(0)

    # make target label binary integer
    y = list(map(lambda c: [1, 0] if c == 0 else [0, 1], y))  # one-hot encode labels

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


def preprocess_humor_detection(filepath):
    """
    Preprocess humor detection data from CSV file.
    1 - True (is humor), 0 – False (is not humor).

    :param filepath: string path to the folder with data
    :return: data in the shape of (X, y, mask)
    """
    data = pd.read_csv(filepath + 'dataset.csv')

    X = data["text"]
    y = data["humor"]

    # make target label binary integer
    X = list(map(lambda s: str(s), X))
    y = list(map(lambda c: [1, 0] if c is False else [0, 1], y))  # one-hot encode labels

    # take only the first 50.000 samples to make it fit into memory
    X = X[:50000]
    y = y[:50000]

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


def get_data(s):
    """
    Get data for string s abbreviation.

    :param s: string of dataset abbreviation name
    :return: X, y, mask
    """
    if s == 'HS':
        X = torch.load('Word2Vec_embeddings/X_hate_speech.pt').float()
        y = torch.load('Word2Vec_embeddings/y_hate_speech.pt')
        mask = torch.load('Word2Vec_embeddings/mask_hate_speech.pt').float()
    elif s == 'SA':
        X = torch.load('Word2Vec_embeddings/X_IMDB_sentiment_analysis.pt').float()
        y = torch.load('Word2Vec_embeddings/y_IMDB_sentiment_analysis.pt')
        mask = torch.load('Word2Vec_embeddings/mask_IMDB_sentiment_analysis.pt').float()
    elif s == 'S':
        X = torch.load('Word2Vec_embeddings/X_sms_spam.pt').float()
        y = torch.load('Word2Vec_embeddings/y_sms_spam.pt')
        mask = torch.load('Word2Vec_embeddings/mask_sms_spam.pt').float()
    elif s == 'SA_2':
        X = torch.load('Word2Vec_embeddings/X_sentiment_analysis_2.pt').float()
        y = torch.load('Word2Vec_embeddings/y_sentiment_analysis_2.pt')
        mask = torch.load('Word2Vec_embeddings/mask_sentiment_analysis_2.pt').float()
    elif s == 'C':
        X = torch.load('Word2Vec_embeddings/X_clickbait.pt').float()
        y = torch.load('Word2Vec_embeddings/y_clickbait.pt')
        mask = torch.load('Word2Vec_embeddings/mask_clickbait.pt').float()
    elif s == 'HD':
        X = torch.load('Word2Vec_embeddings/X_humor_detection.pt').float()
        y = torch.load('Word2Vec_embeddings/y_humor_detection.pt')
        mask = torch.load('Word2Vec_embeddings/mask_humor_detection.pt').float()

    return X, y, mask

