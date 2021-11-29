import numpy as np
import torch
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
from sklearn.metrics import average_precision_score


def load_data(dataset):
    """
    Load ELMo embeddings for 'dataset'.

    :param dataset: string name of the datast
    :return: X_train, y_train, X_test, y_test
    """
    X_train = np.load('ELMo_embeddings/%s/X_train_all.npy' % dataset, allow_pickle=True)
    y_train = np.load('ELMo_embeddings/%s/train_labels.npy' % dataset, allow_pickle=True)
    X_test = np.load('ELMo_embeddings/%s/X_test_all.npy' % dataset, allow_pickle=True)
    y_test = np.load('ELMo_embeddings/%s/test_labels.npy' % dataset, allow_pickle=True)
    return X_train, y_train, X_test, y_test


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_stats(outputs, y):
    """
    Get statistics for given model outputs and true predictions 'y'.

    :param outputs: list of model predictions, each element of the list has a size (batch_size, 2)
    :param y: tensor of true labels (size equals to number of samples)
    :return: (accuracy, AUROC, AUPRC)
    """
    outputs = torch.cat(outputs, dim=0)
    probs = torch.squeeze(torch.softmax(outputs, dim=1))
    probs = probs.cpu().detach().numpy()
    true = y.cpu().detach().numpy()
    auroc = roc_auc_score(true, probs[:, 1])
    auprc = average_precision_score(true, probs[:, 1])
    predicted = np.argmax(outputs.cpu().detach().numpy(), axis=1).ravel()
    acc = np.sum(true == predicted) / true.shape[0]
    return acc, auroc, auprc

