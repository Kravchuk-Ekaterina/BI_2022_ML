import numpy as np


def binary_classification_metrics(y_pred, y_true):
    """
    Computes metrics for binary classification
    Arguments:
    y_pred, np array (num_samples) - model predictions
    y_true, np array (num_samples) - true labels
    Returns:
    precision, recall, f1, accuracy - classification metrics
    """
    TP = 0
    TN = 0
    FP = 0
    FN = 0
    for i in range(len(y_pred)):
        if int(y_pred[i]) == 1:
            if int(y_true[i]) == 1:
                TP += 1
            else:
                FP += 1
        else:
            if int(y_true[i]) == 0:
                TN += 1
            else:
                FN += 1
    precision = TP/(TP + FP)
    recall = TP/(TP + FN)
    f1 = 2*(precision*recall)/(precision + recall)
    accuracy = (TP + TN)/(TP + TN + FN + FP)
    return precision, recall, f1, accuracy


def multiclass_accuracy(y_pred, y_true):
    """
    Computes metrics for multiclass classification
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true labels
    Returns:
    accuracy - ratio of accurate predictions to total samples
    """
    T = 0
    ALL = len(y_pred)
    for i in range(ALL):
        if y_pred[i] == y_true[i]:
            T += 1
    accuracy = T/ALL
    return accuracy


def r_squared(y_pred, y_true):
    """
    Computes r-squared for regression
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    r2 - r-squared value
    """

    return 1 - np.sum((y_pred - y_true)**2)/np.sum((y_true - np.mean(y_true))**2)


def mse(y_pred, y_true):
    """
    Computes mean squared error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mse - mean squared error
    """

    return sum((y_pred - y_true)**2)/len(y_pred)


def mae(y_pred, y_true):
    """
    Computes mean absolut error
    Arguments:
    y_pred, np array of int (num_samples) - model predictions
    y_true, np array of int (num_samples) - true values
    Returns:
    mae - mean absolut error
    """

    return sum(np.absolute(y_pred - y_true))/len(y_pred)
