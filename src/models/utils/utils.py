from zipfile import ZipFile, ZIP_DEFLATED
import json
import os
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from models import FFNN, MyModel, DREBIN, SecSVM
from sklearn.metrics import roc_curve, auc, confusion_matrix, f1_score, \
    precision_score

__all__ = ["load_features", "load_labels", "load_sha256_list", "plot_roc",
           "get_metrics", "load_samples_features", "parse_model"]

def parse_model(classifier_str: str, n_features=1461078, vocab=None, use_RS=False):
    '''
    Returns a tuple of three elements:
        The classifier
        The path to the classifier
        The path to the vectorizer
    '''
    print(f"parsing model {classifier_str}")

    model_base_path = os.path.join(os.path.dirname(__file__), "../../..")
    file_extension = "pth" if "FFNN" in classifier_str else "pkl"
    clf_path = os.path.join(model_base_path, f"pretrained/{classifier_str}_classifier.{file_extension}")
    vect_path = os.path.join(model_base_path, f"pretrained/{classifier_str}_vectorizer.pkl")

    if classifier_str == "MyModel":
        classifier = MyModel()
    elif classifier_str == "secsvm":
        classifier = SecSVM(C=0.1, lb=-0.5, ub=0.5)
    elif classifier_str == "drebin":
        classifier = DREBIN(C=0.1)
    elif "FFNN" in classifier_str:
        aux = classifier_str.split("_")
        training = aux[1]
        structure = aux[2]
        cel = True if "CEL" in classifier_str else False
        cel_pos_class = float(aux[3][3:5])/10 if cel else 0
        cel_neg_class = float(aux[3][5:])/10 if cel else 0
        dense = True if "dense" in classifier_str else False
        classifier = FFNN(training=training, structure=structure, use_CEL=cel,
                      CEL_weight_pos_class=cel_pos_class,
                      CEL_weight_neg_class=cel_neg_class, dense=dense,
                      n_features=n_features, vocabulary=vocab,
                      use_rand_smoothing=use_RS)
    else:
        raise ValueError(f"Error: {classifier_str} does not exist!")
    return (classifier, clf_path, vect_path)


def load_samples_features(features_path, labels_path, type_of_ware):
    """
    Parameters
    ----------
    features_path : str
        Absolute path of the features compressed file.
    labels_path : str
        Absolute path of the data file (json or compressed csv) containing
        the labels.
    type_of_ware : int
        If 0, then return all goodware samples, if 1, return all malware ones.
    ----------
    The training data is not divided with positive class on one file
    and negative on the other.
    """
    with ZipFile(labels_path, "r", ZIP_DEFLATED) as z:
        ds_csv = pd.concat(
            [pd.read_csv(z.open(f))[["sha256", "label"]]
             for f in z.namelist()], ignore_index=True)
        labels_json = {k: v for k, v in zip(ds_csv.sha256.values,
                                            ds_csv.label.values)}
    with ZipFile(features_path, "r", ZIP_DEFLATED) as z:
        for filename in z.namelist():
            if labels_json[filename.split(".")[0].lower()] == type_of_ware:
                with z.open(filename) as fp:
                    js = json.load(fp)
                    yield [f"{k}::{v}" for k in js for v in js[k] if js[k]]


def load_features(features_path):
    """

    Parameters
    ----------
    features_path :
        Absolute path of the features compressed file.

    Returns
    -------
    generator of list of strings
        Iteratively returns the textual feature vector of each sample.
    """
    with ZipFile(features_path, "r", ZIP_DEFLATED) as z:
        for filename in z.namelist():
            with z.open(filename) as fp:
                js = json.load(fp)
                yield [f"{k}::{v}" for k in js for v in js[k] if js[k]]


def load_labels(features_path, ds_data_path, i=1):
    """

    Parameters
    ----------
    features_path : str
        Absolute path of the features compressed file.
    ds_data_path : str
        Absolute path of the data file (json or compressed csv) containing
        the labels.
    i : int
        If a json file is provided, specify the index to select.

    Returns
    -------
    np.ndarray
        Array of shape (n_samples,) containing the class labels.
    """
    if ds_data_path.endswith(".json"):
        with open(ds_data_path, "r") as f:
            labels_json = {k: v for k, v in json.load(f)[i].items()}
    else:
        with ZipFile(ds_data_path, "r", ZIP_DEFLATED) as z:
            ds_csv = pd.concat(
                [pd.read_csv(z.open(f))[["sha256", "label"]]
                 for f in z.namelist()], ignore_index=True)
            labels_json = {k: v for k, v in zip(ds_csv.sha256.values,
                                                ds_csv.label.values)}

    with ZipFile(features_path, "r", ZIP_DEFLATED) as z:
        labels = [labels_json[f.split(".json")[0].lower()]
                  for f in z.namelist()]
    return np.array(labels)


def load_sha256_list(features_path):
    """

    Parameters
    ----------
    features_path :
        Absolute path of the features compressed file.

    Returns
    -------
    list of strings
        List containing the sha256 hash of the APK files.
    """
    with ZipFile(features_path, "r", ZIP_DEFLATED) as z:
        return [filename.split(".")[0] for filename in z.namelist()]


def plot_roc(y_true, scores, img_path="", title="Roc"):
    fpr, tpr, th = roc_curve(y_true, scores)
    roc_auc = auc(fpr, tpr)

    plt.semilogx(fpr, tpr, color="darkorange", lw=2,
                 label=f"AUC = {roc_auc:0.2f}")
    plt.axvline(fpr[np.argmin(np.abs(th))], color="k", linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    plt.legend(loc="lower right")
    if img_path > "":
        plt.savefig(img_path)
    plt.show()
    plt.clf()


def get_metrics(y_true, y_pred):

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    print(f"F1 Score: {f1_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred):.4f}")
    print(f"TPR (aka Recall): {tp / (tp + fn):.4f}")
    print(f"FPR: {fp / (fp + tn):.4f}")
