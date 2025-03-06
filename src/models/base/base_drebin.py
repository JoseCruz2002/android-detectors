from sklearn.feature_extraction.text import CountVectorizer
from models.base import BaseModel
import dill as pkl
from feature_extraction import DREBINFeatureExtractor
from feature_selection import f_sel_methods
import logging
import json
import torch

class BaseDREBIN(BaseModel):
    """
    Base class for any scikit-learn or secml classifier that can be trained on
    the DREBIN feature set.
    Features are parsed with a CountVectorizer, supporting both sparse and
    dense formats.
    It must be extended by implementing the _fit and predict methods.
    method.
    """

    def __init__(self, vocabulary=None):
        self._vectorizer = CountVectorizer(
            input="content", lowercase=False,
            tokenizer=lambda x: x, binary=True, token_pattern=None,
            vocabulary=vocabulary)
        self._feat_extractor = DREBINFeatureExtractor(
            logging_level=logging.ERROR)
        self._input_features = None

    def fit(self, features, y, fit=True, feat_sel=False, FS_args={}):
        """

        Parameters
        ----------
        features: iterable of iterables of strings
            Iterable of shape (n_samples, n_features) containing textual
            features in the format <feature_type>::<feature_name>.
        y : np.ndarray
            Array of shape (n_samples,) containing the class labels.
        """
        if fit:
            X = self._vectorizer.fit_transform(features)
        else:
            X = self._vectorizer.transform(features)
        self._input_features = (self._vectorizer.get_feature_names_out()
                                .tolist())

        if feat_sel:
            X, self._input_features = (f_sel_methods.feature_selection(X,
                                            self._input_features, y, FS_args))
            self._vectorizer = CountVectorizer(
                        input="content", lowercase=False,
                        tokenizer=lambda x: x, binary=True, token_pattern=None,
                        vocabulary=self._input_features)
            self._vectorizer.fit(features)
            self._input_features = self._input_features.tolist()

        print(f"shape of input: {X.shape}")
        print(f"size of input_features list: {len(self._input_features)}")
        #for el in self._input_features:
            #print(el)

        self._fit(X, y)

    def _fit(self, X, y):
        """

        Parameters
        ----------
        X: scipy sparse matrix
            Sparse matrix of shape (n_samples, n_features) containing the
            features.
        y : np.ndarray
            Array of shape (n_samples,) containing the class labels.
        """
        return NotImplemented
    
    def vectorizer_fit(self, X, transform=False):
        """
        Method to fit a model's vectorizer independently of actually training.
        """
        X_new = []
        if transform:
            X_new = self._vectorizer.fit_transform(X)
        else:
            self._vectorizer.fit(X)
        self._input_features = (self._vectorizer.get_feature_names_out()
                                .tolist())
        print(f"Number of input features: {len(self._input_features)}")
        return X_new

    def extract_features(self, apk_list):
        """

        Parameters
        ----------
        apk_list : list of str
            List with the absolute path of each APK file to classify.

        Returns
        -------
        iterable of iterables of strings
            Iterable of shape (n_samples, n_features) containing textual
            features in the format <feature_type>::<feature_name>.
        """
        return self._feat_extractor.extract_features(apk_list)

    def classify(self, apk_list):
        features = self.extract_features(apk_list)
        return self.predict(features)

    def save(self, vectorizer_path, classifier_path):
        """

        Parameters
        ----------
        vectorizer_path : str
        classifier_path : str
        """
        with open(vectorizer_path, "wb") as f:
            pkl.dump(self._vectorizer, f)
        vectorizer = self._vectorizer
        self._vectorizer = None
        with open(classifier_path, "wb") as f:
            pkl.dump(self, f)
        self._vectorizer = vectorizer

    @staticmethod
    def load(vectorizer_path, classifier_path):
        """

        Parameters
        ----------
        vectorizer_path : str
        classifier_path : str

        Returns
        -------
        BaseDREBIN
        """
        with open(classifier_path, "rb") as f:
            classifier = pkl.load(f)
        with open(vectorizer_path, "rb") as f:
            classifier._vectorizer = pkl.load(f)
        return classifier
    
    def set_input_features(self, features):
        #print(f"input_features = {self.input_features}")
        if self.input_features == None:
            self._vectorizer.transform(features)
            self._input_features = (self._vectorizer.get_feature_names_out().tolist())

    @property
    def input_features(self):
        return self._input_features
    
    @property
    def vectorizer(self):
        return self._vectorizer
