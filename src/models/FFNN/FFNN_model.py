
import torch
import torch.nn as nn
import numpy

import dill as pkl

from sklearn.feature_extraction.text import CountVectorizer
from models.base import BaseDREBIN
from models.FFNN import FeedForwardNN

class FFNN(BaseDREBIN):

    def __init__(self, features):
        '''
        features: iterable of iterables of strings
            Iterable of shape (n_samples, n_features) containing textual
            features in the format <feature_type>::<feature_name>.
            This is necessary to know the number of features for the model.
        '''
        
        BaseDREBIN.__init__(self)

        #self.n_samples, n_features = self._vectorizer.fit_transform(features).get_shape()
        #print(n_features)
        #FeedForwardNN(self, n_classes=2, n_features=n_features)
        self.model = FeedForwardNN(n_classes=2, n_features=1461078)
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Putting model in {DEVICE}")
        self.model.to(DEVICE)

        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr=0.01,
                                         weight_decay=0)
        self.criterion = nn.CrossEntropyLoss()
        self.batch_size = 300
        self.n_samples = 75000

    def _fit(self, X, y):
        print(f"X.shape = {X.shape}")
        for batch in range(self.n_samples // self.batch_size):
            loss = self.train_batch(X[batch*self.batch_size : batch*self.batch_size+self.batch_size, :],
                                    y[batch*self.batch_size : batch*self.batch_size+self.batch_size])
            print(f"batch number = {batch}; has loss = {loss}")
            return
    
    def train_batch(self, X, y, **kwargs):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        model: a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
        """
        self.model.train()

        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Putting tensors in {DEVICE}")
        X_tensor = csr_matrix_to_sparse_tensor(X)
        X_tensor.to(DEVICE)        
        y_tensor = torch.Tensor(y).type(torch.LongTensor)
        y_tensor.to(DEVICE)

        self.optimizer.zero_grad()
        outputs = self.model(X_tensor)
        loss = self.criterion(outputs, y_tensor)
        loss.backward()
        self.optimizer.step()
        return loss.item()  
    
    def predict(self, features):
        """
        Given the textual features of the samples to classify, returns the
        predicted labels and scores.

        Parameters
        ----------
        features: iterable of iterables of strings
            Iterable of shape (n_samples, n_features) containing textual
            features in the format <feature_type>::<feature_name>.

        Returns
        -------
        labels : numpy.ndarray
            Flat dense array of shape (n_samples,) with the label assigned
            to each test pattern. The classification label is the label of
            the class associated with the highest score.
        scores : numpy.ndarray
            Array of shape (n_samples,) with classification
            score of each test pattern with respect to the positive class.
        """
        X = self._vectorizer.transform(features)
        scores = self.model(csr_matrix_to_sparse_tensor(X))  # (n_examples x n_classes)
        predicted_labels = scores.argmax(dim=-1)  # (n_examples)
        return predicted_labels, scores.max(dim=1)[0]

    def classify(self, apk_list):
        """
        Given a list of APK file paths, extracts the features and classifies
        them, returning the predicted labels and scores.

        Parameters
        ----------
        apk_list : list of str
            List with the absolute path of each APK file to classify.

        Returns
        -------
        labels : numpy.ndarray
            Flat dense array of shape (n_samples,) with the label assigned
            to each test pattern. The classification label is the label of
            the class associated with the highest score.
        scores : numpy.ndarray
            Array of shape (n_samples,) with classification
            score of each test pattern with respect to the positive class.
        """
        return NotImplemented
    
    def save(self, vectorizer_path, classifier_path):

        with open(vectorizer_path, "wb") as f:
            pkl.dump(self.vectorizer, f)
        
        torch.save(self.model.state_dict(), classifier_path)

    def load(self, vectorizer_path, classifier_path):

        self.model.load_state_dict(torch.load(classifier_path, weights_only=True))

        with open(vectorizer_path, "rb") as f:
            self._vectorizer = pkl.load(f)
        return self


def csr_matrix_to_sparse_tensor(csr_matrix):
    coo_matrix = csr_matrix.tocoo()
    values = coo_matrix.data
    indices = numpy.vstack((coo_matrix.row, coo_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_matrix.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))