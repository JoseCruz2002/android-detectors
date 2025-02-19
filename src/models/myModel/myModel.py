from sklearn.utils._array_api import get_namespace
from models.base import BaseDREBIN
from sklearn.neural_network import MLPClassifier
import numpy

class MyModel(BaseDREBIN, MLPClassifier):
    
    def __init__(self):
        BaseDREBIN.__init__(self)
        self.MLP = MLPClassifier(hidden_layer_sizes=[10, 5],
                                 activation='relu', 
                                 solver='sgd',
                                 batch_size=16)

    def _fit(self, X, y, rand_smoothing=False, noise=0.0):
        self.MLP.fit(X, y)

    def predict(self, features):
        X = self._vectorizer.transform(features)
        labels = self.MLP.predict(X)
        #scores = self.decision_function(X)
        scores = self.MLP.predict_proba(X)
        scores_final = []
        for i in range(len(scores)):
            scores_final += [scores[i][labels[i]]]
        return labels, numpy.array(scores_final)
    
    def get_model(self):
        """Return the MLPClassifier instance for ONNX conversion."""
        return self.MLP