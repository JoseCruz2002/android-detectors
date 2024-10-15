from sklearn.utils._array_api import get_namespace
from models.base import BaseDREBIN
from sklearn.neural_network import MLPClassifier

class MyModel(BaseDREBIN, MLPClassifier):
    
    def __init__(self):
        BaseDREBIN.__init__(self)
        MLPClassifier.__init__(self,
                               hidden_layer_sizes=[10, 5],
                               activation='relu', 
                               solver='sgd',
                               batch_size=16)

    def _fit(self, X, y):
        MLPClassifier._fit(self, X, y)

    def predict(self, features):
        X = self._vectorizer.transform(features)
        labels = MLPClassifier.predict(self, X)
        #scores = self.decision_function(X)
        scores = MLPClassifier.predict_proba(self, X)
        scores_final = []
        for i in range(len(scores)):
            scores_final += [scores[i][labels[i]]]
        return labels, scores_final