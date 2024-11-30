
import torch
import torch.nn as nn
import numpy

import subprocess
import re
import time

import dill as pkl
from scipy.sparse import csr_matrix

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from models.base import BaseDREBIN
from models.FFNN import FeedForwardNN

class FFNN(BaseDREBIN):

    def __init__(self, training="normal", structure="small", use_CEL=False, CEL_weight_pos_class=0.1, 
                 CEL_weight_neg_class=0.9, dense=False, features=[]):
        '''
        features: iterable of iterables of strings
            Iterable of shape (n_samples, n_features) containing textual
            features in the format <feature_type>::<feature_name>.
            This is necessary to know the number of features for the model.
        '''
        print(f"Model parameters:\n\
              training: {training}\n\
              structure: {structure}\n\
              use_CEL: {use_CEL}\n\
                CEL_weight_pos_class: {CEL_weight_pos_class}\n\
                CEL_weight_neg_class: {CEL_weight_neg_class}\n\
              dense: {dense}")

        DEVICE = get_free_gpu() if torch.cuda.is_available() else "cpu"
        print(f"Putting everything in {DEVICE}")
        self.device = DEVICE
        
        BaseDREBIN.__init__(self)

        #self.n_samples, n_features = self._vectorizer.fit_transform(features).get_shape()
        #print(n_features)
        hidden_size, layers = (10, 2) if structure == "small" else (150, 3)
        self.model = FeedForwardNN.FeedForwardNN(n_classes=2, n_features=1461078,
                                                 hidden_size=hidden_size, layers=layers)
        self.model = self.model.to(self.device)

        self.optimizer = torch.optim.SGD(self.model.parameters(), 
                                         lr=0.01,
                                         weight_decay=0)
        
        if use_CEL:
            self.CEL = True
            self.CEL_weight_pos_class = CEL_weight_pos_class
            self.CEL_weight_neg_class = CEL_weight_neg_class
            weight_ = torch.tensor([CEL_weight_pos_class, CEL_weight_neg_class]).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weight_)
        else: 
            self.criterion = nn.CrossEntropyLoss()
        
        self.training = training
        self.structure = structure
        self.dense = dense
        self.batch_size = 30
        self.n_samples = 75000


    def _fit(self, X, y):
        print(f"Which cuda is the model in? {next(self.model.parameters()).device}")
        if self.training == "normal":
            self.normal_training(X, y)
        elif self.training == "ratioed":
            self.fifty_fifty_rationed_training(X, y)
        elif self.training == "only_mal":
            self.only_malware_training(X, y)
        else:
            print(f"ERROR!! No training method -{self.training}- exists!")

    def only_malware_training(self, X, y):
        print("Only malware training")
        self.n_samples //= 10 # only malware apks
        malware_pos = []
        for i in range(y.shape[0]):
            malware_pos += [i] if y[i] == 1 else []
        print(len(malware_pos))
        time.sleep(4)
        for batch in range(self.n_samples // self.batch_size):
            row_indices = malware_pos[batch*self.batch_size : (batch+1)*self.batch_size]
            input_ = X[row_indices]
            labels = numpy.array(list(1 for _ in range(self.batch_size)))
            print(f"shape of input = {input_.shape}")
            print(f"shape of labels = {labels.shape}")
            loss = self.train_batch(input_, labels)
            print(f"batch number = {batch}; has loss = {loss}")

    def fifty_fifty_rationed_training(self, X, y):
        print("Fifty-fifty training with class balancing")
        sm = SMOTE(sampling_strategy="minority", random_state=42, n_jobs=1)
        X_smote, y_smote = sm.fit_resample(X, y)
        X, y = self.stratified_downsample(X_smote, y_smote, self.n_samples)
        print(f"X_smote shape = {X.shape}")
        print(f"y_smote shape = {y.shape}")
        print(f"Amount of malware = {numpy.count_nonzero(y)}")
        print(f"Amount of goodware = {y.shape[0]-numpy.count_nonzero(y)}")
        time.sleep(4)
        self.normal_training(X, y)

    def stratified_downsample(self, X, y, n_samples):
        assert X.shape[0] == y.shape[0]
        print(n_samples / X.shape[0])
        X_train, _, y_train, _ = train_test_split(
                X, y, test_size=1 - (n_samples / X.shape[0]), random_state=42)
        return X_train, y_train
    
    def normal_training(self, X, y):
        print("Normal training")
        print(X[:2])
        print(y[:2])
        time.sleep(4)
        for batch in range(self.n_samples // self.batch_size):
            input_ = X[batch*self.batch_size : (batch+1)*self.batch_size, :]
            labels = y[batch*self.batch_size : (batch+1)*self.batch_size]
            print(f"shape of input = {input_.shape}")
            print(f"shape of labels = {labels.shape},\nlabels: {labels}")
            loss = self.train_batch(input_, labels)
            print(f"batch number = {batch}; has loss = {loss}")

    def train_batch(self, X, y, **kwargs):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        model: a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
        """
        assert X.shape[0] == y.shape[0]
        self.model.train()

        X_tensor = csr_matrix_to_sparse_tensor(X)
        if self.dense:
            X_tensor = X_tensor.to_dense()
        X_tensor = X_tensor.to(self.device)
        y_tensor = torch.Tensor(y).type(torch.LongTensor)
        y_tensor = y_tensor.to(self.device)

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
        self.model.eval()
        X = self._vectorizer.transform(features)
        X_tensor = csr_matrix_to_sparse_tensor(X).to(self.device)
        scores = torch.nn.functional.softmax(self.model(X_tensor), dim=1)
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

        self.model.load_state_dict(torch.load(classifier_path, 
                                              weights_only=False, 
                                              map_location=torch.device(self.device)))
        self.model.to(self.device)

        with open(vectorizer_path, "rb") as f:
            self._vectorizer = pkl.load(f)

        return self

    def set_input_features(self, features):
        #print(f"input_features = {self.input_features}")
        if self.input_features == None:
            self._vectorizer.transform(features)
            self._input_features = (self._vectorizer.get_feature_names_out().tolist())
        
    def toString(self):
        CEL_str = "CEL" + str(self.CEL_weight_pos_class).replace(".", "") + \
                str(self.CEL_weight_neg_class).replace(".", "") if self.CEL else ""
        dense_str = "dense" if self.dense else ""
        return f"FFNN_{self.training}_{self.structure}_{CEL_str}_{dense_str}"


def csr_matrix_to_sparse_tensor(csr_matrix):
    coo_matrix = csr_matrix.tocoo()
    values = coo_matrix.data
    indices = numpy.vstack((coo_matrix.row, coo_matrix.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo_matrix.shape
    return torch.sparse_coo_tensor(i, v, torch.Size(shape))

def get_free_gpu():
    try:
        # Run nvidia-smi and get the GPU memory usage
        nvidia_smi_output = subprocess.check_output(["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,nounits,noheader"]).decode("utf-8")
        # Parse the output
        memory_usage = [line.strip().split(",") for line in nvidia_smi_output.strip().split("\n")]
        memory_usage = [(int(used), int(total)) for used, total in memory_usage]

        # Find GPU with maximum available memory
        free_memory = [(total - used, i) for i, (used, total) in enumerate(memory_usage)]
        best_gpu = max(free_memory, key=lambda x: x[0])[1]
        return best_gpu
    except Exception as e:
        print("Error finding the best GPU:", e)
        return None
