import torch
from torch.utils.data import DataLoader
import torch.nn as nn

class FeedForwardNN(nn.Module):

    def __init__(self, n_classes, n_features, hidden_size = 4, layers = 2, **kwargs):
        '''
        n_classes: two, one for malware and other for goodware
        n_features: size of input
        hidden_size: number of neurons in the hidden layers
        hidden_layers: number of hidden layers on the NN 
        '''
        super(FeedForwardNN, self).__init__()
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(n_features, hidden_size))
        for _ in range(layers-1):
            self.layers.append(nn.Linear(hidden_size, hidden_size))
        self.output_layer = nn.Linear(hidden_size, n_classes)
        self.activation = nn.functional.relu


    def forward(self, x, **kwargs):
        for layer in self.layers:
            x = self.activation(layer(x))
        x = self.output_layer(x)
        return x
    
    def train_batch(X, y, model, optimizer, criterion, **kwargs):
        """
        X (n_examples x n_features)
        y (n_examples): gold labels
        model: a PyTorch defined model
        optimizer: optimizer used in gradient step
        criterion: loss function
        """
        model.train()
        X_tensor = torch.Tensor(X)
        y_tensor = torch.Tensor(y)
        optimizer.zero_grad()
        outputs = model(X_tensor)
        loss = criterion(outputs, y_tensor)
        loss.backward()
        optimizer.step()
        return loss.item()