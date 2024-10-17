import torch
from torch.utils.data import DataLoader
import torch.nn as nn

class FeedForwardNN(nn.Module):

    def __init__(self, n_classes, n_features, hidden_size = 150, layers = 3, **kwargs):
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
