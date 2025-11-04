from sklearn.linear_model import Ridge, Lasso
import torch
from torch import nn

def get_baseline_model(model_type='ridge'):
    if model_type == 'ridge':
        return Ridge(alpha=1.0)
    elif model_type == 'lasso':
        return Lasso(alpha=0.001)
    else:
        raise ValueError('model_type must be ridge or lasso')

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[128,64,32], dropout=0.2):
        super().__init__()
        layers = []
        dims = [input_dim] + hidden_dims
        for i in range(len(dims)-1):
            layers.append(nn.Linear(dims[i], dims[i+1]))
            layers.append(nn.BatchNorm1d(dims[i+1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(dims[-1], 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)
