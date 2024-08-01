import torch
import torch.nn as nn

def create_layer(in_size, out_size, activation_function=nn.ReLU, p=0.25):
    return nn.Sequential(
        nn.Dropout(p=p),
        nn.Linear(in_size, out_size),
        activation_function()
    )


class DNN(nn.Module):
    def __init__(self, input_layer, hidden_layers, output_layer, activation_function=nn.ReLU, p=0.25):
        super().__init__()
        
        self.input_layer = nn.Linear(input_layer, hidden_layers[0])
        self.act1 = activation_function()
        self.dropout1 = nn.Dropout(p=p)
        
        layers = [create_layer(hl_in, hl_out) for hl_in, hl_out in zip(hidden_layers, hidden_layers[1:])]
        
        self.hidden_layers = nn.Sequential(*layers)
        
        self.output_layer = nn.Linear(hidden_layers[-1], output_layer)
        
    def forward(self, x):
        x = self.dropout1(x)
        x = self.act1(self.input_layer(x))
        x = self.hidden_layers(x)
        x = self.output_layer(x)
        return x
