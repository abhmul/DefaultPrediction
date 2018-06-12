import torch.nn.functional as F

from pyjet.layers import FullyConnected
from pyjet.models import SLModel


class SimpleModel(SLModel):
    def __init__(self, input_size, output_size=1, hidden_size=128):
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = FullyConnected(
            input_size,
            hidden_size,
            activation='relu',
            batchnorm=True,
            input_dropout=0.2,
            dropout=0.5)
        self.fc2 = FullyConnected(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        self.loss_in = self.fc2(x)
        return F.sigmoid(self.loss_in)
