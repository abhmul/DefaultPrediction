import torch.nn as nn
import torch.nn.functional as F

from pyjet.layers import FullyConnected
from pyjet.models import SLModel


class SimpleModel(SLModel):
    def __init__(self, input_size, output_size=1, hidden_size=128):
        super(SimpleModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = FullyConnected(
            input_size,
            hidden_size,
            activation='relu',
            # batchnorm=True,
            input_dropout=0.05,
            num_layers=2
            # dropout=0.2
        )
        self.fc2 = FullyConnected(hidden_size, output_size)

        self.add_loss(F.binary_cross_entropy_with_logits)

    def forward(self, x):
        x = self.fc1(x)
        self.loss_in = self.fc2(x)
        return F.sigmoid(self.loss_in)


class SNNModel(SLModel):
    def __init__(self,
                 input_size,
                 output_size=1,
                 hidden_size=128,
                 num_hidden=16):
        super(SNNModel, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size

        self.fc1 = FullyConnected(
            input_size,
            hidden_size,
            activation='selu',
            input_dropout=0.05)
        # hidden_layers = [nn.AlphaDropout(p=0.05)]
        hidden_layers = []
        for _ in range(num_hidden):
            hidden_layers.append(
                FullyConnected(
                    hidden_size, hidden_size, activation='selu'))
            # hidden_layers.append(nn.AlphaDropout(p=0.05))
        self.hidden_fc = nn.ModuleList(hidden_layers)
        self.fc2 = FullyConnected(hidden_size, output_size)

        self.add_loss(F.binary_cross_entropy_with_logits)

    def forward(self, x):
        x = self.fc1(x)
        for i in range(0, len(self.hidden_fc), 2):
            inp = x
            x = self.hidden_fc[i](inp)
            x = self.hidden_fc[i+1](x)
            # Residual connectino
            x = (x + inp) / 2.
        self.loss_in = self.fc2(x)
        return F.sigmoid(self.loss_in)
