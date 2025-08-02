import torch.nn as nn
import torch
from control.Enums import TrainingMode4LSTMAE
from models.LSTMAE import Encoder as LSTMEncoder
from models.MLP import MLP2, MLP
from torch import Tensor


class LSTMDIS(nn.Module):
    def __init__(self, input_size, hidden_size, name="global;LSTMDIS") -> None:
        super(LSTMDIS, self).__init__()
        hidden_size = 32
        self.name = name
        self.encoder = LSTMEncoder(
            input_size=input_size,
            hidden_size=hidden_size,
            dropout=0,
            training_mode=TrainingMode4LSTMAE.UNCONDITIONED.value,
            encoder_only=True,
            layer_num=2,
        )
        self.fc1 = nn.Linear(hidden_size, 8)
        self.fc2 = nn.Linear(8, 2)
        nn.init.normal_(self.fc1.weight.data, mean=0, std=0.1)
        nn.init.normal_(self.fc2.weight.data, mean=0, std=0.1)
        self.activation = nn.ELU(0.2)

    def forward(self, x):
        # print(x)
        # print(torch.transpose(x, 0, 1))
        hidden: Tensor = self.encoder(x)
        # print(hidden.shape)
        # assert False
        # hidden = hidden.squeeze(0)

        # hidden = torch.std(hidden, dim=1, keepdim=True)
        # print(hidden)
        # assert False
        out = self.fc1(hidden)
        out = self.activation(out)
        out = self.fc2(out)
        # out = torch.sigmoid(out)
        # assert False
        return out

    def set_name(self, name):
        self.name = name
