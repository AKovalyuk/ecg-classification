import numpy as np
import torch.nn as nn
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


def test(net, samples):
    with torch.no_grad():
        return net.forward(torch.Tensor(samples))


class ConvNet(nn.Module):
    def __init__(
            self,
            fc_layers_size,
            conv_layers
    ) -> None:
        super(ConvNet, self).__init__()
        layers = []
        in_channels = 12
        conv_out_size = 1000
        for kernel_size, out_channels, stride in conv_layers:
            layers += [
                nn.Conv1d(in_channels, out_channels, kernel_size, stride),
                nn.ReLU(),
                nn.MaxPool1d(2, 2)
            ]
            in_channels = out_channels
            # for Conv1d
            conv_out_size = self.conv_eval(conv_out_size, kernel_size, stride, 1)
            # for MaxPool1d
            conv_out_size = self.conv_eval(conv_out_size, 2, 2, 1)
        layers.append(nn.Flatten())
        features_in = conv_out_size * out_channels
        for fc_out in fc_layers_size[:-1]:
            layers += [
                nn.Linear(features_in, fc_out),
                nn.ReLU()
            ]
            features_in = fc_out
        layers += [
            nn.Linear(features_in, fc_layers_size[-1]),
            nn.Sigmoid()
        ]
        self.seq = nn.Sequential(*layers)

    @classmethod
    def conv_eval(self, len_in, kernel_size, stride, dilation):
        return int((len_in - 1 - dilation * (kernel_size - 1)) / stride + 1)

    def forward(self, x):
        return self.seq(x)

    def train_net(
            self,
            criterion,
            optimizer,
            objects,
            labels,
            epochs,
            batch_size
    ):
        objects = torch.Tensor(objects)
        labels = torch.Tensor(np.array([labels]).T)
        dataset = TensorDataset(objects, labels)
        dataloader = DataLoader(dataset, batch_size, shuffle=True)

        for epoch in range(epochs):
            loss_value = 0.0
            for i, batch in enumerate(dataloader):
                inputs, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
                optimizer.zero_grad()
                outputs = self(inputs)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                loss_value += loss.item()

    def test(self, samples):
        with torch.no_grad():
            return self.forward(torch.Tensor(samples))
