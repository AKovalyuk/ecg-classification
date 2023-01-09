from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import optim

from src.genetic.chromosome import Chromosome
from src.genetic.enviroment import PopulationEnvironment
from src.model import ConvNet
from src.preprocessing import preprocess_data


class Evolution:
    def __init__(self, x_train, y_train, x_test, y_test):
        self._env = PopulationEnvironment()
        self._population: List[Chromosome] = self._env.generate_population()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    def fitness_calculation(self):
        return self._calculate_fitness(self._population[0])

    def selection(self) -> list:
        pass

    def point_crossover(self, mather: Chromosome, father: Chromosome) -> Chromosome:
        pass

    def _calculate_fitness(self, chromosome: Chromosome) -> float:
        net = self._create_net(chromosome)
        self._net_to_device(net)
        self._fit_net(net)
        res = self._calculate_train_metric(net)
        del net
        return res

    def _create_net(self, chromosome: Chromosome) -> ConvNet:
        genomes_part = [tuple(x) for x in np.array_split(chromosome.genes_data, self._env.layers)]
        print(genomes_part)
        return ConvNet(
            fc_layers_size=self._env.layers_size,
            conv_layers=genomes_part
        )

    def _net_to_device(self, net: ConvNet):
        net.to(self._device)

    def _fit_net(self, net: ConvNet):
        criterion = self._env.criterion
        optimizer = optim.Adam(net.parameters(), lr=self._env.learn_rate)
        net.train_net(
            criterion=criterion,
            optimizer=optimizer,
            objects=self._x_train,
            labels=self._y_train,
            epochs=self._env.epoch,
            batch_size=self._env.batch_size
        )

    @staticmethod
    def test(net, samples):
        with torch.no_grad():
            return net.forward(torch.Tensor(samples))

    def _calculate_train_metric(self, net: ConvNet) -> float:
        device = torch.device('cpu')
        return self._get_accuracy(net, device, self._x_train, self._y_train)

    def _calculate_test_metric(self, net: ConvNet):
        pass

    def _get_accuracy(self, net, device, x, y):
        return accuracy_score(
            y,  # .to(device, non_blocking=True),
            [(1 if pred[0] > 0.5 else 0) for pred in
             Evolution.test(net.to(device), torch.Tensor(x).to(device, non_blocking=True))]
        )


x_train, x_test, y_train, y_test = preprocess_data(path='/home/twoics/py-proj/ecg-classification/plt')
evolution = Evolution(x_train, y_train, x_test, y_test)
print(evolution.fitness_calculation())
