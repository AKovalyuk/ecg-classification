from typing import List

import numpy as np
import torch
from sklearn.metrics import accuracy_score
from torch import optim
import random

from .chromosome import Chromosome
from .enviroment import PopulationEnvironment
from ..model import ConvNet


# from src.preprocessing import preprocess_data


class Evolution:
    def __init__(self, x_train, y_train, x_test, y_test):
        self.means = []
        self.maxes = []
        self.mines = []

        self._env = PopulationEnvironment()
        self._population: List[Chromosome] = self._env.generate_population()
        self._device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self._metrics = []

        self._x_train = x_train
        self._y_train = y_train
        self._x_test = x_test
        self._y_test = y_test

    def fit(self, n_epoch: int):
        for epoch in range(n_epoch):
            best_parents = self._selection()

            metrics = self._metrics.copy()
            self.maxes.append(max(self._metrics))
            self.mines.append(min(self._metrics))
            mean = self._get_generation_metric()
            self.means.append(mean)

            print(f'Epoch {epoch}')
            for chromosome in self._population:
                print(f'\t\t{chromosome.genes_data}, Epoch: {chromosome.epoch}')
            print(f'\t Metrics {metrics}')
            print(f'\t Mean metric: {mean}')
            print('\n')

            self._generate_new_population(best_parents)
            self._make_mutation()

    def _selection(self) -> List[Chromosome]:
        self._fitness_calculation()
        return self._select()

    def _generate_new_population(self, parents: List[Chromosome]):
        mothers = parents[:len(parents) // 2]
        fathers = parents[len(parents) // 2:]

        new_population = []
        for mother in mothers:
            for father in fathers:
                child = mother.point_crossover(father)
                new_population.append(child)
        new_population.append(self._env.generate_chromosome())
        self._population = new_population

    def _make_mutation(self):
        if random.random() > self._env.mutation_chance:
            chromosome = random.choice(self._population)
            chromosome.make_mutation()

    def _get_generation_metric(self) -> float:
        res = sum(self._metrics) / len(self._metrics)
        self._metrics = []
        return res

    def _get_relative_chromosome_fitness(self) -> List[float]:
        relative_fitness = []
        total_fitness = self._get_total_fitness()
        for chromosome in self._population:
            relative_fit = chromosome.fitness / total_fitness
            relative_fitness.append(relative_fit)
        return relative_fitness

    def _select(self):
        next_population = []
        population = self._population.copy()
        relative_fitness = self._get_relative_chromosome_fitness()
        for _ in range(self._env.population - 1):
            chromosome = random.choices(population, relative_fitness)[0]
            chromosome_index = population.index(chromosome)
            population.pop(chromosome_index)
            relative_fitness.pop(chromosome_index)
            next_population.append(chromosome)
            # print(f'-- {population} --', f'||{relative_fitness}||')
        return next_population

    # def _select(self):
    #     population = self._population.copy()
    #     population.sort()
    #     return population[:len(population) - 2]

    def _get_total_fitness(self):
        return sum([chromosome.fitness for chromosome in self._population])

    def _fitness_calculation(self):
        for chromosome_idx in range(len(self._population)):
            while True:
                try:
                    fitness = self._calculate_fitness(self._population[chromosome_idx])
                    break
                except Exception:
                    print('WARNING')
                    self._population[chromosome_idx] = self._env.generate_chromosome()

            self._population[chromosome_idx].fitness = fitness

    def _calculate_fitness(self, chromosome: Chromosome) -> float:
        net = self._create_net(chromosome)
        self._net_to_device(net)

        try:
            self._fit_net(net, chromosome)
        except Exception:
            del net
            raise Exception()

        res = self._calculate_test_metric(net)
        self._metrics.append(res)
        del net
        return res

    def _create_net(self, chromosome: Chromosome) -> ConvNet:
        genomes_part = [tuple(x) for x in np.array_split(chromosome.genes_data, self._env.layers)]
        # print(genomes_part)
        return ConvNet(
            fc_layers_size=self._env.layers_size,
            conv_layers=genomes_part
        )

    def _net_to_device(self, net: ConvNet):
        net.to(self._device)

    def _fit_net(self, net: ConvNet, chromosome: Chromosome):
        criterion = self._env.criterion
        optimizer = optim.Adam(net.parameters(), lr=self._env.learn_rate)
        net.train_net(
            criterion=criterion,
            optimizer=optimizer,
            objects=self._x_train,
            labels=self._y_train,
            epochs=chromosome.epoch,
            batch_size=self._env.batch_size
        )

    def _calculate_test_metric(self, net: ConvNet) -> float:
        device = torch.device('cpu')
        return self._get_accuracy(net, device, self._x_test, self._y_test)

    @staticmethod
    def _test(net, samples):
        with torch.no_grad():
            return net.forward(torch.Tensor(samples))

    @staticmethod
    def _get_accuracy(net, device, x, y):
        return accuracy_score(
            y,  # .to(device, non_blocking=True),
            [(1 if pred[0] > 0.5 else 0) for pred in
             Evolution._test(net.to(device), torch.Tensor(x).to(device, non_blocking=True))]
        )

# x_train, x_test, y_train, y_test = preprocess_data(path='/home/twoics/py-proj/ecg-classification/plt')
# evolution = Evolution(x_train, y_train, x_test, y_test)
# evolution.fit(1000)
