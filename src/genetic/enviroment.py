import random
from torch import nn
from typing import List
from dataclasses import dataclass
from chromosome import Chromosome, Gene


@dataclass
class LayerSettings:
    kernel_size: int
    out_channels: int
    step: int


class PopulationEnvironment:
    def __init__(self):
        self._step_threshold = 5
        self._kernel_threshold = 21
        self._out_threshold = 300

        self._layers_count = 3
        self._layers_size = [500, 1]
        self._criterion = nn.BCELoss()
        self._epoch = 5
        self._batch_size = 128
        self._learn_rate = 0.0001

        self._population_count = 5
        self._mutation_chance = 0.1

    @property
    def population(self):
        return self._population_count

    @property
    def learn_rate(self):
        return self._learn_rate

    @property
    def epoch(self):
        return self._epoch

    @property
    def batch_size(self):
        return self._batch_size

    @property
    def criterion(self):
        return self._criterion

    @property
    def layers(self):
        return self._layers_count

    @property
    def layers_size(self):
        return self._layers_size

    @property
    def mutation_chance(self):
        return self._mutation_chance

    def generate_population(self) -> List[Chromosome]:
        population = []
        for population_number in range(self._population_count):
            chromosome = self.generate_chromosome()
            population.append(chromosome)

        return population

    def generate_chromosome(self) -> Chromosome:
        chromosome_genes = []
        for _ in range(self._layers_count):
            genes = self._generate_random_genes()
            chromosome_genes.extend(genes)

        return Chromosome(chromosome_genes)

    def _generate_random_genes(self) -> tuple:
        kernel = self._random_kernel()
        out = self._random_out()
        step = self._random_step()

        return Gene(kernel), Gene(out), Gene(step)

    def _random_out(self) -> int:
        return random.randrange(12, self._out_threshold, 12)

    def _random_step(self) -> int:
        return random.randint(1, self._step_threshold)

    def _random_kernel(self) -> int:
        return random.randrange(5, self._kernel_threshold, 2)
