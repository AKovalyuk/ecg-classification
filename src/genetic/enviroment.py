import random
from dataclasses import dataclass

from chromosome import Chromosome, Gene


@dataclass
class LayerSettings:
    kernel_size: int
    out_channels: int
    step: int


class PopulationEnvironment:
    def __init__(self):
        self._step_threshold = 10
        self._kernel_threshold = 15
        self._out_threshold = 288

        self._layers_count = 3
        self._population_count = 5
        self._mutation_chance = 0.1

    @property
    def mutation_chance(self):
        return self._mutation_chance

    def generate_population(self):
        population = []
        for population_number in range(self._population_count):
            chromosome = self._generate_chromosome()
            population.append(chromosome)

        return population

    def _generate_chromosome(self) -> Chromosome:
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
        return random.randint(2, self._step_threshold)

    def _random_kernel(self) -> int:
        return random.randrange(3, self._kernel_threshold, 2)


env = PopulationEnvironment()
z = env.generate_population()
print('adf')
