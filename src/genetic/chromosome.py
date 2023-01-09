from typing import List
from random import randint


class Gene:
    def __init__(self, value: int):
        Gene.int_check(value)
        self._genome = value

    @property
    def data(self) -> int:
        return self._genome

    @data.setter
    def data(self, value: int):
        Gene.int_check(value)
        self._genome = value

    @staticmethod
    def int_check(value):
        if type(value) == bool:
            raise ValueError('Genome data must be integer')


class Chromosome:
    def __init__(self, genomes: list):
        if not all(isinstance(genome, Gene) for genome in genomes):
            raise ValueError('All elements must be Genome type')
        self._fitness: int = 0
        self._genomes = genomes

    @property
    def genes(self) -> List[Gene]:
        return self._genomes

    @genes.setter
    def genes(self, value: List[Gene]):
        self._genomes = value

    @property
    def genes_data(self) -> List[int]:
        genes_data = []
        for gene in self.genes:
            genes_data.append(gene.data)
        return genes_data

    @property
    def fitness(self):
        return self._fitness

    @fitness.setter
    def fitness(self, value: int):
        self._fitness = value

    def point_crossover(self, chromosome):
        idx = randint(1, len(chromosome.genes))

        target_genomes = self.genes[:idx + 1]
        other_genomes = chromosome.genes[idx + 1:]
        return Chromosome(target_genomes + other_genomes)
