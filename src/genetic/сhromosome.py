from .genome import Genome


class Chromosome:
    def __init__(self, genomes: list):
        if not all(isinstance(genome, Genome) for genome in genomes):
            raise ValueError('All elements must be Genome type')

        self._genomes = genomes

    