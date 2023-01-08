class Chromosome:
    def __init__(self, genomes: list):
        if not all(isinstance(genome, Gene) for genome in genomes):
            raise ValueError('All elements must be Genome type')

        self._genomes = genomes

    @property
    def genes(self) -> list:
        return self._genomes


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
