from .chromosome import Chromosome
from .enviroment import PopulationEnvironment


class Population:
    def __init__(self):
        self._env = PopulationEnvironment()
        self._population = self._env.generate_population()

    def selection(self) -> list:
        pass

    def point_crossover(self, mather: Chromosome, father: Chromosome) -> Chromosome:
        pass
