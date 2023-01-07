class Genome:
    def __init__(self, value: int):
        self._genome = value

    @property
    def data(self) -> int:
        return self._genome

    @data.setter
    def data(self, value: int):
        if not isinstance(value, int):
            raise ValueError('Genome data must be integer')
        self._genome = value
