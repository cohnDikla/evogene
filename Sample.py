class Sample:
    """
    This Class represents a Sample object,
    which holds its sample index,
    a field indicates whether it is healthy or not,
    value in each strain,
    average value in each species,
    and number of strains in each species.
    """
    NUMBER_OF_SPECIES = 10

    def __init__(self, idx, is_healthy):
        """
        Constructor of the Sample Class
        :param idx: the sample index
        :param is_healthy: whether this sample is healthy or not
        """
        self._sample_idx = idx
        self._is_healthy = is_healthy
        self._strains = dict()
        self._species_avgs = [0] * self.NUMBER_OF_SPECIES
        self._num_strains_in_species = [0] * self.NUMBER_OF_SPECIES

    def add_value(self, strain, species, value):
        self._strains[strain] = value
        self._num_strains_in_species[species] += 1
        self._species_avgs[species] = (self._species_avgs[species]+value) / \
                                       self._num_strains_in_species[species]

    def get_species_avgs(self):
        return self._species_avgs

    def is_healthy(self):
        return self._is_healthy
