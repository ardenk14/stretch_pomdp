from pomdp_py.framework import basics
import numpy as np


class Observation(basics.Observation):
    def __init__(self, pos_reading=None):
        """
        pos_reading: The position reading of the observation.
        """
        self._pos_reading = pos_reading
        self._hash = hash(tuple(self._pos_reading))

    def __hash__(self):
        return self._hash

    def __eq__(self, other):
        if isinstance(other, Observation):
            return self._hash == other._hash
        return False

    def __str__(self):
        return f"<pos_reading {self._pos_reading}>"

    def __repr__(self):
        return self.__str__()

    @property
    def to_vector(self):
        return np.array(self._pos_reading)

    @property
    def get_reading(self):
        return self._pos_reading