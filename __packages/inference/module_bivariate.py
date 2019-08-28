# -*- coding: utf-8 -*-

# import built in libarys
from itertools import permutations

# import 3rd party libarys
import numpy as np

# import local libarys
from whypy.__packages.utils import utils
from whypy.__packages.utils import stats


###############################################################################
class Bivariate():
    """
    Causal Inference methods for the two variable case. General SCMs are not
    identifiable in the two variable case. Additional Assumptions are required,
    given by the modelclass restrictions. Only acyclic graphs are considered.
    """
    attr_variate = 'bivariate'

    def __init__(self):
        """
        Class Constructor for the Bivariate Case. Assigns Bivariate specific
        Methods to the Inference Model. Instance Attributes are assigned in
        the inference Class.
        """

    def get_combinations(self):
        """
        Method to get a list of combinations for the Bivariate Case.
        """
        variable_names = np.arange(self._xi.shape[1])
        variable_names = list(variable_names)
        combinations = [x for x in permutations(variable_names, 2)]
        # Order combinations by forward, backward equality
        combinations_array = np.array(combinations) + 1
        sort_order = np.argsort(np.multiply(combinations_array[:,0], combinations_array[:,1]))
        combinations = [combinations[i] for i in sort_order]
        self._combinations = utils.trans_nestedlist_to_tuple(combinations)

    def check_combinations(self):
        """
        Method to check combinations for the Bivariate Case.
        """
        assert np.array(self._combinations).shape[1] == 2, 'Shape of combinations must be (m, 2)'
        self._combinations = utils.trans_nestedlist_to_tuple(self._combinations)

###############################################################################
