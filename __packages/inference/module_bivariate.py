# -*- coding: utf-8 -*-

# import built in libarys
from itertools import permutations
from copy import deepcopy

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

    def get_all_combinations(self):
        """
        Method to get a list of combinations for the Bivariate Case.
        """
        var_names = np.arange(self.obs.shape[1]).tolist()
        comb = [x for x in permutations(var_names, 2)]
        # Order combinations by forward, backward equality
        comb_ary = np.array(comb) + 1
        sort_order = np.argsort(np.multiply(comb_ary[:, 0], comb_ary[:, 1]))
        comb = [comb[i] for i in sort_order]
        self._combs = utils.trans_nestedlist_to_tuple(comb)

    def check_combinations(self):
        """
        Method to check combinations for the Bivariate Case.
        """
        assert np.array(self.combs).shape[1] == 2,'Shape of combinations must be (m, 2)'
        assert isinstance(self.combs[0], list),'Combinations must be nested list [[combination 1], [combination 2], ...]'

    def init_combinations(self):
        """
        Method to init combinations for the Bivariate Case.
        """
        if self.combs is 'all':
            self.get_all_combinations()
        else:
            self._combs = deepcopy(self.combs)
            self._combs = utils.trans_nestedlist_to_tuple(self._combs)


###############################################################################
