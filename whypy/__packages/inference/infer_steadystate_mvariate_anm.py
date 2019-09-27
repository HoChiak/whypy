# -*- coding: utf-8 -*-

# import built in libarys

# import 3rd party libarys

# import local libarys
from .module_general import General as parent0
from .module_time import SteadyState as parent1
from .module_variate import Mvariate as parent2
from .module_anm import ANM as parent3

from whypy.__packages.utils import utils

###############################################################################


class Model(parent0, parent1, parent2, parent3):
    """
    Parent Class for Causal Inference Methods. Different Cases
    (Bivariate <-> Multivariate | SteadyState <-> Transient) are defined by
    parent1 and parent2. The additive noise model causal inference Methods
    are loaded by parent3. Other imports are the general class (using
    class and instance attributes) and functions loaded by utils.
    """
    # Global dictionary
    attr_dict = {'LikelihoodVariance': 'likelihood-ratio',
                 'LikelihoodEntropy': 'likelihood-ratio',
                 'KolmogorovSmirnoff': 'p-value',
                 'MannWhitney': 'p-value',
                 'HSIC': 'unknown'
                 }

    def __init__(self, obs=None, combinations='all', regmod=None, scaler=None,
                 obs_name=None, t0=None, stride=None):
        """
        Class constructor for causal inference methods.

        INPUT (inherent from parent):
        obs:        observations (data) as pandas DataFrame or numpy array.
                    shape -> (observations, variables)
        combs:      nested list of combinations or codeword. List logic
                    regarding the following rule [comb1, comb2, ... combn],
                    where combi is another list. To test x2 = f(x1, x5) the
                    combi list is defined as follows: [x2, x1, x5]. The first
                    variable is always the dependent variable extended by an
                    abritary number of independent variables (to be tested).
        model:      model for regression, must be callable with model.fit()
                    and model.predict(). Can be single object or list of
                    objects. If List is given, models should be independent
                    initialized and list must have length according to the
                    number n of combinations: [model1, model2, ... model_n]
        scaler:     single scaler object (optional), must be callable with
                    transform and inverse_transform.
        """
        parent0.__init__(self)
        parent1.__init__(self, t0, stride)
        parent2.__init__(self)
        parent3.__init__(self)
        self.obs = obs
        self.combs = combinations
        self.regmod = regmod
        self.scaler = scaler
        self.obs_name = obs_name
        self._obs = None
        self._combs = None
        self._regmods = None
        self._scaler = None
        self._obs_name = None
###############################################################################
