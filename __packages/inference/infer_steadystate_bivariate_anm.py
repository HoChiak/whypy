# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import local libarys
from whypy.__packages.inference.module_general import General as parent0
from whypy.__packages.inference.module_steadystate import SteadyState as parent1
from whypy.__packages.inference.module_bivariate import Bivariate as parent2
from whypy.__packages.inference.module_anm import RunANM as parent3
from whypy.__packages.inference.module_anm import PlotANM as parent4
from whypy.__packages.inference.module_anm import ResultsANM as parent5

from whypy.__packages.utils import utils

###############################################################################
class Model(parent0, parent1, parent2, parent3, parent4, parent5):
    """
    Causal Inference methods for the two variable case. General SCMs are not
    identifiable in the two variable case. Additional Assumptions are required,
    given by the modelclass restrictions. Only acyclic graphs are considered.
    """
    attr_dict = {'Likelihood': 'likelihood-ratio',
                 'KolmogorovSmirnoff': 'p-value',
                 'MannWhitney': 'p-value',
                 'HSIC': 'unknown'
                 }


    def __init__(self, xi=None, combinations='all', regmod=None, scaler=None):
        """
        Child class constructor for causal inference methods in the 2 variable
        case. Xi may consist of an abritary number of variables, but only one
        variable is mapped to one other

        INPUT (Inherent from parent):
        Xi:         observations (data)
                    (columns are variables)
        model:      List of regression models.
                    model[i][j] maps Xi[i] ~ f(Xi[j])
                    while i is the dependent and j is the independent variable
                    (must be callable with model.fit() and model.predict())
        scaler:     List of scaler.

        INPUT (Child specific):
        dict_2V:    needed to store additional information for each mapping
                    dict_2V[i][j] maps Xi[i] ~ f(Xi[j])

        """
        parent0.__init__(self)
        parent1.__init__(self)
        parent2.__init__(self)
        parent3.__init__(self)
        parent4.__init__(self)
        parent5.__init__(self)
        self.xi = np.array(xi)
        self.combinations = combinations
        self.regmod = regmod
        self.scaler = scaler
        self._xi = None
        self._combinations = None
        self._regmod = None
        self._scaler = None
        self._results = {}
        self._results2V = {}




    def run(self,
            testtype='Likelihood',# Likelihood KolmogorovSmirnoff MannWhitney HSIC
            scale=True,
            modelpts=50,
            plot_inference=True,
            plot_results=True
            ):
        """
        Method to test independence of residuals.
        Theorem: In causal direction, the noise is independent of the input
        Valid for Additive Noise Models e.g. LiNGAM, NonLinear GaussianAM
        """
        # Count Number of runs +1
        self._numberrun += 1
        # Check and Initialisation of Attributes
        self.check_and_init_attr(scale)
        self.check_combinations()
        # Check Function Arguments
        ### TBD go on here (assign testtype as atttribute?)
        ### only run for one testtype (kolmogorov or the other or likelihoodration)
        ### placeholder for bootstrape, time, holdout set, different environments
        assert testtype in ('Likelihood', 'KolmogorovSmirnoff', 'MannWhitney', 'HSIC'), 'Wrong Argument given for TestType'
        # Add information to config
        self._config['%i' % (self._numberrun)] = {'testtype': testtype,
                                                  'scale': scale,
                                                  'modelpts': modelpts,
                                                  'shape_observations': self._xi.shape,
                                                  'shape_combinations': np.array(self._combinations).shape,
                                                  'regression_model:': str(self._regmod[0]),
                                                  'scaler_model:': str(self._scaler[0]),
                                                  }
        # Do the math
        self.run_inference()
        # Plot the math of inference
        if plot_inference is True:
            self.plot_inference()
        # Plot results
        if plot_results is True:
            self.plot_results()


###############################################################################
