# -*- coding: utf-8 -*-

# import built in libarys
from copy import deepcopy

# import 3rd party libarys
import numpy as np
from sklearn.utils import resample


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
    attr_dict = {'LikelihoodVariance': 'likelihood-ratio',
                 'LikelihoodEntropy': 'likelihood-ratio',
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
        self.obs = np.array(xi)
        self.combs = combinations
        self.regmod = regmod
        self.scaler = scaler
        self._obs = None
        self._combs = None
        self._regmods = None
        self._scaler = None

    def run(self,
            testtype='LikelihoodVariance',
            scale=True,
            bootstrap=False,
            holdout=False,
            plot_inference=True,
            plot_results=True,
            **kwargs):
        """
        Method to test independence of residuals.
        Theorem: In causal direction, the noise is independent of the input
        Valid for Additive Noise Models e.g. LiNGAM, NonLinear GaussianAM
        """
        # Count Number of runs +1
        self._runi = 0
        # Check and Initialisation of Attributes
        self.check_instance_model_attr(scale)
        self.init_instance_model_attr()
        # Clear Arguments from previous caclulations
        parent3.__del__(self)
        # Check Method "Run" Arguments
        self.check_and_init_arg_run(testtype, bootstrap, holdout)
        # Add information to config
        self._config = {'testtype': testtype,
                        'scale': scale,
                        'bootstrap': bootstrap,
                        'holdout': holdout,
                        'shape_observations': self._obs.shape,
                        'shape_combinations': np.array(self._combs).shape,
                        'regression_model': str(self._regmods[0]),
                        'scaler_model': str(self._scaler[0]),
                        }
        # Check and Init Kwargs
        self.check_init_kwargs(kwargs)
        # Check and Init Holdout Lists
        self.check_init_holdout_ids()
        # Check and display warnings
        self.check_warnings()
        # Display Start of Causal Inference
        utils.display_text_predefined(what='inference header')
        # TBD Add Time shift / Adress different environments
        for boot_i, _ in enumerate(self._bootstrap):
            if bootstrap > 0:
                # Display the current bootstrap number
                utils.display_text_predefined(what='count bootstrap',
                                              current=boot_i, sum=bootstrap)
                # Init fresh _regmod from regmod -> otherwise fit will fail
                self._regmods = utils.trans_object_to_list(self.regmod,
                                                          len(self._combs),
                                                          dcopy=True)
                # Do the Bootstrap
                self._obs = resample(deepcopy(self.obs), replace=True,
                                    n_samples=int(self.obs.shape[0] * self._kwargs['bootstrap_ratio']),
                                    random_state=self._kwargs['bootstrap_seed']+boot_i)
            self._runi = boot_i
            # Do the math
            self.run_inference()
        # Plot the math of inference
        if plot_inference is True:
            self.plot_inference()
        # Plot results
        if plot_results is True:
            self.plot_results()


###############################################################################
