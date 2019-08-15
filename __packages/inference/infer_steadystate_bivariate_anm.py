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
from whypy.__packages.inference.module_anm import ANM as parent3

from whypy.__packages.utils import utils

###############################################################################
class Model(parent0, parent1, parent2, parent3):
    """
    Causal Inference methods for the two variable case. General SCMs are not
    identifiable in the two variable case. Additional Assumptions are required,
    given by the modelclass restrictions. Only acyclic graphs are considered.
    """
    attr_dict = {'independence': 'p-value',
                 'likelihood': 'likelihood'}


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
        self._xi = np.array(xi)
        self._combinations = combinations
        self._regmod = regmod
        self._scaler = scaler
        self._results = {}
        self._results2V = {}


    def run(self,
            testtype='likelihood', #independence, likelihood
            scale=True,
            modelpts=50,
            out_Regr_Model=True,
            out_Regr_Model_info=True,
            out_X_Residuals_NormalityTest=True,
            out_X_vs_Residuals=True,
            out_X_vs_Residuals_info=True,
            out_Results_testtype=True,
            out_CausalGraph=True,
            CGmetric='Combined'):
        """
        Method to test independence of residuals.
        Theorem: In causal direction, the noise is independent of the input
        Valid for Additive Noise Models e.g. LiNGAM, NonLinear GaussianAM
        """
        # Check and Initialisation of Attributes
        if self._numberrun == 0:
            self.check_and_init_attr(scale)
        self.check_combinations()
        # Check Function Arguments
        ### TBD go on here (assign testtype as atttribute?)
        ### only run for one testtype (kolmogorov or the other or likelihoodration)
        ### placeholder for bootstrape, time, holdout set
        assert testtype in ('independence', 'likelihood'), 'testtype must be either "independence" or "likelihood"'
        # Add information to config
        self._config['%i' % (self._numberrun)] = {'testtype': testtype,
                                                  'scale': scale,
                                                  'modelpts': modelpts,
                                                  'shape_observations': self._xi.shape,
                                                  'shape_combinations': np.array(self._combinations).shape,
                                                  'regression_model:': str(self._regmod[0]),
                                                  'scaler_model:': str(self._scaler[0]),
                                                  }
        # Global translater
        ### Remove this TBD when attr_dict is ready
        dic = {'independence': 'p-value', 'likelihood': 'likelihood'}
        namecode = dic[testtype]
        # Do the math
        self.run_inference()
        # Loop trough possible combinations of tdep and tindep for plots/logs
        # Define a list of do's (dolist) for plots sorted by tdep/tindep
        # combinations. Start dolist:
        dolist = []
        if out_Regr_Model is True:
            dolist.append('out_Regr_Model')
        if out_Regr_Model_info is True:
            dolist.append('out_Regr_Model_info')
        if out_X_Residuals_NormalityTest is True:
            dolist.append('out_X_Residuals_NormalityTest')
        if out_X_vs_Residuals_info is True:
            if 'p-value' in namecode:
                dolist.append('out_X_vs_Residuals_info')
            else:
                print('X vs Residual log only available for independence test')
        if len(dolist) != 0:
            self.loop_and_do(do=dolist)
        # end dolist
        utils.print_in_console(what='result header')
        # Plot independence/likelihood tests results
        if out_X_vs_Residuals is True:
            if 'p-value' in namecode:
                self.plt_2metrics_groupedby(namecode)
            self.plt_1metric_groupedby(namecode)
        # Print independence/likelihood tests results
        if out_Results_testtype is True:
            rs = self._results2V['%i' % (self._numberrun)]
            print(rs[['TestType', '2V-case', 'pval/likel',
                      'rank pval/likel', '2V-direction']].to_string())
        # plot the Causal Graph
        if out_CausalGraph is True:
            utils.print_in_console(what='CG Warning')
            self.predict_CG(testtype, CGmetric=CGmetric)
        self._numberrun += 1

###############################################################################
