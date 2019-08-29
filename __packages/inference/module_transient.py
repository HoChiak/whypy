# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import local libarys
from whypy.__packages.utils import utils

from importlib import reload
utils=reload(utils)
###############################################################################
class Transient():
    """
    Causal methods... tbd
    """
    def __init__(self, xi=None, regmod=None, scaler=None):
        """
        Parent class constructor for causal inference methods

        INPUT
        _xi:        observations
                    (columns are variables)
        regmod:    List of regression models.
                    model[0][3] maps Xi[0] ~ f(Xi[3])
                    (must be callable with model.fit() and model.predict())
        scaler:     List of scaler
                    (structure as _regmod)
        """


    def predict(self, testvariant,
                scale=True,
                modelpts=50,
                out_Regr_Model=True,
                out_Regr_Model_info=True,
                out_X_Residuals_NormalityTest=True,
                out_X_vs_Residuals=True,
                out_X_vs_Residuals_info=True,
                out_Results_Testvariant=True,
                out_CausalGraph=True,
                CGmetric='Combined'):
        """
        Method to test independence of residuals.
        Theorem: In causal direction, the noise is independent of the input
        Valid for Additive Noise Models e.g. LiNGAM, NonLinear GaussianAM
        """
        assert testvariant in ('independence', 'likelihood'), 'TestVariant must be either "independence" or "likelihood"'
        # Global translater
        dic = {'independence': 'p-value',
               'likelihood': 'likelihood'}
        namecode = dic[testvariant]
        # Do the math
        self.regress(scale, testvariant, modelpts)
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
        if out_Results_Testvariant is True:
            rs = self._results2V
            print(rs[['TestType', '2V-case', 'pval/likel',
                      'rank pval/likel', '2V-direction']].to_string())
        # plot the Causal Graph
        if out_CausalGraph is True:
            utils.print_in_console(what='CG Warning')
            self.predict_CG(testvariant, CGmetric=CGmetric)

###############################################################################
