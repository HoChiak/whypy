# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import local libarys
from whypy.__packages.utils import utils
from whypy.__packages.utils import stats


###############################################################################
class General():
    """
    General Causal Inference methods.
    """

    def __init__(self):
        """
        Class Constructor for General CI Methods
        """

    def check_instance_attr(self, scale):
        """
        Method to check the instance attributes
        """
        assert self._xi is not None, 'Observations are None type'
        assert not(np.isnan(self._xi).any()), 'Observations contain np.nan'
        assert not(np.isinf(self._xi).any()), 'Observations contain np.inf'
        assert self._regmod is not None, 'Regression Model is None type'
        assert not(hasattr(type(self._regmod), '__iter__')), 'Regression Model should be passed as single object. Attribute __iter__ detected.'
        assert not(hasattr(type(self._scaler), '__iter__')), 'Scaler Model should be passed as single object. Attribute __iter__ detected.'
        assert ((scale is False) or ((scale is True) and (self._scaler is not None))), 'If scale is True, a scaler must be assigned'

    def object_to_list(self, object1, n):
        """
        Method to expand one object to a list of length n from this object.
        """
        objectn = [object1 for i in range(n)]
        return(objectn)

    def fit_scaler(self):
        """
        Method to fit a choosen list of scalers to all Xi
        """
        for i in range(len(self._scaler)):
            self._scaler[i].fit(self._xi[:, i].reshape(-1, 1))

    def transform_with_scaler(self, data, idx):
        """
        Method to fit a choosen list of scalers to all Xi
        """
        # Univariate Case
        if idx.size == 1:
            idx = self.array_to_scalar(idx)
            self._scaler[idx].transform(data)
        else:
        # Multivariate Case
            for temp_id, temp_val in enumerate(idx):
                self._scaler[temp_val].transform(data[:, temp_id])
        return(data)

    def inverse_transform_with_scaler(self, data, idx):
        """
        Method to fit a choosen list of scalers to all Xi
        """
        # Univariate Case
        if idx.size == 1:
            idx = self.array_to_scalar(idx)
            self._scaler[idx].inverse_transform(data)
        # Multivariate Case
        else:
            for temp_id, temp_val in enumerate(idx):
                self._scaler[temp_val].inverse_transform(data[:, temp_id])
        return(data)

    def get_tINdep(self, i):
        """
        Method to get the index of the dependent (tdep) and the independent
        (tindep) variable by index i from combinations.
        """
        tdep = self._combinations[i][0]
        tindep = self._combinations[i][1:]
        return(tdep, tindep)

    def array_to_scalar(self, array):
        """
        Method to turn a np.array([scalar]) into scalar.
        """
        if array.size == 1:
            scalar = array.item()
            return(scalar)
        else:
            return(array)

    def get_Xmodel(self, X_data, modelpts):
        """
        Method to linspaced x_model data from the limits of x_data.
        """
        X_range = np.max(X_data, axis=0) - np.min(X_data, axis=0)
        # TBD not working vor mvariate case
        X_model = np.linspace(np.min(X_data, axis=0) - (X_range * 0.05),
                              np.max(X_data, axis=0) + (X_range * 0.05),
                              num=modelpts)
        X_model = X_model.reshape(-1, 1)
        return(X_model)

    def loop_and_do(self, do, **kwargs):
        """
        Method to scale (if scale==True) and loop trough possible combinations
        of tdep and tindep for modelfit of residuals. Save result in _dict2V.
        """
        # Loop trough possible combinations of tdep and tindep for modelfit
        for i in range(self._combinations.shape[0]):
            tdep, tindep = self.get_tINdep(i)
            # fit regmod on observations
            if 'fit' in do:
                self.fit_model2xi(i, tdep, tindep,
                                  kwargs['scale'],
                                  kwargs['modelpts'])
            # do normality test
            tindep = self.array_to_scalar(tindep)
            if 'normality' in do:
                self.do_normality(i, tdep, tindep)
            # do independence test
            if 'independence' in do:
                self.do_independence(i, tdep, tindep)
            # do calculate the likelihood
            if 'likelihood' in do:
                self.do_likelihood(i, tdep, tindep)
            # print header for 2V
            if ((('out_Regr_Model' in do) or
                 ('out_Regr_Model_info' in do) or
                 ('out_X_Residuals_NormalityTest' in do) or
                 ('out_X_vs_Residuals_info' in do))):
                utils.print_in_console(what='regmod header',
                                       tdep=tdep, tindep=tindep)
            # plot joint and marginal together with model and hist
            if 'out_Regr_Model' in do:
                self.plt_1model_adv(i, tdep, tindep)
                self.plt_1hist(i, tdep, tindep)
            # print/plot model informations
            if 'out_Regr_Model_info' in do:
                try:
                    self.plt_GAMlog(i, tdep, tindep)
                except:
                    print('An exception occurred using -plt_GAMlog()-')
                try:
                    utils.print_in_console(what='model summary')
                    self._regmod[i].summary()
                except:
                    print('An exception occurred using -summary()-')
            # print the normality log
            if 'out_X_Residuals_NormalityTest' in do:
                self.print_log_st(i, tdep, tindep, 'normality')
            # print the independence log
            if 'out_X_vs_Residuals_info' in do:
                self.print_log_st(i, tdep, tindep, 'independence')
