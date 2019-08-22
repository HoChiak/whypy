# -*- coding: utf-8 -*-

# import built in libarys
from copy import deepcopy


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
        self._figsize = (10, 7.071)
        self._cmap = plt.get_cmap('Pastel1', 100)
        self._numberrun = 0
        self._config = {}

    def check_and_init_attr(self, scale):
        """
        Method to check correctness of instance attributes as well as init
        missing attributs.
        """
        # Do Checks on global attributes
        self.check_instance_attr(scale)
        # Init Observations
        self._xi = deepcopy(self.xi)
        # Get Combinations of dependent and independent variable if not given
        if self.combinations is 'all':
            self.get_combinations()
        else:
            self._combinations = deepcopy(self.combinations)
        # Initiate a regmod for each combination
        no_combinations = len(self._combinations)
        self._regmod = utils.trans_object_to_list(self.regmod, no_combinations, dcopy=True)
        # Initiate a scaler for each variable
        no_variables = self._xi.shape[1]
        self._scaler = utils.trans_object_to_list(self.scaler, no_variables, dcopy=True)

    def check_instance_attr(self, scale):
        """
        Method to check the instance attributes
        """
        assert self.xi is not None, 'Observations are None type'
        assert not(np.isnan(self.xi).any()), 'Observations contain np.nan'
        assert not(np.isinf(self.xi).any()), 'Observations contain np.inf'
        assert self.regmod is not None, 'Regression Model is None type'
        assert hasattr(self.regmod, 'fit'), 'Regression Model has no attribute "fit"'
        assert hasattr(self.regmod, 'predict'), 'Regression Model has no attribute "predict"'
        assert not(hasattr(type(self.regmod), '__iter__')), 'Regression Model should be passed as single object. Attribute __iter__ detected.'
        assert not(hasattr(type(self.scaler), '__iter__')), 'Scaler Model should be passed as single object. Attribute __iter__ detected.'
        assert hasattr(self.scaler, 'fit'), 'Scaler Model has no attribute "fit"'
        assert hasattr(self.scaler, 'transform'), 'Scaler Model has no attribute "transform"'
        assert hasattr(self.scaler, 'inverse_transform'), 'Scaler Model has no attribute "inverse_transform"'
        # assert self.scaler.copy is False, 'Scaler Model doesnt support inplace transformation (maybe set "copy=False")'
        assert ((scale is False) or ((scale is True) and (self.scaler is not None))), 'If scale is True, a scaler must be assigned'

    def check_kwargs_declaration(self, kwargs, kwargskey, default_value):
        """
        Method to test wheter a keyword in kwargs exist. If not, set kwargs
        keyword to given default_value.
        """
        try:
            kwargs[kwargskey]
        except:
            kwargs[kwargskey] = default_value
        finally:
            return(kwargs)

    def check_and_init_arg_run(self, testtype, bootstrap):
        """
        Method to check the run-methods arguments
        """
        assert testtype in ('Likelihood', 'KolmogorovSmirnoff', 'MannWhitney', 'HSIC'), 'Wrong Argument given for TestType'
        if bootstrap is False:
            _bootstrap = (1,)
        elif bootstrap == 1:
            bootstrap = False
            _bootstrap = (1,)
        else:
            try:
                assert bootstrap > 0, 'Argument bootstrap must be positive integer > 0'
                _bootstrap = tuple([x for x in range(1, bootstrap+1)])
            except:
                raise ValueError('Argument bootstrap must be either False or type integer (int>0)')
        return(bootstrap, _bootstrap)

    def check_init_kwargs(self, kwargs):
        """
        Method to check and init the run-methods kwargs
        """
        new_kwargs = {}
        if self._config['bootstrap'] > 0:
            new_kwargs = self.check_kwargs_declaration(kwargs, kwargskey='bootstrap_ratio', default_value=1)
            assert 0 < kwargs['bootstrap_ratio'] <=1 , 'Bootstrap Ratio must be in range [0, 1]'
            new_kwargs = self.check_kwargs_declaration(kwargs, kwargskey='bootstrap_seed', default_value=1)
            assert isinstance(kwargs['bootstrap_seed'], int), 'Bootstrap Seed must be integer'
        return(new_kwargs)

    def scaler_fit(self):
        """
        Method to fit a choosen list of scalers to all Xi
        """
        for i in range(len(self._scaler)):
            self._scaler[i].fit(self._xi[:, i].reshape(-1, 1))

    def scaler_transform(self, data, idx):
        """
        Method to fit a choosen list of scalers to all Xi
        idx must be in tuple.
        """
        for temp_id, temp_val in enumerate(idx):
            data = self._scaler[temp_val].transform(data[:, temp_id].reshape(-1, 1))
        return(data)

    def scaler_inverse_transform(self, data, idx):
        """
        Method to fit a choosen list of scalers to all Xi
        idx must be in tuple.
        """
        # Univariate Case
        for temp_id, temp_val in enumerate(idx):
            data = self._scaler[temp_val].inverse_transform(data[:, temp_id].reshape(-1, 1))
        return(data)

    def get_tINdep(self, combno):
        """
        Method to get the index of the dependent (tdep) and the independent
        (tindep) variable by index i from combinations.
        """
        tdep = tuple([self._combinations[combno][0],])
        tindep = self._combinations[combno][1:]
        return(tdep, tindep)

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
