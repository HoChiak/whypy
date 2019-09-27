# -*- coding: utf-8 -*-

# import built in libarys
from copy import deepcopy
from warnings import warn

# import 3rd party libarys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from pandas import DataFrame

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
        self._colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
        self._runi = 0
        self._config = {}
        self._kwargs = None

    def check_instance_model_attr(self, scale):
        """
        Method to check the instance attributes
        """
        # Check Observations
        assert self.obs is not None, 'Observations are None type'
        assert not(np.isnan(np.array(self.obs)).any()), 'Observations contain np.nan'
        assert not(np.isinf(np.array(self.obs)).any()), 'Observations contain np.inf'
        # Check Regression Model
        assert self.regmod is not None, 'Regression Model is None type'
        # Differ wheter a list of regmod or a single regmod is given
        if hasattr(type(self.regmod), '__iter__'):
            assert hasattr(self.regmod[0], 'fit'), 'Regression Model has no attribute "fit"'
            assert hasattr(self.regmod[0], 'predict'), 'Regression Model has no attribute "predict"'
        else:
            assert hasattr(self.regmod, 'fit'), 'Regression Model has no attribute "fit"'
            assert hasattr(self.regmod, 'predict'), 'Regression Model has no attribute "predict"'
        # Check Scaler
        assert isinstance(scale, bool), 'Scale must be Bool'
        if scale is True:
            assert self.scaler is not None, 'If scale is True, a scaler must be assigned'
            if hasattr(type(self.scaler), '__iter__'):
                assert hasattr(self.scaler[0], 'fit'), 'Scaler Model has no attribute "fit"'
                assert hasattr(self.scaler[0], 'transform'), 'Scaler Model has no attribute "transform"'
                assert hasattr(self.scaler[0], 'inverse_transform'), 'Scaler Model has no attribute "inverse_transform"'
            else:
                assert hasattr(self.scaler, 'fit'), 'Scaler Model has no attribute "fit"'
                assert hasattr(self.scaler, 'transform'), 'Scaler Model has no attribute "transform"'
                assert hasattr(self.scaler, 'inverse_transform'), 'Scaler Model has no attribute "inverse_transform"'
        if self.combs is not 'all':
            self.check_combinations()
        if self.attr_time is 'transient':
            assert self._t0 is not None, 'In transient model parameter t0 must be defined'
            assert self._stride is not None, 'In transient model parameter stride must be defined'
            assert isinstance(self._t0, int), 'Parameter t0 must be type int'
            assert (isinstance(self._stride, int) and (self._stride > 0)), 'Parameter stride must be type int and >0'

    def init_instance_model_attr(self):
        """
        Method to check correctness of instance attributes as well as init
        missing attributs.
        """
        # Get Combinations of dependent and independent variable if not given
        self.init_combinations()
        # Init Observations
        self._obs = np.array(deepcopy(self.obs))
        # Initiate a regmod for each combination if not already a list is given
        if hasattr(type(self.regmod), '__iter__'):
            # Make deepcopy to ensure independence
            self._regmods = [deepcopy(regmod) for regmod in self.regmod]
        else:
            no_combs = len(self._combs)
            self._regmods = utils.object2list(self.regmod, no_combs, dcopy=True)
        # Initiate a scaler for each variable if not already a list is given
        no_var = self._obs.shape[1]
        if hasattr(type(self.scaler), '__iter__'):
            # Make deepcopy to ensure independence
            self._scaler = [deepcopy(scaler) for scaler in self.scaler]
        else:
            self._scaler = utils.object2list(self.scaler, no_var, dcopy=True)
        # Init observation names
        if self.obs_name is None:
            self._obs_name = [r'X%i' % (i) for i in range(no_var)]
        else:
            self._obs_name = deepcopy(self.obs_name)

    def check_and_init_arg_run(self, testtype, bootstrap, holdout):
        """
        Method to check the run-methods arguments
        """
        assert testtype in ('LikelihoodVariance', 'LikelihoodEntropy', 'KolmogorovSmirnoff', 'MannWhitney', 'HSIC'), 'Wrong Argument given for TestType'
        assert isinstance(bootstrap, bool) or (isinstance(bootstrap, int) and (bootstrap > 0)), 'Bootstrap must be Bool or Int (int>0)'
        assert isinstance(holdout, bool), 'Holdout must be Bool'
        if bootstrap < 2:  # If bootstrap is False or 1
            self._bootstrap = (1,)
        else:
            self._bootstrap = tuple([x for x in range(1, bootstrap+1)])

    def check_kwargs_declaration(self, key, default):
        """
        Method to test wheter a keyword in kwargs exist. If not, set kwargs
        keyword to given default.
        """
        if key not in self._kwargs:
            self._kwargs[key] = default

    def check_init_kwargs(self, kwargs):
        """
        Method to check and init the run-method kwargs
        """
        # Delete and init new self._kwargs
        del self._kwargs
        self._kwargs = kwargs
        # Check Bootstrap Kwargs
        if self._config['bootstrap'] > 0:
            self.check_kwargs_declaration(key='bootstrap_ratio', default=1)
            assert 0 < self._kwargs['bootstrap_ratio'] <= 1, 'Bootstrap Ratio must be in range [0, 1]'
            self.check_kwargs_declaration(key='bootstrap_seed', default=None)
            assert (isinstance(self._kwargs['bootstrap_seed'], int) or (self._kwargs['bootstrap_seed'] is None)), 'Bootstrap Seed must be integer or None type'
        # Check Holdout Kwargs
        if self._config['holdout'] > 0:
            self.check_kwargs_declaration(key='holdout_ratio', default=0.2)
            assert 0 < self._kwargs['holdout_ratio'] <= 1, 'Holdout Ratio must be in range [0, 1]'
            self.check_kwargs_declaration(key='holdout_seed', default=None)
            assert (isinstance(self._kwargs['holdout_seed'], int) or (self._kwargs['holdout_seed'] is None)), 'Holdout Seed must be integer or None type'
        # Check Other Kwargs
        self.check_kwargs_declaration(key='modelpts', default=50)
        assert isinstance(self._kwargs['modelpts'], int), 'Modelpts must be Int'
        self.check_kwargs_declaration(key='gridsearch', default=False)
        assert isinstance(self._kwargs['gridsearch'], bool), 'Gridsearch must be Bool, if defined'
        # Check Gridsearch Kwargs
        if ((self._kwargs['gridsearch'] is True) and
           not('pygam' in str(self._regmods[0].__class__))):
            if not('param_grid' in kwargs):
                raise AssertionError('If "gridsearch" is True, argument params must be specified')
        # Add a list of Kwargs to config
        self._config['**kwargs'] = str(self._kwargs)

    def check_warnings(self):
        """
        Method to display warnings
        """
        if self.obs.shape[0] < 50:
            warn('WARNING: Less than 50 values remaining to fit the regression model')
        else:
            if (self._config['bootstrap'] > 0) and (self._config['holdout'] > 0):
                if self._kwargs['bootstrap_ratio'] * (1 - self._kwargs['holdout_ratio']) * self.obs.shape[0] < 50:
                    warn('WARNING: Less than 50 values remaining to fit the regression model, from bootstrap- and holdout_ratio')
                if self._kwargs['bootstrap_ratio'] * (self._kwargs['holdout_ratio']) * self.obs.shape[0] < 50:
                    warn('WARNING: Less than 50 values remaining to estimate the test statistics, from bootstrap- and holdout_ratio')
            elif (self._config['bootstrap'] > 0):
                if self._kwargs['bootstrap_ratio'] * self.obs.shape[0] < 50:
                    warn('WARNING: Less than 50 values remaining to fit the regression model, from bootstrap_ratio')
            elif (self._config['holdout'] > 0):
                if (1 - self._kwargs['holdout_ratio']) * self.obs.shape[0] < 50:
                    warn('WARNING: Less than 50 values remaining to fit the regression model, from holdout_ratio')
                if (self._kwargs['holdout_ratio']) * self.obs.shape[0] < 50:
                    warn('WARNING: Less than 50 values remaining to estimate the test statistics, from holdout_ratio')

    def get_seed(self, seed=None):
        """
        Method to get a consistent seed for dependent and indpendent split.
        """
        if seed is not None:
            seed = seed
        else:
            seed = np.random.randint(low=0, high=999999999, size=1)[0]
        return(seed)

    def ids_split(self, ids, seed):
        """
        Method to split in regress and test set, with keeping index order.
        Based on given Holout Parameters
        """
        # Holdout for dependent variables
        fit, test = train_test_split(ids,
                                     test_size=self._kwargs['holdout_ratio'],
                                     random_state=seed,
                                     shuffle=True)
        fit = np.sort(fit).tolist()
        test = np.sort(test).tolist()
        return(fit, test)

    def ids_init4holdout(self):
        """
        Method to init ids based on holdout parameters. Split ids in regress
        and test set, ordered sequence.
        """
        if self._config['holdout'] is True:
            # Get a consistent seed for dependent and indpendent split
            seed = self.get_seed(self._kwargs['holdout_seed'])
            ids_fit_tdep, ids_test_tdep = self.ids_split(self._ids_tdep,
                                                         seed)
            ids_fit_tindep, ids_test_tindep = self.ids_split(self._ids_tindep,
                                                             seed)
        else:
            ids_fit_tdep = self._ids_tdep
            ids_test_tdep = self._ids_tdep
            ids_fit_tindep = self._ids_tindep
            ids_test_tindep = self._ids_tindep
        # Get shared fit and test ids
        ids_fit = np.concatenate([ids_fit_tdep, ids_fit_tindep], axis=0)
        ids_fit = np.unique(ids_fit)
        ids_test = np.concatenate([ids_test_tdep, ids_test_tindep], axis=0)
        ids_test = np.unique(ids_test)
        # Assign to instance Attributes
        self._ids_fit_tdep = ids_fit_tdep
        self._ids_test_tdep = ids_test_tdep
        self._ids_fit_tindep = ids_fit_tindep
        self._ids_test_tindep = ids_test_tindep
        self._ids_fit = ids_fit
        self._ids_test = ids_test

    def scaler_fit(self):
        """
        Method to fit a choosen list of scalers to all Xi
        """
        for i in range(len(self._scaler)):
            self._scaler[i].fit(self._obs[self._ids_fit, i].reshape(-1, 1))

    def scaler_transform(self, data, idx):
        """
        Method to fit a choosen list of scalers to all Xi
        idx must be in tuple.
        """
        for i, val in enumerate(idx):
            data[:, i] = self._scaler[val].transform(data[:, i].reshape(-1, 1)).reshape(-1)
        return(data)

    def scaler_inverse_transform(self, data, idx):
        """
        Method to fit a choosen list of scalers to all Xi
        idx must be in tuple.
        """
        # Univariate Case
        for i, val in enumerate(idx):
            data[:, i] = self._scaler[val].inverse_transform(data[:, i].reshape(-1, 1)).reshape(-1)
        return(data)

    def get_tINdeps(self, combno):
        """
        Method to get the index of the dependent (tdep) and the independent
        (tindep) variable by index i from combinations.
        """
        tdep = tuple([self._combs[combno][0], ])
        tindeps = self._combs[combno][1:]
        return(tdep, tindeps)

    def get_Xmodel(self, X_data, modelpts):
        """
        Method to linspaced x_model data from the limits of x_data.
        """
        # Get Key Indicators
        X_max = np.max(X_data, axis=0)
        X_min = np.min(X_data, axis=0)
        X_range = X_max - X_min
        # Init empty Array of Shape Modelpoints, Number Variables
        X_model = np.ndarray(shape=(modelpts, X_data.shape[1]))
        # Iter over all variables in
        for i in range(X_data.shape[1]):
            X_model[:, i] = np.linspace(X_min[i] - (X_range[i] * 0.05),
                                        X_max[i] + (X_range[i] * 0.05),
                                        num=modelpts)
        return(X_model)
