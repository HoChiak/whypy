# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np

# import local libarys
from whypy.__packages.utils import utils
from whypy.__packages.utils import stats

from importlib import reload
utils=reload(utils)
###############################################################################
class ANM():
    """
    Class for "Additive Noise Model" Methods.
    Please be aware of the assumptions for models of these categorie.
    """
    def __init__(self):
        """
        Class constructor.
        """

    def iter_CM(self, tdep, tindep):
        """
        Method to return [X, Y and the regarding model], controlled by
        parameters tdep (dependent variable) and tindep (independent variable)
        y = Xi[tdep]
        X = Xi[tindep]
        Copy values to make original values independent from scaling
        """
        assert self._xi is not None, 'Xi is None type'
        assert self._regmod is not None, 'Regression Model is None type'
        model = self._regmod[tdep][tindep]
        Y_data = np.copy(self._xi[:, tdep].reshape(-1, 1))
        X_data = np.copy(self._xi[:, tindep].reshape(-1, 1))
        return(model, X_data, Y_data)

    def do_dict2V(self, tdep, tindep, scale, modelpts):
        """
        Method to create further information on a fit. Returns a list for each
        fit including the following values:
        X_model:    # X values in linspace to plot the fitted model
        Y_model:    # Y values in linspace to plot the fitted model
        Y_predict:  predicted values of y given x
        Residuals:  Y_data - Y_predict
        """
        model, X_data, Y_data = self.iter_CM(tdep, tindep)
        # Scale data for predict()
        if scale is True:
            self._scaler[tdep].transform(Y_data)
            self._scaler[tindep].transform(X_data)
        X_range = np.max(X_data) - np.min(X_data)
        # Get model data and y_prediction
        X_model = np.linspace(np.min(X_data) - (X_range * 0.05),
                              np.max(X_data) + (X_range * 0.05),
                              num=modelpts)
        X_model = X_model.reshape(-1, 1)
        Y_model = model.predict(X_model).reshape(-1, 1)
        Y_predict = model.predict(X_data).reshape(-1, 1)
        # Scale data back
        if scale is True:
            self._scaler[tindep].inverse_transform(X_model)
            self._scaler[tdep].inverse_transform(Y_data)
            self._scaler[tdep].inverse_transform(Y_model)
            self._scaler[tdep].inverse_transform(Y_predict)
        # Scale data back
        Residuals = Y_data - Y_predict
        self._dict2V[tdep][tindep] = {'X_model': X_model,
                                      'Y_model': Y_model,
                                      'Y_predict': Y_predict,
                                      'Residuals': Residuals}

    def get_model_stats(self, tdep, tindep):
        """
        Method to get the statistics of the regression model.
        TBD for other models than GAM.
        """
        model, X_data, Y_data = self.iter_CM(tdep, tindep)
        # Differentiate between different models
        if 'pygam' in str(self._regmod[0][1].__class__):
            stat = model.statistics_['p_values']
            self._dict2V[tdep][tindep]['Model_Statistics'] = stat
        else:
            self._dict2V[tdep][tindep]['Model_Statistics'] = 'NaN'

    def fit_model2xi(self, tdep, tindep, scale, modelpts):
        """
        Method to fit model to Xi in the two variable case
        """
        model, X_data, Y_data = self.iter_CM(tdep, tindep)
        if scale is True:
            self._scaler[tdep].transform(Y_data)
            self._scaler[tindep].transform(X_data)
        # Use gridsearch instead of predict if model is pyGAM
        if 'pygam' in str(self._regmod[0][1].__class__):
            model.gridsearch(X_data.reshape(-1, 1), Y_data)
        else:
            model.predict(X_data.reshape(-1, 1), Y_data)
        self.do_dict2V(tdep, tindep, scale, modelpts)
        self.get_model_stats(tdep, tindep)

    def regress(self, scale, testvariant, modelpts):
        """
        Method to do the math. Run trough all possible 2V combinations of
        observations and calculate the inference.
        """
        # Fit Scaler
        if scale is True:
            self.fit_scaler()
        # Initialize empty dictionary to be filled
        self._dict2V = utils.init_2V_list(self._xi.shape[1])
        # Fit (scaled) models and do statistical tests
        self.loop_and_do(do=('fit', 'normality', testvariant),
                         scale=scale, modelpts=modelpts)
        # Get results from independence test
        self.restructure_results(testvariant)
        # PairGrid of all Observations
        self.plt_PairGrid()
