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

    def iter_combination(self, i, tdep, tindep):
        """
        Method to return [X, Y and the regarding model], controlled by index i,
        where i is in (0, number_of_combinations, 1).
        In Combinations, the first value of the nested list is always the
        dependent variable whereas the other values are the independent
        variables. Copy values to make original values independent from
        scaling.
        """
        model = self._regmod[i]
        Y_data = np.copy(self._xi[:, tdep].reshape(-1, 1))
        X_data = np.copy(self._xi[:, tindep].reshape(-1, tindep.shape[0]))
        return(model, X_data, Y_data)

    def get_results(self, i, tdep, tindep, scale, modelpts):
        """
        Method to create further information on a fit. Returns a list for each
        fit including the following values:
        X_model:    # X values in linspace to plot the fitted model
        Y_model:    # Y values in linspace to plot the fitted model
        Y_predict:  predicted values of y given x
        Residuals:  Y_data - Y_predict
        """
        model, X_data, Y_data = self.iter_combination(i, tdep, tindep)
        if scale is True:
            X_data = self.transform_with_scaler(X_data, tindep)
            Y_data = self.transform_with_scaler(Y_data, tdep)
        # Get independent model data
        X_model = self.get_Xmodel(X_data, modelpts)
        # Do Prediction
        Y_model = model.predict(X_model).reshape(-1, 1)
        Y_predict = model.predict(X_data).reshape(-1, 1)
        # Scale data back
        if scale is True:
            X_model = self.inverse_transform_with_scaler(X_model, tindep)
            Y_data = self.inverse_transform_with_scaler(Y_data, tdep)
            Y_model = self.inverse_transform_with_scaler(Y_model, tdep)
            Y_predict = self.inverse_transform_with_scaler(Y_predict, tdep)
        # Scale data back
        Residuals = Y_data - Y_predict
        self._results[i] = {'X_model': X_model,
                            'Y_model': Y_model,
                            'Y_predict': Y_predict,
                            'Residuals': Residuals}

    def get_model_stats(self, i, tdep, tindep):
        """
        Method to get the statistics of the regression model.
        TBD for other models than GAM.
        """
        model, X_data, Y_data = self.iter_combination(i, tdep, tindep)
        # Differentiate between different models
        if 'pygam' in str(self._regmod[0].__class__):
            stat = model.statistics_['p_values']
            self._results[i]['Model_Statistics'] = stat
        else:
            self._results[i]['Model_Statistics'] = 'NaN'

    def fit_model2xi(self, i, tdep, tindep, scale, modelpts):
        """
        Method to fit model to Xi in the two variable case
        """
        model, X_data, Y_data = self.iter_combination(i, tdep, tindep)
        if scale is True:
            X_data = self.transform_with_scaler(X_data, tindep)
            Y_data = self.transform_with_scaler(Y_data, tdep)
        # Use gridsearch instead of predict if model is pyGAM
        if 'pygam' in str(self._regmod[0].__class__):
            model.gridsearch(X_data, Y_data)
        else:
            model.predict(X_data.reshape(-1, 1), Y_data)
        self.get_results(i, tdep, tindep, scale, modelpts)
        self.get_model_stats(i, tdep, tindep)

    def run_inference(self, scale, testvariant, modelpts):
        """
        Method to do the math. Run trough all possible 2V combinations of
        observations and calculate the inference.
        """
        # Fit Scaler
        if scale is True:
            self.fit_scaler()
        # Initialize empty dictionary to be filled
        self._results = self.object_to_list(None, self._combinations.shape[0])
        # Fit (scaled) models and do statistical tests
        self.loop_and_do(do=('fit', 'normality', testvariant),
                         scale=scale, modelpts=modelpts)
        # Get results from independence test
        self.restructure_results(testvariant)
        # PairGrid of all Observations
        self.plt_PairGrid()
