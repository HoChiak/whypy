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
    attr_method = 'anm'

    def __init__(self):
        """
        Class constructor.
        """

    def get_combination_objects(self, i, tdep, tindep):
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
        X_data = np.copy(self._xi[:, tindep].reshape(-1, len(tindep)))
        return(model, X_data, Y_data)

    def get_model_stats(self, i, tdep, tindep, model, X_data, Y_data):
        """
        Method to get the statistics of the regression model.
        TBD for other models than GAM.
        """
        # Differentiate between different models
        if 'pygam' in str(self._regmod[0].__class__):
            stat = model.statistics_['p_values']
            self._results['%i' % (self._numberrun)][i]['Model_Statistics'] = stat
        else:
            self._results['%i' % (self._numberrun)][i]['Model_Statistics'] = 'NaN'

    def fit_model2xi(self, i, tdep, tindep, model, X_data, Y_data):
        """
        Method to fit model to Xi in the two variable case
        """
        if self._config['%i' % (self._numberrun)]['scale'] is True:
            X_data = self.scaler_transform(X_data, tindep)
            Y_data = self.scaler_transform(Y_data, tdep)
        # Use gridsearch instead of fit if model is pyGAM
        if 'pygam' in str(self._regmod[0].__class__):
            model.gridsearch(X_data, Y_data)
        else:
            model.fit(X_data.reshape(-1, len(tindep)), Y_data)

    def predict_results(self, i, tdep, tindep, model, X_data, Y_data):
        """
        Method to create further information on a fit. Returns a list for each
        fit including the following values:
        X_model:    # X values in linspace to plot the fitted model
        Y_model:    # Y values in linspace to plot the fitted model
        Y_predict:  predicted values of y given x
        Residuals:  Y_data - Y_predict
        """
        if self._config['%i' % (self._numberrun)]['scale'] is True:
            X_data = self.scaler_transform(X_data, tindep)
            #Y_data = self.scaler_transform(Y_data, tdep)
        # Get independent model data
        X_model = self.get_Xmodel(X_data, self._config['%i' % (self._numberrun)]['modelpts'])
        # Do Prediction
        Y_model = model.predict(X_model).reshape(-1, 1)
        Y_predict = model.predict(X_data).reshape(-1, 1)
        # Scale data back
        if self._config['%i' % (self._numberrun)]['scale'] is True:
            X_model = self.scaler_inverse_transform(X_model, tindep)
            Y_data = self.scaler_inverse_transform(Y_data, tdep)
            Y_model = self.scaler_inverse_transform(Y_model, tdep)
            Y_predict = self.scaler_inverse_transform(Y_predict, tdep)
        # Scale data back
        Residuals = Y_data - Y_predict
        self._results['%i' % (self._numberrun)][i] = {'X_model': X_model,
                                                      'Y_model': Y_model,
                                                      'Y_predict': Y_predict,
                                                      'Residuals': Residuals}

    def test_normality(self, i, tdep, tindep):
        """
        Method to perform normality test on Xi independent and the
        corresponding Residuals
        """
        ob1 = self._results['%i' % (self._numberrun)][i]['Residuals']
        ob2 = self._xi[:, tindep]
        # normality test on Xi
        tn, tr = stats.normality(ob1)
        self._results['%i' % (self._numberrun)][i]['X_Names'] = tn
        self._results['%i' % (self._numberrun)][i]['X_Results'] = tr
        # normality test on Residuals
        tn, tr = stats.normality(ob2)
        self._results['%i' % (self._numberrun)][i]['Residuals_Names'] = tn
        self._results['%i' % (self._numberrun)][i]['Residuals_Results'] = tr

    def test_independence(self, i, tdep, tindep):
        """
        Method to perform independence test on Xi independent and the
        corresponding Residuals
        """
        ob1 = self._results['%i' % (self._numberrun)][i]['Residuals']
        ob2 = self._xi[:, tindep]
        # independence test on Xi and Residuals
        tn, tr = stats.independence(ob1, ob2)
        self._results['%i' % (self._numberrun)][i]['X-Residuals_Names'] = tn
        self._results['%i' % (self._numberrun)][i]['X-Residuals_Results'] = tr

    def test_likelihood(self, i, tdep, tindep):
        """
        Method to perform
        a) Calculate the likelihood based on variance (only valid for Gaussian)
        """
        ob1 = self._results['%i' % (self._numberrun)][i]['Residuals']
        ob2 = self._xi[:, tindep]
        # Get likelihood based on variance
        tn, tr = stats.likelihood(ob1, ob2)
        # Write in _dict2V
        self._results['%i' % (self._numberrun)][i]['X-Residuals_Names'] = tn
        self._results['%i' % (self._numberrun)][i]['X-Residuals_Results'] = tr

    def run_inference(self):
        """
        Method to do the math. Run trough all possible 2V combinations of
        observations and calculate the inference.
        """
        # Fit Scaler
        if self._config['%i' % (self._numberrun)]['scale'] is True:
            self.scaler_fit()
        # Initialize empty dictionary to be filled
        self._results['%i' % (self._numberrun)] = utils.trans_object_to_list(None, len(self._combinations), dcopy=True)
        # Fit (scaled) models and do statistical tests
        for i in range(len(self._combinations)):
            tdep, tindep = self.get_tINdep(i)
            model, X_data, Y_data = self.get_combination_objects(i, tdep, tindep)
            # fit regmod on observations
            self.fit_model2xi(i, tdep, tindep, model, X_data, Y_data)
            # predict results
            self.predict_results(i, tdep, tindep, model, X_data, Y_data)
            self.get_model_stats(i, tdep, tindep, model, X_data, Y_data)
            # do normality test
            self.test_normality(i, tdep, tindep)
            # do independence test
            if self._config['%i' % (self._numberrun)]['testtype'] is 'independence':
                self.test_independence(i, tdep, tindep)
            # do calculate the likelihood
            if self._config['%i' % (self._numberrun)]['testtype'] is 'likelihood':
                self.test_likelihood(i, tdep, tindep)
        # Get results from independence test
        self.restructure_results()
        # PairGrid of all Observations
        self.plt_PairGrid()
