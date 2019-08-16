# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

# import local libarys
from whypy.__packages.utils import utils
from whypy.__packages.utils import stats

###############################################################################
class RunANM():
    """
    Class for "Additive Noise Model" Methods.
    Please be aware of the assumptions for models of these categorie.
    """
    attr_method = 'anm'

    def __init__(self):
        """
        Class constructor.
        """

    def get_combination_objects(self, combno, tdep, tindep):
        """
        Method to return [X, Y and the regarding model], controlled by index combno,
        where i is in (0, number_of_combinations, 1).
        In Combinations, the first value of the nested list is always the
        dependent variable whereas the other values are the independent
        variables. Copy values to make original values independent from
        scaling.
        """
        model = self._regmod[combno]
        Y_data = np.copy(self._xi[:, tdep]).reshape(-1, 1)
        X_data = np.copy(self._xi[:, tindep]).reshape(-1, len(tindep))
        return(model, X_data, Y_data)

    def fit_model2xi(self, combno, tdep, tindep, model, X_data, Y_data):
        """
        Method to fit model to Xi in the two variable case
        """
        # Scale data forward
        if self._config['%i' % (self._numberrun)]['scale'] is True:
            X_data = self.scaler_transform(X_data, tindep)
            Y_data = self.scaler_transform(Y_data, tdep)
        # Use gridsearch instead of fit if model is pyGAM
        if 'pygam' in str(self._regmod[0].__class__):
            model.gridsearch(X_data.reshape(-1, len(tindep)), Y_data)
        else:
            model.fit(X_data.reshape(-1, len(tindep)), Y_data)
        # Scale data back
        if self._config['%i' % (self._numberrun)]['scale'] is True:
            X_data = self.scaler_inverse_transform(X_data, tindep)
            Y_data = self.scaler_inverse_transform(Y_data, tdep)

    def predict_results(self, combno, tdep, tindep, model, X_data, Y_data):
        """
        Method to create further information on a fit. Returns a list for each
        fit including the following values:
        X_model:    # X values in linspace to plot the fitted model
        Y_model:    # Y values in linspace to plot the fitted model
        Y_predict:  predicted values of y given x
        Residuals:  Y_data - Y_predict
        """
        # Scale data forward
        if self._config['%i' % (self._numberrun)]['scale'] is True:
            X_data = self.scaler_transform(X_data, tindep)
            Y_data = self.scaler_transform(Y_data, tdep)
        # Get independent model data
        modelpts = self._config['%i' % (self._numberrun)]['modelpts']
        X_model = self.get_Xmodel(X_data, modelpts)
        # Do Prediction
        Y_model = model.predict(X_model).reshape(-1, 1)
        Y_predict = model.predict(X_data).reshape(-1, 1)
        # Scale data back
        if self._config['%i' % (self._numberrun)]['scale'] is True:
            X_data = self.scaler_inverse_transform(X_data, tindep)
            Y_data = self.scaler_inverse_transform(Y_data, tdep)
            X_model = self.scaler_inverse_transform(X_model, tindep)
            Y_model = self.scaler_inverse_transform(Y_model, tdep)
            Y_predict = self.scaler_inverse_transform(Y_predict, tdep)
        # Scale data back
        Residuals = Y_data - Y_predict
        self._results['%i' % (self._numberrun)][combno] = {'X_model': X_model,
                                                           'Y_model': Y_model,
                                                           'Y_predict': Y_predict,
                                                           'Residuals': Residuals}

    def do_statistics(self, combno, obs_name, test_stat,
                      obs_valu1, obs_valu2=None):
        """
        Method to comprehense statistical tests
        """
        if test_stat is 'Normality':
            tr = stats.normality(obs_valu1)
        elif test_stat is 'Likelihood':
            tr = stats.likelihood(obs_valu1, obs_valu2)
        elif test_stat is 'KolmogorovSmirnoff':
            tr = stats.kolmogorov(obs_valu1, obs_valu2)
        elif test_stat is 'MannWhitney':
            tr = stats.mannwhitneyu(obs_valu1, obs_valu2)
        else:
            print('Given test_stat argument is not defined.')
        self._results['%i' % (self._numberrun)][combno]['%s' % (obs_name)] = tr


    def test_statistics(self, combno, tdep, tindep, model, X_data, Y_data):
        """
        Method to perform statistical tests on the given and predicted data.
        """
        for temp_i, temp_tindep in enumerate(tindep):
            # Normality Test on X_data
            self.do_statistics(combno, 'Normality_X_data_%i' % (temp_tindep), 'Normality',
                               obs_valu1=X_data[:, temp_i], obs_valu2=None)
            # Test Independence of Residuals
            self.do_statistics(combno, 'IndepResiduals_%i' % (temp_tindep),
                               self._config['%i' % (self._numberrun)]['testtype'],
                               obs_valu1=self._results['%i' % (self._numberrun)][combno]['Residuals'],
                               obs_valu2=X_data[:, temp_i])
        # Normality Test on Residuals
        self.do_statistics(combno, 'Normality_Residuals', 'Normality',
                           obs_valu1=self._results['%i' % (self._numberrun)][combno]['Residuals'],
                           obs_valu2=None)
        # Normality Test on Y_data
        self.do_statistics(combno, 'Normality_Y_data', 'Normality',
                           obs_valu1=Y_data,
                           obs_valu2=None)
        # Test Goodness of Fit
        self.do_statistics(combno, 'GoodnessFit',
                           self._config['%i' % (self._numberrun)]['testtype'],
                           obs_valu1=self._results['%i' % (self._numberrun)][combno]['Y_predict'],
                           obs_valu2=Y_data)

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
        for combno in range(len(self._combinations)):
            tdep, tindep = self.get_tINdep(combno)
            model, X_data, Y_data = self.get_combination_objects(combno, tdep, tindep)
            # fit regmod on observations
            self.fit_model2xi(combno, tdep, tindep, model, X_data, Y_data)
            # predict results
            self.predict_results(combno, tdep, tindep, model, X_data, Y_data)
            # do statistical tests
            self.test_statistics(combno, tdep, tindep, model, X_data, Y_data)
            # do independence test

class PlotANM():
    """
    Class for "Additive Noise Model" Methods.
    Please be aware of the assumptions for models of these categorie.
    """

    def __init__(self):
        """
        Class constructor.
        """

    def get_std_txt(self, combno, tdep, tindep):
        """
        Libary of some standard text phrases
        """
        txt = r'X_%i ~ f(X_%combno, E_X)' % (tdep, tindep)
        return(txt)

    def get_math_txt(self, combno, tdep, tindep):
        """
        Libary of some standard text phrases
        """
        txt = r'X_{%i} \approx f\left( X_{%i}, E_{X}\right)' % (tdep, tindep)
        return(txt)

    def plt_PairGrid(self):
        """
        Method to plot a PairGrid scatter of the observations.
        """
        plt.figure(r'PairGrid',
                   figsize=self._figsize)
        df = pd.DataFrame(self._xi)
        df.columns = [r'$X_%i$' % (i) for i in range(self._xi.shape[1])]
        g = sns.PairGrid(df)
        g = g.map(plt.scatter)
        plt.show();

    def plt_1model_adv(self, combno, tdep, temp_i, tindep):
        """
        Method to plot a scatter of the samples, the fitted model and the
        residuals. Plot joint distribution and marginals.
        """
        txt = self.get_math_txt(combno, tdep, tindep)
        g = sns.JointGrid(self._xi[:, tindep], self._xi[:, tdep],
                          height=self._figsize[0]*5/6,
                          ratio=int(5)
                          )
        g.plot_joint(plt.scatter)
        plt.plot(self._results['%i' % (self._numberrun)][combno]['X_model'][:, temp_i],
                 self._results['%i' % (self._numberrun)][combno]['Y_model'],
                 c='r')
        plt.scatter(self._xi[:, tindep],
                    self._results['%i' % (self._numberrun)][combno]['Residuals'])
        plt.legend([r'$Model\ %s$' % (txt),
                    r'$Observations$',
                    r'$Residuals\ (X_{%i}-\hatX_{%i})$' % (tdep, tdep)])
        plt.xlabel(r'$X_{%i}$' % (tindep))
        plt.ylabel(r'$X_{%i}$' % (tdep))
        g.plot_marginals(sns.distplot, kde=True)
        plt.show();

    def plt_hist_IndepResiduals(self, combno, tdep, temp_i, tindep):
        """
        Method to plot a histogramm of both the independent sample and the
        Residuals
        """
        txt = self.get_math_txt(combno, tdep, tindep)
        plt.figure(r'Independence of Residuals: %s' % (txt),
                   figsize=self._figsize)
        sns.distplot(self._xi[:, tindep],
                     norm_hist=True)
        sns.distplot(self._results['%i' % (self._numberrun)][combno]['Residuals'],
                     norm_hist=True)
        plt.legend([r'$X_{%i}$' % (tindep),
                    r'$Residuals\ (X_{%i}-\hatX_{%i})$' % (tdep, tdep)])
        plt.title(r'$Independence\ of\ Residuals:\ %s$' % (txt),
                  fontweight='bold')
        plt.xlabel(r'$X_{i}$')
        plt.ylabel(r'$f\left(X_{i}\right)$')
        plt.show()

    def plt_hist_GoodnessFit(self, combno, tdep, temp_i, tindep):
        """
        Method to plot a histogramm of both the independent sample and the
        Residuals
        """
        txt = self.get_math_txt(combno, tdep, tindep)
        plt.figure(r'Goodness of Fit: %s' % (txt),
                   figsize=self._figsize)
        sns.distplot(self._xi[:, tdep],
                     norm_hist=True)
        sns.distplot(self._results['%i' % (self._numberrun)][combno]['Y_predict'],
                     norm_hist=True)
        plt.legend([r'$X_{%i}$' % (tdep),
                    r'$\hatX_{%i}$' % (tdep)])
        plt.title(r'$Goodness\ of\ Fit:\ %s$' % (txt),
                  fontweight='bold')
        plt.xlabel(r'$X_{%i}$' % (tdep))
        plt.ylabel(r'$f\left(X_{i}\right)$')
        plt.show()

    def plot_inference(self):
        """
        Method to visualize the interference
        """
        # Pairgrid Plot of Observations
        utils.print_in_console(what='pairgrid header')
        self.plt_PairGrid()
        # Iterate over combinations
        for combno in range(len(self._combinations)):
            tdep, tindep = self.get_tINdep(combno)
            tdep = tdep[0]
            utils.print_in_console(what='combination major header',
                                   tdep=tdep, tindep=tindep)
           # Iterate over independent variables
            for temp_i, temp_tindep in enumerate(tindep):
                # Plot Tindep vs Tdep
                if self.attr_variate is 'mvariate':
                    utils.print_in_console(what='combination minor header',
                                           tdep=tdep, tindep=temp_tindep)
                self.plt_1model_adv(combno, tdep, temp_i, temp_tindep)
                self.plt_hist_IndepResiduals(combno, tdep, temp_i, temp_tindep)
            self.plt_hist_GoodnessFit(combno, tdep, temp_i, temp_tindep)


    def loop_and_do(self, do, **kwargs):
        """
        Method to scale (if scale==True) and loop trough possible combinations
        of tdep and tindep for modelfit of residuals. Save result in _dict2V.
        """
        # Loop trough possible combinations of tdep and tindep for modelfit
        for i in range(len(self._combinations)):
            tdep, tindep = self.get_tINdep(combno)
            # print/plot model informations
            if 'out_Regr_Model_info' in do:
                try:
                    self.plt_GAMlog(combno, tdep, tindep)
                except:
                    print('An exception occurred using -plt_GAMlog()-')
                try:
                    utils.print_in_console(what='model summary')
                    self._regmod[combno].summary()
                except:
                    print('An exception occurred using -summary()-')
            # print the normality log
            if 'out_X_Residuals_NormalityTest' in do:
                self.print_log_st(combno, tdep, tindep, 'normality')
            # print the independence log
            if 'out_X_vs_Residuals_info' in do:
                self.print_log_st(combno, tdep, tindep, 'independence')

class ResultsANM():
    """
    Class for "Additive Noise Model" Methods.
    Please be aware of the assumptions for models of these categorie.
    """

    def __init__(self):
        """
        Class constructor.
        """

    def plot_results(self):
        """
        Method to display the results of the interference
        """
        print('Method not defined yet')
        #         self.restructure_results()
        # # Plot results
        # self.plot_results()
        #     # Loop trough possible combinations of tdep and tindep for plots/logs
        #     # Define a list of do's (dolist) for plots sorted by tdep/tindep
        #     # combinations. Start dolist:
        #     dic = {'KolmogorovSmirnoff': 'p-value', 'Likelihood': 'likelihood'}
        #     namecode = dic[testtype]
        #     dolist = []
        #     if out_Regr_Model is True:
        #         dolist.append('out_Regr_Model')
        #     if out_Regr_Model_info is True:
        #         dolist.append('out_Regr_Model_info')
        #     if out_X_Residuals_NormalityTest is True:
        #         dolist.append('out_X_Residuals_NormalityTest')
        #     if out_X_vs_Residuals_info is True:
        #         if 'p-value' in namecode:
        #             dolist.append('out_X_vs_Residuals_info')
        #         else:
        #             print('X vs Residual log only available for independence test')
        #     if len(dolist) != 0:
        #         self.loop_and_do(do=dolist)
        #     # end dolist
        #     utils.print_in_console(what='result header')
        #     # Plot independence/likelihood tests results
        #     if out_X_vs_Residuals is True:
        #         if 'p-value' in namecode:
        #             self.plt_2metrics_groupedby(namecode)
        #         self.plt_1metric_groupedby(namecode)
        #     # Print independence/likelihood tests results
        #     if out_Results_testtype is True:
        #         rs = self._results2V['%i' % (self._numberrun)]
        #         print(rs[['TestType', '2V-case', 'pval/likel',
        #                   'rank pval/likel', '2V-direction']].to_string())
        #     # plot the Causal Graph
        #     if out_CausalGraph is True:
        #         utils.print_in_console(what='CG Warning')
        #         self.predict_CG(testtype, CGmetric=CGmetric)
