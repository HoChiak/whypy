# -*- coding: utf-8 -*-

# import built in libarys
from json import dumps as jdump
from json import loads as jload
from copy import deepcopy

# import 3rd party libarys
import numpy as np
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split

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
        self._results = {}
        self._results_df = {}

    def __del__(self):
        """
        Class deconstructor.
        """
        self._results = {}
        self._results_df = {}

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
        if self._config['scale'] is True:
            X_data = self.scaler_transform(X_data, tindep)
            Y_data = self.scaler_transform(Y_data, tdep)
        # Use gridsearch instead of fit if model is pyGAM
        if 'pygam' in str(self._regmod[0].__class__):
            model.gridsearch(X_data.reshape(-1, len(tindep)), Y_data)
        else:
            model.fit(X_data.reshape(-1, len(tindep)), Y_data)
        # Scale data back
        if self._config['scale'] is True:
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
        if self._config['scale'] is True:
            X_data = self.scaler_transform(X_data, tindep)
            Y_data = self.scaler_transform(Y_data, tdep)
        # Get independent model data
        modelpts = self._config['modelpts']
        X_model = self.get_Xmodel(X_data, modelpts)
        # Do Prediction
        Y_model = model.predict(X_model).reshape(-1, 1)
        Y_predict = model.predict(X_data).reshape(-1, 1)
        # Scale data back
        if self._config['scale'] is True:
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
        elif test_stat is 'HSIC':
            tr = stats.hsic_gam(obs_valu1, obs_valu2)
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
                               self._config['testtype'],
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
                           self._config['testtype'],
                           obs_valu1=self._results['%i' % (self._numberrun)][combno]['Y_predict'],
                           obs_valu2=Y_data)

    def run_inference(self):
        """
        Method to do the math. Run trough all possible 2V combinations of
        observations and calculate the inference.
        """
        # Fit Scaler
        if self._config['scale'] is True:
            self.scaler_fit()
        # Initialize empty dictionary to be filled
        self._results['%i' % (self._numberrun)] = utils.trans_object_to_list(None, len(self._combinations), dcopy=True)
        # Holdout if defined
        if self._config['holdout'] is True:
            xi_original = deepcopy(self._xi)
            xi_fit, xi_test = train_test_split(self._xi,
                                               test_size = self._kwargs['holdout_ratio'],
                                               random_state = self._kwargs['holdout_seed'],
                                               shuffle = False)
        # Fit (scaled) models and do statistical tests
        for combno in range(len(self._combinations)):
            # Holdout if defined
            if self._config['holdout'] is True:
                self._xi = xi_fit
            # Get Constants
            tdep, tindep = self.get_tINdep(combno)
            model, X_data, Y_data = self.get_combination_objects(combno, tdep, tindep)
            # fit regmod on observations
            self.fit_model2xi(combno, tdep, tindep, model, X_data, Y_data)
            # predict results
            self.predict_results(combno, tdep, tindep, model, X_data, Y_data)
            # do statistical tests
            if self._config['holdout'] is True:
                self._xi = xi_test
            self.test_statistics(combno, tdep, tindep, model, X_data, Y_data)
            # Regain original _xi
            if self._config['holdout'] is True:
                self._xi = deepcopy(xi_original)

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
        g = sns.PairGrid(df);
        g = g.map(plt.scatter);
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
        plt.title(r'$\bf{Independence\ of\ Residuals:\ %s}$' % (txt))
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
        plt.title(r'$\bf{Goodness\ of\ Fit:\ %s}$' % (txt))
        plt.xlabel(r'$X_{%i}$' % (tdep))
        plt.ylabel(r'$f\left(X_{i}\right)$')
        plt.show()

    def plot_inference(self):
        """
        Method to visualize the interference
        """
        # Holdout if defined
        if self._config['holdout'] is True:
            xi_fit, xi_test = train_test_split(self._xi,
                                               test_size = self._kwargs['holdout_ratio'],
                                               random_state = self._kwargs['holdout_seed'],
                                               shuffle = False)
        utils.display_text_predefined(what='visualization header')
        # Pairgrid Plot of Observations
        utils.display_text_predefined(what='pairgrid header')
        self.plt_PairGrid()
        # Iterate over combinations
        for combno in range(len(self._combinations)):
            tdep, tindep = self.get_tINdep(combno)
            tdep = tdep[0]
            utils.display_text_predefined(what='combination major header',
                                          tdep=tdep, tindep=tindep)
           # Iterate over independent variables
            for temp_i, temp_tindep in enumerate(tindep):
                # Holdout if defined
                if self._config['holdout'] is True:
                    self._xi = xi_fit
                # Plot Tindep vs Tdep
                utils.display_text_predefined(what='combination minor header',
                                              tdep=tdep, tindep=temp_tindep)
                self.plt_1model_adv(combno, tdep, temp_i, temp_tindep)
                # Holdout if defined
                if self._config['holdout'] is True:
                    self._xi = xi_test
                self.plt_hist_IndepResiduals(combno, tdep, temp_i, temp_tindep)
            self.plt_hist_GoodnessFit(combno, tdep, temp_i, temp_tindep)


class ResultsANM():
    """
    Class for "Additive Noise Model" Methods.
    Please be aware of the assumptions for models of these categorie.
    """

    def __init__(self):
        """
        Class constructor.
        """

    def bootstrap_to_mean_var(self, combno, namekey, dict, dictkey):
        """
        Method to get mean and variance from bootstrap runs
        """
        for temp_test in self._results['0'][combno][namekey].keys():
            newlist = list()
            for temp_boot in self._results.keys():
                newlist.append(self._results[temp_boot][combno][namekey][temp_test])
            newarray = np.array(newlist).flatten()
            medianarray = np.median(newarray)
            vararray = np.std(newarray)
            dict[dictkey+' '+str(temp_test)+' [List]'] = jdump(newlist)
            dict[dictkey+' '+str(temp_test)+' [Median]'] = medianarray
            dict[dictkey+' '+str(temp_test)+' [SD]'] = vararray
        return(dict)

    def restructure_results(self):
        """
        Method to extract a readable DataFrame from the self._results attribute
        """
        # Init new DataFrame
        results_df = pd.DataFrame()
        # Iterate over all possible combinations
        for combno in range(len(self._combinations)):
            tdep, tindep = self.get_tINdep(combno)
            tdep = tdep[0]
            # Iterate over bivariate comparisons
            for temp_i, temp_tindep in enumerate(tindep):
                # Init new dict
                df_dict = {}
                df_dict['Fitted Combination'] = r'$X_%i \sim f(X_{%s})$' % (tdep, tindep)
                df_dict['tdep'] = tdep
                df_dict['tindep'] = tindep
                df_dict['Bivariate Comparison'] = r'$X_%i \sim f(X_%s)$' % (tdep, temp_tindep)
                df_dict['bivariate tindep'] = temp_tindep
                # Get Mean and Variance Value out of all bootstrap examples
                df_dict = self.bootstrap_to_mean_var(combno, 'Normality_X_data_%i' % (temp_tindep), df_dict, 'Normality Indep. Variable')
                df_dict = self.bootstrap_to_mean_var(combno, 'Normality_Y_data', df_dict, 'Normality Depen. Variable')
                df_dict = self.bootstrap_to_mean_var(combno, 'Normality_Residuals', df_dict, 'Normality Residuals')
                df_dict = self.bootstrap_to_mean_var(combno, 'IndepResiduals_%i' % (temp_tindep), df_dict, 'Dependence: Indep. Variable - Residuals')
                df_dict = self.bootstrap_to_mean_var(combno, 'GoodnessFit', df_dict, 'Dependence: Depen. Variable - Prediction (GoF)')
            # Append current bivariate comparison to DF
            new_df = pd.DataFrame(df_dict)
            results_df = pd.concat([results_df, new_df], ignore_index=True, axis=0)
        self._results_df = results_df

    def get_df_normality(self, testkey):
        """
        Method to return the DF summarizing the normality test
        """
        columns = ['Fitted Combination',
                   'Bivariate Comparison',
                   'Normality Indep. Variable SW_pvalue [Median]',
                   'Normality Indep. Variable SW_pvalue [SD]',
                   'Normality Indep. Variable Pearson_pvalue [Median]',
                   'Normality Indep. Variable Pearson_pvalue [SD]',
                   'Normality Indep. Variable Combined_pvalue [Median]',
                   'Normality Indep. Variable Combined_pvalue [SD]',
                   'Normality Depen. Variable SW_pvalue [Median]',
                   'Normality Depen. Variable SW_pvalue [SD]',
                   'Normality Depen. Variable Pearson_pvalue [Median]',
                   'Normality Depen. Variable Pearson_pvalue [SD]',
                   'Normality Depen. Variable Combined_pvalue [Median]',
                   'Normality Depen. Variable Combined_pvalue [SD]',
                   'Normality Residuals SW_pvalue [Median]',
                   'Normality Residuals SW_pvalue [SD]',
                   'Normality Residuals Pearson_pvalue [Median]',
                   'Normality Residuals Pearson_pvalue [SD]',
                   'Normality Residuals Combined_pvalue [Median]',
                   'Normality Residuals Combined_pvalue [SD]']
        # Extract only testtype of interest
        _columns = [key for key in columns if testkey in key]
        columns = columns[0:2]
        columns.extend(_columns)
        # Remove "fitted Combination" if bivariate case
        if self.attr_variate is 'bivariate':
            columns = columns[1:]
        # Remove [SD] if no boostrap is done
        if self._config['bootstrap'] is False:
            columns = [key for key in columns if '[SD]' not in key]
        df = self._results_df[columns]
        # Clean up Columns
        columns = [key.replace('Normality ', '') for key in columns]
        columns = [key.replace(testkey+str(' '), '') for key in columns]
        df.columns = columns
        return(df)

    def get_df_dependence(self, testkey, removeList=True):
        """
        Method to return the DF summarizing the normality test
        """
        columns = ['Fitted Combination',
                   'Bivariate Comparison',
                   'Dependence: Indep. Variable - Residuals %s [List]' % (self._config['testtype']),
                   'Dependence: Indep. Variable - Residuals %s [Median]' % (self._config['testtype']),
                   'Dependence: Indep. Variable - Residuals %s [SD]' % (self._config['testtype']),
                   'Dependence: Depen. Variable - Prediction (GoF) %s [List]' % (self._config['testtype']),
                   'Dependence: Depen. Variable - Prediction (GoF) %s [Median]' % (self._config['testtype']),
                   'Dependence: Depen. Variable - Prediction (GoF) %s [SD]' % (self._config['testtype'])]
        # Extract only testtype of interest
        _columns = [key for key in columns if testkey in key]
        columns = columns[0:2]
        columns.extend(_columns)
        # Remove [SD] if no boostrap is done
        if self._config['bootstrap'] is False:
            columns = [key for key in columns if '[SD]' not in key]
        # Remove [SD] if no boostrap is done
        if removeList is True:
            columns = [key for key in columns if '[List]' not in key]
            # Remove "fitted Combination" if bivariate case
            if self.attr_variate is 'bivariate':
                columns = columns[1:]
        else:
            columns = [key for key in columns if '[Median]' not in key]
            columns = [key for key in columns if '[SD]' not in key]
        df = self._results_df[columns]
        # Clean up Columns
        columns = [key.replace('Dependence: ', '') for key in columns]
        columns = [key.replace('(GoF) ', '') for key in columns]
        columns = [key.replace( self._config['testtype']+str(' '), '') for key in columns]
        df.columns = columns
        return(df)

    def plt_combinations_boxplot(self, testkey):
        """
        Method to plot the results of the independence test.
        """
        # Get Dictionary
        df_dependence = self.get_df_dependence(testkey, removeList=False)
        # Init Plot
        plt.figure('Combination Boxplot', figsize=[self._figsize[0], self._figsize[1]/1.5])
        if 'p-value' in self.attr_dict[self._config['testtype']]:
            plt.yscale('log')
            lbl = r'$dependence \leftarrow\ p-value\ \rightarrow independence$'
        elif 'likelihood-ratio' in self.attr_dict[self._config['testtype']]:
            lbl = r'$not favored \leftarrow\ likelihood-ratio\ \rightarrow favored$'
        # Get Data for each bivariate case from dictionary
        x_data = np.arange(0.5, df_dependence.shape[0]+0.5, 1)
        y_data = [jload(bivacomp[-1]) for i, bivacomp in df_dependence.iterrows()]
        labels_box = [r'$\bf{%i}:$' % (i) + '\n%s' % (bivacomp[1]) for i, bivacomp in df_dependence.iterrows()]
        combnos = [bivacomp[0] for i, bivacomp in df_dependence.iterrows()]
        # Get Unique combinations and give number to them
        combno_unique = set(combnos)
        combno_unique = {key: i for i, key in enumerate(combno_unique)}
        # Background Color different bivariate cases from same combinations
        for i, combno in enumerate(combnos):
            plt.axvspan(x_data[i]-1/3, x_data[i]+1/3,
                        facecolor=self._cmap(combno_unique[combno]/len(combno_unique)), alpha=0.5)
        plt.legend(combnos,
                   loc='upper center',
                   bbox_to_anchor=(0.5, -0.25),
                   ncol=3)
        # Boxplot
        plt.boxplot(y_data, positions=x_data, labels=labels_box)
        # Further Plot Settings
        plt.title(r'BoxPlot', fontweight='bold')
        plt.tick_params(labelbottom=True)
        plt.tick_params(right=False, top=False, left=True, bottom=False)
        plt.ylabel(lbl)
        plt.show()

    def plot_results(self, number_run=False):
        """
        Method to display the results of the interference. If number_run is
        False the current run will be plotted
        """
        # Create self._results_df to get results in a handy way
        self.restructure_results()
        # Plot Header and Configuration:
        utils.display_text_predefined(what='result header', dict=self._config)
        # Plot Normality DataFrame
        utils.display_text_predefined(what='normality')
        utils.display_text_predefined(what='thirdlevel', key='Pearsons p-value')
        utils.display_df(self.get_df_normality(testkey='Pearson_pvalue'), fontsize='6pt')
        utils.display_text_predefined(what='thirdlevel', key='Shapiro Wilk p-value')
        utils.display_df(self.get_df_normality(testkey='SW_pvalue'), fontsize='6pt')
        utils.display_text_predefined(what='thirdlevel', key='Combined p-value')
        utils.display_df(self.get_df_normality(testkey='Combined_pvalue'), fontsize='6pt')
        # Plot Goodness of Fit Test
        utils.display_text_predefined(what='dependence prediction')
        utils.display_text_predefined(what='thirdlevel', key='%s: %s' % (self._config['testtype'], self.attr_dict[self._config['testtype']]))
        utils.display_df(self.get_df_dependence('GoF'))
        self.plt_combinations_boxplot('GoF')
        # Plot Indepndence of Residuals
        utils.display_text_predefined(what='dependence residuals')
        utils.display_text_predefined(what='thirdlevel', key='%s: %s' % (self._config['testtype'], self.attr_dict[self._config['testtype']]))
        utils.display_df(self.get_df_dependence('Residuals'))
        self.plt_combinations_boxplot('Residuals')




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
        #     utils.display_text_predefined(what='result header')
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
        #         utils.display_text_predefined(what='CG Warning')
        #         self.predict_CG(testtype, CGmetric=CGmetric)
