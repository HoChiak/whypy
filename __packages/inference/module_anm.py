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
from sklearn.model_selection import GridSearchCV

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

    def get_combination_objects(self, combno, tdep, tindep, ids_list):
        """
        Method to return [X, Y and the regarding model], controlled by index combno,
        where i is in (0, number_of_combinations, 1).
        In Combinations, the first value of the nested list is always the
        dependent variable whereas the other values are the independent
        variables. Copy values to make original values independent from
        scaling.
        """
        model = self._regmod[combno]
        Y_data = np.copy(self._xi[ids_list, tdep]).reshape(-1, 1)
        X_data = np.copy(self._xi[ids_list, tindep]).reshape(-1, len(tindep))
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
        if self._kwargs['gridsearch'] is True:
            if 'pygam' in str(self._regmod[0].__class__):
                model.gridsearch(X_data.reshape(-1, len(tindep)), Y_data)
            else:
                grid_search = GridSearchCV(model, self._kwargs['param_grid'])
                grid_search.fit(X_data.reshape(-1, len(tindep)), Y_data)
                #### TBD check if redundant
                model.set_params(grid_search.best_params_)
                model.fit(X_data.reshape(-1, len(tindep)), Y_data)
        else:
            model.fit(X_data.reshape(-1, len(tindep)), Y_data)
        # Scale data back
        if self._config['scale'] is True:
            X_data = self.scaler_inverse_transform(X_data, tindep)
            Y_data = self.scaler_inverse_transform(Y_data, tdep)

    def predict_model(self, combno, tdep, tindep, model, X_data, Y_data):
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
        modelpts = self._kwargs['modelpts']
        X_model = self.get_Xmodel(X_data, modelpts)
        # Do Prediction
        Y_model = model.predict(X_model).reshape(-1, 1)
        # Scale data back
        if self._config['scale'] is True:
            X_data = self.scaler_inverse_transform(X_data, tindep)
            Y_data = self.scaler_inverse_transform(Y_data, tdep)
            X_model = self.scaler_inverse_transform(X_model, tindep)
            Y_model = self.scaler_inverse_transform(Y_model, tdep)
        self._results['%i' % (self._numberrun)][combno]['X_model'] = X_model
        self._results['%i' % (self._numberrun)][combno]['Y_model'] = Y_model

    def predict_residuals(self, combno, tdep, tindep, model, X_data, Y_data):
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
        # Do Prediction
        Y_predict = model.predict(X_data).reshape(-1, 1)
        # Scale data back
        if self._config['scale'] is True:
            X_data = self.scaler_inverse_transform(X_data, tindep)
            Y_predict = self.scaler_inverse_transform(Y_predict, tdep)
        # Get residuals
        Residuals = Y_data - Y_predict
        self._results['%i' % (self._numberrun)][combno]['Y_predict'] = Y_predict
        self._results['%i' % (self._numberrun)][combno]['Residuals'] = Residuals

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
        # Initialize empty list to be filled
        self._results['%i' % (self._numberrun)] = utils.trans_object_to_list(None, len(self._combinations), dcopy=True)
        # Fit (scaled) models and do statistical tests
        for combno in range(len(self._combinations)):
            # Initialize empty dictionary to be filled
            self._results['%i' % (self._numberrun)][combno] = {}
            # Get Constants
            tdep, tindep = self.get_tINdep(combno)
            # Get Constants
            model, X_data, Y_data = self.get_combination_objects(combno, tdep, tindep, self._ids_fit)
            # fit regmod on observations
            self.fit_model2xi(combno, tdep, tindep, model, X_data, Y_data)
            # predict model points
            self.predict_model(combno, tdep, tindep, model, X_data, Y_data)
            # Get Constants
            model, X_data, Y_data = self.get_combination_objects(combno, tdep, tindep, self._ids_test)
            # predict residuals
            self.predict_residuals(combno, tdep, tindep, model, X_data, Y_data)
            # do statistical tests
            self.test_statistics(combno, tdep, tindep, model, X_data, Y_data)

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
        txt = r'X_{%i} ~ f(X_{%combno}, E_X)' % (tdep, tindep)
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
        # Differentiate between holdout case
        if self._config['holdout'] is True:
            # Get Holdout for hue
            hue = np.zeros((self._xi.shape[0], ))
            hue[self._ids_fit] = 1
            hue = hue.tolist()
            hue = ['test' if i==0 else 'fit' for i in hue]
        else:
            hue = ['fit/test' for i in range(self._xi.shape[0])]
        df = pd.DataFrame(self._xi)
        df.columns = [r'$X_{%i}$' % (i) for i in range(self._xi.shape[1])]
        df = pd.concat([df, pd.DataFrame(hue, columns=['holdout'])], axis=1)
        g = sns.PairGrid(df, hue='holdout');
        g = g.map_diag(plt.hist, edgecolor="w")
        g = g.map_offdiag(plt.scatter, edgecolor="w", s=40, alpha=0.5)
        g = g.add_legend()
        g.fig.set_size_inches(self._figsize)
        plt.show();

    def plt_1model_adv(self, combno, tdep, temp_i, tindep):
        """
        Method to plot a scatter of the samples, the fitted model and the
        residuals. Plot joint distribution and marginals.
        """
        txt = self.get_math_txt(combno, tdep, tindep)
        # Jointgrid of Observations for Fit
        g = sns.JointGrid(self._xi[self._ids_fit, tindep], self._xi[self._ids_fit, tdep],
                          height=self._figsize[0]*5/6,
                          ratio=int(5)
                          )
        g.plot_joint(plt.scatter, edgecolor="w", s=40, alpha=0.5,
                     c=self._colors[0])
        # Plot of Model
        plt.plot(self._results['%i' % (self._numberrun)][combno]['X_model'],
                 self._results['%i' % (self._numberrun)][combno]['Y_model'],
                 c='r')
        # Differentiate between holdout case
        if self._config['holdout'] is True:
            # Scatter of Observations for Test
            plt.scatter(self._xi[self._ids_test, tindep],
                        self._xi[self._ids_test, tdep],
                        marker='s', edgecolor="w", s=40, alpha=0.5,
                        c=self._colors[1])
            legend = [r'$Model\ %s$' % (txt),
                      r'$Observations\ for\ Fit$',
                      r'$Observations\ for\ Test$',
                      r'$Residuals\ (X_{%i}-\hatX_{%i})$' % (tdep, tdep)]
        else:
            legend = [r'$Model\ %s$' % (txt),
                      r'$Observations\ for\ Fit/Test$',
                      r'$Residuals\ (X_{%i}-\hatX_{%i})$' % (tdep, tdep)]
        # Scatter of Residuals
        plt.scatter(self._xi[self._ids_test, tindep],
                    self._results['%i' % (self._numberrun)][combno]['Residuals'],
                    marker='D', edgecolor="w", s=40, alpha=0.8,
                    c=self._colors[2])
        # Further Plot Options
        plt.legend(legend)
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
        if self._config['holdout'] is True:
            sns.distplot(self._xi[self._ids_test, tindep],
                         norm_hist=True,
                         color=self._colors[1])
        else:
            sns.distplot(self._xi[self._ids_test, tindep],
                         norm_hist=True,
                         color=self._colors[0])
        sns.distplot(self._results['%i' % (self._numberrun)][combno]['Residuals'],
                     norm_hist=True,
                     color=self._colors[2])
        plt.legend([r'$X_{%i}$' % (tindep),
                    r'$Residuals\ (X_{%i}-\hatX_{%i})$' % (tdep, tdep)])
        plt.title(r'$\bf{Independence\ of\ Residuals:\ %s}$' % (txt))
        plt.xlabel(r'$X_{i}$')
        plt.ylabel(r'$p\left(X_{i}\right)$')
        plt.show()

    def plt_hist_GoodnessFit(self, combno, tdep, temp_i, tindep):
        """
        Method to plot a histogramm of both the independent sample and the
        Residuals
        """
        txt = self.get_math_txt(combno, tdep, tindep)
        plt.figure(r'Goodness of Fit: %s' % (txt),
                   figsize=self._figsize)
        sns.distplot(self._xi[self._ids_fit, tdep],
                     norm_hist=True,
                     hist_kws={"alpha": 0.3},
                     color=self._colors[0])
        sns.distplot(self._results['%i' % (self._numberrun)][combno]['Y_predict'],
                     norm_hist=False,
                     hist_kws={"alpha": 0.7},
                     color=self._colors[0],
                     kde_kws={"linestyle": "--"})
        plt.legend([r'$X_{%i}$' % (tdep),
                    r'$\hatX_{%i}$' % (tdep)])
        plt.title(r'$\bf{Goodness\ of\ Fit:\ %s}$' % (txt))
        plt.xlabel(r'$X_{%i}$' % (tdep))
        plt.ylabel(r'$p\left(X_{i}\right)$')
        plt.show()

    def plot_inference(self):
        """
        Method to visualize the interference
        """
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
                # Plot Tindep vs Tdep
                utils.display_text_predefined(what='combination minor header',
                                              tdep=tdep, tindep=temp_tindep)
                self.plt_1model_adv(combno, tdep, temp_i, temp_tindep)
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
                # Differ between bivariate and mvariate for fitted combination
                if self.attr_variate is 'bivariate':
                    if tdep < tindep:
                        df_dict['Fitted Combination'] = r'$X_{%i}, X_{%s}$' % (tdep, temp_tindep)
                    elif tdep >= tindep:
                        df_dict['Fitted Combination'] = r'$X_{%s}, X_{%i}$' % (temp_tindep, tdep)
                elif self.attr_variate is 'mvariate':
                    df_dict['Fitted Combination'] = r'$X_{%i} \sim f(X_{%s})$' % (tdep, tindep)
                df_dict['tdep'] = tdep
                df_dict['tindep'] = tindep
                df_dict['Bivariate Comparison'] = r'$X_{%i} \sim f(X_{%s})$' % (tdep, temp_tindep)
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
        plt.figure('Combination Boxplot', figsize=[self._figsize[0], self._figsize[1]*(len(self._combinations)/9)])
        if 'p-value' in self.attr_dict[self._config['testtype']]:
            plt.xscale('log')
            lbl = r'$dependence \leftarrow\ p-value\ \rightarrow independence$'
        elif 'likelihood-ratio' in self.attr_dict[self._config['testtype']]:
            lbl = r'$not favored \leftarrow\ likelihood-ratio\ \rightarrow favored$'
        # Get Data for each bivariate case from dictionary
        x_data = np.arange(-0.5, -(df_dependence.shape[0]+0.5), -1)
        y_data = [jload(bivacomp[-1]) for i, bivacomp in df_dependence.iterrows()]
        labels_box = [r'$\bf{%i}:$' % (i) + ' %s' % (bivacomp[1]) for i, bivacomp in df_dependence.iterrows()]
        combnos = [bivacomp[0] for i, bivacomp in df_dependence.iterrows()]
        # Get Unique combinations and give number to them
        combno_unique = set(combnos)
        combno_unique = {key: i for i, key in enumerate(combno_unique)}
        # Background Color different bivariate cases from same combinations
        for i, combno in enumerate(combnos):
            plt.axhspan(x_data[i]-1/3, x_data[i]+1/3,
                        facecolor=self._cmap(combno_unique[combno]/(len(combno_unique)-1)), alpha=1)
        plt.legend(combnos,
                   loc='center right',
                   bbox_to_anchor=(1.2, 0.5),
                   ncol=1)
        # Boxplot
        plt.boxplot(y_data, positions=x_data, labels=labels_box, vert=False,
                    patch_artist=True)
        # Further Plot Settings
        plt.title(r'BoxPlot', fontweight='bold')
        plt.tick_params(labelbottom=True)
        plt.tick_params(right=False, top=False, left=True, bottom=False)
        plt.xlabel(lbl)
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
