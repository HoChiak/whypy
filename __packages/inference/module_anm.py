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
from sklearn.utils import resample

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

    def get_combination_objects(self, combi, tdep, tindeps, ids_list):
        """
        Method to return [X, Y and the regarding model], controlled by index
        combi, where i is in (0, number_of_combinations, 1).
        In Combinations, the first value of the nested list is always the
        dependent variable whereas the other values are the independent
        variables. Copy values to make original values independent from
        scaling.
        """
        model = self._regmods[combi]
        Y_data = np.copy(self._obs[ids_list, tdep]).reshape(-1, 1)
        # TBD check if there is a better solution to index over two axis
        X_data = np.copy(self._obs[:, tindeps][ids_list, :]).reshape(-1, len(tindeps))
        return(model, X_data, Y_data)

    def fit_model2xi(self, combi, tdep, tindeps, model, X_data, Y_data):
        """
        Method to fit model to Xi in the two variable case
        """
        # Scale data forward
        if self._config['scale'] is True:
            X_data = self.scaler_transform(X_data, tindeps)
            Y_data = self.scaler_transform(Y_data, tdep)
        # Use gridsearch instead of fit if model is pyGAM
        if self._kwargs['gridsearch'] is True:
            if 'pygam' in str(self._regmods[0].__class__):
                model.gridsearch(X_data.reshape(-1, len(tindeps)), Y_data)
            else:
                grid_search = GridSearchCV(model, self._kwargs['param_grid'])
                grid_search.fit(X_data.reshape(-1, len(tindeps)),
                                Y_data.reshape(-1,))
                # TBD check if redundant
                model.set_params(**grid_search.best_params_)
                model.fit(X_data.reshape(-1, len(tindeps)),
                          Y_data.reshape(-1,))
                # Clean up
                del grid_search
        else:
            model.fit(X_data.reshape(-1, len(tindeps)), Y_data)
        # Scale data back
        if self._config['scale'] is True:
            X_data = self.scaler_inverse_transform(X_data, tindeps)
            Y_data = self.scaler_inverse_transform(Y_data, tdep)

    def predict_model(self, combi, tdep, tindeps, model, X_data, Y_data):
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
            X_data = self.scaler_transform(X_data, tindeps)
            Y_data = self.scaler_transform(Y_data, tdep)
        # Get independent model data
        modelpts = self._kwargs['modelpts']
        X_model = self.get_Xmodel(X_data, modelpts)
        # Do Prediction
        Y_model = model.predict(X_model).reshape(-1, 1)
        # Scale data back
        if self._config['scale'] is True:
            X_data = self.scaler_inverse_transform(X_data, tindeps)
            Y_data = self.scaler_inverse_transform(Y_data, tdep)
            X_model = self.scaler_inverse_transform(X_model, tindeps)
            Y_model = self.scaler_inverse_transform(Y_model, tdep)
        self._results['%i' % (self._runi)][combi]['X_model'] = X_model
        self._results['%i' % (self._runi)][combi]['Y_model'] = Y_model

    def predict_residuals(self, combi, tdep, tindeps, model, X_data, Y_data):
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
            X_data = self.scaler_transform(X_data, tindeps)
        # Do Prediction
        Y_predict = model.predict(X_data).reshape(-1, 1)
        # Scale data back
        if self._config['scale'] is True:
            X_data = self.scaler_inverse_transform(X_data, tindeps)
            Y_predict = self.scaler_inverse_transform(Y_predict, tdep)
        # Get residuals
        Residuals = Y_data - Y_predict
        self._results['%i' % (self._runi)][combi]['Y_predict'] = Y_predict
        self._results['%i' % (self._runi)][combi]['Residuals'] = Residuals

    def do_statistics(self, combi, obs_name, test_stat, obs1, obs2=None):
        """
        Method to comprehense statistical tests
        """
        if test_stat is 'Normality':
            tr = stats.normality(obs1)
        elif test_stat is 'LikelihoodVariance':
            tr = stats.likelihoodvariance(obs1, obs2)
        elif test_stat is 'LikelihoodEntropy':
            tr = stats.likelihoodentropy(obs1, obs2)
        elif test_stat is 'KolmogorovSmirnoff':
            tr = stats.kolmogorov(obs1, obs2)
        elif test_stat is 'MannWhitney':
            tr = stats.mannwhitneyu(obs1, obs2)
        elif test_stat is 'HSIC':
            tr = stats.hsic_gam(obs1, obs2)
        else:
            print('Given test_stat argument is not defined.')
        self._results['%i' % (self._runi)][combi]['%s' % (obs_name)] = tr

    def test_statistics(self, combi, tdep, tindeps, model, X_data, Y_data):
        """
        Method to perform statistical tests on the given and predicted data.
        """
        for tindepi, tindepv in enumerate(tindeps):
            # Normality Test on X_data
            self.do_statistics(combi,
                               'Normality_X_data_%i' % (tindepv),
                               'Normality',
                               obs1=X_data[:, tindepi],
                               obs2=None)
            # Test Independence of Residuals
            self.do_statistics(combi,
                               'IndepResiduals_%i' % (tindepv),
                               self._config['testtype'],
                               obs1=self._results['%i' % (self._runi)][combi]['Residuals'],
                               obs2=X_data[:, tindepi])
        # Normality Test on Residuals
        self.do_statistics(combi,
                           'Normality_Residuals',
                           'Normality',
                           obs1=self._results['%i' % (self._runi)][combi]['Residuals'],
                           obs2=None)
        # Normality Test on Y_data
        self.do_statistics(combi,
                           'Normality_Y_data',
                           'Normality',
                           obs1=Y_data,
                           obs2=None)
        # Test Goodness of Fit
        self.do_statistics(combi,
                           'GoodnessFit',
                           self._config['testtype'],
                           obs1=self._results['%i' % (self._runi)][combi]['Y_predict'],
                           obs2=Y_data)

    def run_inference(self):
        """
        Method to do the math. Run trough all possible 2V combinations of
        observations and calculate the inference.
        """
        # Fit Scaler
        if self._config['scale'] is True:
            self.scaler_fit()
        # Initialize empty list to be filled
        self._results['%i' % (self._runi)] = utils.trans_object_to_list(None, len(self._combs), dcopy=True)
        # Fit (scaled) models and do statistical tests
        for combi in range(len(self._combs)):
            # Initialize empty dictionary to be filled
            self._results['%i' % (self._runi)][combi] = {}
            # Get Constants
            tdep, tindeps = self.get_tINdeps(combi)
            # Get Constants
            model, X_data, Y_data = self.get_combination_objects(combi, tdep, tindeps, self._ids_fit)
            # fit regmod on observations
            self.fit_model2xi(combi, tdep, tindeps, model, X_data, Y_data)
            # predict model points
            self.predict_model(combi, tdep, tindeps, model, X_data, Y_data)
            # Get Constants
            model, X_data, Y_data = self.get_combination_objects(combi, tdep, tindeps, self._ids_test)
            # predict residuals
            self.predict_residuals(combi, tdep, tindeps, model, X_data, Y_data)
            # do statistical tests
            self.test_statistics(combi, tdep, tindeps, model, X_data, Y_data)


###############################################################################
class PlotANM():
    """
    Class for "Additive Noise Model" Methods.
    Please be aware of the assumptions for models of these categorie.
    """

    def __init__(self):
        """
        Class constructor.
        """

    def get_std_txt(self, combi, tdep, tindeps):
        """
        Libary of some standard text phrases
        """
        txt = r'X_{%i} ~ f(X_{%combi}, E_X)' % (tdep, tindeps)
        return(txt)

    def get_math_txt(self, combi, tdep, tindeps):
        """
        Libary of some standard text phrases
        """
        txt = r'X_{%i} \approx f\left( X_{%i}, E_{X}\right)' % (tdep, tindeps)
        return(txt)

    def plt_PairGrid(self):
        """
        Method to plot a PairGrid scatter of the observations.
        """
        # Differentiate between holdout case
        if self._config['holdout'] is True:
            # Get Holdout for hue
            hue = np.zeros((self._obs.shape[0], ))
            hue[self._ids_fit] = 1
            hue = hue.tolist()
            hue = ['test' if i == 0 else 'fit' for i in hue]
        else:
            hue = ['fit/test' for i in range(self._obs.shape[0])]
        df = pd.DataFrame(self._obs)
        df.columns = [r'$X_{%i}$' % (i) for i in range(self._obs.shape[1])]
        df = pd.concat([df, pd.DataFrame(hue, columns=['holdout'])], axis=1)
        g = sns.PairGrid(df, hue='holdout');
        g = g.map_diag(plt.hist, edgecolor="w")
        g = g.map_offdiag(plt.scatter, edgecolor="w", s=40, alpha=0.5)
        g = g.add_legend()
        g.fig.set_size_inches(self._figsize)
        plt.show();

    def plt_1model_adv(self, combi, tdep, tindepi, tindepv):
        """
        Method to plot a scatter of the samples, the fitted model and the
        residuals. Plot joint distribution and marginals.
        """
        txt = self.get_math_txt(combi, tdep, tindepv)
        # Jointgrid of Observations for Fit
        g = sns.JointGrid(self._obs[self._ids_fit, tindepv],
                          self._obs[self._ids_fit, tdep],
                          height=self._figsize[0]*5/6,
                          ratio=int(5)
                          )
        g.plot_joint(plt.scatter, edgecolor="w", s=40, alpha=0.5,
                     c=self._colors[0])
        # Plot of Model
        plt.plot(self._results['%i' % (self._runi)][combi]['X_model'][:, tindepi],
                 self._results['%i' % (self._runi)][combi]['Y_model'],
                 c='r')
        # Differentiate between holdout case
        if self._config['holdout'] is True:
            # Scatter of Observations for Test
            plt.scatter(self._obs[self._ids_test, tindepv],
                        self._obs[self._ids_test, tdep],
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
        plt.scatter(self._obs[self._ids_test, tindepv],
                    self._results['%i' % (self._runi)][combi]['Residuals'],
                    marker='D', edgecolor="w", s=40, alpha=0.8,
                    c=self._colors[2])
        # Further Plot Options
        plt.legend(legend)
        plt.xlabel(r'$X_{%i}$' % (tindepv))
        plt.ylabel(r'$X_{%i}$' % (tdep))
        g.plot_marginals(sns.distplot, kde=True)
        plt.show();

    def plt_hist_IndepResiduals(self, combi, tdep, tindepi, tindepv):
        """
        Method to plot a histogramm of both the independent sample and the
        Residuals
        """
        txt = self.get_math_txt(combi, tdep, tindepv)
        plt.figure(r'Independence of Residuals: %s' % (txt),
                   figsize=self._figsize)
        if self._config['holdout'] is True:
            sns.distplot(self._obs[self._ids_test, tindepv],
                         norm_hist=True,
                         color=self._colors[1])
        else:
            sns.distplot(self._obs[self._ids_test, tindepv],
                         norm_hist=True,
                         color=self._colors[0])
        sns.distplot(self._results['%i' % (self._runi)][combi]['Residuals'],
                     norm_hist=True,
                     color=self._colors[2])
        plt.legend([r'$X_{%i}$' % (tindepv),
                    r'$Residuals\ (X_{%i}-\hatX_{%i})$' % (tdep, tdep)])
        plt.title(r'$\bf{Independence\ of\ Residuals:\ %s}$' % (txt))
        plt.xlabel(r'$X_{i}$')
        plt.ylabel(r'$p\left(X_{i}\right)$')
        plt.show()

    def plt_hist_GoodnessFit(self, combi, tdep, tindepi, tindepv):
        """
        Method to plot a histogramm of both the independent sample and the
        Residuals
        """
        txt = self.get_math_txt(combi, tdep, tindepv)
        plt.figure(r'Goodness of Fit: %s' % (txt),
                   figsize=self._figsize)
        sns.distplot(self._obs[self._ids_fit, tdep],
                     norm_hist=True,
                     hist_kws={"alpha": 0.3},
                     color=self._colors[0])
        sns.distplot(self._results['%i' % (self._runi)][combi]['Y_predict'],
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
        for combi in range(len(self._combs)):
            tdep, tindeps = self.get_tINdeps(combi)
            tdep = tdep[0]
            utils.display_text_predefined(what='combination major header',
                                          tdep=tdep, tindeps=tindeps)
            # Iterate over independent variables
            for tindepi, tindepv in enumerate(tindeps):
                # Plot tindeps vs Tdep
                utils.display_text_predefined(what='combination minor header',
                                              tdep=tdep, tindepv=tindepv)
                self.plt_1model_adv(combi, tdep, tindepi, tindepv)
                self.plt_hist_IndepResiduals(combi, tdep, tindepi, tindepv)
            self.plt_hist_GoodnessFit(combi, tdep, tindepi, tindepv)


###############################################################################
class ResultsANM():
    """
    Class for "Additive Noise Model" Methods.
    Please be aware of the assumptions for models of these categorie.
    """

    def __init__(self):
        """
        Class constructor.
        """

    def boots_to_med_sd(self, combi, namekey, dict, dictkey):
        """
        Method to get mean and variance from bootstrap runs
        """
        for testi in self._results['0'][combi][namekey].keys():
            newlist = list()
            for booti in self._results.keys():
                newlist.append(self._results[booti][combi][namekey][testi])
            newarray = np.array(newlist).flatten()
            medianarray = np.median(newarray)
            vararray = np.std(newarray)
            dict[dictkey+' '+str(testi)+' [List]'] = jdump(newlist)
            dict[dictkey+' '+str(testi)+' [Median]'] = medianarray
            dict[dictkey+' '+str(testi)+' [SD]'] = vararray
        return(dict)

    def restructure_results(self):
        """
        Method to extract a readable DataFrame from the self._results attribute
        """
        # Init new DataFrame
        results_df = pd.DataFrame()
        # Iterate over all possible combinations
        for combi in range(len(self._combs)):
            tdep, tindeps = self.get_tINdeps(combi)
            tdep = tdep[0]
            # Iterate over bivariate comparisons
            for tindepi, tindepv in enumerate(tindeps):
                # Init new dict
                df_dict = {}
                # Differ between bivariate and mvariate for fitted combination
                if self.attr_variate is 'bivariate':
                    if tdep < tindepv:
                        df_dict['Fitted Combination'] = r'$X_{%i}, X_{%s}$' % (tdep, tindepv)
                    elif tdep >= tindepv:
                        df_dict['Fitted Combination'] = r'$X_{%s}, X_{%i}$' % (tindepv, tdep)
                elif self.attr_variate is 'mvariate':
                    df_dict['Fitted Combination'] = r'$X_{%i} \sim f(X_{%s})$' % (tdep, tindeps)
                df_dict['Bivariate Comparison'] = r'$X_{%i} \sim f(X_{%s})$' % (tdep, tindepv)
                df_dict['tdep'] = tdep
                df_dict['tindeps'] = jdump(tindeps)
                df_dict['tindep'] = tindepv
                # Get Mean and Variance Value out of all bootstrap examples
                df_dict = self.boots_to_med_sd(combi,
                                               'Normality_X_data_%i' % (tindepv),
                                               df_dict,
                                               'Normality Indep. Variable')
                df_dict = self.boots_to_med_sd(combi,
                                               'Normality_Y_data',
                                               df_dict,
                                               'Normality Depen. Variable')
                df_dict = self.boots_to_med_sd(combi,
                                               'Normality_Residuals',
                                               df_dict,
                                               'Normality Residuals')
                df_dict = self.boots_to_med_sd(combi,
                                               'IndepResiduals_%i' % (tindepv),
                                               df_dict,
                                               'Dependence: Indep. Variable - Residuals')
                df_dict = self.boots_to_med_sd(combi,
                                               'GoodnessFit',
                                               df_dict,
                                               'Dependence: Depen. Variable - Prediction (GoF)')
                # Append current bivariate comparison to DF
                results_df = pd.concat([results_df, pd.Series(df_dict)],
                                       ignore_index=False, axis=1, sort=False)
            self._results_df = results_df.T

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
        columns = [key.replace(self._config['testtype']+str(' '), '') for key in columns]
        df.columns = columns
        return(df)

    def plt_combinations_boxplot(self, testkey):
        """
        Method to plot the results of the independence test.
        """
        # Get Dictionary
        df_dependence = self.get_df_dependence(testkey, removeList=False)
        # Get Data for each bivariate case from dictionary
        positions = np.arange(-0.5, -(df_dependence.shape[0]+0.5), -1)
        y_data = [jload(bivacomp[-1]) for i, bivacomp in df_dependence.iterrows()]
        labels_box = [r'$\bf{%i}:$' % (i) + ' %s' % (bivacomp[1]) for i, bivacomp in df_dependence.iterrows()]
        combis = [bivacomp[0] for i, bivacomp in df_dependence.iterrows()]
        # Get Unique combinations and give number to them
        combi_unique = set(combis)
        combi_unique = {key: i for i, key in enumerate(combi_unique)}
        # Init Plot
        plt.figure('Combination Boxplot',
                   figsize=[self._figsize[0],
                            self._figsize[1]*(len(combis)/9)])
        if 'p-value' in self.attr_dict[self._config['testtype']]:
            plt.xscale('log')
            lbl = r'$dependence \leftarrow\ p-value\ \rightarrow independence$'
        elif 'likelihood-ratio' in self.attr_dict[self._config['testtype']]:
            lbl = r'$not favored \leftarrow\ likelihood-ratio\ \rightarrow favored$'
        # Background Color different bivariate cases from same combinations
        for i, combi in enumerate(combis):
            facecolor = self._cmap(combi_unique[combi]/(len(combi_unique)))
            plt.axhspan(positions[i]-1/3, positions[i]+1/3,
                        facecolor=facecolor, alpha=1)
        plt.legend(combis,
                   loc='center right',
                   bbox_to_anchor=(1.2, 0.5),
                   ncol=1)
        # Boxplot
        plt.boxplot(y_data, positions=positions, labels=labels_box, vert=False,
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
        utils.display_text_predefined(what='thirdlevel',
                                      key='Pearsons p-value')
        utils.display_df(self.get_df_normality(testkey='Pearson_pvalue'))
        utils.display_text_predefined(what='thirdlevel',
                                      key='Shapiro Wilk p-value')
        utils.display_df(self.get_df_normality(testkey='SW_pvalue'))
        utils.display_text_predefined(what='thirdlevel',
                                      key='Combined p-value')
        utils.display_df(self.get_df_normality(testkey='Combined_pvalue'))
        # Plot Goodness of Fit Test
        utils.display_text_predefined(what='dependence prediction')
        key = '%s: %s' % (self._config['testtype'],
                          self.attr_dict[self._config['testtype']])
        utils.display_text_predefined(what='thirdlevel', key=key)
        utils.display_df(self.get_df_dependence('GoF'))
        self.plt_combinations_boxplot('GoF')
        # Plot Indepndence of Residuals
        utils.display_text_predefined(what='dependence residuals')
        utils.display_text_predefined(what='thirdlevel', key=key)
        utils.display_df(self.get_df_dependence('Residuals'))
        self.plt_combinations_boxplot('Residuals')


###############################################################################
class ANM(RunANM, PlotANM, ResultsANM):
    """
    Causal Inference methods for the two variable case. General SCMs are not
    identifiable in the two variable case. Additional Assumptions are required,
    given by the modelclass restrictions. Only acyclic graphs are considered.
    """
    attr_dict = {'LikelihoodVariance': 'likelihood-ratio',
                 'LikelihoodEntropy': 'likelihood-ratio',
                 'KolmogorovSmirnoff': 'p-value',
                 'MannWhitney': 'p-value',
                 'HSIC': 'unknown'
                 }

    def __init__(self, xi=None, combinations='all', regmod=None, scaler=None):
        """
        Parent class constructor for causal inference methods in the 2 variable
        case. Xi may consist of an abritary number of variables, but only one
        variable is mapped to one other

        INPUT (Inherent from parent):
        """
        RunANM.__init__(self)
        PlotANM.__init__(self)
        ResultsANM.__init__(self)

    def run(self,
            testtype='LikelihoodVariance',
            scale=True,
            bootstrap=False,
            holdout=False,
            plot_inference=True,
            plot_results=True,
            **kwargs):
        """
        Method to test independence of residuals.
        Theorem: In causal direction, the noise is independent of the input
        Valid for Additive Noise Models e.g. LiNGAM, NonLinear GaussianAM
        """
        # Count Number of runs +1
        self._runi = 0
        # Check and Initialisation of Attributes
        self.check_instance_model_attr(scale)
        self.init_instance_model_attr()
        # Clear Arguments from previous caclulations
        RunANM.__del__(self)
        # Check Method "Run" Arguments
        self.check_and_init_arg_run(testtype, bootstrap, holdout)
        # Add information to config
        self._config = {'testtype': testtype,
                        'scale': scale,
                        'bootstrap': bootstrap,
                        'holdout': holdout,
                        'shape_observations': self._obs.shape,
                        'shape_combinations': np.array(self._combs).shape,
                        'regression_model': str(self._regmods[0]),
                        'scaler_model': str(self._scaler[0]),
                        }
        # Check and Init Kwargs
        self.check_init_kwargs(kwargs)
        # Check and Init Holdout Lists
        self.check_init_holdout_ids()
        # Check and display warnings
        self.check_warnings()
        # Display Start of Causal Inference
        utils.display_text_predefined(what='inference header')
        # TBD Add Time shift / Adress different environments
        for boot_i, _ in enumerate(self._bootstrap):
            if bootstrap > 0:
                # Display the current bootstrap number
                utils.display_text_predefined(what='count bootstrap',
                                              current=boot_i, sum=bootstrap)
                # Init fresh _regmod from regmod -> otherwise fit will fail
                self._regmods = utils.trans_object_to_list(self.regmod,
                                                          len(self._combs),
                                                          dcopy=True)
                # Do the Bootstrap
                self._obs = resample(deepcopy(self.obs), replace=True,
                                    n_samples=int(self.obs.shape[0] * self._kwargs['bootstrap_ratio']),
                                    random_state=self._kwargs['bootstrap_seed']+boot_i)
            self._runi = boot_i
            # Do the math
            self.run_inference()
        # Plot the math of inference
        if plot_inference is True:
            self.plot_inference()
        # Plot results
        if plot_results is True:
            self.plot_results()


###############################################################################
