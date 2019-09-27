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
    Class for running the calculations of "Additive Noise Model" methods.
    Please be aware of the assumptions for models of these categorie.
    """
    attr_method = 'anm'

    def __init__(self):
        """
        Class constructor.
        """
        self._results = {}
        self._results_df = {}
        self.results = []

    def __del__(self):
        """
        Class deconstructor.
        """
        self._results = {}
        self._results_df = {}
        self.results = []

    def get_combi(self, combi, tdep, tindeps, ids_tdep, ids_tindep):
        """
        Method to return [X, Y and the regarding model], controlled by index
        combi, where i is in (0, number_of_combinations, 1).
        In Combinations, the first value of the nested list is always the
        dependent variable whereas the other values are the independent
        variables. Copy values to make original values independent from
        scaling.
        """
        model = self._regmods[combi]
        ydata = np.copy(self._obs[ids_tdep, tdep]).reshape(-1, 1)
        xdata = np.copy(self._obs[:, tindeps][ids_tindep, :])
        xdata = xdata.reshape(-1, len(tindeps))
        return(model, xdata, ydata)

    def fit_model2xi(self, combi, tdep, tindeps, model, xdata, ydata):
        """
        Method to fit model regarding to combi, which defines tdep, tindep,
        model, xdata and Ydata.
        """
        # Scale data forward
        if self._config['scale'] is True:
            xdata = self.scaler_transform(xdata, tindeps)
            ydata = self.scaler_transform(ydata, tdep)
        # Reshape Data (optimized for fit)
        x_shape = xdata.shape
        y_shape = ydata.shape
        xdata = xdata.reshape(-1, len(tindeps))
        ydata = ydata.reshape(-1,)
        # Use gridsearch instead of fit
        if self._kwargs['gridsearch'] is True:
            if 'pygam' in str(self._regmods[0].__class__):
                model.gridsearch(xdata, ydata)
            else:
                grid_search = GridSearchCV(model, self._kwargs['param_grid'])
                grid_search.fit(xdata, ydata)
                # TBD check if redundant
                model.set_params(**grid_search.best_params_)
                model.fit(xdata, ydata)
                # Clean up (otherwise memory might overload)
                del grid_search
        else:
            model.fit(xdata, ydata)
        # Shape data back
        xdata = xdata.reshape(x_shape)
        ydata = ydata.reshape(y_shape)
        # Scale data back
        if self._config['scale'] is True:
            xdata = self.scaler_inverse_transform(xdata, tindeps)
            ydata = self.scaler_inverse_transform(ydata, tdep)

    def predict_model(self, combi, tdep, tindeps, model, xdata, ydata):
        """
        Method to create further information on a fit. Returns a list for each
        fit including the following values:
        X_model:    # X values in linspace to plot the fitted model
        Y_model:    # Y values in linspace to plot the fitted model
        (If Holdout is True: ids for "fit" are used)
        """
        # Scale data forward
        if self._config['scale'] is True:
            xdata = self.scaler_transform(xdata, tindeps)
            ydata = self.scaler_transform(ydata, tdep)
        # Get independent model data
        modelpts = self._kwargs['modelpts']
        X_model = self.get_Xmodel(xdata, modelpts)
        # Do Prediction
        Y_model = model.predict(X_model).reshape(-1, 1)
        # Scale data back
        if self._config['scale'] is True:
            xdata = self.scaler_inverse_transform(xdata, tindeps)
            ydata = self.scaler_inverse_transform(ydata, tdep)
            X_model = self.scaler_inverse_transform(X_model, tindeps)
            Y_model = self.scaler_inverse_transform(Y_model, tdep)
        # Add information to self._results
        self._results['%i' % (self._runi)][combi]['X_model'] = X_model
        self._results['%i' % (self._runi)][combi]['Y_model'] = Y_model

    def predict_residuals(self, combi, tdep, tindeps, model, xdata, ydata):
        """
        Method to create further information on a fit. Returns a list for each
        fit including the following values:
        Y_predict:  predicted values of y given x
        Residuals:  ydata - Y_predict
        (If Holdout is True: ids for "test" are used)
        """
        # Scale data forward
        if self._config['scale'] is True:
            xdata = self.scaler_transform(xdata, tindeps)
        # Do Prediction
        Y_predict = model.predict(xdata).reshape(-1, 1)
        # Scale data back
        if self._config['scale'] is True:
            xdata = self.scaler_inverse_transform(xdata, tindeps)
            Y_predict = self.scaler_inverse_transform(Y_predict, tdep)
        # Get residuals
        Residuals = ydata - Y_predict
        # Add information to self._results
        self._results['%i' % (self._runi)][combi]['Y_predict'] = Y_predict
        self._results['%i' % (self._runi)][combi]['Residuals'] = Residuals

    def do_statistics(self, combi, obs_name, test_stat, obs1, obs2=None):
        """
        Method to summarize statistical tests
        """
        # Do statistics depending on the key_word "test_stat"
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
        # Add information to self._results
        self._results['%i' % (self._runi)][combi]['%s' % (obs_name)] = tr

    def test_statistics(self, combi, tdep, tindeps, model, xdata, ydata):
        """
        Method to perform statistical tests on the given and predicted data.
        """
        # Get Data for combination i
        Residuals = self._results['%i' % (self._runi)][combi]['Residuals']
        Y_predict = self._results['%i' % (self._runi)][combi]['Y_predict']
        for tindepi, tindepv in enumerate(tindeps):
            # Get Data for independent variable i
            obs_tindepi = xdata[:, tindepi]
            # Normality Test on xdata
            self.do_statistics(combi,
                               'Normality_xdata_%i' % (tindepv),
                               'Normality',
                               obs1=obs_tindepi,
                               obs2=None)
            # Test Independence of Residuals
            self.do_statistics(combi,
                               'IndepResiduals_%i' % (tindepv),
                               self._config['testtype'],
                               obs1=Residuals,
                               obs2=obs_tindepi)
        # Normality Test on Residuals
        self.do_statistics(combi,
                           'Normality_Residuals',
                           'Normality',
                           obs1=Residuals,
                           obs2=None)
        # Normality Test on ydata
        self.do_statistics(combi,
                           'Normality_ydata',
                           'Normality',
                           obs1=ydata,
                           obs2=None)
        # Test Goodness of Fit
        self.do_statistics(combi,
                           'GoodnessFit',
                           self._config['testtype'],
                           obs1=Y_predict,
                           obs2=ydata)

    def run_inference(self):
        """
        Method to do the math. Run trough all given combinations
        """
        # fit scaler
        if self._config['scale'] is True:
            self.scaler_fit()
        # initialize empty list to be filled
        empty_list = utils.object2list(None, len(self._combs), dcopy=True)
        self._results['%i' % (self._runi)] = empty_list
        # Fit (scaled) models and do statistical tests
        for combi in range(len(self._combs)):
            # initialize empty dictionary to be filled
            self._results['%i' % (self._runi)][combi] = {}
            # get objects
            tdep, tindeps = self.get_tINdeps(combi)
            # get objects
            model, xdata, ydata = self.get_combi(combi, tdep, tindeps,
                                                 self._ids_fit_tdep,
                                                 self._ids_fit_tindep)
            # fit regmod on observations
            self.fit_model2xi(combi, tdep, tindeps, model, xdata, ydata)
            # predict model points
            self.predict_model(combi, tdep, tindeps, model, xdata, ydata)
            # get objects
            model, xdata, ydata = self.get_combi(combi, tdep, tindeps,
                                                 self._ids_test_tdep,
                                                 self._ids_test_tindep)
            # predict residuals
            self.predict_residuals(combi, tdep, tindeps, model, xdata, ydata)
            # do statistical tests
            self.test_statistics(combi, tdep, tindeps, model, xdata, ydata)
###############################################################################


class PlotANM():
    """
    Class for plotting the inference of RunANM methods.
    """

    def __init__(self):
        """
        Class constructor.
        """

    # def get_std_txt(self, combi, tdep, tindeps):
    #     """
    #     Libary of some standard text phrases
    #     """
    #     txt = r'X_{%i} ~ f(X_{%combi}, E_X)' % (tdep, tindeps)
    #     return(txt)

    def get_math_txt(self, combi, tdep, tindepv):
        """
        Libary of some standard text phrases
        """
        txt = r'%s \sim f\left(%s, E_{X}\right)' % (self._obs_name[tdep],
                                                    self._obs_name[tindepv])
        return(txt)

    def plt_PairGrid(self):
        """
        Method to scatter a PairGrid of all observations.
        """
        # differentiate between holdout cases
        if self._config['holdout'] is True:
            # get hue for holdout is true
            hue = np.zeros((self._obs.shape[0], ))
            hue[self._ids_fit] = 1
            hue = hue.tolist()
            hue = ['test' if i == 0 else 'fit' for i in hue]
            hue_order = ['fit', 'test']
        else:
            # get hue if holdout is false
            hue = ['fit/test' for i in range(self._obs.shape[0])]
            hue_order = ['fit/test']
        # create datframe for easy use of seaborn pairgrid
        df = pd.DataFrame(self._obs)
        df.columns = self._obs_name
        df = pd.concat([df, pd.DataFrame(hue, columns=['holdout'])], axis=1)
        # seaborn pairgrid scatter
        g = sns.PairGrid(df, hue='holdout', hue_order=hue_order)
        g = g.map_diag(plt.hist, edgecolor="w")
        g = g.map_offdiag(plt.scatter, edgecolor="w", s=40, alpha=0.5)
        g = g.add_legend()
        g.fig.set_size_inches(self._figsize)
        plt.show()

    def plt_1model_adv(self, combi, tdep, tindepi, tindepv):
        """
        Method to plot a scatter of the samples, the fitted model and the
        residuals. Plot joint distribution and marginals.
        """
        txt = self.get_math_txt(combi, tdep, tindepv)
        # Jointgrid of Observations for Fit
        g = sns.JointGrid(self._obs[self._ids_fit_tindep, tindepv],
                          self._obs[self._ids_fit_tdep, tdep],
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
            plt.scatter(self._obs[self._ids_test_tindep, tindepv],
                        self._obs[self._ids_test_tdep, tdep],
                        marker='s', edgecolor="w", s=40, alpha=0.5,
                        c=self._colors[1])
            legend = [r'$Model\ %s$' % (txt),
                      r'$Observations\ for\ Fit$',
                      r'$Observations\ for\ Test$',
                      r'$Residuals\ (%s-\hat{%s})$' % (self._obs_name[tdep],
                                                       self._obs_name[tdep])]
        else:
            legend = [r'$Model\ %s$' % (txt),
                      r'$Observations\ for\ Fit/Test$',
                      r'$Residuals\ (%s-\hat{%s})$' % (self._obs_name[tdep],
                                                       self._obs_name[tdep])]
        # Scatter of Residuals
        plt.scatter(self._obs[self._ids_test_tindep, tindepv],
                    self._results['%i' % (self._runi)][combi]['Residuals'],
                    marker='D', edgecolor="w", s=40, alpha=0.8,
                    c=self._colors[2])
        # Further Plot Options
        plt.legend(legend)
        plt.xlabel(self._obs_name[tindepv])
        plt.ylabel(self._obs_name[tdep])
        g.plot_marginals(sns.distplot, kde=True)
        plt.show()

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
        plt.legend([r'$%s$' % (self._obs_name[tindepv]),
                    r'$Residuals\ (%s-\hat{%s})$' % (self._obs_name[tdep],
                                                     self._obs_name[tdep])])
        plt.title(r'$\bf{Independence\ of\ Residuals:\ %s}$' % (txt))
        plt.xlabel(r'$%s,\ %s$' % (self._obs_name[tindepv],
                                   self._obs_name[tdep]))
        plt.ylabel(r'$p\left(%s,\ %s\right)$' % (self._obs_name[tindepv],
                                                 self._obs_name[tdep]))
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
        plt.legend([r'$%s$' % (self._obs_name[tdep]),
                    r'$\hat{%s}$' % (self._obs_name[tdep])])
        plt.title(r'$\bf{Goodness\ of\ Fit:\ %s}$' % (txt))
        plt.xlabel(r'$%s$' % (self._obs_name[tdep]))
        plt.ylabel(r'$p\left(%s\right)$' % (self._obs_name[tdep]))
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
            # Independent Variable Names to string
            tindeps_list = [self._obs_name[tindepv2] for tindepv2 in tindeps]
            tindeps_str = ', '.join(tindeps_list)
            # Displau the mvariate combination to be plotted
            utils.display_text_predefined(what='combination major header',
                                          tdep=self._obs_name[tdep],
                                          tindeps=tindeps_str)
            # Iterate over independent variables
            for tindepi, tindepv in enumerate(tindeps):
                # Plot tindeps vs Tdep
                # Displau the mvariate combination to be plotted
                utils.display_text_predefined(what='combination minor header',
                                              tdep=self._obs_name[tdep],
                                              tindepv=self._obs_name[tindepv])
                self.plt_1model_adv(combi, tdep, tindepi, tindepv)
                self.plt_hist_IndepResiduals(combi, tdep, tindepi, tindepv)
                self.plt_hist_GoodnessFit(combi, tdep, tindepi, tindepv)
###############################################################################


class ResultsANM():
    """
    Class to display the results of RunANM methods.
    """

    def __init__(self):
        """
        Class constructor.
        """

    def get_df_normality(self, testkey):
        """
        Method to return a DF summarizing the normality test
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
        Method to return a DF summarizing the normality test
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
        Method to plot the results of the independence test as BoxPlot.
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
            lbl = r'$not \ favored \leftarrow\ likelihood-ratio\ \rightarrow favored$'
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

    def plot_results(self):
        """
        Method to display the results of the interference.
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

    def bootstrap_obs(self):
        """
        Method to bootstrap observations.
        """
        # Get bootstrap seed
        seed = self.get_seed(self._kwargs['bootstrap_seed'])
        # Get bootstrap number of n_samples
        n_samples = int(self.obs.shape[0] * self._kwargs['bootstrap_ratio'])
        # Get ids of observations
        ids = np.arange(0, self.obs.shape[0], 1)
        # Bootstrap ids
        bids = resample(ids, replace=True, n_samples=n_samples,
                        random_state=seed)
        # Order sequence if transient case
        if self.attr_time is 'transient':
            bids = np.sort(bids)
        # Assign bootstraped observations
        self._obs = deepcopy(self.obs[bids])

    def boots2stats(self, combi, namekey, dict, dictkey):
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
            tindeps_list = [self._obs_name[tindepv2] for tindepv2 in tindeps]
            tindeps_str = ', '.join(tindeps_list)
            # Iterate over bivariate comparisons
            for tindepi, tindepv in enumerate(tindeps):
                # Init new dict
                df_dict = {}
                # Differ between bivariate and mvariate for fitted combination
                if self.attr_variate is 'bivariate':
                    if tdep < tindepv:
                        txt = r'$%s,\ %s$' % (self._obs_name[tdep],
                                              self._obs_name[tindepv])
                        df_dict['Fitted Combination'] = txt
                    elif tdep >= tindepv:
                        txt = r'$%s,\ %s$' % (self._obs_name[tindepv],
                                              self._obs_name[tdep])
                        df_dict['Fitted Combination'] = txt
                elif self.attr_variate is 'mvariate':
                    txt = r'$%s \sim f(%s)$' % (self._obs_name[tdep],
                                                tindeps_str)
                    df_dict['Fitted Combination'] = txt
                txt = r'$%s \sim f(%s)$' % (self._obs_name[tdep],
                                            self._obs_name[tindepv])
                df_dict['Bivariate Comparison'] = txt
                df_dict['tdep'] = tdep
                df_dict['tindeps'] = jdump(tindeps)
                df_dict['tindep'] = tindepv
                # Get Mean and Variance Value out of all bootstrap examples
                txt = 'Normality Indep. Variable'
                df_dict = self.boots2stats(combi,
                                           'Normality_xdata_%i' % (tindepv),
                                           df_dict,
                                           txt)
                txt = 'Normality Depen. Variable'
                df_dict = self.boots2stats(combi,
                                           'Normality_ydata',
                                           df_dict,
                                           txt)
                txt = 'Normality Residuals'
                df_dict = self.boots2stats(combi,
                                           'Normality_Residuals',
                                           df_dict,
                                           txt)
                txt = 'Dependence: Indep. Variable - Residuals'
                df_dict = self.boots2stats(combi,
                                           'IndepResiduals_%i' % (tindepv),
                                           df_dict,
                                           txt)
                txt = 'Dependence: Depen. Variable - Prediction (GoF)'
                df_dict = self.boots2stats(combi,
                                           'GoodnessFit',
                                           df_dict,
                                           txt)
                # Append current bivariate comparison to DF
                results_df = pd.concat([results_df, pd.Series(df_dict)],
                                       ignore_index=False, axis=1, sort=False)
        results_df = results_df.T.reset_index()
        self._results_df = results_df

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
        # Check and display warnings
        self.check_warnings()
        # Init ids based depending on SteadyState and Transient case
        self.ids_init4time()
        # Display Start of Causal Inference
        if ((plot_inference is True) or (plot_results is True)):
            utils.display_text_predefined(what='inference header')
        # Bootstrap | If False, run only once
        for boot_i, _ in enumerate(self._bootstrap):
            # Check and Init ids (based on holdout if True)
            if ((holdout is True) or (boot_i == 0)):
                self.ids_init4holdout()
            if bootstrap > 0:
                # Display the current bootstrap number
                utils.prgr_bar(boot_i+1, bootstrap, txt='Bootstrap')
                # Init fresh _regmod from regmod -> otherwise fit will fail
                if hasattr(type(self.regmod), '__iter__'):
                    # Make deepcopy to ensure independence
                    self._regmods = [deepcopy(regmod) for regmod in self.regmod]
                else:
                    no_combs = len(self._combs)
                    self._regmods = utils.object2list(self.regmod,
                                                      no_combs, dcopy=True)
                # Do the Bootstrap
                self.bootstrap_obs()
            self._runi = boot_i
            # Do the math
            self.run_inference()
        # Restructure results to make them more accesible
        self.restructure_results()
        # Pass restructered results to attribute results
        self.results = deepcopy(self._results_df)
        # Plot the math of inference
        if plot_inference is True:
            self.plot_inference()
        # Plot results
        if plot_results is True:
            self.plot_results()
###############################################################################
