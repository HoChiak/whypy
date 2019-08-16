# -*- coding: utf-8 -*-

# import built in libarys
from itertools import permutations

# import 3rd party libarys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# import local libarys
from whypy.__packages.utils import utils
from whypy.__packages.utils import stats


###############################################################################
class Bivariate():
    """
    Causal Inference methods for the two variable case. General SCMs are not
    identifiable in the two variable case. Additional Assumptions are required,
    given by the modelclass restrictions. Only acyclic graphs are considered.
    """
    attr_variate = 'bivariate'

    def __init__(self):
        """
        Class Constructor for the Bivariate Case. Assigns Bivariate specific
        Methods to the Inference Model. Instance Attributes are assigned in
        the inference Class.
        """

    def get_combinations(self):
        """
        Method to get a list of combinations for the Bivariate Case.
        """
        variable_names = np.arange(self._xi.shape[1])
        variable_names = list(variable_names)
        combinations = [x for x in permutations(variable_names, 2)]
        self._combinations = utils.trans_nestedlist_to_tuple(combinations)

    def check_combinations(self):
        """
        Method to check combinations for the Bivariate Case.
        """
        assert np.array(self._combinations).shape[1] == 2, 'Shape of combinations must be (m, 2)'
        self._combinations = utils.trans_nestedlist_to_tuple(self._combinations)

class CurrentlyExcluded1():
    """
    Class for "Additive Noise Model" Methods.
    Please be aware of the assumptions for models of these categorie.
    """

    def __init__(self):
        """
        Class constructor.
        """



    def get_testnames(self):
        """
        Method to get and return all testnames for X vs Residuals tests
        """
        # Get a list of all test names
        testn = self._results['%i' % (self._numberrun)][0]['X-Residuals_Names']
        # check if testn is list (its no list if only one stat test performed)
        if not isinstance(testn, list):
            # convert type to list
            testn = [testn]
        return(testn)

    def restructure_results(self, fav_dir='high'):
        """
        Method to get the results of the independence/likelihood test
        structured by testtype.
        """
        # Local dict for to translate fav_dir
        dic = {'high': False,
               'low': True}
        # Get a list of all test names
        testn = self.get_testnames()
        # Loop over all tests
        for i, tp in enumerate(testn):
            # Create a placeholder for each TestType
            temp = list()
            # Loop trough all possible combinations of tdep and tindep
            for ti in range(len(self._combinations)):
                tdep, tindep = self.get_tINdep(ti)
                tindep = self.array_to_scalar(tindep)
                test = self._results['%i' % (self._numberrun)][ti]
                txt = self.get_std_txt(what='std-Y~f(X)',
                                       tdep=tdep,
                                       tindep=tindep)
                if 'independence' in self._config['%i' % (self._numberrun)]['testtype']:
                    temp.append((tp, (tdep+1) * (tindep+1), txt,
                                 test['X-Residuals_Results'][(i*2)+1],
                                 test['X-Residuals_Results'][i*2],
                                 np.array(test['Model_Statistics']),
                                 tdep, tindep))
                if 'likelihood' in self._config['%i' % (self._numberrun)]['testtype']:
                    temp.append((tp, (tdep+1) * (tindep+1), txt,
                                 test['X-Residuals_Results'],
                                 np.array(test['Model_Statistics']),
                                 tdep, tindep))
            # Create DataFrame for single TestType and add extra information
            temp = pd.DataFrame(temp)
            # assign names to columns of DF
            if 'independence' in self._config['%i' % (self._numberrun)]['testtype']:
                temp.columns = ('TestType', '2V-no', '2V-case',
                                'pval/likel',
                                'statistics',
                                'Model Statistics',
                                'tdep', 'tindep')
            if 'likelihood' in self._config['%i' % (self._numberrun)]['testtype']:
                temp.columns = ('TestType', '2V-no', '2V-case',
                                'pval/likel',
                                'Model Statistics',
                                'tdep', 'tindep')
            # Rank testresults (pvalue or likelihood) for one TestType
            rank1 = temp.iloc[:, 3].rank(method='max', ascending=dic[fav_dir])
            temp['rank pval/likel'] = rank1
            # Rank testresults (pvalue or likelihood) for one 2V_no
            rank2 = temp.groupby('2V-no')['rank pval/likel'].rank(method='max')
            temp['2V-direction'] = rank2
            # Rename it for understanding
            temp['2V-direction'].replace(1, 'favored', inplace=True)
            temp['2V-direction'].replace(2, 'NOT favored', inplace=True)
            # Add TestType to final DataFrame
            if i == 0:
                rs = temp
            else:
                rs = pd.concat([rs, temp], axis=0)
        self._results2V['%i' % (self._numberrun)] = rs

    def do_CGM_favored(self, rs):
        """
        Method to do Causal Graph Metric "favored direction".
        """
        # Add a column 'CGmetric' as the choosen metric
        metric = (rs['2V-direction'] == 'favored')
        metric = np.array(metric)
        return(metric)

    def do_CGM_positive(self, rs):
        """
        Method to do Causal Graph Metric "positive values".
        """
        # Add a column 'CGmetric' as the choosen metric
        metric = (rs['pval/likel'] >= 0)
        metric = np.array(metric)
        return(metric)

    def do_CGM_interception(self, rs):
        """
        Method to do Causal Graph Metric "compare p-value to interception
        model".
        """
        try:
            # Do some type transformation
            modstat = rs['Model Statistics'].values
            modstat = np.stack(modstat, axis=0)
            # Choose models which p-value is <= interception-model the p-value
            p_interception = modstat[:, 0]
            p_regmod = modstat[:, 1]
            metric = (p_interception >= p_regmod)
        except:
            print('Failed: Compare p-value to interception model')
            metric = np.ones(shape=rs.shape[0])
        metric = np.array(metric)
        return(metric)

    def do_CGM_variance(self, rs, alpha=1):
        """
        Method to do Causal Graph Metric "compare the variance (approximated
        by empirical values)".
        """
        # Calculation
        abso = rs['pval/likel']
        mean = rs.groupby('2V-no')['pval/likel'].transform(np.mean)
        stde = np.std(abso-mean)
        rnge = rs.groupby('2V-no')['pval/likel'].transform(np.ptp)
        metric = alpha * rnge >= stde
        metric = np.array(metric)
        return(metric)

    def do_CGM_range(self, rs, percentile=0.05):
        """
        Method to do Causal Graph Metric "compare the local range to the global
        range (approximated by empirical values)".
        """
        # Calculation
        abso = rs['pval/likel']
        loc_range = rs.groupby('2V-no')['pval/likel'].transform(np.ptp)
        glo_range = np.ptp(rs['pval/likel'])
        glo_range = np.ones(shape=loc_range.shape[0]) * glo_range
        metric = percentile * glo_range <= loc_range
        metric = np.array(metric)
        return(metric)

    def do_CGM(self, testtype, CGmetric, testname):
        """
        Method to vote for the structure of the Causal Graph based on metrics
        Dummy tbd
        """
        rs = self._results2V['%i' % (self._numberrun)].copy()
        # Extract only these rows with the current TestType
        rs = rs[rs['TestType'] == testname]
        if 'Favored' in CGmetric:
            metric1 = self.do_CGM_favored(rs)
            # Combine metrics
            metric = metric1
        elif 'Model_Statistics' in CGmetric:
            metric1 = self.do_CGM_favored(rs)
            metric2 = self.do_CGM_interception(rs)
            # Combine metrics
            metric = metric1 & metric2
        elif 'Result_Variance' in CGmetric:
            metric1 = self.do_CGM_favored(rs)
            # Choose Results with range between both directions greater than
            # standard deviation of all ranges.
            if 'independence' in testtype:  # transform pvalue
                rs['pval/likel'] = rs['pval/likel'].transform(np.log)
            metric2 = self.do_CGM_variance(rs)
            # Combine metrics
            metric = metric1 & metric2
        elif 'Combined' in CGmetric:
            metric1 = self.do_CGM_favored(rs)
            metric2 = self.do_CGM_positive(rs)
            # Check if there are pos and neg values in favored direction
            metric3 = any(metric1 & metric2) & any(metric1 & ~metric2)
            metric3 = np.array([metric3])
            metric4 = self.do_CGM_interception(rs)
            if 'independence' in testtype:  # transform pvalue
                rs['pval/likel'] = rs['pval/likel'].transform(np.log)
                metric5 = self.do_CGM_range(rs, percentile=0.05)
                # Combine metrics
                metric = metric1 & metric4 & metric5
                metricdf = pd.DataFrame([metric1, metric4, metric5, metric])
                metriccolumns = ('Favored', '_&_Model_Statistics',
                                 '_&_Result_Range', '_=_Combined')
            elif 'likelihood' in testtype:
                metric5 = self.do_CGM_range(rs, percentile=0.05)
                # Combine metrics
                metric = ((metric1 & metric2 & metric3 & metric5) |
                          (metric1 & metric4 & metric5))
                metricdf = pd.DataFrame([metric1, metric2,
                                         metric3, metric5,
                                         metric1, metric4,
                                         metric5, metric])
                metriccolumns = ('(Favored', '_&_Positive',
                                 '_&_PosNeg', '_&_Result_Range)_|_',
                                 '(Favored', '_&_Model_Statistics',
                                 '_&_Result_Range)', '_=_Combined')
            # Create DataFrame
            metricdf = metricdf.T
            metricdf.columns = metriccolumns
            # Add metric dataframe to Results (self._results2V)
            metricdf = pd.concat((rs, metricdf), axis=1)
        else:
            print('No proper metric defined')
            metricdf = None
            metric = None
        return(metric, metricdf)

    def do_create_CG(self, testtype, CGmetric, testname):
        """
        Method to get single CG based on results of self.do_CGM
        """
        # Get a metric to jugde the relative propability for an edge
        metric, metricdf = self.do_CGM(testtype, CGmetric, testname)
        # Filter results by TestType and metric
        tempr = self._results2V['%i' % (self._numberrun)]
        tempr = tempr[tempr['TestType'] == testname]
        tempr = tempr[metric]
        # Extract Edge_list
        Node1 = tempr['tindep'].values
        Node1 = [r'X_%i' % (t) for t in Node1]
        Node2 = tempr['tdep'].values
        Node2 = [r'X_%i' % (t) for t in Node2]
        Label = tempr['pval/likel'].values
        Label = [r'%.3E' % (t) for t in Label]
        Edge_list = [Node1, Node2, Label]
        Edge_list = np.array(Edge_list).T.tolist()
        # Extract Node_list by total number of observations
        Node_list = np.arange(0, self._xi.shape[1], 1)
        Node_list = [r'X_%i' % (t) for t in Node_list]
        # Create CG dict
        return(Edge_list, Node_list, metricdf)

    def predict_CG(self, testtype, CGmetric):
        """
        Method to get CGs based on self.do_create_CG and self.do_CGM
        to pass it to utils.plot_DAG
        """
        # Get a list of all test names
        testn = self.get_testnames()
        # Loop over all independence tests
        for i, tp in enumerate(testn):
            El, Nl, metricdf = self.do_create_CG(testtype, CGmetric, tp)
            utils.print_in_console(what='CG Info',
                                   testname=tp, testmetric=CGmetric)
            utils.print_DF(metricdf)
            utils.plot_DAG(El, Nl)


    def plt_1hist(self, i, tdep, tindep):
        """
        Method to plot a histogramm of both the independent sample and the
        Residuals
        """
        txt = self.get_std_txt(what='math-Y~f(X)',
                               tdep=tdep,
                               tindep=tindep)
        plt.figure(r'Histogramm: %s' % (txt),
                   figsize=self._figsize)
        sns.distplot(self._xi[:, tindep],
                     norm_hist=True
                     )
        sns.distplot(self._results['%i' % (self._numberrun)][i]['Residuals'],
                     norm_hist=True
                     )
        plt.legend([r'$X_{%i}$' % (tindep),
                    r'$Residuals\ (X_{%i}-\hatX_{%i})$' % (tdep, tdep)])
        plt.title(r'$Histogramm:\ %s$' % (txt),
                  fontweight='bold')
        plt.xlabel(r'$X_{i}$')
        plt.ylabel(r'$f\left(X_{i}\right)$')
        plt.show()

    def plt_2metrics_groupedby(self, namecode,
                               metric1='statistics',
                               metric2='pval/likel',
                               groupedby='TestType'):
        """
        Method to plot the results of the independence test.
        Show p-value and statistic results grouped by TestType.
        Testnames:   source: self._results[.][.]['X-Residuals_Names']
                     shape:  n
        Testresults: source: self._results[.][.]['X-Residuals_Results']
                     shape:  2n (Value 0,1 -> Test 1 | Value 2,3 -> Test 2 ...)
        """
        # Get result array
        rs = self._results2V['%i' % (self._numberrun)]
        # Loop over all independence tests
        if 'p-value' in namecode:
            lbl = r'$dependence \leftarrow\ p-value\ \rightarrow independence$'
        if 'likelihood' in namecode:
            lbl = r'$not favored \leftarrow\ likelihood\ \rightarrow favored$'
        # selsect groupedby-combination one after the other
        for no in rs[groupedby].unique():
            data = rs[rs[groupedby] == no]
            x = data[metric1].values.tolist()
            y = data[metric2].values.tolist()
            tdep = data['tdep'].values.tolist()
            tindep = data['tindep'].values.tolist()
            # start plot
            plt.figure('%s, %s grouped by %s' % (metric1, metric2, groupedby),
                       figsize=self._figsize)
            if 'p-value' in namecode:
                plt.yscale('log')
            plt.scatter(x, y, marker='x', color='red')
            # annotate points
            for i, tp in enumerate(tdep):
                anx = x[i]
                any = y[i]
                antdep = tdep[i]
                antindep = tindep[i]
                txt = self.get_std_txt(what='math-Y~f(X)',
                                       tdep=antdep,
                                       tindep=antindep)
                plt.annotate('$%s$' % (txt), xy=(anx, any))
            plt.title(r'$Test Results: Grouped by "%s"$' % (str(no)),
                      fontweight='bold')
            plt.xlabel(r'$Test-Statistics$')
            plt.ylabel(lbl)
            plt.show()

    def plt_1metric_groupedby(self, namecode,
                              metric='pval/likel',
                              groupedby='2V-no'):
        """
        Method to plot the results of the independence test.
        Show p-value results grouped by variable combinations (2V-no).
        Get data from restructure_results()
        """
        # Get result array
        rs = self._results2V['%i' % (self._numberrun)]
        # settings for positioning data in plot
        diff_2Vno = 1
        diff_testtype = 1
        bbox_props = dict(boxstyle='round,pad=0.3',
                          fc='white', ec='r', alpha=0.5)
        # start plot
        plt.figure('Test Results: %s grouped by %s' % (metric, groupedby),
                   figsize=self._figsize)
        if 'p-value' in namecode:
            plt.yscale('log')
            lbl = r'$dependence \leftarrow\ p-value\ \rightarrow independence$'
        if 'likelihood' in namecode:
            lbl = r'$not favored \leftarrow\ likelihood\ \rightarrow favored$'
        # set start position
        x_pos = diff_2Vno/2
        # selsect groupedby-combination one after the other
        for no in rs[groupedby].unique():
            data1 = rs[rs[groupedby] == no]
            # select Testtype one after the other
            for i, tt in enumerate(rs['TestType'].unique()):
                data2 = data1[data1['TestType'] == tt]
                y = data2[metric].values.tolist()
                x = [x_pos for i in range(len(y))]
                tdep = data2['tdep'].values.tolist()
                tindep = data2['tindep'].values.tolist()
                plt.plot(x, y, c=plt.cm.tab10(i))
                # annotate min and max
                for i, tp in enumerate(y):
                    anx = x[i]
                    any = y[i]
                    antdep = tdep[i]
                    antindep = tindep[i]
                    txt = self.get_std_txt(what='math-Y~f(X)',
                                           tdep=antdep,
                                           tindep=antindep)
                    plt.text(anx, any, '$%s$' % (txt),
                             ha='center', va='center', rotation=0,
                             fontsize=8, bbox=bbox_props)
                # increase x positon for next TestType
                x_pos += diff_testtype
            # increase x positon for next 2V-no
            x_pos += diff_2Vno
        # further plot settings
        plt.xlim(0, x_pos-diff_2Vno*1.5)
        plt.title(r'$Test Results: %s grouped by %s$' % (metric, groupedby),
                  fontweight='bold')
        plt.tick_params(labelbottom=False)
        plt.tick_params(right=False, top=False, left=True, bottom=False)
        plt.legend(rs['TestType'].unique(),
                   loc='upper center',
                   bbox_to_anchor=(0.5, 0),
                   ncol=rs['TestType'].unique().shape[0])
        plt.ylabel(lbl)
        plt.show()

    def plt_GAMlog(self, i, tdep, tindep):
        """
        Method to plot the logs (callbacks) of pyGAM
        """
        # fetch not empty data
        listlog = [self._regmod[i].logs_['deviance'],
                   self._regmod[i].logs_['diffs'],
                   self._regmod[i].logs_['accuracy'],
                   self._regmod[i].logs_['coef']]
        listnames = ['Deviance', 'Diffs', 'Accuracy', 'Coef']
        listlog = [(i, x) for i, x in enumerate(listlog) if len(x) != 0]
        # start plot
        txt = self.get_std_txt(what='math-Y~f(X)',
                               tdep=tdep,
                               tindep=tindep)
        plt.figure(r'$GAMlog:\ %s$' % (txt),
                   figsize=self._figsize)
        for i in range(len(listlog)):
            plt.subplot(1, len(listlog), i+1)
            plt.plot(np.arange(0, len(listlog[i][1]), 1),
                     listlog[i][1])
            plt.xlabel(r'$Optimization\ Iteration$')
            plt.ylabel(r'$%s$' % (str(listnames[listlog[i][0]])))
        plt.suptitle(r'$GAMlog:\ %s$' % (txt))
        plt.show()

    def print_log_st(self, i, tdep, tindep, testtype):
        """
        Method to 'plot' a txt log-file regarding the results of the
        statistical tests on normality/independence.
        INPUT:  testtype (normality/independence), testnames and testresults
        """
        nm1 = 'X_%i' % (tindep)
        nm2 = 'Residuals'
        if 'normality' in testtype:
            testn = self._results['%i' % (self._numberrun)][i]['X_Names']
            testr = self._results['%i' % (self._numberrun)][i]['X_Results']
            print(stats.log_st('normality', testn, testr, nm1))
            testn = self._results['%i' % (self._numberrun)][i]['Residuals_Names']
            testr = self._results['%i' % (self._numberrun)][i]['Residuals_Results']
            print(stats.log_st('normality', testn, testr, nm2))
        elif 'independence' in testtype:
            testn = self._results['%i' % (self._numberrun)][i]['X-Residuals_Names']
            testr = self._results['%i' % (self._numberrun)][i]['X-Residuals_Results']
            print(stats.log_st('independence', testn, testr, nm1, nm2))

###############################################################################
