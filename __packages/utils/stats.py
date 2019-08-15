# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
from scipy.stats import normaltest, shapiro, anderson
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu, combine_pvalues


# import local libarys


###############################################################################
def likelihood(sample1, sample2):
    """
    Method to calculate the Likelihood based on the variance (only Valid
    for Gaussian samples) and based on differential entropy of the error term
    """
    testnames = ['Likelihood']
    sample1 = np.array(sample1)
    sample2 = np.array(sample2)
    # calculate likelihood based on variance (only Gaussian)
    log1 = np.log(np.var(sample1.reshape(-1)))
    log2 = np.log(np.var(sample2.reshape(-1)))
    testresults = - log1 - log2
    # calculate likelihood based on entropy (also non Gaussian)
    return(testnames, testresults)


def normality(sample):
    """
    Method to evaluate normality of the given sample by
    Anderson-Darling-, Shapiro-Wilk and D’Agostino-Test.

    INPUT:  Sample to be tested

    RETURN: Test Statistic
    """
    # sample = np.array(5 + 1.5 * np.random.randn(100)) # test sample
    sample = sample.reshape(-1)
    a_stat, a_cv, a_sl = anderson(sample, dist='norm')
    s_stat, s_pv = shapiro(sample)
    d_stat, d_pv = normaltest(sample)
    cp_stat, cp_pv = combine_pvalues([s_pv, d_pv])
    testnames = ['Anderson-Darling', 'Shapiro-Wilk', 'D’Agostino',
                 'Combined p-values']
    testresults = [a_stat, a_sl, s_stat, s_pv, d_stat, d_pv,
                   cp_stat, cp_pv]
    return(testnames, testresults)


def independence(sample1, sample2):
    """
    Method to perform independence test between sample1 and sample2 by
    t-test (Gaussian only) and Kolmogroff-Smirnoff, Mann-Whitney and
    Anderson-Darling.
    (HSIC  and Cramer-von Mises test is to be done)

    INPUT:  Samples to be tested

    RETURN: Test Statistic
    """
    sample1 = sample1.reshape(-1)
    sample2 = sample2.reshape(-1)
    # if yj_transform is True:
    #    sample1, lambda1 = yeojohnson(sample1)
    #    sample2, lambda2 = yeojohnson(sample2)
    #    sample1 = np.array(sample1).reshape(-1)
    #    sample2 = np.array(sample2).reshape(-1)
    # tt_stat, tt_pv = ttest_ind(sample1, sample2)
    ks_stat, ks_pv = ks_2samp(sample1, sample2)
    mw_stat, mw_pv = mannwhitneyu(sample1, sample2)
    # ad_stat, ad_cv, ad_sl = anderson_ksamp([sample1, sample2])
    # cp_stat, cp_pv = combine_pvalues([ks_pv, mw_pv, ad_sl])
    testnames = ['Kolmogroff-Smirnoff', 'Mann-Whitney']
    testresults = [ks_stat, ks_pv, mw_stat, mw_pv]
    testresults = np.array(testresults)
    testresults[np.isinf(testresults)] = 8.888
    testresults[np.isnan(testresults)] = 8.888
    return(testnames, testresults)


def log_st(testtype, testnames, testresults, name1='NA', name2='NA'):
    """
    Method to create a txt log-file regarding the results of the
    statistical tests on normality/independence.

    INPUT:  testtype (normality/independence), testnames and testresults

    RETURN: Test Statistic Logfile
    """
    # Define Header #####################################
    assert testtype == 'normality' or testtype == 'independence', 'To print a logfile the testype must be either normality or independence'
    if testtype == 'normality':
        statistics = """
----------------------------------------------------------------------------
Test Statistics for Normality Test for %s
----------------------------------------------------------------------------
(H0: Data was drawn from Gaussian)
---- ---- ---- ----
                     """ % (str(name1))
    elif testtype == 'independence':
        statistics = """
----------------------------------------------------------------------------
Test Statistics for Independence Test P(%s), P(%s)
----------------------------------------------------------------------------
(Shoud be a independence test of second moment or higher)
---- ---- ---- ----
                     """ % (str(name1), str(name2))
    #####################################################
    # Add results from test
    for i, tp in enumerate(testnames):
        if isinstance(testresults[(i*2)+1], float):
            statistics = """
%s
%s:
statistic: \t %.3f;
p-value: \t %.3e;
---- ---- ---- ----
                         """ % (statistics, tp,
                                testresults[i*2], testresults[(i*2)+1])
        else:
            statistics = """
%s
%s:
statistic: \t %.3f;
c-value: \t %s;
---- ---- ---- ----
                         """ % (statistics, tp,
                                testresults[i*2],
                                str(testresults[(i*2)+1]))
    #####################################################
    # Define Footer with additional info
    if testtype == 'normality':
        statistics = """
%s
Anderson Darling:
(critical values > [5%%, 10%%, 5%%, 2.5%%, 1%%]: H0 can be rejected)
Shapiro-Wilk:
(p-value < alpha: H0 can be rejected)
D’Agostino:
(p-value < alpha: H0 can be rejected)
Combined p-values:
Combination of SW and Ag p-values by Fishers method.
---- ---- ---- ----
                    """ % (statistics)
    elif testtype == 'independence':
        statistics = """
%s
t-Test:
(2-sided test; Gaussian; 2 independent samples; H0: identical average (expected) values); (p-value < alpha: H0 can be rejected)
Kolmogroff-Smirnoff:
(2-sided test; distribution free; 2 independent samples; H0: drawn from the same continuous distribution); (p-value < alpha | statistic is high: H0 can be rejected)
Mann-Whitney:
(2-sided test; distribution free; 2 independent samples; H0: drawn from the same continuous distribution); (p-value < alpha: H0 can be rejected)
Anderson-Darling:
(2-sided test; distribution free; 2 independent samples; H0: drawn from the same continuous distribution); (p-value < alpha: H0 can be rejected)
Combined p-values:
Combination of KS, MW and AD p-values by Fishers method.
---- ---- ---- ----\n
                    """ % (statistics)
    #####################################################
    return(statistics)
