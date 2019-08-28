# -*- coding: utf-8 -*-

# import built in libarys
# import sys
# sys.setrecursionlimit(10000)

# import 3rd party libarys
import numpy as np
from scipy.stats import normaltest, shapiro, anderson
from scipy.stats import ks_2samp
from scipy.stats import mannwhitneyu, combine_pvalues


# import local libarys
from whypy.__packages.utils import utils
from whypy.__packages.utils.hsic_gam import *


###############################################################################
def normality(sample):
    """
    Method to evaluate normality of the given sample by
    Anderson-Darling-, Shapiro-Wilk and D’Agostino-Test.

    INPUT:  Sample to be tested

    RETURN: Test Statistic
    """
    # sample = np.array(5 + 1.5 * np.random.randn(100)) # test sample
    sample = sample.reshape(-1)
    #a_stat, a_cv, a_sl = anderson(sample, dist='norm')
    _, s_pv = shapiro(sample)
    _, d_pv = normaltest(sample)
    _, cp_pv = combine_pvalues([s_pv, d_pv])
    test_results = {'SW_pvalue': s_pv,
                    'Pearson_pvalue': d_pv,
                    'Combined_pvalue': cp_pv,
                    }
    return(test_results)


def likelihoodvariance(sample1, sample2):
    """
    Method to calculate the Likelihood based on the variance (only valid
    for Gaussian samples)
    """
    sample1 = np.array(sample1).reshape(-1)
    sample2 = np.array(sample2).reshape(-1)
    # calculate likelihood based on variance (only Gaussian)
    log1 = np.log(np.var(sample1))
    log2 = np.log(np.var(sample2))
    likeratio = - log1 - log2
    test_results = {'LikelihoodVariance': likeratio}
    return(test_results)


def likelihoodentropy(sample1, sample2):
    """
    Method to calculate the Likelihood based on the differential entropy
    of the error term
    """
    sample1 = np.array(sample1).reshape(-1)
    sample2 = np.array(sample2).reshape(-1)
    # calculate likelihood based on entropy TBD!!!!
    log1 = np.log(np.var(sample1))
    log2 = np.log(np.var(sample2))
    likeratio = - log1 - log2
    # TBD calculate likelihood based on entropy (also non Gaussian)
    test_results = {'LikelihoodEntropy': likeratio}
    return(test_results)


def kolmogorov(sample1, sample2):
    """
    Method to perform independence test between sample1 and sample2 by
    t-test (Gaussian only) and Kolmogorov-Smirnoff, Mann-Whitney and
    Anderson-Darling.
    (HSIC  and Cramer-von Mises test is to be done)

    INPUT:  Samples to be tested

    RETURN: Test Statistic
    """
    sample1 = np.array(sample1).reshape(-1)
    sample2 = np.array(sample2).reshape(-1)
    # calculate KolmogorovSmirnoff
    _, id_pv = ks_2samp(sample1, sample2)
    # process value
    id_pv = utils.check_inf_nan(id_pv)
    test_results = {'KolmogorovSmirnoff': id_pv}
    return(test_results)


def mannwhitneyu(sample1, sample2):
    """
    Method to perform independence test between sample1 and sample2 by
    t-test (Gaussian only) and Kolmogorov-Smirnoff, Mann-Whitney and
    Anderson-Darling.
    (HSIC  and Cramer-von Mises test is to be done)

    INPUT:  Samples to be tested

    RETURN: Test Statistic
    """
    sample1 = np.array(sample1).reshape(-1)
    sample2 = np.array(sample2).reshape(-1)
    # calculate MannWhitney
    _, id_pv = mannwhitneyu(sample1, sample2)
    # process value
    id_pv = utils.check_inf_nan(id_pv)
    test_results = {'MannWhitney': id_pv}
    return(test_results)

def hsic_gam(sample1, sample2):
    """
    Method to perform independence test between sample1 and sample2 by
    t-test (Gaussian only) and Kolmogorov-Smirnoff, Mann-Whitney and
    Anderson-Darling.
    (HSIC  and Cramer-von Mises test is to be done)

    INPUT:  Samples to be tested

    RETURN: Test Statistic
    """
    sample1 = np.array(sample1).reshape(-1)
    sample2 = np.array(sample2).reshape(-1)
    # calculate HSIC
    id_pv, _ = hsic_gam(sample1, sample2)
    # process value
    id_pv = utils.check_inf_nan(id_pv)
    test_results = {'HSIC': id_pv}
    return(test_results)











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
Kolmogorov-Smirnoff:
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
