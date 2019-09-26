# -*- coding: utf-8 -*-

# import built in libarys

# import 3rd party libarys
from sklearn.svm import SVR
from pygam import LinearGAM, s, f, l
from sklearn.preprocessing import PolynomialFeatures, FunctionTransformer
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import Pipeline, make_pipeline
import numpy as np

# import local libarys
from whypy.__packages.utils import utils

###############################################################################


def lingam(term='spline'):
    """
    Method to load unfitted Generalized Additive Models models of
    type modelclass

    INPUT:
    term: 'linear', 'spline' or 'factor'

    RETURN:
    model
    """
    if term is 'linear':
        regmod = LinearGAM(l(0))
    # GAM with spline term
    elif term is 'spline':
        regmod = LinearGAM(s(0))
    # GAM with factor term
    elif term is 'factor':
        regmod = LinearGAM(f(0))
    else:
        raise ValueError('Given Gam term unknown')
    utils.display_get_params('LinearGAM Model Description',
                             regmod.get_params())
    return(regmod)


def svr(term='poly4'):
    """
    Method to load unfitted SVR models of type modelclass

    INPUT:
    term: 'linear', 'poly2' or 'poly4'

    RETURN:
    model
    """
    if term is 'linear':
        regmod = SVR(kernel='linear',
                     gamma='auto_deprecated',
                     C=1.0, epsilon=0.1)
    # SVR with poly kernel
    elif term is 'poly2':
        regmod = SVR(kernel='poly', degree=2,
                     gamma='auto_deprecated',
                     C=1.0, epsilon=0.1)
    # SVR with poly kernel
    elif term is 'poly4':
        regmod = SVR(kernel='poly', degree=4,
                     gamma='auto_deprecated',
                     C=1.0, epsilon=0.1)
    # SVR with rbf kernel
    elif term is 'rbf':
        regmod = SVR(kernel='rbf',
                     gamma='auto_deprecated',
                     C=1.0, epsilon=0.1)
    else:
        raise ValueError('Term unknown')
    utils.display_get_params('SVR Model Description', regmod.get_params())
    return(regmod)


def polyreg(degree=2):
    """
    Method to load unfitted LinearRegression models with polynomial features of
    given degree based on RidgeCV()

    INPUT:
    degree: polynomial degree

    RETURN:
    model
    """
    def polyfeatures(X):
        """
        Function to get polynomial features but no interactions
        """
        return(np.hstack([X**(i) for i in range(4+1)]))
    regmod = make_pipeline(FunctionTransformer(polyfeatures, validate=True),
                           RidgeCV())
    return(regmod)
###############################################################################
