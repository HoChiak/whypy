# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
from sklearn.svm import SVR
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pygam import LinearGAM, s, f, l

# import local libarys
from whypy.__packages.utils import utils


###############################################################################
def observations(modelclass, no_data=100, seed=None):
    """
    Method to load ovbservations. In the 2 variable case there are restrictions
    on the modelclass to needed for identifiable.
                4: linear model, three variables, gaussian noise
                   SCM ->   X0 ~ [G(5, 2.25)]
                            X1 ~ [X0*3 + G(0, 2.25)]
                            X2 ~ [X0*4 - X1*3 + G(0, 0.25)]
                5: polynomial model, three variables, gaussian noise
                   SCM ->   X0 ~ [G(5, 2.25)]
                            X1 ~ [X0**3 + G(0, 2.25)]
                            X2 ~ [X0**4 - X1**3 + G(0, 0.25)]
    no_data:    data points created for each observation
    seed:       set seed for np.random methods

    RETURN:
    Xi
    """
    np.random.seed(seed=seed)
    # model class 1
    if modelclass == '2VLiNGAM':
        desc = """
Modelclass '2VLiNGAM':
No. of variables:\t 2
Class of functions:\t Linear,
Class of noise distr:\t Non-Gaussian (here Uniform), Additive, non equivalent
SCM -> \tX0 ~ [U(0.4)]
       \tX1 ~ [2*X0 + U(0.2)]
               """
        X0 = np.array(0.4 * np.random.uniform(0, 5, no_data))
        X1 = np.array(2 * X0 + 0.2 * np.random.uniform(0, 5, no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        xi = np.concatenate([X0, X1], axis=1)
        no_obs = xi.shape[1]
    # model class 2
    elif modelclass == '2VNLiGAM':
        desc = """
Modelclass '2VNLiGAM':
No. of variables:\t 2
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, equivalent
SCM -> \tX0 ~ [5 + G(0, 2.25)]
       \tX1 ~ [X0^3 + G(0, 0.25)]
               """
        X0 = np.array(5 + 0.5 * np.random.randn(no_data))
        X1 = np.array(X0**3 + 0.5 * np.random.randn(no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        xi = np.concatenate([X0, X1], axis=1)
        no_obs = xi.shape[1]
    # model class 3
    elif modelclass == '3VLiNGAM':
        desc = """
Modelclass '3VLiNGAM':
No. of variables:\t 3
Class of functions:\t Linear,
Class of noise distr:\t Non-Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [5 + U(0.4)]
       \tX1 ~ [X0 + U(0.6)]
       \tX2 ~ [X0 + X1 + U(0.2)]
               """
        X0 = np.array(5 + 0.4 * np.random.uniform(0, 5, no_data))
        X1 = np.array(X0 + 0.6 * np.random.uniform(0, 5, no_data))
        X2 = np.array(X0 + X1 + 0.2 * np.random.uniform(0, 5, no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 4
    elif modelclass == '3VLiGAM':
        desc = """
Modelclass '3VLiGAM':
No. of variables:\t 3
Class of functions:\t Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(5, 0.25)]
       \tX1 ~ [X0*3 + G(0, 0.16)]
       \tX2 ~ [X0*4 + X1*3 + G(0, 0.09)]
               """
        X0 = np.array(5 + 0.5 * np.random.randn(no_data))
        X1 = np.array(X0*3 + 0.4 * np.random.randn(no_data))
        X2 = np.array(X0*4 + X1*3 + 0.3 * np.random.randn(no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.0
    elif modelclass == '3VNLiGAM-collider':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [G(0, 2.25)]
       \tX2 ~ [X0^4 + X1^2 + G(0, 0.25)]
               """
        Edge_list = [['X_0', 'X_2'],
                     ['X_1', 'X_2']]
        X0 = np.array(1 + 0.5 * np.random.randn(no_data))
        X1 = np.array(1.5 * np.random.randn(no_data))
        X2 = np.array(X0**4 + X1**2 + 0.5 * np.random.randn(no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.1
    elif modelclass == '3VNLiGAM-rev-collider':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [X0^4 + G(0, 2.25)]
       \tX2 ~ [X0^3 + G(0, 0.25)]
               """
        Edge_list = [['X_0', 'X_1'],
                     ['X_0', 'X_2']]
        X0 = np.array(1 + 0.5 * np.random.randn(no_data))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_data))
        X2 = np.array(X0**3 + 0.5 * np.random.randn(no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.2
    elif modelclass == '3VNLiGAM-confounded':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [X0^4 + G(0, 2.25)]
       \tX2 ~ [X0^4 + X1^2 + G(0, 0.25)]
               """
        Edge_list = [['X_0', 'X_1'],
                     ['X_0', 'X_2'],
                     ['X_1', 'X_2']]
        X0 = np.array(1 + 0.5 * np.random.randn(no_data))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_data))
        X2 = np.array(X0**4 + X1**2 + 0.5 * np.random.randn(no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.3
    elif modelclass == '3VNLiGAM-series':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [X0^4 + G(0, 2.25)]
       \tX2 ~ [X1^2 + G(0, 0.25)]
               """
        Edge_list = [['X_0', 'X_1'],
                     ['X_1', 'X_2']]
        X0 = np.array(1 + 0.5 * np.random.randn(no_data))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_data))
        X2 = np.array(X1**2 + 0.5 * np.random.randn(no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 5.4
    elif modelclass == '3VNLiGAM-none':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [G(4, 2.25)]
       \tX2 ~ [G(2, 0.25)]
               """
        Edge_list = []
        X0 = np.array(1 + 0.5 * np.random.randn(no_data))
        X1 = np.array(4 + 1.5 * np.random.randn(no_data))
        X2 = np.array(2 + 0.5 * np.random.randn(no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2], axis=1)
        no_obs = xi.shape[1]
    # model class 6
    elif modelclass == '4VNLiGAM':
        desc = """
Modelclass '3VNLiGAM':
No. of variables:\t 3
Class of functions:\t Non-Linear,
Class of noise distr:\t Gaussian, Additive, non equivalent
SCM -> \tX0 ~ [G(1, 2.25)]
       \tX1 ~ [X0^3 + G(0, 2.25)]
       \tX2 ~ [X0^3 + X1^2 + G(0, 0.25)]
       \tX3 ~ [G(2, 0.125)]
               """
        X0 = np.array(1 + 0.5 * np.random.randn(no_data))
        X1 = np.array(X0**4 + 1.5 * np.random.randn(no_data))
        X2 = np.array(X0**4 + X1**2 + 0.5 * np.random.randn(no_data))
        X3 = np.array(2 + 0.5 * np.random.randn(no_data))
        X0 = X0.reshape(-1, 1)
        X1 = X1.reshape(-1, 1)
        X2 = X2.reshape(-1, 1)
        X3 = X3.reshape(-1, 1)
        xi = np.concatenate([X0, X1, X2, X3], axis=1)
        no_obs = xi.shape[1]
    else:
        raise ValueError('Modelclass unknown')
    # Extract Node_list by total number of observations
    Node_list = np.arange(0, no_obs, 1)
    Node_list = [r'X_%i' % (t) for t in Node_list]
    # Plot correct Causal Graph
    print('CAUSAL GRAPH:')
    utils.plot_DAG(Edge_list, Node_list)
    return(xi, no_obs, desc)


def modelGam(no_obs, term='spline'):
    """
    Method to load unfitted Generalized Additive Models models of
    type modelclass

    INPUT:
    term: 'linear', 'spline' or 'factor'

    RETURN:
    [[model, ...] ... ]
    """
    # Create empty list size Xi times Xi
    regrmodl = utils.init_2V_list(no_obs)
    # Assign model to empty list
    # Loop trough all possible combinations of tdep and tindep
    for tdep in range(no_obs):
        for tindep in range(no_obs):
            if tdep != tindep:  # no diagonal values
                # GAM with linear term
                if term == 'linear':
                    regrmodl[tdep][tindep] = LinearGAM(l(0))
                # GAM with spline term
                elif term == 'spline':
                    regrmodl[tdep][tindep] = LinearGAM(s(0))
                # GAM with factor term
                elif term == 'factor':
                    regrmodl[tdep][tindep] = LinearGAM(f(0))
                else:
                    raise ValueError('Term unknown')
    return(regrmodl)


def modelSVR(no_obs, term='poly4'):
    """
    Method to load unfitted SVR models of type modelclass

    INPUT:
    term: 'linear', 'poly2' or 'poly4'

    RETURN:
    [[model, ...] ... ]
    """
    # Create empty list size Xi times Xi
    regrmodl = utils.init_2V_list(no_obs)
    # Assign model to empty list
    # Loop trough all possible combinations of tdep and tindep
    for tdep in range(no_obs):
        for tindep in range(no_obs):
            if tdep != tindep:  # no diagonal values
                # SVR with linear kernel
                if term == 'linear':
                    regrmodl[tdep][tindep] = SVR(kernel='linear',
                                                 gamma='auto_deprecated',
                                                 C=1.0, epsilon=0.1)
                # SVR with poly kernel
                elif term == 'poly2':
                    regrmodl[tdep][tindep] = SVR(kernel='poly', degree=2,
                                                 gamma=0.8,
                                                 C=0.8, epsilon=0.1)
                # GAM with factor term
                elif term == 'poly4':
                    regrmodl[tdep][tindep] = SVR(kernel='poly', degree=4,
                                                 gamma=0.8,
                                                 C=0.8, epsilon=0.1)
                else:
                    raise ValueError('Term unknown')
    return(regrmodl)


def scalerMinMax(no_obs):
    """
    Method to load a [0,1] MinMaxScaler

    RETURN:
    scaler
    """
    scaler = list()
    for t in range(no_obs):
        scaler.append(MinMaxScaler(feature_range=(0, 1), copy=False))
    return(scaler)


def scalerStandard(no_obs):
    """
    Method to load a zero mean and unit variance StandardScaler

    RETURN:
    scaler
    """
    scaler = list()
    for t in range(no_obs):
        scaler.append(StandardScaler(copy=False))
    return(scaler)
