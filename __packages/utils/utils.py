# -*- coding: utf-8 -*-

# import built in libarys
from copy import deepcopy


# import 3rd party libarys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import display, HTML
import pandas as pd


# import local libarys


###############################################################################
def trans_nestedlist_to_tuple(nestedlist):
    """
    Function to transform nested list to a tuple of tuple.
    """
    tupletuple = [tuple(x) for x in nestedlist]
    tupletuple = tuple(tupletuple)
    return(tupletuple)


def trans_object_to_list(object1, n, dcopy=False):
    """
    Fuunction to expand one object to a list of length n from this object.
    """
    if dcopy is False:
        objectn = [object1 for i in range(n)]
    elif dcopy is True:
        objectn = [deepcopy(object1) for i in range(n)]
    return(objectn)


def trans_tuple_to_scalar(array):
    """
    Function to turn a np.array([scalar]) into scalar, if possible.
    """
    if array.size == 1:
        scalar = array.item()
        return(scalar)
    else:
        return(array)

def check_inf_nan(value):
    """
    Function to check if value is inf or nan. If True replace by 123.456
    """
    if np.isinf(value) is True:
        value = 123.456
        print('WARNING: a passed value is infinite and set to 123.456')
    elif np.isnan(value) is True:
        value = 123.456
        print('WARNING: a passed value is nan and set to 123.456')
    return(value)

def display_text_predefined(what, **kwargs):
    """
    Fucntion to print text in python console
    """
    if 'inference header' in what:
            text = r"""
<div style="background-color:black;color:white;padding:8px;letter-spacing:1em;" align="center">
<h2>Run   Causal   Inference</h2>
</div>
                    """
    if 'count bootstrap' in what:
            text = r"""
<div style="background-color:grey;color:white;padding:4px;" align="center">
<h3>Bootstrap: %i / %i</h3>
</div>
                    """ % (kwargs['current']+1, kwargs['sum'])
    elif 'visualization header' in what:
            text = r"""
<div style="background-color:black;color:white;padding:8px;letter-spacing:1em;" align="center">
<h2>INFERENCE   VISUALIZATION</h2>
</div>
                    """
    elif 'pairgrid header' in what:
        text = r"""
<div style="background-color:grey;color:white;padding:4px;" align="center">
<h3>PAIRGRID ALL VARIABLES</h3>
</div>
                """
    elif 'combination major header' in what:
        text = r"""

<div style="background-color:grey;color:white;padding:4px;" align="center">
<h3>Fitted Combination: X<sub>%i</sub> ~ f(X<sub>%s</sub>)</h3>
</div>
                """ % (kwargs['tdep'], str(kwargs['tindep']))
    elif 'combination minor header' in what:
        text = r"""
<div style="background-color:lightgrey;color:black;padding:0px;" align="center">
<h4>2D Visualisation for: X<sub>%i</sub> ~ f(X<sub>%s</sub>)</h4>
</div>
                """ % (kwargs['tdep'], kwargs['tindep'])
    elif 'result header' in what:
        text = r"""
<style>
table td, table th, table tr {text-align:left !important;}
</style>
<div style="background-color:black;color:white;padding:8px;letter-spacing:1em;" align="center">
<h2>RESULTS</h2>
</div>
<div style="background-color:grey;color:white;padding:4px;" align="center">
<h3>Configuration</h3>
</div>
<table>
                """
        for temp_key, temp_value in kwargs['dict'].items():
            text = r"""
%s
<tr>
<td align="left"><b>%s:</b></td>
<td>%s</td>
</tr>
                """ %(text, temp_key, str(temp_value))
        text = r"""
 %s
</table>
                """ % (text)
    elif 'normality' in what:
        text = r"""
</div>
<div style="background-color:grey;color:white;padding:4px;" align="center">
<h3>Normality Test</h3>
</div>
                """
    elif 'dependence residuals' in what:
        text = r"""
</div>
<div style="background-color:grey;color:white;padding:4px;" align="center">
<h3>Dependence: Indep. Variable - Residuals</h3>
</div>
                """
    elif 'dependence prediction' in what:
        text = r"""
</div>
<div style="background-color:grey;color:white;padding:4px;" align="center">
<h3>Dependence: Depen. Variable - Prediction (GoF)</h3>
</div>
                """
    elif 'thirdlevel' in what:
        text = r"""
<div style="background-color:lightgrey;color:black;padding:0px;" align="center">
<h4>%s</h4>
</div>
                """ % (kwargs['key'])
    display(HTML(text))

def display_df(df, fontsize='6pt'):
    """
    Function to output DataFrame in IPhython formatted as HTML.
    Adjust how df is displayed in width and fontsize TBD.
    """
    pd.options.display.float_format = '{:,.3e}'.format
    # df.style.set_properties(**{'font-size': fontsize})
    # display(HTML("<style>.container { width:100% !important; }</style>"))
    display(HTML(df.to_html()))











# def init_2V_list(no_obs):
#     """
#     Function returns an empty list of list shape=(no_obs, no_obs)
#     """
#     list_inner = list()
#     list_outer = list()
#     for t in range(no_obs):
#         list_inner.append(None)
#     for t in range(no_obs):
#         list_outer.append(list_inner.copy())
#     return(list_outer)
#
#
# def init_comp_matrix_2V(no_obs):
#     """
#     Function returns an empty list of list shape=(no_obs, no_obs)
#     """
#     matrix = np.eye(no_obs, M=None, k=0, dtype=int)
#     matrix -= 1
#     matrix = matrix * (-1)
#     return(matrix)
#
#
# def random_environments(obs, ration=0.9, method='Uniform'):
#     """
#     tbd
#     """


def plot_DAG(Edge_list, Node_list=None):
    """
    Function to plot a directed acyclic graph (DAG)
    Edge_list = [[Node1, Node2, EdgeLabel],
                 [Node1, Node2, EdgeLabel],
                 ...                      ]
    Node_list = [Node1, Node2, ...]
                List of all Node names (optional)
    """
    try:
        Edge_list = np.array(Edge_list)
        Edges = Edge_list[:, 0:2]
    except:
        Edges = None
    try:
        Labels = Edge_list[:, 2].astype(str)
    except:
        Labels = None
    # Extract Node_list from Edge_list, if not given
    if Node_list is None:
        Node_list = Edge_list[['Node1', 'Node2']].values.ravel()
        Node_list = pd.unique(Node_list)
    # Get length of list
    list_len = len(Node_list)
    # Multiply colors for all entrys in list
    colors = [[0, 0.31765, 0.61961] for i in range(list_len)]
    # Initiate Graph
    G = nx.MultiDiGraph()
    # Add nodes from Node_list
    try:
        G.add_nodes_from(Node_list)
    except:
        if Node_list is None:
            print('An Exeption occured in automatic extracting Node_list')
        else:
            print('Node_list is assigned but wrong type')
    # Add edges
    if Edges is not None:
        G.add_edges_from(Edges)
    # Fix positions
    pos = nx.shell_layout(G)
    # Draw the graph
    nx.draw(G, pos,
            node_size=1500,
            node_color=colors,
            edge_color='black',
            arrows=True)
    # Add Node Labels (Node_list names are used as labels)
    nx.draw_networkx_labels(G, pos,
                            font_size=14,
                            font_color='white',
                            font_weight='bold')
    # Add Edge Labels
    if Labels is not None:
        for i, tp in enumerate(Labels):
            Edge1 = Edges[i][0]
            Edge2 = Edges[i][1]
            Label = tp
            nx.draw_networkx_edge_labels(G, pos,
                                         edge_labels={(Edge1, Edge2): Label},
                                         font_size=12,
                                         font_color='black')
    plt.show()
