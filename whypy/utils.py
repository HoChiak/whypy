# -*- coding: utf-8 -*-

# import built in libarys


# import 3rd party libarys
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from IPython.display import display, HTML


# import local libarys


###############################################################################
def init_2V_list(no_obs):
    """
    Method returns an empty list of list shape=(no_obs, no_obs)
    """
    list_inner = list()
    list_outer = list()
    for t in range(no_obs):
        list_inner.append(None)
    for t in range(no_obs):
        list_outer.append(list_inner.copy())
    return(list_outer)


def init_comp_matrix_2V(no_obs):
    """
    Method returns an empty list of list shape=(no_obs, no_obs)
    """
    matrix = np.eye(no_obs, M=None, k=0, dtype=int)
    matrix -= 1
    matrix = matrix * (-1)
    return(matrix)


def random_environments(obs, ration=0.9, method='Uniform'):
    """
    tbd
    """


def plot_DAG(Edge_list, Node_list=None):
    """
    Method to plot a directed acyclic graph (DAG)
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


def print_DF(df, fontsize='6pt'):
    """
    Method to output DataFrame in IPhython formatted as HTML.
    Adjust how df is displayed in width and fontsize TBD.
    """
    # df.style.set_properties(**{'font-size': fontsize})
    # display(HTML("<style>.container { width:100% !important; }</style>"))
    display(HTML(df.to_html()))


def print_in_console(what, **kwargs):
    """
    Method to print text in python console
    """
    if 'regmod header' in what:
        text = r"""

############################################################################
############################################################################
----------------------------------------------------------------------------
--------                        X_%i ~ f(X_%i)                        --------
----------------------------------------------------------------------------
                """ % (kwargs['tdep'], kwargs['tindep'])
    if 'model summary' in what:
        text = r"""
----------------------------------------------------------------------------
MODEL SUMMARY  RESULTS
----------------------------------------------------------------------------
                """
    if 'result header' in what:
        text = r"""
############################################################################
############################################################################
----------------------------------------------------------------------------
--------                      OVERALL  RESULTS                      --------
----------------------------------------------------------------------------
                """
    if 'CG Warning' in what:
        text = r"""
----------------------------------------------------------------------------
CAUSAL GRAPH PREDICTION
----------------------------------------------------------------------------
WARNING: This graph is predicted under the underlying assumptions and
         only valid under these assumptions. Please check verbose for
         more information.
                """
    if 'CG Info' in what:
        text = r"""
Testtype: %s | Testmetric: %s
                """ % (kwargs['testname'], kwargs['testmetric'])
    print(text)
