from __future__ import print_function

import time
import warnings

import igraph
import numpy as np
import args
import collections
from mna.MNA import MNA
from detect.clique import CPM
from util.graph_helper import load_graph
from util.graph_helper import clone_graph
# from src.detect.Louvain import load_graph
from algorithm.EM import EM
from algorithm.GN import GN
from algorithm.LFM import LFM
from algorithm.LPA import LPA
from algorithm.Louvain import Louvain
from detect.SCAN import SCAN
from src.emd import getEMDCommunity
from recm import RECM
from src.graph import Graph
import networkx as nx
from src.xnetmf import RepMethod
from src.xtadw import XTADW

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

def load_graph_GN(path):
    G = nx.Graph()
    with open(path) as text:
        for line in text:
            vertices = line.strip().split(" ")
            source = int(vertices[0])
            target = int(vertices[1])
            G.add_edge(source, target)
    return G


def clone_graph(G):
    cloned_g = nx.Graph()
    for edge in G.edges():
        cloned_g.add_edge(edge[0], edge[1])
    return cloned_g


def load_graph_LV(path):
    G = collections.defaultdict(dict)
    with open(path) as text:
        for line in text:
            vertices = line.strip().split(" ")
            v_i = int(vertices[0])
            v_j = int(vertices[1])
            G[v_i][v_j] = 1.0
            G[v_j][v_i] = 1.0
    return G


class Vertex:

    def __init__(self, vid, cid, nodes, k_in=0):
        self._vid = vid
        self._cid = cid
        self._nodes = nodes
        self._kin = k_in


def main(arg_one_input="D:\\workspace\\pycharm\\paper_algorithm\\FindSimilarityCommunity\\src\\contrast\\data\\paper\\synthetic\\football_1.net",
         arg_one_feature_file="D:\\workspace\\pycharm\\paper_algorithm\\FindSimilarityCommunity\\src\\contrast\\data\\paper\\synthetic\\football_info_115_1",
         arg_two_input="D:\\workspace\\pycharm\\paper_algorithm\\FindSimilarityCommunity\\src\\contrast\\data\\paper\\synthetic\\football_1-0.05.net",
         arg_two_feature_file="D:\\workspace\\pycharm\\paper_algorithm\\FindSimilarityCommunity\\src\\contrast\\data\\paper\\synthetic\\football_info_115_2"
         ):
    warnings.filterwarnings("ignore", category=FutureWarning)
    t1 = time.time()
    # init graph
    arg_one = args.args()
    arg_one.input = arg_one_input
    arg_one.feature_file = arg_one_feature_file
    nx_graph_one = nx.read_edgelist(arg_one.input, nodetype=int, comments="%")
    adj_matrix_one = nx.adjacency_matrix(nx_graph_one).todense()
    g_one = Graph(adj_matrix_one)

    g_one.read_edgelist(filename=arg_one.input, weighted=arg_one.weighted,
                        directed=arg_one.directed)
    g_one.read_node_features(arg_one.feature_file)

    arg_two = args.args()
    arg_two.input = arg_two_input
    arg_two.feature_file = arg_two_feature_file
    nx_graph_two = nx.read_edgelist(arg_two.input, nodetype=int, comments="%")
    adj_matrix_two = nx.adjacency_matrix(nx_graph_two).todense()
    g_two = Graph(adj_matrix_two)

    g_two.read_edgelist(filename=arg_two.input, weighted=arg_two.weighted,
                        directed=arg_two.directed)
    g_two.read_node_features(arg_two.feature_file)
    # community detection
    # igraph.Graph.community_infomap()
    # SCAN
    algorithm_one = SCAN(g_one.G, 0.5, 3)
    communities_one = algorithm_one.execute()
    algorithm_two = SCAN(g_two.G, 0.5, 3)
    communities_two = algorithm_two.execute()
    return communities_one, communities_two,g_one,g_two


if __name__ == '__main__':
    communities_one,communities_two,g_one,g_two = main()
    mna = MNA(communities_one,communities_two,g_one,g_two)
    pre, recall = mna.execute()
    print("pre: ", pre, "recall: ", recall)