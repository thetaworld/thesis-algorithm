from __future__ import print_function

import time
import warnings

import igraph
import numpy as np
import args
import collections

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


def main(arg_one_input="data/paper/truedata/facebook.net",
             arg_one_feature_file="data/paper/truedata/facebook_info",
             arg_two_input="data/paper/truedata/twitter.net",
             arg_two_feature_file="data/paper/truedata/twitter_info"
         ):
    # syn data input as follow
    # arg_one_input="data/paper/synthetic/football_1.net",
    #          arg_one_feature_file="data/paper/synthetic/football_info_115_1",
    #          arg_two_input="data/paper/synthetic/football_1-0.01.net",
    #          arg_two_feature_file="data/paper/synthetic/football_info_115_2"




    # tue data input as follow
    # arg_one_input="data/paper/truedata/facebook.net",
    #          arg_one_feature_file="data/paper/truedata/facebook_info",
    #          arg_two_input="data/paper/truedata/twitter.net",
    #          arg_two_feature_file="data/paper/truedata/twitter_info"


    # disturb factor input as follow
    # arg_one_input="data/paper/disturb/football_1-0.1.net",
    #              arg_one_feature_file="data/paper/synthetic/football_info_115_1",
    #              arg_two_input="data/paper/disturb/football_1-0.1.net",
    #              arg_two_feature_file="data/paper/synthetic/football_info_115_2"

    warnings.filterwarnings("ignore", category=FutureWarning)
    t1 = time.time()
    # init graph
    # K is maxLayer
    # alpha is discount factor for higher layers
    rep_method = RepMethod(max_layer=4, alpha=0.1)
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
    # algorithm_one = SCAN(g_one.G, 0.5, 3)
    # communities_one = algorithm_one.execute()
    # algorithm_two = SCAN(g_two.G, 0.5, 3)
    # communities_two = algorithm_two.execute()

    # LV
    # G_one = load_graph_LV(arg_one_input)
    # algorithm_one = Louvain(G_one)
    # communities_one = algorithm_one.execute()
    # G_two=load_graph_LV(arg_two_input)
    # algorithm_two = Louvain(G_two)
    # communities_two = algorithm_two.execute()

    # CPM
    # algorithm_one = CPM()
    # communities_one = algorithm_one.execute(g_one.G, 4)
    # algorithm_two = CPM()
    # communities_two = algorithm_two.execute(g_two.G, 4)

    # GN other experiment run on this algorithm
    G_one = load_graph_GN(arg_one_input)
    algorithm_one = GN(G_one)
    communities_one = algorithm_one.execute()
    G_two=load_graph_GN(arg_two_input)
    algorithm_two = GN(G_two)
    communities_two = algorithm_two.execute()

    # LPA can not use this algorithm
    # algorithm_one = LPA(g_one.G)
    # communities_one = algorithm_one.execute()
    # algorithm_two = LPA(g_two.G)
    # communities_two = algorithm_two.execute()

    # LPA

    # algorithm_one = EM(g_one.G, 9)
    # communities_one = algorithm_one.execute()
    # algorithm_two = EM(g_two.G, 2)
    # communities_two = algorithm_two.execute()

    # LV
    # G_one = load_graph_LV(arg_one_input)
    # algorithm_one = Louvain(G_one)
    # communities_one = algorithm_one.execute()
    # G_two=load_graph_LV(arg_two_input)
    # algorithm_two = Louvain(G_two)
    # communities_two = algorithm_two.execute()

    # CPM
    # algorithm_one = CPM()
    # communities_one = algorithm_one.execute(g_one.G, 4)
    # algorithm_two = CPM()
    # communities_two = algorithm_two.execute(g_two.G, 4)

    # LFM
    # algorithm_one = LFM(g_one.G, 0.8)
    # communities_one = algorithm_one.execute()
    # algorithm_two = LFM(g_two.G, 0.8)
    # communities_two = algorithm_two.execute()

    # print(communities_one)
    # print(communities_two)

    # demo
    # algorithm = SCAN(g_one.G)
    # communities = algorithm.execute()
    # print(communities)

    # node embed

    x_tawd_one = XTADW(g_one, arg_one.representation_size)
    structure_feature_one = x_tawd_one.get_features(rep_method)

    x_tawd_two = XTADW(g_two, arg_two.representation_size)
    structure_feature_two = x_tawd_two.get_features(rep_method)

    structure_feature_one, structure_feature_two = completion_vec(structure_feature_one, structure_feature_two)
    combine_future = np.vstack((structure_feature_one, structure_feature_two))
    # S is dim the first para
    # lamb is the second para penalty factor
    recm = RECM(5, 0.1, g_one, g_two)
    recm.getT()
    g_one_node_embeding, g_two_node_embeding = recm.train(5, rep_method, combine_future)
    # print(communities_one)
    # print(communities_two)
    # print("shape", g_one_node_embeding.shape)
    res, len_community_pair = computer_pair(communities_one, communities_two, g_one_node_embeding, g_two_node_embeding)
    # print(res)
    TP_FP = len(res)
    TP = 0
    TP_FN = len_community_pair
    for tuple_ele in res:
        tuple_ele = tuple_ele[0]
        if tuple_ele[0] == tuple_ele[1]:
            TP = TP+1
    pre = TP/TP_FP
    recall = TP/TP_FN
    print("pre: ", pre, "recall: ", recall)
    print("sum time ", time.time() - t1)


def completion_vec(structure_feature_one, structure_feature_two):
    len_one = structure_feature_one.shape[1]
    len_two = structure_feature_two.shape[1]
    if len_one < len_two:
        structure_feature_one = np.pad(structure_feature_one, ((0, 0),
                                                               (0, len_two - len_one)),
                                       'constant', constant_values=(0, 0))

    elif len_one > len_two:
        structure_feature_two = np.pad(structure_feature_two, ((0, 0),
                                                               (0, len_one - len_two)),
                                       'constant', constant_values=(0, 0))
    return structure_feature_one, structure_feature_two


# computer community match pair
def computer_pair(communities_one, communities_two, g_one_node_embeding, g_two_node_embeding):
    dict_community_one = {}
    for i, c in enumerate(communities_one):
        dict_community_one[i] = c
    dict_community_two = {}
    for i, c in enumerate(communities_two):
        dict_community_two[i] = c
    # print(dict_community_one)
    # print(dict_community_two)
    len_one = len(dict_community_one)
    len_two = len(dict_community_two)
    len_community_pair = min(len_one, len_two)
    print("len_one", len_one, "len_two", len_two, "len_community_pair", len_community_pair)
    communities_pair = []
    temp_list = []
    # use space to decrease time
    exclude_list_one = []
    exclude_list_two = []
    communities_pair_dist = {}
    for key_one in dict_community_one.keys():
        for key_two in dict_community_two.keys():
            communities_one_set = list(dict_community_one[key_one])
            communities_two_set = list(dict_community_two[key_two])
            communities_one_set = [str(x) for x in communities_one_set]
            communities_two_set = [str(x) for x in communities_two_set]
            temp = getEMDCommunity(communities_one_set, communities_two_set,
                                   g_one_node_embeding, g_two_node_embeding)
            temp_list.append(key_one)
            temp_list.append(key_two)
            temp_tuple = tuple(temp_list)
            communities_pair_dist[temp_tuple] = temp
            temp_list = []
    communities_pair_dist = sorted(communities_pair_dist.items(), key=lambda x: x[1])
    # print(communities_pair_dist)
    for communities_pair_ele in communities_pair_dist:
        if communities_pair_ele[0][0] not in exclude_list_one \
                and communities_pair_ele[0][1] not in exclude_list_two:
            communities_pair.append(communities_pair_ele)
            # print("len(communities_pair)", len(communities_pair))
            if len(communities_pair) > len_community_pair:
                break
        exclude_list_one.append(communities_pair_ele[0][0])
        exclude_list_two.append(communities_pair_ele[0][1])
    return communities_pair, len_community_pair


if __name__ == "__main__":
    main()
    # init graph
    # rep_method = RepMethod(max_layer=2)
    # arg_one = args.args()
    # arg_one.input = "data/test/Wiki_edgelist_1.txt"
    # arg_one.feature_file = "data/test/wiki_info_2045_1"
    # nx_graph_one = nx.read_edgelist(arg_one.input, nodetype=int, comments="%")
    # adj_matrix_one = nx.adjacency_matrix(nx_graph_one).todense()
    # g_one = Graph(adj_matrix_one)
    # g_one.read_edgelist(filename=arg_one.input, weighted=arg_one.weighted,
    #                     directed=arg_one.directed)
    # g_one.read_node_features(arg_one.feature_file)
    # print(g_one.node_size)


