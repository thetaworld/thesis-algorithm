from __future__ import print_function

import time
import warnings
import numpy as np
import args
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


def main():
    warnings.filterwarnings("ignore", category=FutureWarning)
    t1 = time.time()
    # init graph
    rep_method = RepMethod(max_layer=2)
    arg_one = args.args()
    arg_one.input = "data/test/football_1.txt"
    arg_one.feature_file = "data/test/cora.features_1"
    nx_graph_one = nx.read_edgelist(arg_one.input, nodetype=int, comments="%")
    adj_matrix_one = nx.adjacency_matrix(nx_graph_one).todense()
    g_one = Graph(adj_matrix_one)

    g_one.read_edgelist(filename=arg_one.input, weighted=arg_one.weighted,
                        directed=arg_one.directed)
    g_one.read_node_features(arg_one.feature_file)

    arg_two = args.args()
    arg_two.input = "data/test/football_2.txt"
    arg_two.feature_file = "data/test/cora.features_2"
    nx_graph_two = nx.read_edgelist(arg_two.input, nodetype=int, comments="%")
    adj_matrix_two = nx.adjacency_matrix(nx_graph_two).todense()
    g_two = Graph(adj_matrix_two)

    g_two.read_edgelist(filename=arg_two.input, weighted=arg_two.weighted,
                        directed=arg_two.directed)
    g_two.read_node_features(arg_two.feature_file)
    # community detection

    algorithm_one = SCAN(g_one.G, 0.7, 3)
    communities_one = algorithm_one.execute()
    print(communities_one)

    algorithm_two = SCAN(g_two.G, 0.7, 3)
    communities_two = algorithm_two.execute()
    print(communities_two)

    # algorithm_one = LFM(g_one.G, 0.8)
    # communities_one = algorithm_one.execute()
    # algorithm_two = LFM(g_two.G, 0.8)
    # communities_two = algorithm_two.execute()

    # algorithm_one = GN(g_one.G)
    # communities_one = algorithm_one.execute()
    # algorithm_two = GN(g_two.G)
    # communities_two = algorithm_two.execute()
    # print(communities_one)
    # print(communities_two)
    # algorithm_one = LPA(g_one.G)
    # communities_one = algorithm_one.execute()
    # algorithm_two = LPA(g_two.G)
    # communities_two = algorithm_two.execute()
    # print(communities_one)
    # print(communities_two)
    # algorithm_one = EM(g_one.G, 9)
    # communities_one = algorithm_one.execute()
    # algorithm_two = EM(g_two.G, 2)
    # communities_two = algorithm_two.execute()
    # print(communities_one)
    # print(communities_two)

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
    recm = RECM(6, 0.2, g_one, g_two)
    recm.getT()
    g_one_node_embeding, g_two_node_embeding = recm.train(1, rep_method, combine_future)
    res = computer_pair(communities_one, communities_two, g_one_node_embeding, g_two_node_embeding)
    print(res)
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
    print(dict_community_one)
    print(dict_community_two)
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
    return communities_pair


if __name__ == "__main__":
    main()
