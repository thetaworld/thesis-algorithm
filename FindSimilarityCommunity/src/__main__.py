from __future__ import print_function

import time
import warnings

import numpy as np
import args
from algorithm.LFM import LFM
# from detect.SCAN import SCAN
from src.emd import getEMDCommunity
from recm import RECM
from src.graph import Graph
import networkx as nx
from src.xnetmf import RepMethod
from src.xtadw import XTADW


def main(args):
    warnings.filterwarnings("ignore", category=FutureWarning)
    t1 = time.time()
    # init graph
    rep_method = RepMethod(max_layer=2)
    arg_one = args.args()
    arg_one.input = "data/test/karate.edgelist_1"
    arg_one.feature_file = "data/test/cora.features_1"
    nx_graph_one = nx.read_edgelist(arg_one.input, nodetype=int, comments="%")
    adj_matrix_one = nx.adjacency_matrix(nx_graph_one).todense()
    g_one = Graph(adj_matrix_one)

    g_one.read_edgelist(filename=arg_one.input, weighted=arg_one.weighted,
                        directed=arg_one.directed)
    g_one.read_node_features(arg_one.feature_file)

    arg_two = args.args()
    arg_two.input = "data/test/karate.edgelist_3"
    arg_two.feature_file = "data/test/cora.features_3"
    nx_graph_two = nx.read_edgelist(arg_two.input, nodetype=int, comments="%")
    adj_matrix_two = nx.adjacency_matrix(nx_graph_two).todense()
    g_two = Graph(adj_matrix_two)

    g_two.read_edgelist(filename=arg_two.input, weighted=arg_two.weighted,
                        directed=arg_two.directed)
    g_two.read_node_features(arg_two.feature_file)
    # community detection

    # algorithm_one = SCAN(g_one.G, 0.7, 3)
    # communities_one = algorithm_one.execute()
    # for community in communities_one:
    #     print('community: ', sorted(community))
    # hubs_outliers_one = algorithm_one.get_hubs_outliers(communities_one)
    # print('hubs: ', hubs_outliers_one[0])
    # print('outliers: ', hubs_outliers_one[1])
    #
    # algorithm_two = SCAN(g_two.G, 0.7, 3)
    # communities_two = algorithm_two.execute()
    # for community in communities_two:
    #     print('community: ', sorted(community))
    # hubs_outliers_two = algorithm_two.get_hubs_outliers(communities_one)
    # print('hubs: ', hubs_outliers_two[0])
    # print('outliers: ', hubs_outliers_two[1])

    algorithm_one = LFM(g_one.G, 0.8)
    communities_one = algorithm_one.execute()
    # for c in communities_one:
    #     print(len(c), sorted(c))
    # print("----------------------")
    algorithm_two = LFM(g_two.G, 0.8)
    communities_two = algorithm_two.execute()
    # for c in communities_two:
    #     print(len(c), sorted(c))
    # print(type(communities_one[0]))
    # print(communities_one)
    # print(structure_feature_1)
    # print(structure_feature_1.shape)

    # node embed

    x_tawd_one = XTADW(g_one, arg_one.representation_size)
    structure_feature_one = x_tawd_one.get_features(rep_method)

    x_tawd_two = XTADW(g_two, arg_two.representation_size)
    structure_feature_two = x_tawd_two.get_features(rep_method)
    # print(structure_feature_2)

    structure_feature_one, structure_feature_two = completion_vec(structure_feature_one, structure_feature_two)
    # print(structure_feature_2.shape)
    # print(structure_feature_2)
    combine_future = np.vstack((structure_feature_one, structure_feature_two))
    # print(combineFuture.shape)
    recm = RECM(6, 0.2, g_one, g_two)
    recm.getT()
    g_one_node_embeding, g_two_node_embeding = recm.train(1, rep_method, combine_future)

    # get community pair
    # res = getEMDCommunity(
    #     ['13', '14', '15', '18', '2', '20', '22', '23',
    # '24', '25', '26', '27', '28', '29', '30', '31', '32', '33', '8',
    #      '9'],
    #     ['0', '1', '10', '11', '12', '13', '16', '17', '19', '2', '21', '3', '4', '5', '6', '7', '8', '9'],
    #     g_one_node_embeding, g_two_node_embeding)
    #  print(res)
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
    len_one = len(dict_community_one)
    len_two = len(dict_community_two)
    len_community_pair = min(len_one, len_two)
    communities_pair = []
    temp_list = []
    # use space to decrease time
    excluede_list_one = []
    excluede_list_two = []
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
    for communities_pair_ele in communities_pair_dist:
        if len(communities_pair) >= len_community_pair:
            break
        if communities_pair_ele[0][0] not in excluede_list_one \
                and communities_pair_ele[0][1] not in excluede_list_two:
            communities_pair.append(communities_pair_ele)
        excluede_list_one.append(communities_pair_ele[0][0])
        excluede_list_two.append(communities_pair_ele[0][1])

    return communities_pair


if __name__ == "__main__":
    main(args)
