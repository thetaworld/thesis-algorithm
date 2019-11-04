import time
import warnings
from numpy import linalg as la
import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize
import args
from algorithm.LFM import LFM
from config import RepMethod
# from lsi.LSI import LSI
from detect.SCAN import SCAN
from src.emd import getEMDCommunity, getEMDCommunitys
from src.xtadw import XTADW
# from src.emd import getEMDCommunity
from src.graph import Graph


class RECM(object):
    def __init__(self, dim, lamb, graph1, graph2):
        self.lamb = lamb
        self.dim = dim
        self.node_size = graph1.N + graph2.N
        self.graph1 = graph1
        self.graph2 = graph2
        self.features = None

    def train(self, epochs, rep_method=None, combineFeature=None):
        """

        :param epochs: train epochs
        :param rep_method: rep learn's var
        :param combineFeature: G1 and G2's combination fearture
        :return: xTAWD's result
        """
        # self.adj = self.getAdj()
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        self.M = self.getSimilarityMatrix(rep_method, combineFeature)
        # T is feature_size*node_num, text features
        # get from lsi
        self.T = self.features.T
        self.feature_size = self.features.shape[1]
        self.W = np.random.randn(self.dim, self.node_size)
        self.H = np.random.randn(self.dim, self.feature_size)
        # Update
        for i in range(epochs):
            print('Iteration ', i)
            # Update W
            B = np.dot(self.H, self.T)
            drv = 2 * np.dot(np.dot(B, B.T), self.W) - \
                  2 * np.dot(B, self.M.T) + self.lamb * self.W
            Hess = 2 * np.dot(B, B.T) + self.lamb * np.eye(self.dim)
            drv = np.reshape(drv, [self.dim * self.node_size, 1])
            rt = -drv
            dt = rt
            vecW = np.reshape(self.W, [self.dim * self.node_size, 1])
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.node_size))
                Hdt = np.reshape(np.dot(Hess, dtS), [
                    self.dim * self.node_size, 1])

                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecW = vecW + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.W = np.reshape(vecW, (self.dim, self.node_size))

            # Update H
            drv = np.dot((np.dot(np.dot(np.dot(self.W, self.W.T), self.H), self.T)
                          - np.dot(self.W, self.M.T)), self.T.T) + self.lamb * self.H
            drv = np.reshape(drv, (self.dim * self.feature_size, 1))
            rt = -drv
            dt = rt
            vecH = np.reshape(self.H, (self.dim * self.feature_size, 1))
            while np.linalg.norm(rt, 2) > 1e-4:
                dtS = np.reshape(dt, (self.dim, self.feature_size))
                Hdt = np.reshape(np.dot(np.dot(np.dot(self.W, self.W.T), dtS), np.dot(self.T, self.T.T))
                                 + self.lamb * dtS, (self.dim * self.feature_size, 1))
                at = np.dot(rt.T, rt) / np.dot(dt.T, Hdt)
                vecH = vecH + at * dt
                rtmp = rt
                rt = rt - at * Hdt
                bt = np.dot(rt.T, rt) / np.dot(rtmp.T, rtmp)
                dt = rt + bt * dt
            self.H = np.reshape(vecH, (self.dim, self.feature_size))
        self.Vecs = np.hstack(
            (normalize(self.W.T), normalize(np.dot(self.T.T, self.H.T))))

        # print(self.Vecs)
        # print(self.Vecs.shape)
        # get embeddings
        self.vectors_1 = {}
        self.vectors_2 = {}
        # todo possible has mistake
        look_back_1 = self.graph1.look_back_list
        node_size_1 = self.graph1.N
        # print(look_back_1)
        look_back_2 = self.graph2.look_back_list
        node_size_2 = self.graph2.N
        # print(look_back_2)
        # print("****************")
        for i, embedding in enumerate(self.Vecs):
            # print(i)
            # print("---------------")
            # print(look_back[i])
            if i <= node_size_1 - 1:
                self.vectors_1[look_back_1[i]] = embedding
            else:
                self.vectors_2[look_back_2[i - node_size_1]] = embedding
        return pd.DataFrame(self.vectors_1), pd.DataFrame(self.vectors_2)

    def getSimilarityMatrix(self, rep_method, combineFeature):
        """

        :param rep_method:
        :return: simlar matrix
        """
        C = np.zeros((self.node_size, self.node_size))
        for out_node_index in range(self.node_size):  # for each of N nodes
            for inner_node_index in range(self.node_size):  # for each of p landmarks
                # calcu similar matrix
                C[out_node_index, inner_node_index] = self.compute_similarity(
                    rep_method,
                    combineFeature[out_node_index],
                    combineFeature[inner_node_index])
        return C

    def compute_similarity(self, rep_method, vec1, vec2):
        """
        self.node_attributes: tuple of (same length) vectors of node attributes for corresponding nodes
        :param rep_method:
        :param vec1:
        :param vec2: two vectors of the same length
        # :param node_indices:
        :return: number between 0 and 1 representing their similarity
        """
        dist = rep_method.gammastruc * np.linalg.norm(vec1 - vec2)  # compare distances between structural identities
        return np.exp(-dist)

    def getT(self):
        g = self.graph1.G
        look_back = self.graph1.look_back_list
        features_1 = np.vstack([g.nodes[look_back[i]]['feature']
                                for i in range(g.number_of_nodes())])
        g = self.graph2.G
        look_back = self.graph2.look_back_list
        features_2 = np.vstack([g.nodes[look_back[i]]['feature']
                                for i in range(g.number_of_nodes())])
        self.features = np.vstack((features_1, features_2))
        self.preprocessFeature()
        # print(self.features.T)
        # print(self.features.T.shape)
        return self.features.T

    def preprocessFeature(self):
        if self.features.shape[1] > 200:
            U, S, VT = la.svd(self.features)
            Ud = U[:, 0:200]
            Sd = S[0:200]
            self.features = np.array(Ud) * Sd.reshape(200)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    rep_method = RepMethod(max_layer=2)
    arg_1 = args.args()
    arg_1.input = "data/test/karate.edgelist_1"
    arg_1.feature_file = "data/test/cora.features_1"
    t1 = time.time()
    nx_graph_1 = nx.read_edgelist(arg_1.input, nodetype=int, comments="%")
    adj_matrix_1 = nx.adjacency_matrix(nx_graph_1).todense()
    g_1 = Graph(adj_matrix_1)

    g_1.read_edgelist(filename=arg_1.input, weighted=arg_1.weighted,
                      directed=arg_1.directed)
    g_1.read_node_features(arg_1.feature_file)
    xTawd_1 = XTADW(g_1, arg_1.representation_size)
    structure_feature_1 = xTawd_1.get_features(rep_method)
    # print(structure_feature_1)
    # print(structure_feature_1.shape)

    arg_2 = args.args()
    arg_2.input = "data/test/karate.edgelist_3"
    arg_2.feature_file = "data/test/cora.features_3"
    nx_graph_2 = nx.read_edgelist(arg_2.input, nodetype=int, comments="%")
    adj_matrix_2 = nx.adjacency_matrix(nx_graph_2).todense()
    g_2 = Graph(adj_matrix_2)

    g_2.read_edgelist(filename=arg_2.input, weighted=arg_2.weighted,
                      directed=arg_2.directed)
    g_2.read_node_features(arg_2.feature_file)
    xTawd_2 = XTADW(g_2, arg_2.representation_size)
    structure_feature_2 = xTawd_2.get_features(rep_method)
    # print(structure_feature_2)
    structure_feature_2 = np.pad(structure_feature_2, ((0, 0),
                                                       (0,
                                                        abs(structure_feature_1.shape[1]
                                                            - structure_feature_2.shape[1]))),
                                 'constant', constant_values=(0, 0))
    # print(structure_feature_2.shape)
    # print(structure_feature_2)
    combineFuture = np.vstack((structure_feature_1, structure_feature_2))
    # print(combineFuture.shape)
    recm = RECM(6, 0.2, g_1, g_2)
    recm.getT()
    df1, df2 = recm.train(1, rep_method, combineFuture)
    algorithm = SCAN(g_1.G, 0.7, 3)
    communities = algorithm.execute()
    for community in communities:
        print('community: ', sorted(community))
    hubs_outliers = algorithm.get_hubs_outliers(communities)
    print('hubs: ', hubs_outliers[0])
    print('outliers: ', hubs_outliers[1])

    algorithm = SCAN(g_2.G, 0.7, 3)
    communities = algorithm.execute()
    for community in communities:
        print('community: ', sorted(community))
    hubs_outliers = algorithm.get_hubs_outliers(communities)
    print('hubs: ', hubs_outliers[0])
    print('outliers: ', hubs_outliers[1])
    print(df1)
    res = getEMDCommunity(['1', '13', '3', '7'], ['1', '13', '3', '7'], df1,df2)
    print(res)
    print('Success')


