from __future__ import print_function

import math
import time
import warnings

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import normalize

import args
from config import RepMethod
from lsi.LSI import LSI
from src.emd import getEMDCommunity
from src.graph import Graph

"""
    XTAWD
"""


class XTADW(object):

    def __init__(self, graph, dim, lamb=0.2):
        self.g = graph
        # Penalty Term
        self.lamb = lamb
        # need k/2 todo
        self.dim = int(dim / 2)
        # self.train()

    def getAdj(self):
        graph = self.g.G
        node_size = self.g.node_size
        look_up = self.g.look_up_dict
        adj = np.zeros((node_size, node_size))
        for edge in self.g.G.edges():
            adj[look_up[edge[0]]][look_up[edge[1]]] = 1.0
            adj[look_up[edge[1]]][look_up[edge[0]]] = 1.0
        # ScaleSimMat result is same; adj's normalization
        # res = self.g.G_adj / np.sum(adj, axis=1)
        # res2=adj / np.sum(adj, axis=1)
        return adj / np.sum(adj, axis=1)

    def save_embeddings(self, filename):
        fout = open(filename, 'w')
        node_num = len(self.vectors.keys())
        fout.write("{} {}\n".format(node_num, self.dim * 2))
        for node, vec in self.vectors.items():
            fout.write("{} {}\n".format(node, ' '.join([str(x) for x in vec])))
        fout.close()

    def getT(self, path, fileName, numtopics):
        """
        get xTAWD need T Matrix from LSI
        :return:
        """
        return LSI.getT(path, fileName, numtopics)

    # def getT(self):
    #     """
    #     get xTAWD need T Matrix
    #     :return:
    #     """
    #     g = self.g.G
    #     look_back = self.g.look_back_list
    #     print(look_back)
    #     self.features = np.vstack([g.nodes[look_back[i]]['feature']
    #                                for i in range(g.number_of_nodes())])
    #     self.preprocessFeature()
    #     print(type(self.features.T))
    #     return self.features.T
    #
    # def preprocessFeature(self):
    #     """
    #     deal with text info
    #     :return:
    #     """
    #     print(self.features[1, :])
    #     print(len(self.features[1, :]))
    #     if self.features.shape[1] > 34:
    #         U, S, VT = la.svd(self.features)
    #         Ud = U[:, 0:34]
    #         print(Ud)
    #         Sd = S[0:34]
    #         print(Sd)
    #         self.features = np.array(Ud) * Sd.reshape(34)

    def train(self, epochs, T, rep_method=None, combineFeature=None):
        """

        :param epochs: train epochs
        :param rep_method: rep learn's var
        :param combineFeature: G1 and G2's combination fearture
        :return: xTAWD's result
        """
        # self.adj = self.getAdj()
        # M=(A+A^2)/2 where A is the row-normalized adjacency matrix
        # self.M = (self.adj + np.dot(self.adj, self.adj)) / 2
        # todo
        # (node_size_1+node_size_2)*(node_size_1+node_size_2)
        self.M = self.getSimilarityMatrix(rep_method, combineFeature)
        # T is feature_size*node_num, text features
        # get from lsi
        self.T = T
        node_size_1, node_size_2 = self.get_node_size()
        self.node_size = node_size_1 + node_size_2
        self.feature_size = self.T.shape[1]
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
        # get embeddings
        self.vectors = {}
        # todo possible has mistake
        # look_back = self.g.look_back_list
        for i, embedding in enumerate(self.Vecs):
            # self.vectors[look_back[i]] = embedding
            self.vectors[i] = embedding
        return pd.DataFrame(self.vectors)

    def get_khop_neighbors(self, rep_method):
        """

        :param rep_method: graph, RepMethod
        :return: dictionary of dictionaries: for each node, dictionary containing {node : {layer_num : [list of neighbors]}}
    #        dictionary {node ID: degree}
        """
        if rep_method.max_layer is None:
            rep_method.max_layer = self.g.N  # Don't need this line, just sanity prevent infinite loop

        kneighbors_dict = {}

        # only 0-hop neighbor of a node is itself
        # neighbors of a node have nonzero connections to it in adj matrix
        for node in range(self.g.N):
            neighbors = np.nonzero(self.g.G_adj[node])[-1].tolist()  ###
            if len(neighbors) == 0:  # disconnected node
                print("Warning: node %d is disconnected" % node)
                kneighbors_dict[node] = {0: {node}, 1: set()}
            else:
                if type(neighbors[0]) is list:
                    neighbors = neighbors[0]
                kneighbors_dict[node] = {0: {node}, 1: set(neighbors) - {node}}

        # For each node, keep track of neighbors we've already seen
        all_neighbors = {}
        for node in range(self.g.N):
            all_neighbors[node] = {node}
            all_neighbors[node] = all_neighbors[node].union(kneighbors_dict[node][1])

        # Recursively compute neighbors in k
        # Neighbors of k-1 hop neighbors, unless we've already seen them before
        current_layer = 2  # need to at least consider neighbors
        while True:
            if rep_method.max_layer is not None and current_layer > rep_method.max_layer: break
            reached_max_layer = True  # whether we've reached the graph diameter

            for i in range(self.g.N):
                # All neighbors k-1 hops away
                neighbors_prevhop = kneighbors_dict[i][current_layer - 1]

                khop_neighbors = set()
                # Add neighbors of each k-1 hop neighbors
                for n in neighbors_prevhop:
                    neighbors_of_n = kneighbors_dict[n][1]
                    for neighbor2nd in neighbors_of_n:
                        khop_neighbors.add(neighbor2nd)

                # Correction step: remove already seen nodes (k-hop neighbors reachable at shorter hop distance)
                khop_neighbors = khop_neighbors - all_neighbors[i]

                # Add neighbors at this hop to set of nodes we've already seen
                num_nodes_seen_before = len(all_neighbors[i])
                all_neighbors[i] = all_neighbors[i].union(khop_neighbors)
                num_nodes_seen_after = len(all_neighbors[i])

                # See if we've added any more neighbors
                # If so, we may not have reached the max layer: we have to see if these nodes have neighbors
                if len(khop_neighbors) > 0:
                    reached_max_layer = False

                # add neighbors
                kneighbors_dict[i][current_layer] = khop_neighbors  # k-hop neighbors must be at least k hops away

            if reached_max_layer:
                break  # finished finding neighborhoods (to the depth that we want)
            else:
                current_layer += 1  # move out to next layer
        # print(kneighbors_dict)
        return kneighbors_dict

    def get_degree_sequence(self, rep_method, kneighbors, current_node):
        """
        Turn lists of neighbors into a degree sequence
        :param rep_method:
        :param kneighbors: graph, RepMethod, node's neighbors at a given layer, the node
        :param current_node:
        :return:Output: length-D list of ints (counts of nodes of each degree), where D is max degree in graph
        """
        if rep_method.num_buckets is not None:
            degree_counts = [0] * int(math.log(self.g.max_degree, rep_method.num_buckets) + 1)
        else:
            degree_counts = [0] * (self.g.max_degree + 1)

        # For each node in k-hop neighbors, count its degree
        for kn in kneighbors:
            weight = 1  # unweighted graphs supported here
            degree = self.g.node_degrees[kn]
            if rep_method.num_buckets is not None:
                try:
                    degree_counts[int(math.log(degree, rep_method.num_buckets))] += weight
                except:
                    print("Node %d has degree %d and will not contribute to feature distribution" % (kn, degree))
            else:
                degree_counts[degree] += weight
        return degree_counts

    def get_features(self, rep_method, verbose=True):
        """
        Get structural features for nodes in a graph based on degree sequences of neighbors
        :param rep_method: graph, RepMethod
        :param verbose:
        :return: nxD feature matrix
        """
        before_khop = time.time()
        # Get k-hop neighbors of all nodes
        khop_neighbors_nobfs = self.get_khop_neighbors(rep_method)

        self.g.khop_neighbors = khop_neighbors_nobfs

        if verbose:
            print("max degree: ", self.g.max_degree)
            after_khop = time.time()
            print("got k hop neighbors in time: ", after_khop - before_khop)

        G_adj = self.g.G_adj
        num_nodes = G_adj.shape[0]
        if rep_method.num_buckets is None:  # 1 bin for every possible degree value
            num_features = self.g.max_degree + 1  # count from 0 to max degree
            # ...could change if bucketizing degree sequences
        else:  # logarithmic binning with num_buckets as the base of logarithm (default: base 2)
            num_features = int(math.log(self.g.max_degree, rep_method.num_buckets)) + 1
        feature_matrix = np.zeros((num_nodes, num_features))

        before_degseqs = time.time()
        for n in range(num_nodes):
            for layer in self.g.khop_neighbors[n].keys():  # construct feature matrix one layer at a time
                if len(self.g.khop_neighbors[n][layer]) > 0:
                    # degree sequence of node n at layer "layer"
                    deg_seq = self.get_degree_sequence(rep_method, self.g.khop_neighbors[n][layer], n)
                    # add degree info from this degree sequence, weighted depending on layer and discount factor alpha
                    feature_matrix[n] += [(rep_method.alpha ** layer) * x for x in deg_seq]
        after_degseqs = time.time()

        if verbose:
            print("got degree sequences in time: ", after_degseqs - before_degseqs)
        return feature_matrix

    def compute_similarity(self, rep_method, vec1, vec2, node_indices=None):
        """
        self.node_attributes: tuple of (same length) vectors of node attributes for corresponding nodes
        :param rep_method:
        :param vec1:
        :param vec2: two vectors of the same length
        :param node_indices:
        :return: number between 0 and 1 representing their similarity
        """
        dist = rep_method.gammastruc * np.linalg.norm(vec1 - vec2)  # compare distances between structural identities
        if self.g.node_attributes is not None:
            # distance is number of disagreeing attributes
            attr_dist = np.sum(self.g.node_attributes[node_indices[0]] != self.g.node_attributes[node_indices[1]])
            dist += rep_method.gammaattr * attr_dist
        return np.exp(-dist)  # convert distances (weighted by coefficients on structure and attributes) to similarities

    def getSimilarityMatrix(self, rep_method, feature_matrix):
        """

        :param rep_method:
        :return: simlar matrix
        """
        C = np.zeros((self.g_1.N + self.g_2.N, self.g_1.N + self.g_2.N))
        for out_node_index in range(self.g_1.N + self.g_2.N):  # for each of N nodes
            for inner_node_index in range(self.g_1.N + self.g_2.N):  # for each of p landmarks
                # calcu similar matrix
                C[out_node_index, inner_node_index] = self.compute_similarity(
                    rep_method,
                    feature_matrix[out_node_index],
                    feature_matrix[inner_node_index],
                    None)
        return C

    def get_sample_nodes(self, rep_method):
        """
        sample nodes in RepMethod
        Sample landmark nodes (to compute all pairwise similarities to in Nystrom approx)
        :param rep_method: graph (just need graph size here), RepMethod (just need dimensionality here)
        :return: np array of node IDs
        """
        # Sample uniformly at random
        sample = np.random.permutation(np.arange(self.g.N))[:rep_method.p]
        return sample

    def get_feature_dimensionality(self, rep_method, verbose=True):
        """
        control dimensionality of representations
        Get dimensionality of learned representations Related to rank of similarity matrix approximations
        :param rep_method: Input: graph, RepMethod
        :param verbose:
        :return: dimensionality of representations to learn (tied into rank of similarity matrix approximation)
        """
        p = int(rep_method.k * math.log(self.g.N, 2))  # k*log(n) -- user can set k, default 10
        if verbose:
            print("feature dimensionality is ", min(p, self.g.N))
        rep_method.p = min(p, self.g.N)  # don't return larger dimensionality than # of nodes
        return rep_method.p

    def get_representations(self, rep_method, verbose=True):
        """
        get the final representations of xNetMF
        :param rep_method:
        :param verbose:
        :return: xNetMF pipeline
        """
        # Node identity extraction
        feature_matrix = self.get_features(rep_method, verbose)

        # Efficient similarity-based representation
        # Get landmark nodes
        if rep_method.p is None:
            rep_method.p = self.get_feature_dimensionality(rep_method, verbose=verbose)  # k*log(n), where k = 10
        elif rep_method.p > self.g.N:
            print("Warning: dimensionality greater than number of nodes. Reducing to n")
            rep_method.p = self.g.N
        landmarks = self.get_sample_nodes(rep_method)

        # Explicitly compute similarities of all nodes to these landmarks
        before_computesim = time.time()
        C = np.zeros((self.g.N, rep_method.p))
        for node_index in range(self.g.N):  # for each of N nodes
            for landmark_index in range(rep_method.p):  # for each of p landmarks
                # select the p-th landmark
                C[node_index, landmark_index] = self.compute_similarity(
                    rep_method,
                    feature_matrix[node_index],
                    feature_matrix[landmarks[landmark_index]],
                    (node_index, landmarks[landmark_index]))

        before_computerep = time.time()

        # Compute Nystrom-based node embeddings
        W_pinv = np.linalg.pinv(C[landmarks])
        U, X, V = np.linalg.svd(W_pinv)
        Wfac = np.dot(U, np.diag(np.sqrt(X)))
        reprsn = np.dot(C, Wfac)
        after_computerep = time.time()
        if verbose:
            print("computed representation in time: ", after_computerep - before_computerep)

        # Post-processing step to normalize embeddings (true by default, for use with REGAL)
        if rep_method.normalize:
            reprsn = reprsn / np.linalg.norm(reprsn, axis=1).reshape((reprsn.shape[0], 1))
        return reprsn

    def get_community_similarity(self, comm_one, comm_two, vectors):
        """
        get community similarity
        :return:
        """
        return getEMDCommunity(comm_one, comm_two, vectors)


if __name__ == '__main__':
    pass
    # res = xTawd.train(10, rep_method, structure_feature)
    # print(res)
    # sim = xTawd.get_community_similarity([13, 1], [13, 1], res)
    # print(sim)
    # algorithm = SCAN(g.G, 0.7, 3)
    # communities = algorithm.execute()
    # for community in communities:
    #     print('community: ', sorted(community))
    # hubs_outliers = algorithm.get_hubs_outliers(communities)
    # print('hubs: ', hubs_outliers[0])
    # print('outliers: ', hubs_outliers[1])
