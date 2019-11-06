#!/usr/bin/python
# coding:utf-8
import sys
import networkx as nx
from networkx.algorithms.community import k_clique_communities

class CPM:

    def execute(self, graph, k):
        communities = k_clique_communities(graph, k)
        return communities


if __name__ == '__main__':
    G = nx.karate_club_graph()
    algorithm = CPM()
    communities = algorithm.execute(G, 4)
    for community in communities:
        print(community)
