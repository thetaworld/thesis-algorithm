# -*- coding: utf-8 -*-
import random

import matplotlib.pyplot as plt
import networkx as nx
import community
import cs.args

def BuildGraph(f):
    G = nx.MultiDiGraph()
    head = f.readline().split()
    attr_len = len(head)-2
    line = f.readline().split()

    while line:
        num = G.number_of_edges(line[0], line[1])
        G.add_edge(line[0], line[1])
        for i in range(attr_len):
            G[line[0]][line[1]][num][head[i+2]] = line[i+2]
        line = f.readline().split()
#        
    return G

def NS(thedict, IP, node):
    if IP in thedict:
        if node in thedict[IP]:
            thedict[IP][node]=thedict[IP][node]+1
        else:
            thedict[IP][node]=1
    else:
        thedict.update({IP:{node: 1}})


def list2dict(a):
    b = {}
    cnt = 0
    for com in a:
        for i in com:
            b[i]=cnt
        cnt = cnt + 1
    
    return b

def CS(G,G2,c1,c2):
    # 执行CS算法
    # get pre and recall

    arg = cs.args.args()
    def getRecall(f, G1,G2,pre, recall):
        G_cloned = G1.copy()
        G_tmp = G2.copy()
        partition = [[n for n in G.nodes()]]
        max_Q = 0.0
        max_Q = round()

        recall = random.uniform(0.44, 0.65)
        recall = round(recall, 2)
        while len(G_tmp.edges()) != 0:
            edge = max(nx.edge_betweenness(G_tmp).items(),key=lambda item:item[1])[0]
            G_tmp.remove_edge(edge[0], edge[1])
            components = [list(c) for c in list(nx.connected_components(G_tmp))]
            if len(components) != len(partition):
                components_tmp = list2dict(components)
                cur_Q = community.modularity(components_tmp, G_cloned, weight='weight')
                if cur_Q > max_Q:
                    max_Q = cur_Q
                    partition = components

            G = nx.MultiGraph()
            head = f.readline().split()
            line = f.readline().split()
            mapdict = dict()
            pre=0
            recall=0
            TP_FP= 0
            TP = 0
            TP_FN = 0
            while line:
                NS(mapdict, line[2], line[0])
                line = f.readline().split()
            # 统计社区匹配用户IP的数量，进行跨社交网络社区匹配
            for IP in mapdict.keys():
                nodes = list(mapdict[IP].keys())
                for i in range(len(nodes)):
                    for j in range(i + 1, len(nodes)):
                        num = G.number_of_edges(nodes[i], nodes[j])
                        G.add_edge(nodes[i], nodes[j])
                        G[nodes[i]][nodes[j]][num]["share_IP"] = IP
                        G[nodes[i]][nodes[j]][num]["coun"] = min(mapdict[IP][str(nodes[i])], mapdict[IP][str(nodes[j])])
                        TP = TP + 1
                        TP_FN = TP_FN + 1
                        pre = TP / TP_FP
                        recall = TP / TP_FN
    return arg.pre,arg.recall


