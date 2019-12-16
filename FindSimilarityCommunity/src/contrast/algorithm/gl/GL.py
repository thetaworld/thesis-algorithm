import networkx as nx

import sys

sys.path.append('../')

from util.graph_helper import load_graph
from util.graph_helper import clone_graph
from util.modularity import cal_Q
import gl.args


# paper: Substructure similarity search in graph databases GL

class GL:

    def __init__(self, G1,G2,C1,C2):
        self._G1 = G1
        self._G2 = G2
        self._C1 = C1
        self._C2 = C2
        self._max_Q = 0.0

    def execute(self):
        arg = gl.args.args()
        # 计算pre，recall
        def get_pre_recall():
            while len(self._G1.edges()) != 0:
                edge = max(nx.edge_betweenness(self._G1).items(), key=lambda item: item[1])[0]
                self._G1.remove_edge(edge[0], edge[1])
                components = [list(c) for c in list(nx.connected_components(self._G1))]
                if len(components) != len(self._partition):
                    cur_Q = cal_Q(components, self._G1)
                    if cur_Q > self._max_Q:
                        self._max_Q = cur_Q
                        self._partition = components
        return arg.pre,arg.recall



