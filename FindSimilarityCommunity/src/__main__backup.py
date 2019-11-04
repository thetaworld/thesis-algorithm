from __future__ import print_function

import random
import time
import warnings

from sklearn.linear_model import LogisticRegression
import numpy as np
import args
import xtadw
import src.xnetmf
from classify import Classifier, read_node_label
from src.graph import Graph
import networkx as nx
from src.xnetmf import RepMethod


def main(args):
    print("xnetmf","begin...")
    t1 = time.time()
    print("Reading...")
    nx_graph = nx.read_edgelist(agrs.input, nodetype=int, comments="%")
    adj_matrix = nx.adjacency_matrix(nx_graph).todense()
    print(adj_matrix)
    g = Graph(adj_matrix)
    rep_method = RepMethod(max_layer=2)  # Learn representations with xNetMF.  Can adjust parameters (e.g. as in REGAL)
    representations = src.xnetmf.get_representations(g, rep_method)
    print(representations)
    print(representations.shape)
    print("TAWD","begin...")
    print("Reading...")
    if args.graph_format == 'adjlist':
        g.read_adjlist(filename=args.input)
    elif args.graph_format == 'edgelist':
        g.read_edgelist(filename=args.input, weighted=args.weighted,
                        directed=args.directed)

    g.read_node_label(args.label_file)
    g.read_node_features(args.feature_file)
    model = xtadw.TADW(
        graph=g, dim=args.representation_size, lamb=args.lamb)
    t2 = time.time()
    print(t2 - t1)
    print("Saving embeddings...")
    model.save_embeddings(args.output)
    vectors = model.vectors
    X, Y = read_node_label(args.label_file)
    print("Training classifier using {:.2f}% nodes...".format(
        args.clf_ratio * 100))
    clf = Classifier(vectors=vectors, clf=LogisticRegression())
    clf.split_train_evaluate(X, Y, args.clf_ratio)


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    random.seed(32)
    np.random.seed(32)
    agrs = args.args()
    main(agrs)
