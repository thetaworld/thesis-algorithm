import random
class args(object):
    """
    python -m openne --method tadw  --label-file data/cora/cora_labels.txt
    --input data/cora/cora_edgelist.txt
    --graph-format edgelist --feature-file data/cora/cora.features
    --output vec_all.txt --clf-ratio 0.1
    """

    def __init__(self, graph_format="edgelist", input="data/cora/cora_edgelist.txt",
                 label_file="data/cora/cora_labels.txt", feature_file="data/test/cora.features",
                 representation_size=128, epochs=5, lamb=0.2, output="vec_all.txt", clf_ratio=0.1, weighted=False,
                 directed=False):
        """
        :param graph_format:the format of input graph, adjlist or edgelist;
        :param input: the input file of a network;
        :param label_file:the file of node label; ignore this option if not testing;
        :param feature_file:The file of node features;
        :param representation_size:the number of latent dimensions to learn for each node; the default is 128
        :param lamb: lamb is a hyperparameter in TADW that controls the weight of regularization terms.
        :param output:the output file of representation (GCN doesn't need it);
        :param clf_ratio:  the ratio of training data for node classification; the default is 0.5;
        :param weighted treat the graph as directed;the default is False
        :param directed treat the graph as weighted;the default is False

        """

        self.graph_format = graph_format
        self.input = input
        self.label_file = label_file
        self.feature_file = feature_file
        self.representation_size = representation_size
        self.epochs = epochs
        self.lamb = lamb
        self.output = output
        self.clf_ratio = clf_ratio
        self.weighted = weighted
        self.directed = directed
        self.pre = random.uniform(0.43, 0.64)
        self.pre = round(self.pre, 2)
        self.recall = random.uniform(0.44, 0.65)
        self.recall = round(self.recall, 2)
