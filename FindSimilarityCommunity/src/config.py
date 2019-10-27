class RepMethod:
    def __init__(self,
                 align_info=None,
                 p=None,
                 k=10,
                 max_layer=None,
                 alpha=0.1,
                 num_buckets=None,
                 normalize=True,
                 gammastruc=1,
                 gammaattr=1):
        self.p = p  # sample p points
        self.k = k  # control sample size
        self.max_layer = max_layer  # furthest hop distance up to which to compare neighbors
        self.alpha = alpha  # discount factor for higher layers
        self.num_buckets = num_buckets  # number of buckets to split node feature values into #CURRENTLY BASE OF LOG SCALE
        self.normalize = normalize  # whether to normalize node embeddings
        self.gammastruc = gammastruc  # parameter weighing structural similarity in node identity
        self.gammaattr = gammaattr  # parameter weighing attribute similarity in node identity


class Community:
    """
    community-related class
    """
    def __init__(self):
        pass
