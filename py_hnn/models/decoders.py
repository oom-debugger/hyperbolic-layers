"""Graph decoders."""
import manifolds
import torch.nn as nn
import torch.nn.functional as F

from layers.att_layers import GraphAttentionLayer
from layers.layers import GraphConvolution, Linear
from layers.poincare_layers import GraphConvolution as PGraphConvolution
from layers.poincare_layers import GraphAttentionLayerV0 as PGraphAttentionLayerV0
from layers.poincare_layers import GraphAttentionLayerV1 as PGraphAttentionLayerV1
 

from layers.hyp_att_layers import MultiHeadGraphAttentionLayer as MultiHeadHGAT

from manifolds.poincare import PoincareBall
from manifolds.hyperboloid import Hyperboloid


class Decoder(nn.Module):
    """
    Decoder abstract class for node classification tasks.
    """

    def __init__(self, c):
        super(Decoder, self).__init__()
        self.c = c

    def decode(self, x, adj):
        if self.decode_adj:
            input = (x, adj)
            probs, _ = self.cls.forward(input)
        else:
            probs = self.cls.forward(x)
        return probs


class GCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(GCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = GraphConvolution(args.dim, args.n_classes, args.dropout, act, args.bias)
        self.decode_adj = True

class PGCNDecoder(Decoder):
    """
    Graph Convolution Decoder.
    """

    def __init__(self, c, args):
        super(PGCNDecoder, self).__init__(c)
        act = lambda x: x
        self.cls = PGraphConvolution(in_features=args.dim, 
                                     out_features=args.n_classes, 
                                     dropout=args.dropout, 
                                     act=act, 
                                     curvature=self.c, 
                                     use_bias=args.bias)
        self.decode_adj = True

#    def decode(self, x, adj):
#        return super(PGCNDecoder, self).decode(x, adj)
#        return PoincareBall.euclidean2poincare(out, c=self.c)
#        return PoincareBall.poincare2euclidean(out, c=self.c)


class GATDecoder(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(GATDecoder, self).__init__(c)
         # (self, , , , , , , , , ):
        self.cls = GraphAttentionLayer(input_dim=args.dim, 
                                       output_dim=args.n_classes, 
                                       dropout=args.dropout, 
                                       activation=F.elu, 
                                       alpha=args.alpha, 
                                       nheads=1, 
                                       concat=True)
        self.decode_adj = True

    def decode(self, x, adj):
        return super(GATDecoder, self).decode(x, adj)

class PGATDecoderV0(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(PGATDecoderV0, self).__init__(c)
        self.cls = PGraphAttentionLayerV0(input_dim=args.dim, 
                                       output_dim=args.n_classes, 
                                       dropout=args.dropout, 
                                       activation=F.elu, 
                                       alpha=args.alpha, 
                                       nheads=1, 
                                       concat=True,  
                                       curvature=self.c, 
                                       use_bias= False)
#                                       use_bias= args.bias)
        self.decode_adj = True

class PGATDecoderV1(Decoder):
    """
    Graph Attention Decoder.
    """

    def __init__(self, c, args):
        super(PGATDecoderV1, self).__init__(c)
        self.cls = PGraphAttentionLayerV1(input_dim=args.dim, 
                                       output_dim=args.n_classes, 
                                       dropout=args.dropout, 
                                       activation=F.elu, 
                                       alpha=args.alpha, 
                                       nheads=1, 
                                       concat=True,  
                                       curvature=self.c, 
                                       use_bias= False)
#                                       use_bias= args.bias)
        self.decode_adj = True

#    def decode(self, x, adj):
#        out = super(PGATDecoderV1, self).decode(x, adj)
#        return PoincareBall.poincare2euclidean(out, c=self.c)


class HGATDecoderV0(Decoder):
    """
    Hyperbolic Graph Attention Decoder V0.
    """

    def __init__(self, c, args):
        super(HGATDecoderV0, self).__init__(c)
        self.cls = MultiHeadHGAT(input_dim=args.dim,
                                 output_dim=args.n_classes,
                                 dropout=args.dropout, 
                                 activation=F.elu, 
                                 alpha=args.alpha, 
                                 nheads=1, 
                                 self_attention_version='v0')
        self.decode_adj = True
        
    def decode(self, x, adj):
        h = super(GATDecoder, self).decode(x, adj)
        return self.manifold.proj_tan0(self.manifold.logmap0(h, c=self.c), c=self.c)


class LinearDecoder(Decoder):
    """
    MLP Decoder for Hyperbolic/Euclidean node classification models.
    """

    def __init__(self, c, args):
        super(LinearDecoder, self).__init__(c)
        self.manifold = getattr(manifolds, args.manifold)()
        self.input_dim = args.dim
        self.output_dim = args.n_classes
        self.bias = args.bias
        self.cls = Linear(self.input_dim, self.output_dim, args.dropout, lambda x: x, self.bias)
        self.decode_adj = False

    def decode(self, x, adj):
        h = self.manifold.proj_tan0(self.manifold.logmap0(x, c=self.c), c=self.c)
        return super(LinearDecoder, self).decode(h, adj)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, c={}'.format(
                self.input_dim, self.output_dim, self.bias, self.c
        )


model2decoder = {
    'GCN': GCNDecoder,
    'GAT': GATDecoder,
    'PGCN': PGCNDecoder,
    'PGATV0': PGATDecoderV0,
    'PGATV1': PGATDecoderV1,
    'HGATV0': LinearDecoder,
    'HNN': LinearDecoder,
    'HGCN': LinearDecoder,
    'MLP': LinearDecoder,
    'Shallow': LinearDecoder,
}

