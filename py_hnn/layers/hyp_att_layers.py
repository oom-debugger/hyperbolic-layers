#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperbolic Attention layer
@author: mehrdad khatir
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.module import Module

from layers.hyp_layers import HypLinear, HypAct
from manifolds.poincare import PoincareBall
from manifolds.hyperboloid import Hyperboloid

torch.autograd.set_detect_anomaly(True)


def _make_mask_from_edges(edges, n):
    """Make adjacency matrix from edge tensor.
    
    Args:
        edges: A tensor of dim(N ,2) where N is the number of edges.
        n: number of nodes in the graph.
    Returns:
        adjacency matrix.
    """
    return torch.sparse_coo_tensor(indices=edges.transpose(0,1), values=torch.ones(edges.shape[0], dtype=int), size=(n,n)).to_dense()

class SharedSelfAttention(Module):
    """
    Hyperbolic attention layer with self-attention matrix.
    """
    def __init__(self, manifold, input_dim, output_dim, curvature, activation=None, alpha=0.2, dropout=0.1, use_bias=True):
        super(SharedSelfAttention, self).__init__()
        self.curvature = curvature
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.manifold = manifold

        # As the first step, we create an attention matrix using a linear layer
        # followed by a leakyReLU. 
        # inspired from "Graph Attention Networks" by P. Veickovic ICLR 2018.

        # Note: the paper uses shared attention matrix, which means it is the same W
        # for all inputs of all nodes. W_dim(in_dim=d_model, out_dim=d_k)        
        # However, if we want to have node-specific attention then
        # W_dim(graph_nodes * d_model, graph_nodes * graph_nodes * self.d_k)
        
        #TODO(mehrdad): mobius.mat_vec is more efficient that HypLinear.
        self.att_input_linear = HypLinear(
                manifold=self.manifold,
                in_features=self.input_dim,
                out_features=self.output_dim, 
                c=self.curvature, 
                dropout=dropout, 
                use_bias=use_bias)   
        self.data_input_linear = HypLinear(
                manifold=self.manifold,
                in_features=self.input_dim,
                out_features=self.output_dim, 
                c=self.curvature, 
                dropout=dropout, 
                use_bias=use_bias)   
        # self.att_leakyrelu = nn.LeakyReLU()
        self.att_out_linear = HypLinear(
                manifold=self.manifold, 
                in_features=self.output_dim,
                out_features=self.output_dim,
                c=self.curvature,
                dropout=dropout,
                use_bias=use_bias)
        nn.init.xavier_uniform(self.att_input_linear.weight)
        nn.init.xavier_uniform(self.data_input_linear.weight)
        nn.init.xavier_uniform(self.att_out_linear.weight)


        self.hyp_act = None
        if activation:
            self.hyp_act = HypAct(
                    manifold=self.manifold, 
                    c_in=self.curvature,
                    c_out=self.curvature, 
                    act=activation)


    def __repr__(self):
        return self.__class__.__name__ + ' (' + str(self.input_dim) + ' -> ' + str(self.output_dim) + ')'         

class SharedSelfAttentionV0(SharedSelfAttention):
    """
    Hyperbolic attention layer with self-attention matrix.
    
    Uses mobius midpoint for calculating attention coefficient.
    """
    def __init__(self, manifold, input_dim, output_dim, curvature, activation=None, alpha=0.2, dropout=0.1, use_bias=True):
        super(SharedSelfAttentionV0, self).__init__(manifold, input_dim, 
             output_dim, curvature, activation, alpha, dropout, use_bias)
       
    def forward(self, hyp_features, edges):
        if torch.any(torch.isnan(hyp_features)):
            raise ValueError('input to SharedSelfAttentionV0 has NaaN values')
        att_per_node = self.att_input_linear(hyp_features)
        reduced_hyp_features = self.data_input_linear(hyp_features)
        
        # create adjaceny matrix from edge info
        mask = edges.to_dense().transpose(0,1)
        if torch.nonzero(mask).numel() == 0:
            raise ValueError('adjacency matrix must have at least 1 edge.')
        hyp_att_embeddings = []
        # TODO(mehrdad): if adjaceny matrix is sparse matrix, we can optimize these nested for loops.
        for src_node, incoming_edges in enumerate(mask):
            # calculate the activation for each node
            masked_v = []
            masked_a = []
            for tgt_node, val in enumerate(incoming_edges):
                if val > 0.01:
                    # we define attention coefficient with the following formula.
                    coef = -1 * val * Hyperboloid.sqdist(att_per_node[tgt_node], att_per_node[src_node], c=self.curvature)
                    if torch.isnan(coef):
                        raise ValueError('we cannot have attentions coeficinet as NaaN')
                    masked_a.append(coef)
                    masked_v.append(reduced_hyp_features[tgt_node])
            if not masked_a and not masked_v:
                raise ValueError(
                        'A graph node must have at least one incoming edge.')
            # note that the attention coefficient do not have hyperbolic properties
            # masked_a = torch.softmax(torch.FloatTensor(torch.stack(masked_a).squeeze(-1)), dim=-1)
            masked_a = torch.nn.functional.normalize(torch.FloatTensor(torch.stack(masked_a).squeeze(-1)), dim=-1)
            masked_v = torch.stack(masked_v)
            # Note since for attention matrix we use linear layer which includes 
            # droupout rate as well. we omit the separate drop out layer.
            # project the hyperbolic vector to poincare model.
            poincare_v = PoincareBall.from_hyperboloid(x=masked_v, c=self.curvature)
            # calculate attention embeddings for each node.          
            att_embed = PoincareBall.mobius_midpoint(a=masked_a, v=poincare_v)
# =============================================================================
#             # This will cause a memory leak (OOM) during back propagation
#             vect = PoincareBall.mobius_mul(x=poincare_v, t=masked_a, dim=-1)
#             att_embed = vect[0]
#             for i in range(1, vect.shape[0]):
#                 att_embed = PoincareBall.mobius_add(att_embed, vect[i])
# =============================================================================                
            hyp_att_embeddings.append(Hyperboloid.from_poincare(att_embed, c=self.curvature))
        
        hyp_att_embeddings = torch.stack(hyp_att_embeddings)   
        # TODO(khatir): activation function makes it NaaN...
        if self.hyp_act:
            hyp_att_embeddings = self.hyp_act(hyp_att_embeddings)    
        return self.att_out_linear(hyp_att_embeddings)


class SharedSelfAttentionV1(SharedSelfAttention):
    """
    Hyperbolic attention layer with self-attention matrix.
    
    Uses mobius midpoint for calculating attention coefficient.
    """
    def __init__(self, input_dim, output_dim, curvature, activation=None, alpha=0.2, dropout=0.1, use_bias=True):
        super(SharedSelfAttentionV1, self).__init__(input_dim, 
             output_dim, curvature, activation, alpha, dropout, use_bias)
        self.att_linear = HypLinear(
                manifold=Hyperboloid,
                # re-size the attention dim to input size (since we concat 
                # 2 WVs we need to re-adjust the dimension)
                # Note that concat in hyperbloid does makes dim 2N - 1. (time dimension still remains the same).
                in_features=2 * self.output_dim - 1,
                out_features=self.output_dim,  # outputs the attnetion coefficient in hyperbolic plane (which will be a scalar in poincare model)
                c=self.curvature, 
                dropout=0.0, 
                use_bias=False)
        self.hyp_leakyrelu = HypAct(
                manifold=Hyperboloid, 
                c_in=self.curvature,  
                c_out=self.curvature, 
                act=nn.LeakyReLU(negative_slope=alpha))
        self.hyp_softmax = HypAct(
                manifold=Hyperboloid, 
                c_in=self.curvature,
                c_out=self.curvature, 
                act=torch.nn.Softmax(dim=0))
        
    def forward(self, in_nodes_features, sp_edges):
        in_nodes_features = torch.Tensor(in_nodes_features)
        n = in_nodes_features.shape[0]

        # adjacency matrix
        mask = sp_edges.to_dense()
        edges = sp_edges.coalesce().indices()

        in_embeddings = self.input_linear(in_nodes_features)
        w_ = torch.flatten(in_embeddings.unsqueeze(0).repeat(n, 1, 1), start_dim=0, end_dim=1)
        # this is for masking the a_. it should be normal mul (not mobius).
        w_masked = torch.mul(mask.flatten().unsqueeze(0).transpose(0,1), w_)
        
        # making edge based attention vectors
        tgt_e = PoincareBall.from_hyperboloid(
                torch.index_select(in_embeddings, 0, edges[0, :]))
        src_e = PoincareBall.from_hyperboloid(
                torch.index_select(in_embeddings, 0, edges[1, :]))
        concat = Hyperboloid.from_poincare(PoincareBall.concat(torch.stack([tgt_e, src_e], dim=-2)))
        a = self.att_linear(concat)
        # TODO(mehrdad): make sure softmax does not make the layer unstable.
        a = self.hyp_softmax(self.hyp_leakyrelu(a))
        a_masked = torch.flatten(
                torch.sparse_coo_tensor(
                        indices=edges, 
                        values=a, 
                        size=(n,n, self.output_dim)).to_dense(), 
                start_dim=0, 
                end_dim=1)
        # This is for masking the a_. it should be normal mul (not mobius).
        # This should be mobius_bmm.
        p_a = PoincareBall.from_hyperboloid(a_masked)
        p_w = PoincareBall.from_hyperboloid(w_masked)

        # Note that att coef vector is not poincare vector. It is a matrix
        # of scalars where each index contains a coefficient that is resulted
        # by poincare operation.   
        att_coef = torch.reshape(PoincareBall.mobius_bmm(p_a.unsqueeze(1), p_w.unsqueeze(-1).transpose(1, 2)), [n,n])
        # att_coef is a n*n matrix.
        p_in_embeddings = PoincareBall.from_hyperboloid(in_embeddings)
        p_out_embedding = []
        for a in att_coef:
            p_out_embedding.append(PoincareBall.mobius_midpoint(a=a, v=p_in_embeddings))
        hyp_att_embeddings = Hyperboloid.from_poincare(torch.stack(p_out_embedding))
        if self.hyp_act:
            hyp_att_embeddings = self.hyp_act(hyp_att_embeddings)
        return self.att_out_linear(hyp_att_embeddings)
        
        
class MultiHeadGraphAttentionLayer(Module):

    def __init__(self, input_dim, output_dim, dropout, activation=None, alpha=0.2, nheads=1, concat=None, self_attention_version='v0'):
        """Sparse version of GAT."""
        super(MultiHeadGraphAttentionLayer, self).__init__()
        if self_attention_version == 'v0':
            self_attention_layer_class = SharedSelfAttentionV0
        elif self_attention_version == 'v1':
            self_attention_layer_class = SharedSelfAttentionV1
        else:
            raise ValueError('Unknown self-attention version!')

        self.dropout = dropout
        self.output_dim = output_dim
        self.curvature = 1
        self.manifold = Hyperboloid
        self.attentions = [self_attention_layer_class(
                manifold=self.manifold,
                input_dim=input_dim, 
                output_dim=self.output_dim, 
                curvature=self.curvature, 
                alpha=alpha,
                activation=activation,
                dropout=self.dropout, 
                use_bias=False) for _ in range(nheads)]
        self.linear_out = None
        if nheads > 1:
            self.linear_out = HypLinear(
                manifold=self.manifold,
                in_features=nheads * (self.output_dim - 1) + 1,
                out_features=nheads * self.output_dim,
                c=self.curvature, 
                dropout=0.0, 
                use_bias=False)

        for i, attention in enumerate(self.attentions):
            self.add_module('attention_{}'.format(i), attention)

    def forward(self, input):
        x, adj = input
        if torch.any(torch.isnan(x)):
            raise ValueError('input tensor has NaaN values')
        x = F.dropout(x, self.dropout, training=self.training)
        p_h = PoincareBall.from_hyperboloid(torch.stack([att(x, adj) for att in self.attentions], dim=1))
        p_h = PoincareBall.concat(p_h)
        h = Hyperboloid.from_poincare(p_h)

        # TAdd a linear layer to make the output size equal to the self.output_dim
        if self.linear_out:
            h = self.linear_out(h)    

        h = F.dropout(h, self.dropout, training=self.training)
        return (h, adj)

class HypGraphSelfAttentionLayer(Module):
    """
    Hyperbolic attention layer with node-specific attention matrix.
    """
    def __init__(self, graph_size, vector_dim, curvature, dropout=0.1, use_bias=False):
        super(HypGraphSelfAttentionLayer, self).__init__()
        self.curvature = curvature
        self.vector_dim = vector_dim
        self.graph_dim = graph_size
        # As the first step, we create an attention matrix using a linear layer
        # followed by a leakyReLU. 
        # inspired from "Graph Attention Networks" by P. Veickovic ICLR 2018.
        self.input_linear = HypLinear(
                manifold=PoincareBall,
                in_features=self.graph_dim * self.vector_dim, 
                out_features=self.graph_dim * self.graph_dim * self.vector_dim, 
                c=self.curvature, 
                dropout=dropout, 
                use_bias=use_bias)
        # TODO(mehrdad): investigate if it is needed and maybe remove. I don't
        # know the effect of LeakyRelU on hyperbolic vectors.
        # self.att_leakyrelu = nn.LeakyReLU(negative_slope=1e-2, inpace=True)
        self.att_out_linear = HypLinear(
                manifold=PoincareBall,
                in_features=self.graph_dim * self.vector_dim,
                out_features=self.graph_dim * self.vector_dim,
                c=self.curvature,
                dropout=False,
                use_bias=use_bias)

    @classmethod
    def graph_attention(self, a, v, mask):
        """calculare the graph attention for a single node.
        
        Note: it is based on the eq.8, eq.9 in "Hyperbolic graph attention network" paper.
        
        args:
            a: attention coefficient vector of dim(M,M,N).
            v: M*N dimensional matrix, where M is the number of 
                vectors of dim N. 
            mask: a vector of dim (M, M) that indicates the connection map of
                the nodes in the graph.
        returns:
            a vector of dim(M,N) corresponding to the attention embeddings for
            the given node.
        """
        # TODO(mehrdad): investigate if graph connection can be uploaded at compile time.
        masked_v = []
        masked_a = []
        h = []
        for i, _ in enumerate(v):
            a_i = a[i].view()
            mask_i = mask[i].view()
            # For each node we extract the nodes in the connection map, then 
            # we calculate the mid-point for that node. This needs to be
            # repeated for all nodes in the graph.
            for idx, mask_bit in enumerate(mask_i):
                if mask_bit == 1:
                    masked_v.append(v[idx])
                    masked_a.append(a_i[idx])
            h.append(PoincareBall._mobius_midpoint(v=torch.stack(masked_v)), a=torch.stack(masked_a))
        return torch.stack(h)

    def forward(self, input_vectors, mask):
        # project the hyperbolic vector to poincare model.
        poincare_in = PoincareBall.proj(x=input_vectors, c=self.curvature)
        att_coeff = self.input_linear(poincare_in)
        att_vectors = self.graph_attention(a=att_coeff, v=poincare_in, mask=self.mask)
        return PoincareBall.to_hyperboloid(x=self.att_out_linear(att_vectors), c=self.curvature)