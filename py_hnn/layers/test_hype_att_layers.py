#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Hyperbolic Attention layer unittests.
@author: mehrdad khatir
"""
import torch
import unittest

from layers.hyp_att_layers import HypGraphSharedSelfAttentionLayer

class TestStringMethods(unittest.TestCase):
    def setUp(self):
        self.curvature = 1
        self.input_dim = 3
        self.output_dim = 3
        self.hyp_att = HypGraphSharedSelfAttentionLayer(self.input_dim, self.output_dim, self.curvature)

    def test_HypGraphSharedSelfAttentionLayer(self):
        t_in= torch.tensor([[1.0192, 0.0800, 0.1799], 
                            [1.1018, 0.2379, 0.3966],
                            [1.1018, 0.2379, 0.3966]])
        mask = torch.FloatTensor([[0, 1, 0], [1, 1, 0], [0, 1, 1]])
        self.assertEqual(torch.all(torch.isnan(self.hyp_att(t_in, mask))), False)


if __name__ == '__main__':
    unittest.main()