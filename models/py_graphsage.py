import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F
import torch


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(
            in_feats=in_feats,
            out_feats=hid_feats,
            aggregator_type='mean'
        )
        self.conv2 = dglnn.SAGEConv(
            in_feats=hid_feats,
            out_feats=out_feats,
            aggregator_type='mean'
        )

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        h = torch.nn.Softmax(dim=1)(h) + 1e-10
        h = torch.log(h)
        return h
