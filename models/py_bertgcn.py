from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.base import DGLError
from dgl.utils import expand_as_pair
from dgl.nn.pytorch import GraphConv
import torch


class GraphConvEdgeWeight(GraphConv):

    def forward(self, graph, feat, weight=None, edge_weights=None):
        with graph.local_scope():
            if not self._allow_zero_in_degree:
                if (graph.in_degrees() == 0).any():
                    raise DGLError('There are 0-in-degree nodes in the graph, '
                                   'output for those nodes will be invalid. '
                                   'This is harmful for some applications, '
                                   'causing silent performance regression. '
                                   'Adding self-loop on the input graph by '
                                   'calling `g = dgl.add_self_loop(g)` will resolve '
                                   'the issue. Setting ``allow_zero_in_degree`` '
                                   'to be `True` when constructing this module will '
                                   'suppress the check and let the code run.')

            # (BarclayII) For RGCN on heterogeneous graphs we need to support GCN on bipartite.
            feat_src, feat_dst = expand_as_pair(feat, graph)
            if self._norm == 'both':
                degs = graph.out_degrees().float().clamp(min=1)
                norm = torch.pow(degs, -0.5)
                shp = norm.shape + (1,) * (feat_src.dim() - 1)
                norm = torch.reshape(norm, shp)
                feat_src = feat_src * norm

            if weight is not None:
                if self.weight is not None:
                    raise DGLError('External weight is provided while at the same time the'
                                   ' module has defined its own weight parameter. Please'
                                   ' create the module with flag weight=False.')
            else:
                weight = self.weight

            if self._in_feats > self._out_feats:
                # mult W first to reduce the feature size for aggregation.
                if weight is not None:
                    feat_src = torch.matmul(feat_src, weight)
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
            else:
                # aggregate first then mult W
                graph.srcdata['h'] = feat_src
                if edge_weights is None:
                    graph.update_all(fn.copy_src(src='h', out='m'),
                                     fn.sum(msg='m', out='h'))
                else:
                    graph.edata['a'] = edge_weights
                    graph.update_all(fn.u_mul_e('h', 'a', 'm'),
                                     fn.sum(msg='m', out='h'))
                rst = graph.dstdata['h']
                if weight is not None:
                    rst = torch.matmul(rst, weight)

            if self._norm != 'none':
                degs = graph.in_degrees().float().clamp(min=1)
                if self._norm == 'both':
                    norm = torch.pow(degs, -0.5)
                else:
                    norm = 1.0 / degs
                shp = norm.shape + (1,) * (feat_dst.dim() - 1)
                norm = torch.reshape(norm, shp)
                rst = rst * norm

            if self.bias is not None:
                rst = rst + self.bias

            if self._activation is not None:
                rst = self._activation(rst)

            return rst


class GCN(nn.Module):
    def __init__(self,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 normalization='none'):
        super(GCN, self).__init__()
        self.layers = nn.ModuleList()
        # input layer
        self.layers.append(GraphConvEdgeWeight(in_feats, n_hidden, activation=activation, norm=normalization))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConvEdgeWeight(n_hidden, n_hidden, activation=activation, norm=normalization))
        # output layer
        self.layers.append(GraphConvEdgeWeight(n_hidden, n_classes, norm=normalization))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features, g, edge_weight):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(g, h, edge_weights=edge_weight)
        return h


class BERTGCN(nn.Module):
    def __init__(self, bert_model_name, classes=3, m=0.7, n_hidden=200, dropout=0.5):
        super(BERTGCN, self).__init__()
        self.max_length = 768
        self.bert = AutoTokenizer.from_pretrained(bert_model_name)
        self.bert_model = AutoModelForSequenceClassification.from_pretrained(bert_model_name, num_labels=classes)
        self.feat_dim = self.max_length
        self.gcn = GCN(
            in_feats=self.feat_dim,
            n_hidden=n_hidden,
            n_classes=classes,
            n_layers=1,
            activation=F.elu,
            dropout=dropout
        )
        self.m = m

    def forward(self, g, idx):
        input_ids, attention_mask = g.ndata['input_ids'][idx], g.ndata['attention_mask'][idx]
        if self.training:
            cls_feats = self.bert_model(input_ids, attention_mask, output_hidden_states=True).hidden_states[-1][:, 0]
            g.ndata['cls_feats'][idx] = cls_feats
        else:
            cls_feats = g.ndata['cls_feats'][idx]
        outputs = self.bert_model(input_ids, attention_mask)
        cls_pred = nn.Softmax(dim=1)(outputs.logits)
        gcn_logit = self.gcn(g.ndata['cls_feats'], g, g.edata["weight"])[idx]
        gcn_pred = torch.nn.Softmax(dim=1)(gcn_logit)
        pred = (gcn_pred + 1e-10) * self.m + cls_pred * (1 - self.m)
        pred = torch.log(pred)
        return pred
