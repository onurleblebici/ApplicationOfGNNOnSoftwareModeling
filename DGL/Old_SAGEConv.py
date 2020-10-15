import torch as th
from torch import nn
from torch.nn import init

from .... import function as fn
from ....base import DGLError
from ....utils import expand_as_pair, check_eq_shape

class SAGEConv(nn.Module):

    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type,
                 bias=True,
                 norm=None,
                 activation=None):
        super(SAGEConv, self).__init__()

        self._in_src_feats, self._in_dst_feats = expand_as_pair(in_feats)
        self._out_feats = out_feats
        self._aggre_type = aggregator_type
        self.norm = norm

        # aggregator type: mean, max_pool, lstm, gcn
        if aggregator_type not in ['mean', 'max_pool', 'lstm', 'gcn']:
            raise KeyError('Aggregator type {} not supported.'.format(aggregator_type))
        if aggregator_type == 'max_pool':
            self.fc_pool = nn.Linear(self._in_src_feats, self._in_src_feats)
        if aggregator_type == 'lstm':
            self.lstm = nn.LSTM(self._in_src_feats, self._in_src_feats, batch_first=True)
        if aggregator_type in ['mean', 'max_pool', 'lstm']:
            self.fc_self = nn.Linear(self._in_dst_feats, out_feats, bias=bias)
        self.fc_neigh = nn.Linear(self._in_src_feats, out_feats, bias=bias)
        self.reset_parameters()


    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'max_pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.reset_parameters()
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, feat):
        with graph.local_scope():
            # Specify graph type then expand input feature according to graph type
            feat_src, feat_dst = expand_as_pair(feat, graph)
        if self._aggre_type == 'mean':
            graph.srcdata['h'] = feat_src
            graph.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        elif self._aggre_type == 'gcn':
            check_eq_shape(feat)
            graph.srcdata['h'] = feat_src
            graph.dstdata['h'] = feat_dst     # same as above if homogeneous
            graph.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'neigh'))
            # divide in_degrees
            degs = graph.in_degrees().to(feat_dst)
            h_neigh = (graph.dstdata['neigh'] + graph.dstdata['h']) / (degs.unsqueeze(-1) + 1)
        elif self._aggre_type == 'max_pool':
            graph.srcdata['h'] = F.relu(self.fc_pool(feat_src))
            graph.update_all(fn.copy_u('h', 'm'), fn.max('m', 'neigh'))
            h_neigh = graph.dstdata['neigh']
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(self._aggre_type))

        # GraphSAGE GCN does not require fc_self.
        if self._aggre_type == 'gcn':
            rst = self.fc_neigh(h_neigh)
        else:
            rst = self.fc_self(h_self) + self.fc_neigh(h_neigh)
            
        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)
        return rst

    
    def expand_as_pair(input_, g=None):
        if isinstance(input_, tuple):
            # Bipartite graph case
            return input_
        elif g is not None and g.is_block:
            # Subgraph block case
            if isinstance(input_, Mapping):
                input_dst = {
                    k: F.narrow_row(v, 0, g.number_of_dst_nodes(k))
                    for k, v in input_.items()}
            else:
                input_dst = F.narrow_row(input_, 0, g.number_of_dst_nodes())
            return input_, input_dst
        else:
            # Homograph case
            return input_, input_