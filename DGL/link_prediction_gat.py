
import time
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import dgl
import dgl.function as fn
from dgl.nn.pytorch import edge_softmax, GATConv


def build_karate_club_graph():
    # All 78 edges are stored in two numpy arrays. One for source endpoints
    # while the other for destination endpoints.
    src = np.array([
        1, 2, 2, 3, 3, 3, 4, 5, 6, 6, 6, 7, 7, 7, 7, 8, 8, 9, 10, 10,
        10, 11, 12, 12, 13, 13, 13, 13, 16, 16, 17, 17, 19, 19, 21, 21,
        25, 25, 27, 27, 27, 28, 29, 29, 30, 30, 31, 31, 31, 31, 32, 32,
        32, 32, 32, 32, 32, 32, 32, 32, 32, 33, 33, 33, 33, 33, 33, 33,
        33, 33, 33, 33, 33, 33, 33, 33, 33, 33])
    dst = np.array([0, 0, 1, 0, 1, 2, 0, 0, 0, 4, 5, 0, 1, 2, 3, 0, 2, 2, 0, 4,
                    5, 0, 0, 3, 0, 1, 2, 3, 5, 6, 0, 1, 0, 1, 0, 1, 23, 24, 2, 23,
                    24, 2, 23, 26, 1, 8, 0, 24, 25, 28, 2, 8, 14, 15, 18, 20, 22, 23,
                    29, 30, 31, 8, 9, 13, 14, 15, 18, 19, 20, 22, 23, 26, 27, 28, 29, 30,
                    31, 32])
    # Edges are directional in DGL; Make them bi-directional.
    u = np.concatenate([src, dst])
    v = np.concatenate([dst, src])
    # Construct a DGLGraph
    return dgl.graph((u, v))

class GAT(nn.Module):
    def __init__(self,
                 num_layers,
                 in_dim,
                 num_hidden,
                 num_classes,
                 heads,
                 activation=None,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False):
        super(GAT, self).__init__()
        self.num_layers = num_layers
        self.gat_layers = nn.ModuleList()
        self.activation = activation
        # input projection (no residual)
        #self.gat_layers.append(GATConv(in_dim, num_hidden, heads[0],feat_drop, attn_drop, negative_slope, False, self.activation))
        self.gat_layers.append(GATConv(in_feats=in_dim, out_feats=num_hidden, num_heads=heads[0],feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=False, activation=self.activation))
        # hidden layers
        for l in range(1, num_layers):
            # due to multi-head, the in_dim = num_hidden * num_heads
            #self.gat_layers.append(GATConv(num_hidden * heads[l-1], num_hidden, heads[l],feat_drop, attn_drop, negative_slope, residual, self.activation))
            self.gat_layers.append(GATConv(in_feats=num_hidden * heads[l-1], out_feats=num_hidden, num_heads=heads[l],feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=residual, activation=self.activation))
        # output projection
        #self.gat_layers.append(GATConv(num_hidden * heads[-2], num_classes, heads[-1],feat_drop, attn_drop, negative_slope, residual, None))
        self.gat_layers.append(GATConv(in_feats=num_hidden * heads[-2], out_feats=num_classes, num_heads=heads[-1],feat_drop=feat_drop, attn_drop=attn_drop, negative_slope=negative_slope, residual=residual, activation=None))

    def forward(self, graph, inputs):
        h = inputs
        for l in range(self.num_layers):
            h = self.gat_layers[l](graph, h).flatten(1)
        # output projection
        logits = self.gat_layers[-1](graph, h).mean(1)
        return logits

class DotProductPredictor(nn.Module):
    def forward(self, graph, h):
        # h contains the node representations computed from the GNN defined
        # in the node classification section (Section 5.1).
        with graph.local_scope():
            graph.ndata['h'] = h
            graph.apply_edges(fn.u_dot_v('h', 'h', 'score'))
            return graph.edata['score']

class Model(nn.Module):
    def __init__(self, num_features, num_hidden, num_classes):
        super().__init__()
        #num_layers,in_dim,num_hidden,num_classes,heads,
        self.gat = GAT(num_layers=3,in_dim=num_features,num_hidden=num_hidden,num_classes=num_classes,heads=[2,2,2])
        self.pred = DotProductPredictor()

    def forward(self, g, neg_g, x):
        h = self.gat(g, x)
        return self.pred(g, h), self.pred(neg_g, h)

def construct_negative_graph(graph, k):
    src, dst = graph.edges()
    neg_src = src.repeat_interleave(k)
    neg_dst = torch.randint(0, graph.number_of_nodes(), (len(src) * k,))
    return dgl.graph((neg_src, neg_dst), num_nodes=graph.number_of_nodes())


def compute_loss(pos_score, neg_score):
    # Margin loss
    n_edges = pos_score.shape[0]
    return (1 - neg_score.view(n_edges, -1) + pos_score.unsqueeze(1)).clamp(min=0).mean()





graph = build_karate_club_graph()

embed = nn.Embedding(34, 1)  # 34 nodes with embedding dim equal to 5
#print(embed.weight)
feat = torch.tensor([[-0.5054],[ 1.3402],[-0.3230],[ 1.2102],[-1.8930],[ 1.1408],[-1.1150],[-0.7627],[ 1.2084],[-0.0549],[ 1.5347],[ 0.1806],[ 0.1131],[ 1.6221],
                     [ 0.3696], [ 0.6203], [-0.2875], [ 0.6158], [ 0.4358], [ 1.3153], [-1.3646], [ 1.5194], [ 0.0499], [-1.6464], [ 0.3114], [-0.2818], [ 1.0689],                     
                     [ 1.2322], [-1.4995], [ 0.0612], [-0.6617], [-1.0485], [-0.8325], [ 0.8061]], requires_grad=True)

feat2 = torch.tensor([[2],[4],[2],[4],[1],[3],[1],[2],[3],[-0.0549],[ 2],[ 5],[ 2],[ 5],
                     [ 3], [ 3], [2], [ 3], [ 3], [ 4], [1], [ 5], [ 2], [1], [ 3], [2], [ 4],                     
                     [ 4], [1], [ 2], [2], [1], [2], [ 3]], requires_grad=True)

graph.ndata['feat'] = feat2 #embed.weight

node_features = graph.ndata['feat']
n_features = node_features.shape[1]
#n_features=5
k = 5
model = Model(n_features, 4, 2)
opt = torch.optim.Adam(model.parameters(), lr=0.01)
for epoch in range(50):
    negative_graph = construct_negative_graph(graph, k)
    pos_score, neg_score = model(graph, negative_graph, node_features)
    loss = compute_loss(pos_score, neg_score)
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())

node_embeddings = model.gat(graph, node_features)

print(node_embeddings)


exit(1)

g, features, labels, mask = load_cora_data()

# create the model, 2 heads, each head has hidden size 8
net = GAT(g,
          in_dim=features.size()[1],
          hidden_dim=8,
          out_dim=7,
          num_heads=2)

# create optimizer
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)

# main loop
dur = []
for epoch in range(30):
    if epoch >= 3:
        t0 = time.time()

    logits = net(features)
    logp = F.log_softmax(logits, 1)
    loss = F.nll_loss(logp[mask], labels[mask])

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch >= 3:
        dur.append(time.time() - t0)

    print("Epoch {:05d} | Loss {:.4f} | Time(s) {:.4f}".format(epoch, loss.item(), np.mean(dur)))