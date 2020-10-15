# Contruct a two-layer GNN model
import dgl
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn.functional as F

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


class SAGE(nn.Module):
    def __init__(self, in_feats, hid_feats, out_feats):
        super().__init__()
        self.conv1 = dglnn.SAGEConv(in_feats=in_feats, out_feats=hid_feats, aggregator_type='mean')
        self.conv2 = dglnn.SAGEConv(in_feats=hid_feats, out_feats=out_feats, aggregator_type='mean')

    def forward(self, graph, inputs):
        # inputs are features of nodes
        h = self.conv1(graph, inputs)
        h = F.relu(h)
        h = self.conv2(graph, h)
        return h

def evaluate(model, graph, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(graph, features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

graph = build_karate_club_graph()
numberOfNodeFeatures = 2
#word embedding gibi düşünmek lazım, örn;

embed = nn.Embedding(graph.num_nodes(), numberOfNodeFeatures)  # 34 nodes with embedding dim equal to 5
_input = torch.LongTensor([[1,2,4,5],[4,3,2,9]])
_result = embed(_input)
#print(_result)
print(embed.weight)
exit(1)
graph.ndata['feat'] = embed.weight

graph.ndata['label'] = torch.ones(graph.num_nodes(),dtype=int)

graph.ndata['train_mask'] =torch.ones(graph.num_nodes(), 1,dtype=int) #torch.randn(graph.num_nodes(), 1,dtype=byte)#torch.zeros(graph.num_nodes(), dtype=torch.int32)

graph.ndata['val_mask'] = torch.ones(graph.num_nodes(), 1,dtype=int)

for i in range(34):
    if i % 2 == 0:    
        graph.ndata['label'][i] = 0
    else:
        graph.ndata['label'][i] = 1
    if i % 5 == 0:
        graph.ndata['val_mask'][i] = 1
        graph.ndata['train_mask'][i] = 0
    else:
        graph.ndata['train_mask'][i] = 1
        graph.ndata['val_mask'][i] = 0
    

node_features = graph.ndata['feat']
node_labels = graph.ndata['label']
train_mask = graph.ndata['train_mask']
valid_mask = graph.ndata['val_mask']
#test_mask = graph.ndata['test_mask']
n_features = node_features.shape[1]
n_labels = int(node_labels.max().item() + 1)

#print("node_features : ", node_features)
#print("n_features : ", n_features )
#print("node_labels : ", node_labels)
#print("n_labels : ", n_labels)
#print("train_mask : ", train_mask)
#print("valid_mask : ", valid_mask)



model = SAGE(in_feats=n_features, hid_feats=100, out_feats=n_labels)
opt = torch.optim.Adam(model.parameters())

for epoch in range(10):
    model.train()
    # forward propagation by using all nodes
    logits = model(graph, node_features)
    # compute loss
    
    a = logits[train_mask]
    b = node_labels[train_mask]

    print("logits[train_mask] : ", logits[train_mask])
    print("node_labels[train_mask] : " , node_labels[train_mask])

    loss = F.cross_entropy(a,b)
    # compute validation accuracy
    acc = evaluate(model, graph, node_features, node_labels, valid_mask)
    # backward propagation
    opt.zero_grad()
    loss.backward()
    opt.step()
    print(loss.item())