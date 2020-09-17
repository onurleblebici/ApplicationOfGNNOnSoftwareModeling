import stellargraph as sg

from stellargraph import StellarGraph
from stellargraph.data import EdgeSplitter
from stellargraph.mapper import GraphSAGELinkGenerator
from stellargraph.layer import GraphSAGE, HinSAGE, link_classification

from tensorflow import keras
from sklearn import preprocessing, feature_extraction, model_selection

from stellargraph import globalvar
from stellargraph import datasets
from IPython.display import display, HTML
import numpy as np
import pandas as pd
import argparse

def load_from_file(filePrefix):
    nodes_filename = filePrefix + "_nodes.txt"
    edges_filename = filePrefix + "_edges.txt"
    
    node_features = None
    edge_features = None


    #https://stellargraph.readthedocs.io/en/stable/demos/basics/loading-numpy.html
    with open(nodes_filename) as f:
        num_nodes, num_node_features = map(int, f.readline().split('\t')[:-1])
        if num_node_features > 0:
            node_features = np.zeros((num_nodes, num_node_features))
            for i, line in enumerate(f.readlines()):
                features = np.array(list(map(float,line.split('\t')[1:-1])))
                node_features[i] = features

    # read edge features
    with open(edges_filename) as f:
        num_edges, num_edge_features = map(int, f.readline().split('\t')[:-1])
        senders = np.zeros(num_edges, dtype=int)
        receivers = np.zeros(num_edges, dtype=int)    
        if num_edge_features > 0:
            edge_features = np.zeros((num_edges, num_edge_features))

        for i, line in enumerate(f.readlines()):
            elements = line.split('\t')
            senders[i] = int(elements[0])
            receivers[i] = int(elements[1])
            if edge_features is not None:
                edge_features[i] = np.array(list(map(float, elements[2:-1])))

    
    square_numeric_edges = pd.DataFrame( {"source": senders, "target": receivers})
    #feature_array = np.array([[1.0, -0.2], [2.0, 0.3], [3.0, 0.0], [4.0, -0.5]], dtype=np.float32)
    print("node_features")
    print(node_features)
    print("square_numeric_edges")
    print(square_numeric_edges)
    square_numeric = StellarGraph(node_features, square_numeric_edges)
    return square_numeric


#dataset = datasets.Cora()
#display(HTML(dataset.description))
#G, _ = dataset.load(subject_as_feature=True)

parser = argparse.ArgumentParser(description='mxe file parser')
parser.add_argument('--input-prefix', default=None, help='path of the mxe file')
args = parser.parse_args()

if args.input_prefix is None:
    print('Please specify input prefix argument')
    exit(0)

G = load_from_file(args.input_prefix)
print(G.info())

#print(G)
#exit(1)

edge_splitter_test = EdgeSplitter(G)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G, and obtain the
# reduced graph G_test with the sampled links removed:
G_test, edge_ids_test, edge_labels_test = edge_splitter_test.train_test_split(
    p=0.1, method="global", keep_connected=True
)

# Define an edge splitter on the reduced graph G_test:
edge_splitter_train = EdgeSplitter(G_test)

# Randomly sample a fraction p=0.1 of all positive links, and same number of negative links, from G_test, and obtain the
# reduced graph G_train with the sampled links removed:
G_train, edge_ids_train, edge_labels_train = edge_splitter_train.train_test_split(
    p=0.1, method="global", keep_connected=True
)

print(G_train.info())

print(G_test.info())

batch_size = 20
epochs = 20

num_samples = [20, 10]

train_gen = GraphSAGELinkGenerator(G_train, batch_size, num_samples)
train_flow = train_gen.flow(edge_ids_train, edge_labels_train, shuffle=True)

test_gen = GraphSAGELinkGenerator(G_test, batch_size, num_samples)
test_flow = test_gen.flow(edge_ids_test, edge_labels_test)

layer_sizes = [20, 20]
graphsage = GraphSAGE(
    layer_sizes=layer_sizes, generator=train_gen, bias=True, dropout=0.3
)

# Build the model and expose input and output sockets of graphsage model
# for link prediction
x_inp, x_out = graphsage.in_out_tensors()

prediction = link_classification(
    output_dim=1, output_act="relu", edge_embedding_method="ip"
)(x_out)

model = keras.Model(inputs=x_inp, outputs=prediction)

model.compile(
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss=keras.losses.binary_crossentropy,
    metrics=["acc"],
)

init_train_metrics = model.evaluate(train_flow)
init_test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the initial (untrained) model:")
for name, val in zip(model.metrics_names, init_test_metrics):
    print("\t{}: {:0.4f}".format(name, val))

history = model.fit(train_flow, epochs=epochs, validation_data=test_flow, verbose=2)

sg.utils.plot_history(history)

train_metrics = model.evaluate(train_flow)
test_metrics = model.evaluate(test_flow)

print("\nTrain Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, train_metrics):
    print("\t{}: {:0.4f}".format(name, val))

print("\nTest Set Metrics of the trained model:")
for name, val in zip(model.metrics_names, test_metrics):
    print("\t{}: {:0.4f}".format(name, val))