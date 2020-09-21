import numpy as np
import pandas as pd
from stellargraph import StellarGraph
from stellargraph import datasets

def load_from_file(filePrefix):
    nodes_filename = filePrefix + "_nodes.txt"
    edges_filename = filePrefix + "_edges.txt"
    
    node_features = None
    edge_features = None


    #https://stellargraph.readthedocs.io/en/stable/demos/basics/loading-numpy.html
    with open(nodes_filename) as f:
        num_nodes, num_node_features = map(int, f.readline().split('\t')[:-1])
        if num_node_features > 0:
            node_features = np.zeros((num_node_features,num_nodes))
            for i, line in enumerate(f.readlines()):
                features = np.array(list(map(float,line.split('\t')[1:-1])))
                for fIndex in range(num_node_features):
                    node_features[fIndex][i] = features[fIndex]
                #node_features[i] = features

    # read edge features
    with open(edges_filename) as f:
        num_edges, num_edge_features = map(int, f.readline().split('\t')[:-1])
        senders = np.zeros(num_edges, dtype=int)
        receivers = np.zeros(num_edges, dtype=int)  
        if num_edge_features > 0:
            edge_features = np.zeros((num_edge_features, num_edges))

        for i, line in enumerate(f.readlines()):
            elements = line.split('\t')
            senders[i] = int(elements[0])
            receivers[i] = int(elements[1])
            if edge_features is not None:
                features = np.array(list(map(float, elements[2:-1])))
                for fIndex in range(num_edge_features):
                    edge_features[fIndex][i] = features[fIndex]
                #edge_features[i] = np.array(list(map(float, elements[2:-1])))

    square_numeric_edges = pd.DataFrame( {"source": senders, "target": receivers})
    square_node_data = pd.DataFrame( { "x": node_features[0].tolist(), "y": node_features[1].tolist(), "z" : node_features[2].tolist()  } )

    #feature_array = np.array([[1.0, -0.2], [2.0, 0.3], [3.0, 0.0], [4.0, -0.5]], dtype=np.float32)
    #print("node_features")
    #print(node_features)
    #print("square_numeric_edges")
    #print(square_numeric_edges)
    square_numeric = StellarGraph(square_node_data, edges = square_numeric_edges)
    return square_numeric

 
gx = load_from_file("./output/directed_iselta")
print(gx.info())

dataset = datasets.Cora()
#display(HTML(dataset.description))
G, _ = dataset.load(subject_as_feature=True)
print(G.info())