import xml.etree.ElementTree as ET
import argparse
import os
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='mxe file parser')
    parser.add_argument('--input', default=None, help='path of the mxe file')
    parser.add_argument('--output', default=None, help='prefix of the output files')
    parser.add_argument('--number-of-node-features', default=0, help='prefix of the output files')
    parser.add_argument('--number-of-edge-features', default=0, help='prefix of the output files')
    parser.add_argument('--as-undirected', default=False, help='Splits the original nodes as in-node and out-node and converts the graph to undirected')
    parser.add_argument('--generate-edge-symmetry', default=False, help='For directed graphs, keeps its original nodes but coverts the edge as undirected')
    args = parser.parse_args()

    if args.input is None:
        print('Please specify input and output arguments')
        exit(0)
    elif args.output is None:
        print('Please specify input and output arguments')
        exit(0)
    return args

def findNode(nodes, cellId):
    for i in range(len(nodes)):
        if nodes[i] == cellId:
            return i
    return -1

def createMatrix(rowNo,colNo):
    matrix=[] #define empty matrix
    for i in range(rowNo): 
        row=[] 
        for j in range(colNo):
            row.append(0) #adding 0 value for each column for this row
        matrix.append(row) #add fully defined column into the row
    return matrix

def parse(xmlRoot):
    nodes = []
    edges = []
    
    for mxCell in xmlRoot.iter('mxCell'):
        mxCellId = mxCell.get('id')
        if mxCell.get('vertex') == "1":
            mxCellName = mxCell.find('de.upb.adt.tsd.EventNode').get("name")
            if mxCellName == "[":
                entryNodeId = mxCellId
            elif mxCellName == "]":
                exitNodeId = mxCellId
            else:
                #skip virtual entry and exit nodes
                nodes.append(mxCellId)
                print('Vertex with cellId:',mxCellId , 'named:',mxCellName, ' nodeId:'+ str(len(nodes)-1))

    adjacencyMatrix = createMatrix(len(nodes),len(nodes))
    print(adjacencyMatrix)
    for mxCell in xmlRoot.iter('mxCell'):
        mxCellId = mxCell.get('id')
        if mxCell.get('edge') == "1":
            sourceCellId = mxCell.get('source')
            targetCellId = mxCell.get('target')
            sourceNodeId = findNode(nodes,sourceCellId)
            targetNodeId = findNode(nodes,targetCellId)
            if sourceNodeId != -1 and targetNodeId != -1:
                adjacencyMatrix[sourceNodeId][targetNodeId] = mxCellId
                edges.append([sourceNodeId,targetNodeId])
                print('Edge from-to cell(', sourceCellId,',',targetCellId, ') node(',sourceNodeId,',', targetNodeId,')')

    

    return nodes,edges,adjacencyMatrix

def convertToUndirectedGraph(nodes,edges,nodeFeatures,edgeFeatures):
    undirectedNodes = [0] * len(nodes) * 2
    undirectedNodeFeatures = [0] * len(nodes) * 2
    
    undirectedEdges = []
    undirectedEdgeFeatures =[]
    
    #split each node as in and out nodes
    for i in range(len(nodes)):
        # add an extra node (out node) for each defined vertex
        # in this way each node has its out node 
        #out (source) node
        undirectedNodes[i] = nodes[i]
        undirectedNodeFeatures[i] = nodeFeatures[i]
        #in (target) node
        undirectedNodes[i+len(nodes)] = nodes[i]
        undirectedNodeFeatures[i+len(nodes)] = nodeFeatures[i]
        #always add a edge between splited in - out nodes
        undirectedEdges.append([i,i+len(nodes)])
        undirectedEdges.append([i+len(nodes),i])
        #there is no feature availeable for in - out connection
        undirectedEdgeFeatures.append([0]*len(edgeFeatures[0]))
        undirectedEdgeFeatures.append([0]*len(edgeFeatures[0]))

    undirectedAdjacencyMatrix = createMatrix(len(undirectedNodes),len(undirectedNodes))
    for edgeIndex in range(len(edges)):
        #out node
        sourceId = edges[edgeIndex][0]
        #in node
        targetId = edges[edgeIndex][1]

        undirectedAdjacencyMatrix[sourceId][targetId+len(nodes)] = 1
        undirectedAdjacencyMatrix[targetId+len(nodes)][sourceId] = 1
        #no need to add (to prevent duplicate edges) self loop its already added on edge definition
        if sourceId != targetId:
            undirectedEdges.append([sourceId,targetId+len(nodes)])
            undirectedEdges.append([targetId+len(nodes),sourceId])
            undirectedEdgeFeatures.append(edgeFeatures[edgeIndex])
            undirectedEdgeFeatures.append(edgeFeatures[edgeIndex])

    return undirectedNodes,undirectedEdges,undirectedAdjacencyMatrix,undirectedNodeFeatures,undirectedEdgeFeatures

def writeToDisk(filenamePrefix,nodes,edges,nodeFeatures,edgeFeatures,generateEdgeSymmetry):
    nodesFileName = os.path.join("output",filenamePrefix + "_nodes.txt")
    edgesFilename = os.path.join("output",filenamePrefix + "_edges.txt")

    with open(nodesFileName, "w") as f:
        f.write("{0}\t{1}\t\n".format(len(nodes),len(nodeFeatures[0])))
        for i in range(len(nodes)):
            f.write("{0}\t".format(i))
            for j in range(len(nodeFeatures[0])):
                f.write("{0}\t".format(nodeFeatures[i][j]))
            f.write("\n")

    edges_clone = edges
    edgeFeatures_clone = edgeFeatures
    if generateEdgeSymmetry == True:
        edges_clone = edges.copy()
        edgeFeatures_clone = edgeFeatures.copy()
        for i in range(len(edges)):
            if edges[i][1] != edges[i][0]:
                edges_clone.append([ edges[i][1],edges[i][0] ])
                edgeFeatures_clone.append(edgeFeatures[i])

    with open(edgesFilename, "w") as f:
        f.write("{0}\t{1}\t\n".format(len(edges_clone),len(edgeFeatures_clone[0])))
        for i in range(len(edges_clone)):
            f.write("{0}\t{1}\t".format(edges_clone[i][0],edges_clone[i][1]))
            for j in range(len(edgeFeatures_clone[0])):
                f.write("{0}\t".format(edgeFeatures_clone[i][j]))
            f.write("\n")

def main():
    args = parse_args()
    tree = ET.parse(args.input)
    root = tree.getroot()

    nodes,edges,adjacencyMatrix = parse(root)
    
    nodeFeatures = []
    for i in range(len(nodes)):
        nodeFeatures.append([])
        for j in range(int(args.number_of_node_features)):
            nodeFeatures[i].append(0)
    
    edgeFeatures = []
    for i in range(len(edges)):
        edgeFeatures.append([])
        for j in range(int(args.number_of_edge_features)):
            edgeFeatures[i].append(0)
    

    if eval(args.as_undirected) == True:
        nodes,edges,adjacencyMatrix,nodeFeatures,edgeFeatures = convertToUndirectedGraph(nodes,edges,nodeFeatures,edgeFeatures)
    
    generateEdgeSymmetry = (eval(args.generate_edge_symmetry)==True) & (eval(args.as_undirected)==False)
    if generateEdgeSymmetry:
        for i in range(len(adjacencyMatrix)):
            for j in range(len(adjacencyMatrix[i])):
                if i < j:
                    adjacencyMatrix[j][i] = adjacencyMatrix[i][j]

    print("nodes")
    print(nodes)
    print("nodeFeatures")
    print(nodeFeatures)
    print("edges")
    print(edges)
    print("edgeFeatures")
    print(edgeFeatures)
    print("adjacencyMatrix")
    print(adjacencyMatrix)
    writeToDisk(args.output,nodes,edges,nodeFeatures,edgeFeatures,generateEdgeSymmetry)



    #output.write("{0}\t{1}".format(a, am))
    


if __name__ == "__main__":
    main()