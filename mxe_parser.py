import xml.etree.ElementTree as ET
import argparse
import os

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

        elif mxCell.get('edge') == "1":
            sourceCellId = mxCell.get('source')
            targetCellId = mxCell.get('target')
            sourceNodeId = findNode(nodes,sourceCellId)
            targetNodeId = findNode(nodes,targetCellId)
            if sourceNodeId != -1 and targetNodeId != -1:
                edges.append([sourceNodeId,targetNodeId])
                print('Edge from-to cell(', sourceCellId,',',targetCellId, ') node(',sourceNodeId,',', targetNodeId,')')

        else:
            print('none')

    return nodes,edges

def convertToUndirectedGraph(nodes,edges,nodeFeatures,edgeFeatures):
    undirectedNodes = []
    undirectedEdges = []
    #split each node as in and out nodes
    for i in range(numberOfNodes):
        # add an extra node (out node) for each defined vertex
        # in this way each node has its out node 
        #in node
        undirectedNodes.append(i)
        #out node
        undirectedNodes.append(i+1)
    
    for e in edges:
        sourceId = e[0]
        targetId = e[1]
        #out node
        sourceIndex =  nodes.index(sourceId)
        #in node
        targetIndex =  nodes.index(targetId)

        undirectedNodes[sourceIndex]    
    #numberOfNodes = 0
    #entryNodeId = -1
    #exitNodeId = -1
    return nodes,edges,nodeFeatures,edgeFeatures

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

    with open(edgesFilename, "w") as f:
        f.write("{0}\t{1}\t\n".format(len(edges),len(edgeFeatures[0])))
        for i in range(len(edges)):
            f.write("{0}\t{1}\t".format(edges[i][0],edges[i][1]))
            for j in range(len(edgeFeatures[0])):
                f.write("{0}\t".format(edgeFeatures[i][j]))
            f.write("\n")
            if generateEdgeSymmetry == True and edges[i][1] != edges[i][0]:
                f.write("{0}\t{1}\t".format(edges[i][1],edges[i][0]))
                for j in range(len(edgeFeatures[0])):
                    f.write("{0}\t".format(edgeFeatures[i][j]))
                f.write("\n")


def main():
    args = parse_args()
    tree = ET.parse(args.input)
    root = tree.getroot()

    nodes,edges = parse(root)
    
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
        nodes,edges,nodeFeatures,edgeFeatures = convertToUndirectedGraph(nodes,edges,nodeFeatures,edgeFeatures)
    
    generateEdgeSymmetry = (eval(args.generate_edge_symmetry)==True) & (eval(args.as_undirected)==False)
    writeToDisk(args.output,nodes,edges,nodeFeatures,edgeFeatures,generateEdgeSymmetry)



    #output.write("{0}\t{1}".format(a, am))
    


if __name__ == "__main__":
    main()