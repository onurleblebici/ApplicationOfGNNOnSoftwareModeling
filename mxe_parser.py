import xml.etree.ElementTree as ET
import argparse
import os
import numpy as np
from tabulate import tabulate
import re

def parse_args():
    parser = argparse.ArgumentParser(description='mxe file parser')
    parser.add_argument('--input', default=None, help='path of the mxe file')
    parser.add_argument('--output', default=None, help='prefix of the output files')
    parser.add_argument('--number-of-node-features', default=1, help='prefix of the output files')
    parser.add_argument('--number-of-edge-features', default=0, help='prefix of the output files')
    parser.add_argument('--as-undirected', default=False, help='Splits the original nodes as in-node and out-node and converts the graph to undirected')
    parser.add_argument('--generate-edge-symmetry', default=False, help='For directed graphs, keeps its original nodes but coverts the edge as undirected')
    parser.add_argument('--embeddings', default="embeddings.txt",help="mxe cell embeddings for automatic node feature extraction")
    parser.add_argument('--tab-to-eol', default=True,help="adds an extra tab to the end of each line")
    parser.add_argument('--add-info-firstline', default=False,help="Adds number of nodes,node features and edges,edge features as first line of output files.")
    parser.add_argument('--add-node-labels', default=False, help="Adds the node labels to the last column of the nodes output file.")
    parser.add_argument('--duplicate-node-features', default=1, help="if arg > 1 then clones same feature given n times")
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
    print("cell not found" + cellId)
    return -1

def findEdgesByTargetCell(edges, targetCellId, includeSelfLoops):
    foundEdges = []
    for e in edges:
        if e[1] == targetCellId and (includeSelfLoops == True or e[0] != e[1]):
            foundEdges.append(e) 
    return foundEdges

def findEdgesBySourceCell(edges, sourceCellId, includeSelfLoops):
    foundEdges = []
    for e in edges:
        if e[0] == sourceCellId and (includeSelfLoops == True or e[0] != e[1]):
            foundEdges.append(e) 
    return foundEdges

def hasSelfLoopEdge(edges,cellId):
    for e in edges:
        #if the node is self connection and it also has a sub graph
        if e[0] == e[1] and cellId == e[0]:
            return True
    return False

def createMatrix(rowNo,colNo):
    matrix=[] #define empty matrix
    for i in range(rowNo): 
        row=[] 
        for j in range(colNo):
            row.append(0) #adding 0 value for each column for this row
        matrix.append(row) #add fully defined column into the row
    return matrix

def createAdjacencyMatrix(edges,nodes,generate_edge_symmetry):
    adjacencyMatrix = createMatrix(len(nodes),len(nodes))
    for i in range(len(edges)):
        sourceCellId = edges[i][0]
        targetCellId = edges[i][1]
        sourceNode =findNode(nodes,sourceCellId)
        targetNode = findNode(nodes,targetCellId)
        adjacencyMatrix[sourceNode][targetNode] = 1
        if generate_edge_symmetry == True:
            adjacencyMatrix[targetNode][sourceNode] = 1
    return adjacencyMatrix

def parseGraph(xmlRoot, cellIdPathPrefix, cells):
    nodes = []
    edges = []
    entryCellId = None
    exitCellId = None
    subGraphs = []

    for mxCell in xmlRoot.findall('root/mxCell'):
        mxCellId = cellIdPathPrefix + mxCell.get('id')
        
        if mxCell.get('vertex') == "1":
            eventNode = mxCell.find('de.upb.adt.tsd.EventNode')
            #some elements has newline char
            mxCellName = eventNode.get("name").replace("\n","").strip()
            #some elements has &# like suffixes on their names
            if mxCellName == "[" or mxCellName.startswith("[&#"):
                entryCellId = mxCellId
                mxCellName = "["
            elif mxCellName == "]" or mxCellName.startswith("]&#"):
                mxCellName = "]"
                exitCellId = mxCellId
            #else:
                #skip virtual entry and exit nodes
            
            nodes.append(mxCellId)
            cells[mxCellId] = mxCellName

            print('Vertex with cellId:',mxCellId , 'named:<',mxCellName, '> nodeId:'+ str(len(nodes)-1))
            
            mxGraphModel = eventNode.find('mxGraphModel')
            if mxGraphModel is not None:
                #if vertex has child graph
                print('HAS CHILD GRAPH->START PARSING')
                childNodes, childEdges, childEntryCellId, childExitCellId, childSubGraphs = parseGraph(mxGraphModel,mxCellId + "-",cells)
                
                # first existing then new nodes; in this way previously added nodes indexes doesn't changes.
                nodes = nodes + childNodes
                edges = edges + childEdges
                subGraphs = subGraphs + childSubGraphs
                #subGraphs.append([mxCellId,childEntryCellId,childExitCellId,findNode(nodes,mxCellId),findNode(nodes,childEntryCellId),findNode(nodes,childExitCellId)])
                subGraphs.append([mxCellId,childEntryCellId,childExitCellId])
                print('HAS CHILD GRAPH->END PARSING')
            else :
                print('HAS NO CHILD')

            

    #adjacencyMatrix = createMatrix(len(nodes),len(nodes))
    #print(adjacencyMatrix)
    for mxCell in xmlRoot.findall('root/mxCell'):
        mxCellId = mxCell.get('id')
        if mxCell.get('edge') == "1":
            sourceCellId = cellIdPathPrefix + mxCell.get('source')
            targetCellId = cellIdPathPrefix + mxCell.get('target')
            #each recursive loop has its own nodes array because of this edges can not use node index for source and target
            #after all parse operation completed we should convert it all cellIds (source and target) to node index (source and target)
            edges.append([sourceCellId,targetCellId])

    return nodes, edges, entryCellId, exitCellId, subGraphs

def convertToUndirectedGraph(nodes,edges,nodeFeatures,edgeFeatures):
    #in ve out node'ların Id'leri aynı olacağı için sistem çalımayacak findNode işlevsiz kalıyor
    raise NotImplementedError("in ve out node'ların Id'leri aynı olacağı için sistem çalımayacak findNode işlevsiz kalıyor")
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
        undirectedEdges.append([nodes[i],nodes[i+len(nodes)]])
        undirectedEdges.append([nodes[i+len(nodes)],nodes[i]])
        #undirectedEdges.append([i,i+len(nodes)])
        #undirectedEdges.append([i+len(nodes),i])

        #there is no feature availeable for in - out connection
        undirectedEdgeFeatures.append([0]*len(edgeFeatures[0]))
        undirectedEdgeFeatures.append([0]*len(edgeFeatures[0]))

    for edgeIndex in range(len(edges)):
        #out node
        sourceCellId = edges[edgeIndex][0]
        #in node
        targetCellId = edges[edgeIndex][1]
        sourceNodeIndex = findNode(nodes,sourceCellId)
        targetNodeIndex = findNode(nodes,targetCellId)
        #no need to add (to prevent duplicate edges) self loop its already added on edge definition
        if sourceId != targetId:
            undirectedEdges.append([sourceCellId,nodes[targetNodeIndex+len(nodes)]])
            #undirectedEdges.append([sourceId,targetId+len(nodes)])
            #undirectedEdges.append([targetId+len(nodes),sourceId])
            undirectedEdgeFeatures.append(edgeFeatures[edgeIndex])
            undirectedEdgeFeatures.append(edgeFeatures[edgeIndex])

    return undirectedNodes,undirectedEdges,undirectedNodeFeatures,undirectedEdgeFeatures

def writeToDisk(filenamePrefix,nodes,edges,nodeFeatures,edgeFeatures,generateEdgeSymmetry,cells,tab_to_eol,add_info_firstline,embeddings,add_node_labels,duplicate_node_features):
    print('GENERATING OUTPUT FILES')
    #nodeMappingsFileName = os.path.join("output",filenamePrefix + "_node_mappings.txt")
    #nodesFileName = os.path.join("output",filenamePrefix + "_nodes.txt")
    #edgesFilename = os.path.join("output",filenamePrefix + "_edges.txt")
    nodeMappingsFileName = "./output/"+filenamePrefix + "_node_mappings.txt"
    nodesFileName = "./output/"+filenamePrefix + "_nodes.txt"
    edgesFilename = "./output/"+filenamePrefix + "_edges.txt"

    with open(nodesFileName, "w") as f:
        if add_info_firstline == True:
            f.write("{0}\t{1}\n".format(len(nodes),len(nodeFeatures[0])))
        for i in range(len(nodes)):
            f.write("{0}".format(i))
            for j in range(len(nodeFeatures[0])):
                for d in range(duplicate_node_features):
                    f.write("\t{0}".format(nodeFeatures[i][j]))
            if add_node_labels == True:
                key = nodeFeatures[i][0]
                print("writing label of node "+ str(key) +" label is " + embeddings[str(key)][0] )
                f.write("\t{0}".format( embeddings[str(key)][0] ))
            f.write("\n")
    
    table = []
    for i in range(len(nodes)):
        table.append([i,nodes[i],cells[nodes[i]]])

    with open(nodeMappingsFileName, "w") as f:
        f.write(tabulate(table, headers=["NodeId","MxeCellId", "MxeCellName"]))
        

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
        if add_info_firstline == True:
            f.write("{0}\t{1}\n".format(len(edges_clone),len(edgeFeatures_clone[0])))
        for i in range(len(edges_clone)):
            f.write("{0}\t{1}".format(findNode(nodes, edges_clone[i][0]), findNode(nodes, edges_clone[i][1])))
            for j in range(len(edgeFeatures_clone[0])):
                f.write("\t{0}".format(edgeFeatures_clone[i][j]))
            f.write("\n")


def removeSubgraphNodes(nodes,subGraphs):
    refinedNodes = nodes.copy()
    for nodeCellId in nodes:
        for subGraph in subGraphs:
            if nodeCellId == subGraph[0] or nodeCellId == subGraph[1] or nodeCellId == subGraph[2]:
                refinedNodes.remove(nodeCellId)
                break
    return refinedNodes

def removeSubgraphEdges(edges,subGraphs):
    refinedEdges = []
    for edge in edges:
        containsSubGraphNode = False
        for subGraph in subGraphs:
            #if the edge source or target node is the subGraph group, entry or exit node then ignore that edge
            if edge[0] == subGraph[0] or edge[0] == subGraph[1] or edge[0] == subGraph[2] or edge[1] == subGraph[0] or edge[1] == subGraph[1] or edge[1] == subGraph[2]:
                print("Removing edge for subGraph:" + str(subGraph[0]) + " edge:" + str(edge))
                containsSubGraphNode = True
                break
        if containsSubGraphNode == False:
            refinedEdges.append(edge)
    return refinedEdges

def hasSubGraph(subGraphs,cellId):
    for sg in subGraphs:
        if sg[0] == cellId:
            return True
    return False


def reOrganizeSubGraphs(nodes, edges, subGraphs):
    
    reOrganizedEdges = removeSubgraphEdges(edges,subGraphs)
    if len(reOrganizedEdges) == len(edges):
        #nothing to re-organize
        return nodes, edges, False


    print("pre clean completed for subgraph")

    #print("reOrganizedNodes: " +str(len(reOrganizedNodes)) + " nodelen:"+str(len(nodes)))
    #print(reOrganizedNodes)
    #print("reOrganizedEdges: " +str(len(reOrganizedEdges)) + " edgelen:"+str(len(edges)))
    #print(reOrganizedEdges)
    #exit(1)
    for g in subGraphs:
        groupNodeCellId = g[0]
        subGraphEntryCellId = g[1]
        subGraphExitCellId = g[2]

        #groupNodeId = g[3] 
        #subGraphEntryNodeId = g[4]
        #subGraphExitNodeId = g[5] 
        
        #entry of the subgraph
        sourceEdges = findEdgesByTargetCell(edges,groupNodeCellId,False)
        targetEdges = findEdgesBySourceCell(edges,subGraphEntryCellId,False)
        print("for target " + groupNodeCellId + " and the source " + subGraphEntryCellId + " new edges will be generating")
        print("sourceEdges " + str(sourceEdges) + " targetEdges" + str(targetEdges))
        for sourceEdge in sourceEdges:
            for targetEdge in targetEdges:
                #find nodes from new reOrganized nodes
                #sourceEdge[0] is the source node
                #targetEdge[1] is the target node
                #reOrganizedSourceNodeId = findNode(reOrganizedNodes, nodes[sourceEdge[0]])
                #reOrganizedTargetNodeId = findNode(reOrganizedNodes, nodes[targetEdge[1]])
                #reOrganizedEdges.append([reOrganizedSourceNodeId,reOrganizedTargetNodeId])
                print("new edge added : " + str([sourceEdge[0],targetEdge[1]]))
                reOrganizedEdges.append([sourceEdge[0],targetEdge[1]])
        
        #exit of the subgraph
        sourceEdges = findEdgesByTargetCell(edges,subGraphExitCellId,False)
        targetEdges = findEdgesBySourceCell(edges,groupNodeCellId,False)
        for sourceEdge in sourceEdges:
            for targetEdge in targetEdges:
                #find nodes from new reOrganized nodes
                #sourceEdge[0] is the source node
                #targetEdge[1] is the target node
                #reOrganizedSourceNodeId = findNode(reOrganizedNodes, nodes[sourceEdge[0]])
                #reOrganizedTargetNodeId = findNode(reOrganizedNodes, nodes[targetEdge[1]])
                #reOrganizedEdges.append([reOrganizedSourceNodeId,reOrganizedTargetNodeId])
                reOrganizedEdges.append([sourceEdge[0],targetEdge[1]])

        
        if hasSelfLoopEdge(edges,groupNodeCellId) == True:
            sourceEdges = findEdgesByTargetCell(edges,subGraphExitCellId,False)
            targetEdges = findEdgesBySourceCell(edges,subGraphEntryCellId,False)
            for sourceEdge in sourceEdges:
                for targetEdge in targetEdges:
                    #find nodes from new reOrganized nodes
                    #sourceEdge[0] is the source node
                    #targetEdge[1] is the target node
                    #reOrganizedSourceNodeId = findNode(reOrganizedNodes, nodes[sourceEdge[0]])
                    #reOrganizedTargetNodeId = findNode(reOrganizedNodes, nodes[targetEdge[1]])
                    #reOrganizedEdges.append([reOrganizedSourceNodeId,reOrganizedTargetNodeId])
                    reOrganizedEdges.append([sourceEdge[0],targetEdge[1]])

    reOrganizedNodes = removeSubgraphNodes(nodes,subGraphs)
    return reOrganizedNodes,reOrganizedEdges, True


def loadEmbeddings(embeddingsFilename):
    nodeFeatureEmbeddings = dict()
    with open(embeddingsFilename) as f:
        for i, line in enumerate(f.readlines()):
            embeddings = re.sub("\n","",  line).split(",")      
            for i in range(len(embeddings)):
                embeddings[i] = embeddings[i].strip()
            print("embeddings key:'"+embeddings[0]+"' values:"+str(embeddings[1:]));      
            nodeFeatureEmbeddings[embeddings[0]] = embeddings[1:]
    return nodeFeatureEmbeddings

def findEmbedding(embeddings,text):
    if text == "[" or text == "]":
        return 0
    for e in embeddings:
        for word in embeddings[e]:
            if re.search(word, text, re.IGNORECASE):
                #print("embedding found for text:'"+text+"' val:'"+str(e)+"'")
                return e    
    print("embedding NOT FOUND for text:'"+text+"'")
    return 0


def main():
    args = parse_args()
    tree = ET.parse(args.input)
    root = tree.getroot()

    cells = dict()
    nodes, edges ,_ ,_ , subGraphs = parseGraph(root,"",cells)
    #edges = convertEdgesCellIdToNodeIndex(edges,nodes)

    #print("BEFORE REORGANIZE")
    #print("nodes")
    #print(nodes)
    #print("edges")
    #print(edges)    
    #print("subGraphs")
    #print(subGraphs)

    continueReOrganization = True
    while continueReOrganization == True:
        nodes, edges, continueReOrganization = reOrganizeSubGraphs(nodes,edges,subGraphs)
        
    #print("AFTER REORGANIZE")
    #print("nodes")
    #print(nodes)
    #print("edges")
    #print(edges)
    nodeFeatureEmbeddings = loadEmbeddings(args.embeddings)
    print("nodeFeatureEmbeddings")
    print(tabulate(nodeFeatureEmbeddings))


    nodeFeatures = []
    for i in range(len(nodes)):
        nodeFeatures.append([])
        for j in range(int(args.number_of_node_features)):
            val = 0
            if j == 0:
                #print("searching embedding for:"+cells[nodes[i]])
                val = findEmbedding(nodeFeatureEmbeddings,cells[nodes[i]])
            nodeFeatures[i].append(val)
    
    edgeFeatures = []
    for i in range(len(edges)):
        edgeFeatures.append([])
        for j in range(int(args.number_of_edge_features)):
            edgeFeatures[i].append(0)
    

    if eval(args.as_undirected) == True:
        nodes,edges,nodeFeatures,edgeFeatures = convertToUndirectedGraph(nodes,edges,nodeFeatures,edgeFeatures)
    
    generateEdgeSymmetry = (eval(args.generate_edge_symmetry)==True) & (eval(args.as_undirected)==False)
    print("createAdjacencyMatrix")
    adjacencyMatrix = createAdjacencyMatrix(edges,nodes,generateEdgeSymmetry)
    
    
    #print("nodes")
    #print(nodes)
    #print("nodeFeatures")
    #print(nodeFeatures)
    #print("edges")
    #print(edges)
    #print("edgeFeatures")
    #print(edgeFeatures)
    #print("adjacencyMatrix")
    #print(adjacencyMatrix)
    print(args.tab_to_eol)
    writeToDisk(args.output,nodes,edges,nodeFeatures,edgeFeatures,generateEdgeSymmetry,cells,args.tab_to_eol,eval(args.add_info_firstline),nodeFeatureEmbeddings,eval(args.add_node_labels),int(args.duplicate_node_features))



    #output.write("{0}\t{1}".format(a, am))
    


if __name__ == "__main__":
    main()