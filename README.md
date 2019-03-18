# minimum-spanning-trees

This Python script generates p=50 randomly
generated complete graphs. It then computes the Minimum Spanning Trees (MSTs) of these graphs using
Kruskal’s algorithm for n = 100, 500, 1000, 5000 respectively, where n = the number of nodes
generated for each graph. Of the n random nodes, each node v_i is a coordinate tuple (x_i, y_i), with x and y being drawn randomly from the closed interval [0, 1]. The average weighted sum of these MSTs is then presented.

The script then shows the average running times for finding a MST in q=50 randomly generated
'connected' graphs using Kruskal’s algorithm. It then does the same using Prim's.

Kruskal's is implemented using the Union-Find data structure in the code. Path compression and union-by-rank heuristic are implemented. A binary heap is used to implement Prim's algorithm.
