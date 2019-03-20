import math
import time
import heapq 
from random import uniform
from random import sample
from random import choice


class Vertex:
    def __init__(self, coordinates):
        self.coordinates = coordinates
        self.parent = None
        self.key = math.inf
        
    def __lt__(self, other):
        return self.key < other.key
        
class Edge:
    def __init__(self, u, v, weight):
        self.u = u
        self.v = v
        self.weight = weight

class CompleteGraph:
    def __init__(self, list_of_vertices):
        self.vertices = list_of_vertices
        self.size = len(self.vertices)
        self.matrix = [[0 for column in range(self.size)]
                    for row in range(self.size)]
        self.edges = []

    def generate_matrix(self):
        # compute the Euclidean distance between each
        # pair of nodes and enter it into the corresponding
        # entry in the matrix
        for i in range(len(self.vertices)):
          for j in range(i, len(self.vertices)):
            v_i = self.vertices[i].coordinates
            v_j = self.vertices[j].coordinates
            dist = compute_euclidean_dist(v_i, v_j)
            self.matrix[i][j] = dist
            
            # if the edge isn't from the same node to itself, record it
            if v_i != v_j:
                edge = Edge(self.vertices[i], self.vertices[j], dist)
                self.edges.append(edge)
        return self.matrix

def compute_euclidean_dist(v_i,v_j):
    '''
        Computes the Euclidean distance between two coordinates
        pairs denoted by v_i(x_i, y_i) and v_j(x_j, y_j)
    '''
    x_i = v_i[0]
    x_j = v_j[0]
    y_i = v_i[1]
    y_j = v_j[1]

    w = math.sqrt(((x_i - x_j)**2) + ((y_i - y_j)**2))
    return w

def generate_random_nodes(n):
      ''' generates n random nodes where each node v_i is a coordinate tuple 
          (x_i, y_i), with x and y being drawn randomly from the closed 
          interval [0, 1]
      '''
      list_of_vertices = []
      for n in range(n):
          random_node = uniform(0,1), uniform(0,1)
          vertex = Vertex(random_node)
          list_of_vertices.append(vertex)
      return list_of_vertices

def make_set(x):
  x.parent = x
  x.rank = 0

def find(x):
  '''
      find_set operation with path compression
  '''
  if x.parent != x:
    x.parent = find(x.parent)
  return x.parent

def union(x, y):
  '''
      Union operation using the rank heuristic
  '''
  x_root = find(x)
  y_root = find(y)
  if x_root.rank > y_root.rank:
      y_root.parent = x_root
  else:
      x_root.parent = y_root
      
  if x_root.rank == y_root.rank:
      y_root.rank = y_root.rank + 1
      
def sort_edges(edges):
    '''
        Takes in a list of edges and sorts them in
        non-decreasing order of edge weight
    '''
    edges.sort(key=lambda x: x.weight, reverse=False)
    return edges

def kruskal_mst(g):
    ''' Given a graph g, runs Kruskal's algorithm on it
        and returns the set of edges in the MST, A
    '''
    vertices = g.vertices
    A = set()
    for vertex in vertices:
        make_set(vertex)    
    sorted_edges = sort_edges(g.edges)
    for edge in sorted_edges:
        if find(edge.u) != find(edge.v):
            A.add(edge)
            union(edge.u, edge.v)
    return A

def weighted_sum_of_edges(mst_edges):
    ''' returns the weighted sum of edges of a mst's edges'''
    count = 0
    for edge in mst_edges:
        count = count + edge.weight
    return count
    
class RandomConnectedGraph:
    def __init__(self, list_of_vertices):
        self.vertices = list_of_vertices
        self.size = len(self.vertices)
        self.graph = dict()
        self.edges = []
        
        
    def generate_graph(self):
        # initialize all vertices in this graph
        for vertex in self.vertices:
            make_set(vertex)
        
        while self.is_connected() == False:
            # choose two nodes u and v randomly
            u, v = sample(self.vertices, 2)          
            # Create an edge between u and v (u,v) with weight = dist
            if u.coordinates != v.coordinates:    
                # compute the Euclidean distance between u and v
                dist = compute_euclidean_dist(u.coordinates, v.coordinates)
                # print("The Euclidean dist between u and v is", dist)
                edge = Edge(u, v, dist)
                union(edge.u, edge.v)
                self.edges.append(edge)
                                               
        self.remove_redundant_edges()
        self.generate_dict()

        
    def remove_redundant_edges(self):
        '''
            Removes all bidirectional/redundant edges
            to ensure that the graph is undirected
        '''
        self.vertices.clear()
        # helper structures
        nodes = set()
        d = dict()
        
        unique_pairs = []
        for edge in self.edges:
            x = edge.u.coordinates
            y = edge.v.coordinates
            s = set()
            s.add(x)
            s.add(y)
            if s not in unique_pairs:
                unique_pairs.append(s)
        self.edges.clear()
    
        for pair in unique_pairs:
            pair = list(pair)
            u = pair[0]
            v = pair[1]
            # ensure that we make only one vertex object correspoding
            # to a single coordinate tuple
            # Do the check for u
            if u not in d.keys():
                d[u] = Vertex(u)
                u = d[u]
                nodes.add(u.coordinates)
                self.vertices.append(u)
                self.graph[u] = []
            else:
                u = d[u]
                if u.coordinates not in nodes:
                    nodes.add(u.coordinates)
                    self.vertices.append(u)
                    self.graph[u] = []
            # Do the check for v
            if v not in d.keys():
                d[v] = Vertex(v)
                v = d[v]
                nodes.add(v.coordinates)
                self.vertices.append(v)
                self.graph[v] = []
            else:
                v = d[v]
                if v.coordinates not in nodes:
                    nodes.add(v.coordinates)
                    self.vertices.append(v)
                    self.graph[v] = []
            # compute the Euclidean distance between u and v
            dist = compute_euclidean_dist(u.coordinates, v.coordinates)
            edge = Edge(u, v, dist)
            self.edges.append(edge)
        nodes.clear()
        d.clear()
                
                    
    def generate_dict(self):
        ''' Helper method to populate dictionary that
            represents the graph node -> list of neighbours
        '''
        for edge in self.edges:
            x = edge.u
            y = edge.v
               
            if x in self.graph.keys():
                self.graph[x].append(y)
            if y in self.graph.keys():
                self.graph[y].append(x)

       
    def is_connected(self):
        representative = find(self.vertices[0])
        for vertex in self.vertices:
            if find(vertex) != representative:
                return False
        return True
   

def prim_mst(g):
    '''
        Given a graph g, executes Prim's algorithm
    '''
    mst = set()
    # initialization
    for vertex in g.vertices:
        vertex.key = math.inf
        vertex.parent = None
        
    # choose an arbitrary root and set its key to zero
    r = choice(g.vertices)
    r.key = 0
    mst.add(r)
    
    Q = g.vertices
    heapq.heapify(Q)
    while len(Q) != 0:
        heapq.heapify(Q)
        u = heapq.heappop(Q)
        mst.add(u)
        # vertices adjacent to u
        neighbours = g.graph[u]
        for v in neighbours:
            weight_uv = compute_euclidean_dist(u.coordinates, v.coordinates)
            if v in Q and weight_uv < v.key:
                v.parent = u
                v.key = weight_uv
    return mst
        

def p_random_complete_graphs(p, n):
    # randomly generate p complete graphs
    # each graph should have n vertices
    weighted_sums_list = []
    for i in range(p):
        list_of_vertices = generate_random_nodes(n)
        g = CompleteGraph(list_of_vertices)
        g.generate_matrix()
        # print("Applying Kruskal's algorithm to graph number:", i+1)
        mst_edges = kruskal_mst(g)
        L = weighted_sum_of_edges(mst_edges)
        weighted_sums_list.append(L)
        
    sum_of_mst_costs = sum(weighted_sums_list)
    average_weighted_sum = (sum_of_mst_costs / p)
    return average_weighted_sum


def p_random_connected_kruskal(q, n):
    # generate p randomly connected graphs, each with n vertices
    running_times = []
    for i in range(q):
        list_of_vertices = generate_random_nodes(n)
        random_g = RandomConnectedGraph(list_of_vertices)
        random_g.generate_graph()
        # start timing before calling kruskal
        start_time = time.clock()
        prim_mst(random_g)
        finish_time = time.clock() - start_time
        running_times.append(finish_time)
    average_running_time = sum(running_times) / len(running_times)
    return average_running_time

def p_random_connected_prim(q, n):
    # generate p randomly connected graphs, each with n vertices
    running_times = []
    for i in range(q):
        list_of_vertices = generate_random_nodes(n)
        random_g = RandomConnectedGraph(list_of_vertices)
        random_g.generate_graph()
        # start timing before calling prim
        start_time = time.clock()
        kruskal_mst(random_g)
        finish_time = time.clock() - start_time
        running_times.append(finish_time)
    average_running_time = sum(running_times) / len(running_times)
    return average_running_time
        
 
print("Q1(i): Average weighted sums of MSTs of p=50 randomly generated complete graphs")
print("The MSTs are generated using Kruskal's algorithm, and n = 100, 500, 1000, 5000 respectively")                        
print("n = 100", ",", "average weighted sum =", p_random_complete_graphs(50, 100))
print("n = 500", ",", "average weighted sum =", p_random_complete_graphs(50, 500))
print("n = 1000", ",", "average weighted sum =", p_random_complete_graphs(50, 1000))
# print("n = 5000", ",", "average weighted sum =", p_random_complete_graphs(50, 5000))

print("Q1(iii): Average running times for finding an MST in q=50 randomly generated connected graphs")
print("Using Kruskal's:")
print("n = 100", ",", "average running time =", p_random_connected_kruskal(50, 100), "sec")
print("n = 500", ",", "average running time =", p_random_connected_kruskal(50, 500), "sec")
print("n = 1000", ",", "average running time =", p_random_connected_kruskal(50, 1000), "sec")
# print("n = 5000", ",", "average running time =", p_random_connected_kruskal(50, 5000), "sec")

print("Using Prim's:")
print("n = 100", ",", "average running time =", p_random_connected_prim(50, 100), "sec")
print("n = 500", ",", "average running time =", p_random_connected_prim(50, 500), "sec")
print("n = 1000", ",", "average running time =", p_random_connected_prim(50, 1000), "sec")
print("n = 5000", ",", "average running time =", p_random_connected_prim(50, 5000), "sec")
