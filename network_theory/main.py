#using virtualenv-optimal_resource

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

data = np.genfromtxt('./network_theory_hw2_graph1.csv', delimiter=',', skip_header=1)
DG=nx.DiGraph()
DG.add_weighted_edges_from(data)

print(DG.nodes(), DG.edges())

def floyd_warshall(DG):
    """Return dictionaries distance and next_v.

    distance[u][v] is the shortest distance from vertex u to v.
    next_v[u][v] is the next vertex after vertex v in the shortest path from u
    to v. It is None if there is no path between them. next_v[u][u] should be
    None for all u.

    g is a Graph object which can have negative edge weights.
    """
    distance = {v:dict.fromkeys(g, float('inf')) for v in DG.nodes()}
    print(distance)
    """
    next_v = {v:dict.fromkeys(g, None) for v in g}

    for v in g:
        for n in v.get_neighbours():
            distance[v][n] = v.get_weight(n)
            next_v[v][n] = n

    for v in g:
         distance[v][v] = 0
         next_v[v][v] = None

    for p in g:
        for v in g:
            for w in g:
                if distance[v][w] > distance[v][p] + distance[p][w]:
                    distance[v][w] = distance[v][p] + distance[p][w]
                    next_v[v][w] = next_v[v][p]

    return distance, next_v
    """