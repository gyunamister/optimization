#!python3

'''
Code for 2018 Network Theory HW#2-2
'''
#using virtualenv-optimal_resource
import networkx as nx
import numpy as np

def read_csv(directory):
    data = np.genfromtxt(directory, delimiter=',', skip_header=1)
    DG=nx.DiGraph()
    DG.add_weighted_edges_from(data)
    return DG

def floyd_warshall(DG):
    #initial setting - distance and pred
    distance = {i:dict.fromkeys(DG.nodes(), float('inf')) for i in DG.nodes()}
    pred = {i:dict.fromkeys(DG.nodes(), 0) for i in DG.nodes()}

    #set loop distance as zero
    for i in DG.nodes():
         distance[i][i] = 0

    #set actual distance
    for i in DG.nodes():
        for j in DG.neighbors(i):
            distance[i][j] = DG[i][j]['weight']
            pred[i][j] = i

    #dynamic programming
    for k in DG.nodes():
        for i in DG.nodes():
            for j in DG.nodes():
                if distance[i][j] > distance[i][k] + distance[k][j]:
                    distance[i][j] = distance[i][k] + distance[k][j]
                    pred[i][j] = pred[k][j]

    return distance, pred

def show_result(DG, distance, pred):
    result = dict()
    negatice_cycle_list = list()
    #to detect negative cycle, first calculate nC
    weight_list=list()
    for i in DG.nodes():
        for j in DG.neighbors(i):
            weight_list.append(DG[i][j]['weight'])
    nC=-len(DG.nodes)*max(weight_list)

    #Print the result
    for i in DG.nodes():
        for j in DG.nodes():
            if i==j:
                #to detect negative cycle
                if distance[i][j] < 0:
                    negatice_cycle_list += [i,j]
                    #print("{}->{} < 0".format(i,j))
                continue
            #to detect negative cycle
            if distance[i][j] < nC:
                negatice_cycle_list += [i,j]
                #print("{}->{} < nC".format(i,j))
            result[(i,j)]={"distance": int(distance[i][j]), "pred": int(pred[i][j])}
    if len(negatice_cycle_list) > 0:
        print("negative cycle: {}".format(list(set(negatice_cycle_list))))
    return result

def solve(filename):
    DG = read_csv(filename)
    distance, pred = floyd_warshall(DG)
    solution = show_result(DG, distance, pred)
    return solution

def main():
    filename = "network_theory_hw2_graph1.csv"
    print(solve(filename))

if __name__ == '__main__':
    main()
