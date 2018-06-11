import cplex
import math
import numpy as np
import pandas as pd



def produce_dist_matrix(_input):
    _input_len = len(_input)
    dist_matrix = np.zeros((_input_len,_input_len))
    for i in range(_input_len):
        for j in range(i+1,_input_len,1):
            vector_i = np.array([_input.loc[i,'XCOORD'], _input.loc[i, 'YCOORD']])
            vector_j = np.array([_input.loc[j,'XCOORD'], _input.loc[j, 'YCOORD']])
            dist = math.trunc(np.linalg.norm(vector_i-vector_j))
            dist_matrix[i][j] = dist
            dist_matrix[j][i] = dist

    return dist_matrix

def set_solver(solver, dist_matrix, size):
    node_size = size
    solver.objective.set_sense(solver.objective.sense.minimize)
    node_size = size

    variables_y = ["y_{}_{}_{}".format(k+1, i+1,j+1) for i in range(node_size) for j in range(node_size) for k in range(node_size)]
    #print(variables_y)
    coefficients = []
    for i, row in enumerate(dist_matrix):
        row = list(row)
        for _element in row:
            coefficients += [_element] * node_size

    solver.variables.add(obj=coefficients, names=variables_y, types=[solver.variables.type.binary]*len(variables_y))

    constraint_1 = []
    for k in range(node_size):
        variables_constraint = ["y_{}_{}_{}".format(k+1, i+1,j+1) for i in range(node_size) for j in range(node_size)]
        constraint_1.append(cplex.SparsePair(ind=variables_constraint, val=[1]*len(variables_constraint)))
    #add constraint #1
    solver.linear_constraints.add(lin_expr=constraint_1, senses=["E"]*len(constraint_1), rhs=[1]*len(constraint_1))

    constraint_2 = []
    for i in range(node_size):
        variables_constraint = ["y_{}_{}_{}".format(k+1, i+1,j+1) for k in range(node_size) for j in range(node_size)]
        constraint_2.append(cplex.SparsePair(ind=variables_constraint, val=[1]*len(variables_constraint)))
    #add constraint #2
    solver.linear_constraints.add(lin_expr=constraint_2, senses=["E"]*len(constraint_2), rhs=[1]*len(constraint_2))

    constraint_3 = []
    for j in range(node_size):
        variables_constraint = ["y_{}_{}_{}".format(k+1, i+1,j+1) for k in range(node_size) for i in range(node_size)]
        constraint_3.append(cplex.SparsePair(ind=variables_constraint, val=[1]*len(variables_constraint)))
    #add constraint #3
    solver.linear_constraints.add(lin_expr=constraint_3, senses=["E"]*len(constraint_3), rhs=[1]*len(constraint_3))

    constraint_4 = []
    for j in range(node_size):
        for k in range(node_size-1):
            variables_constraint = ["y_{}_{}_{}".format(k+1, i+1,j+1) for i in range(node_size)]
            variables_constraint += ["y_{}_{}_{}".format(k+2,j+1,i+1) for i in range(node_size)]
            constraint_4.append(cplex.SparsePair(ind=variables_constraint, val=[1] * node_size + [-1] * node_size))
    #add constraint #4
    solver.linear_constraints.add(lin_expr=constraint_4, senses=["E"]*len(constraint_4), rhs=[0]*len(constraint_4))

    constraint_5 = []
    for j in range(node_size):
        variables_constraint = ["y_{}_{}_{}".format(node_size, i+1,j+1) for i in range(node_size)]
        print(variables_constraint)
        variables_constraint += ["y_{}_{}_{}".format(1,j+1,i+1) for i in range(node_size)]
        constraint_5.append(cplex.SparsePair(ind=variables_constraint, val=[1] * node_size + [-1] * node_size))
    #add constraint #5
    solver.linear_constraints.add(lin_expr=constraint_5, senses=["E"] * len(constraint_5), rhs=[0] * len(constraint_5))
    return solver, variables_y

if __name__ == '__main__':
    _input = pd.read_csv('./input.txt', sep = '\t', index_col = False)
    #calculate distance
    dist_matrix = produce_dist_matrix(_input)
    _input_len = len(_input)
    #initialize solver
    solver = cplex.Cplex()
    #set solver requirements
    solver, variables_y = set_solver(solver, dist_matrix,_input_len)
    solver.solve()
    print("objective value = {}".format(solver.solution.get_objective_value()))
    #find tour sequence
    solution = solver.solution.get_values()
    index_list = np.nonzero(solution)[0]
    pairs = []
    for i in index_list:
        y_chosen=variables_y[i].split('_')
        pair = (int(y_chosen[1]),y_chosen[2])
        pairs.append(pair)
    pairs.sort(key=lambda x: x[0])
    seq = []
    for i in pairs:
        seq.append(i[1])
    print("solution: {}".format(seq))
