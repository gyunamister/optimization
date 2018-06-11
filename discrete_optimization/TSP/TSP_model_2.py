import cplex
import math
import numpy as np
import pandas as pd


def calculate_distance_matrix(data):
    distance_matrix = []
    for i in range(len(data)):
        dummy_vector = []
        for j in range(len(data)):
            if i != j:
                distance = math.trunc(math.sqrt((int(data[i][1]) - int(data[j][1]))**2 + (int(data[i][2]) - int(data[j][2]))**2))
                dummy_vector.append(distance)
        distance_matrix.append(dummy_vector)
    return distance_matrix

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
    solver.objective.set_sense(solver.objective.sense.minimize)
    node_size = size

    variables = ["x_{}_{}".format(i+1,j+1) for i in range(node_size) for j in range(node_size) if i != j]
    coefficients = []
    for i, row in enumerate(dist_matrix):
        row = list(row)
        del row[i]
        coefficients += list(row)
    solver.variables.add(obj=coefficients, names=variables, types=[solver.variables.type.binary]*len(variables))

    #add variable u
    variable_u = ["u_{}".format(i+1) for i in range(node_size)]
    solver.variables.add(names=variable_u)

    out_constraint = []
    for i in range(node_size):
        variables_out_constraint = ["x_{}_{}".format(i+1,j+1) for j in range(node_size) if i != j]
        out_constraint.append(cplex.SparsePair(ind=variables_out_constraint, val=[1]*len(variables_out_constraint)))

    solver.linear_constraints.add(lin_expr=out_constraint, senses=["E"]*len(out_constraint),
                                rhs=[1]*len(out_constraint))

    in_constraint = []
    for j in range(node_size):
        variables_in_constraint = ["x_{}_{}".format(i+1,j+1) for i in range(node_size) if i != j]
        in_constraint.append(cplex.SparsePair(ind=variables_in_constraint, val=[1]*len(variables_in_constraint)))
    solver.linear_constraints.add(lin_expr=in_constraint, senses=["E"]*len(in_constraint),
                                rhs=[1]*len(in_constraint))

    subtour_constraints = []
    for i in range(1, node_size):
        for j in range(1, node_size):
            if i != j:
                variables_subtour_constraints = ["u_{}".format(i+1), "u_{}".format(j+1), "x_{}_{}".format(i+1,j+1)]
                subtour_constraints.append(cplex.SparsePair(ind=variables_subtour_constraints, val=[1, -1, node_size]))
    #add subtour constraints
    solver.linear_constraints.add(lin_expr=subtour_constraints, senses="L"*len(subtour_constraints), rhs=[node_size - 1]*len(subtour_constraints))
    return solver


def identify_sub_tour(solution, length):
    chunks = [solution[i:i + length] for i in range(0, len(solution), length)]
    tour = []
    _current = 0
    tour.append(_current)
    loop_check = True
    while loop_check:
        _index = np.argmax(chunks[_current])
        if _index >= _current:
            _current = _index + 1
        else:
            _current = _index
        if _current in tour:
            loop_check = False
        else:
            tour.append(_current)
    return tour

if __name__ == '__main__':
    _input = pd.read_csv('./input.txt', sep = '\t', index_col = False)
    #calculate distance
    dist_matrix = produce_dist_matrix(_input)
    _input_len = len(_input)
    #initialize solver
    solver = cplex.Cplex()
    #set solver requirements
    solver = set_solver(solver, dist_matrix,_input_len)
    solver.solve()
    x = solver.solution.get_values()
    #find tour sequence
    sub_tour = identify_sub_tour(x, _input_len-1)
    print("solution: {}".format(list(np.array(sub_tour) + 1)))
    print("objective value = {}".format(solver.solution.get_objective_value()))
