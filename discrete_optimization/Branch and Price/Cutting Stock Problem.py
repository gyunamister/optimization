import cplex
import math
import pandas as pd
import numpy as np

def set_master(solver, demand, master_constraint_val,  cost):
	solver.objective.set_sense(solver.objective.sense.minimize)
	variables_obj = ["x_{}".format(i+1) for i in range(len(demand))]
	coefficients = [cost] * len(demand)
	solver.variables.add(obj=coefficients, names=variables_obj)

	for index, val in enumerate(demand):
		variables_constraint = ["x_{}".format(i+1) for i in range(len(demand))]
		constraint = [cplex.SparsePair(ind=variables_constraint, val=master_constraint_val[index])]
		solver.linear_constraints.add(lin_expr=constraint, senses='G', rhs=[val])

	return solver

def set_subproblem(solver, dual_values, width, l):
	solver.objective.set_sense(solver.objective.sense.maximize)
	variables_obj = ["a_{}".format(i+1) for i in range(len(dual_values))]
	solver.variables.add(obj=dual_values, names=variables_obj, types = [solver.variables.type.integer]*len(variables_obj))

	coefficients_constraint = l
	variables_constraint = ["a_{}".format(i+1) for i in range(len(dual_values))]
	constraint = [cplex.SparsePair(ind=variables_constraint, val=coefficients_constraint)]
	solver.linear_constraints.add(lin_expr=constraint, senses='L', rhs=[width])

	return solver

def update_master(solver, demand, master_constraint_val, cost, i):
	variables_obj = ["x_{}".format(i)]
	coefficients = [cost]
	solver.variables.add(obj=coefficients, names=variables_obj)

	solver.linear_constraints.delete()

	for index, val in enumerate(demand):
		variables_constraint = ["x_{}".format(i+1) for i in range(i)]
		constraint = [cplex.SparsePair(ind=variables_constraint, val=master_constraint_val[index])]
		solver.linear_constraints.add(lin_expr=constraint, senses='G', rhs=[val])
	return solver

def solve_stock_problem(master_solver, master_constraint_val, demand, cost, width, dual_values,subsolutions):
	num_iter = 1
	while True:
		sub_solver = cplex.Cplex()
		sub_solver = set_subproblem(sub_solver, dual_values, width,l)
		sub_solver.solve()
		sub_solution = sub_solver.solution.get_values()

		sub_solution_obj_val = sub_solver.solution.get_objective_value()
		print("sub solution: {}".format(sub_solver.solution.get_values()))
		print("sub objective value = {}".format(sub_solver.solution.get_objective_value()))
		if cost - sub_solution_obj_val >= 0:
			break
		subsolutions.append(sub_solution)
		for i, val in enumerate(sub_solution):
			master_constraint_val[i].append(val)
		master_solver = update_master(master_solver, demand, master_constraint_val, cost, len(demand)+num_iter)
		master_solver.solve()
		dual_values = master_solver.solution.get_dual_values()
		print("solution: {}".format(master_solver.solution.get_values()))
		print("objective value = {}".format(master_solver.solution.get_objective_value()))
		num_iter+=1
	return master_solver, master_constraint_val, dual_values,subsolutions
if __name__ == '__main__':
	demand = [100,500,495,250]
	l = [45,36,30,15]
	width = 100
	cost = 2000
	master_constraint_val=[[2,0,0,0], [0,2,0,0], [0,0,3,0],[0,0,0,6]]
	subsolutions = []
	master_solver = cplex.Cplex()
	master_solver = set_master(master_solver, demand, master_constraint_val, cost)
	master_solver.solve()

	#while cost - sub_problem_obj_val < 0:
	dual_values = master_solver.solution.get_dual_values()
	master_solver, master_constraint_val, dual_values, subsolutions = solve_stock_problem(master_solver, master_constraint_val, demand, cost, width, dual_values, subsolutions)
	print("NEW L")
	width = 50
	cost = 1100
	master_solver, master_constraint_val, dual_values,subsolutions = solve_stock_problem(master_solver, master_constraint_val, demand, cost, width, dual_values,subsolutions)
	print(subsolutions)
	master_solver.write("/Users/GYUNAM/Documents/optimization/optimization/HW3/ip4.lp")
	"""
	cost = 1
	width = 20
	master_constraint_val
	"""
	#master_solver, master_constraint_val, dual_values = solve_stock_problem(master_solver, master_constraint_val, demand, cost, width, dual_values)





