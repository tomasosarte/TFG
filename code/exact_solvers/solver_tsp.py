import numpy as np
import torch as th
from python_tsp.exact import solve_tsp_dynamic_programming

def distance_matrix(cities: th.Tensor):
    """Stores the data for the problem."""
    
    n_cities = cities.shape[0]

    # Get matrix of distances
    distance_matrix = th.zeros(n_cities, n_cities)
    for i in range(n_cities):
        for j in range(n_cities):
            if i != j:
                distance_matrix[i][j] = th.norm(cities[i] - cities[j])

    return distance_matrix

def solve_tsp(cities: th.Tensor):
    """Solves the TSP problem."""
    
    distance_matrix = distance_matrix(cities)
    permutation, distance = solve_tsp_dynamic_programming(distance_matrix)
    return permutation, distance

if __name__ == '__main__':

    max_nodes_per_graph = 10
    instance = 0
    cities = th.load(f"../training/tsp/size_{max_nodes_per_graph}/instance_{instance}.pt") 
    permutation, distance = solve_tsp(cities)
    print("Permutation: ", permutation)
    print("Distance: ", distance)