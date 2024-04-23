import torch as th
from python_tsp.exact import solve_tsp_dynamic_programming

def distance_matrix(cities: th.Tensor) -> th.Tensor:
    """
    Computes the distance matrix between cities.
    
    Args:
        cities (th.Tensor): A tensor containing the cities.
    
    Returns:
        th.Tensor: A tensor containing the distance matrix.
    """
    
    n_cities = cities.shape[0]
    distance_matrix = th.zeros(n_cities, n_cities)
    for i in range(n_cities):
        for j in range(i, n_cities):
            distance_matrix[i][j] = th.norm(cities[i] - cities[j])
            distance_matrix[j][i] = distance_matrix[i][j]

    return distance_matrix

def solve_tsp(cities: th.Tensor) -> tuple:
    """
    Solves the TSP problem for a given set of cities.
    
    Args:
        cities (th.Tensor): A tensor containing the cities.
    
    Returns:
        tuple: A tuple containing the permutation of the cities and the total distance.
    """
    
    dist_mtx = distance_matrix(cities)
    permutation, distance = solve_tsp_dynamic_programming(dist_mtx)
    return permutation, distance

if __name__ == '__main__':

    max_nodes_per_graph = 10
    instance = 0
    cities = th.load(f"../training/tsp/size_{max_nodes_per_graph}/instance_{instance}.pt") 
    permutation, distance = solve_tsp(cities)
    print("Permutation: ", permutation)
    print("Distance: ", distance)