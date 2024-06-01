from concorde.tsp import TSPSolver
import torch as th

def concorde_solve_tsp(cities: th.Tensor):
    """
    This function is used to solve the TSP problem using the Concorde solver.

    Args:
        cities (th.Tensor): The list of cities.

    Returns:
        list: The optimal tour.
        float: The optimal cost.
    """
    mult = 1000000  # Factor to scale coordinates to integers
    scaled_x = (cities[:, 0].numpy() * mult).astype(int)
    scaled_y = (cities[:, 1].numpy() * mult).astype(int)

    solver = TSPSolver.from_data(scaled_x, scaled_y, norm="EUC_2D", name="TSP")
    solution = solver.solve(verbose=False)
    
    return solution.tour, solution.optimal_value / mult  # Scale the optimal value back to original scale

if __name__ == "__main__":
    max_nodes_per_graph = 10
    instance = 0
    cities = th.load(f"../training/tsp/size_{max_nodes_per_graph}/instance_{instance}.pt")
    permutation, distance = concorde_solve_tsp(cities)
    print("Permutation: ", permutation)
    print("Distance: ", distance)
