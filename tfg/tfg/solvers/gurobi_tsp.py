import gurobipy as gp
from gurobipy import GRB
import torch as th

# Function to compute the distance matrix between cities
def compute_distance_matrix(cities: th.Tensor) -> th.Tensor:
    num_cities = cities.shape[0]
    dist_matrix = th.zeros((num_cities, num_cities))
    for i in range(num_cities):
        for j in range(num_cities):
            if i != j:
                dist_matrix[i, j] = th.norm(cities[i] - cities[j])
    return dist_matrix

# Function to solve the TSP using Gurobi
def solve_tsp(cities):
    num_cities = cities.shape[0]
    dist_matrix = compute_distance_matrix(cities)

    # Convert the distance matrix to a format that Gurobi can handle
    dist_matrix_np = dist_matrix.numpy()

    # Create a new model
    model = gp.Model("TSP")

    # Suppress Gurobi output
    model.Params.OutputFlag = 0

    # Create variables: x[i,j] = 1 if the path from city i to city j is in the optimal solution
    x = model.addVars(num_cities, num_cities, vtype=GRB.BINARY, name="x")

    # Create variables u for the sub-tour elimination constraints (to eliminate cycles)
    u = model.addVars(num_cities, vtype=GRB.CONTINUOUS, name="u")

    # Set the objective function
    model.setObjective(gp.quicksum(dist_matrix_np[i][j] * x[i,j] for i in range(num_cities) for j in range(num_cities)), GRB.MINIMIZE)

    # Add constraints
    # Each city must have exactly one outgoing path
    model.addConstrs(gp.quicksum(x[i,j] for j in range(num_cities) if i != j) == 1 for i in range(num_cities))

    # Each city must have exactly one incoming path
    model.addConstrs(gp.quicksum(x[i,j] for i in range(num_cities) if i != j) == 1 for j in range(num_cities))

    # Eliminate sub-tours (MTZ constraints)
    for i in range(1, num_cities):
        for j in range(1, num_cities):
            if i != j:
                model.addConstr(u[i] - u[j] + num_cities * x[i,j] <= num_cities - 1)
    
    # Optimize the model
    model.optimize()

    # If an optimal solution is found
    if model.status == GRB.OPTIMAL:
        solution = model.getAttr('x', x)
        tour = []
        current_city = 0
        total_distance = 0.0
        while len(tour) < num_cities:
            tour.append(current_city)
            next_city = [j for i, j in solution.keys() if i == current_city and solution[i, j] > 0.5][0]
            total_distance += dist_matrix_np[current_city, next_city]
            current_city = next_city
        tour.append(tour[0])  # Return to the starting city
        total_distance += dist_matrix_np[current_city, tour[0]]

        # Transform tou to a tensor
        tour = th.tensor(tour)
        return tour, total_distance
    else:
        return None, None

# Main function to run the TSP solver
if __name__ == "__main__":
    max_nodes_per_graph = 20
    instance = 0
    cities = th.load(f"../training/tsp/size_{max_nodes_per_graph}/instance_{instance}.pt")
    best_tour, best_distance = solve_tsp(cities)
    print(f"Best tour: {best_tour}")
    print(f"Total distance: {best_distance}")
