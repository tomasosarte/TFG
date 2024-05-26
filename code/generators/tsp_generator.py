import os
import torch as th
from tqdm import tqdm
import matplotlib.pyplot as plt

class TSPGenerator:
    """
    This class is used to generate TSP instances.
    """

    def generate_instance(self, n_cities: int) -> th.Tensor:
        """
        This method is used to generate a TSP instance. The instance is Tendor cities with their coordinates (x, y), 
        all sitiuated in a 2D space in the region [0, 1] x [0, 1]. 

        Args:
            n_cities (int): The number of cities in the instance.

        Returns:
            Tensor: The list of cities.
        """
        cities = th.rand(n_cities, 2)
        return cities
    
    def generate_batch(self, n_instances: int, n_cities: int) -> list[th.Tensor]:
        """
        This method is used to generate a batch of TSP instances. A batch is a list of instances, 
        each instance is a list of cities with their coordinates (x, y), all sitiuated in a 2D 
        space in the region [0, 1] x [0, 1]. 

        Args:
            n_instances (int): The number of instances in the batch.
            n_cities (int): The number of cities in each instance.

        Returns:
            Tensor: The list of instances.
        """
        return [self.generate_instance(n_cities) for _ in range(n_instances)]
    
    def generate_batch_set(self, n_instances: int, list_n_cities: list[int]) -> list[list[th.Tensor]]:
        """
        This method is used to generate a set of batches of TSP instances. A set is a list of batches, each batch 
        with different city sizes. Each batch is a list of instances, each instance is a list of cities with their 
        coordinates (x, y).

        Args:
            n_instances (int): The number of instances in each batch.
            list_n_cities (list[int]): The list of city sizes for each batch.
        
        Returns:
            Tensor: The list of batches.
        """
        return [self.generate_batch(n_instances, n_cities) for n_cities in list_n_cities]

    def plot_instance(self, cities: th.Tensor):
        """
        Plot an instance of a tsp problem

        Args:
            cities (th.Tensor): Tensor of shape: (n_cities, 2) representingg and instance of tsp problem

        Returns: 
            None
        """
        if cities.is_cuda:
            cities = cities.cpu() 

        # Extract coordinates
        x = cities[:, 0]
        y = cities[:, 1]

        # Create graph
        plt.figure(figsize=(8, 6))
        plt.scatter(x, y, c='blue', marker='o')  # Puedes cambiar el color y el marcador

        # Add title and tags
        plt.title('City')
        plt.xlabel('Coord X')
        plt.ylabel('Coord Y')

        # Show
        plt.show()

    def save_dataset(self, dataset: list[list[th.Tensor]], directory: str, list_of_n_cities: list[int]):
        """
        Saves the dataset to disk.

        Args:
            dataset (list): List of batches of TSP instances.
            directory (str): Directory where the dataset files will be saved.
            list_of_n_cities (list): List of city sizes for each batch.

        Returns:
            None
        """

        # Check if directory exists
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        for i, batch in enumerate(tqdm(dataset, desc="Saving dataset")):
            size = list_of_n_cities[i]
            # Check if instance_size directory exists            
            if not os.path.exists(f"{directory}/size_{size}"):
                os.makedirs(f"{directory}/size_{size}")

            for j, instance in enumerate(tqdm(batch, desc=f'Batch {i+1} - size {size}', leave=False)):
                th.save(instance, f"{directory}/size_{size}/instance_{j}.pt")

if __name__ == "__main__":

    # Create training dataset
    generator = TSPGenerator()

    # Generate a batch set 
    list_of_n_cities = [5, 15, 25, 35, 45, 55, 65, 75, 85, 95]
    n_instances = 10
    dataset = generator.generate_batch_set(n_instances = n_instances, list_n_cities = list_of_n_cities)

    # Save the dataset
    generator.save_dataset(dataset, "../training/tsp", list_of_n_cities)

    # Get inside ../training/tsp(size_10 and open the instance_0.pt file)
    path = "../training/tsp/size_10/instance_0.pt"
    cities = th.load(path)    