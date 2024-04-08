import random as rnd
import torch as th

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

    