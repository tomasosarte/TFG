import torch as th
from torch import tensor

from utils.transition_batch import TransitionBatch
from environments.environment_tsp import EnviornmentTSP
from networks.basic_network import BasicNetwork
from generators.tsp_generator import TSPGenerator
from controllers.ac_controller import ActorCriticController
from runners.runner import Runner
from params import default_params
from learners.reinforce_learner import ReinforceLearner
from experiments.experiment import ACExperiment


def test_transition_batch():
    """
    Test the TransitionBatch class.

    Args:
        None

    Returns:
        None
    """

    def _wrap_transition(action: th.Tensor, state: th.Tensor, next_state: th.Tensor, reward: th.Tensor, done: th.Tensor):
        """
        Wraps the transition in a dictionary.

        Args:
            action (th.Tensor): The action taken.
            state (th.Tensor): The state before the action.
            next_state (th.Tensor): The state after the action.
            reward (th.Tensor): The reward for the action.
            done (th.Tensor): A boolean indicating if the episode is done.

        Returns:
            dict: A dictionary containing the transition.
        """
        return {
            'actions': action,
            'states': state,
            'next_states': next_state,
            'rewards': reward,
            'dones': done,
            'returns': th.zeros(1, dtype=th.float32)
        }
    
    def transition_format() -> dict:
        """
        Returns the format of the transitions: A dictionary of (shape, dtype) entries for each key

        Args:
            None

        Returns:
            (dict) Format transitions
        """
        return {
            'actions': ((1,), th.long),
            'states': ((34,), th.float32),
            'next_states': ((34,), th.float32),
            'rewards': ((1,), th.float32),
            'dones': ((1,), th.bool),
            'returns': ((1,), th.float32)
        }

    def test_add(tb: TransitionBatch, first:int, size:int, wrapped_transition: dict, index:int) -> None:
        # assert tb.first == first, "The first attribute is not correct"
        # assert tb.size == size, "The size of the TransitionBatch is not correct"
        # assert tb.dict['states'][index] == wrapped_transition['states'], "State is not saved correctly"
        # assert tb.dict['next_states'][index] == wrapped_transition['next_states'], "Next state is not saved correctly"
        # assert tb.dict['actions'][index] == wrapped_transition['actions'], "Action is not saved correctly"
        # assert tb.dict['rewards'][index] == wrapped_transition['rewards'], "Reward is not saved correctly"
        # assert tb.dict['dones'][index] == wrapped_transition['dones'], "Done is not saved correctly"
        # assert tb.dict['returns'][index] == wrapped_transition['returns'], "Return is not saved correctly"

        assert tb.first == first, "The first attribute is not correct"
        assert tb.size == size, "The size of the TransitionBatch is not correct"
        assert th.all(tb.dict['states'][index] == wrapped_transition['states']), "State is not saved correctly"
        assert th.all(tb.dict['next_states'][index] == wrapped_transition['next_states']), "Next state is not saved correctly"
        assert th.all(tb.dict['actions'][index] == wrapped_transition['actions']), "Action is not saved correctly"
        assert th.all(tb.dict['rewards'][index] == wrapped_transition['rewards']), "Reward is not saved correctly"
        assert th.all(tb.dict['dones'][index] == wrapped_transition['dones']), "Done is not saved correctly"
        assert th.all(tb.dict['returns'][index] == wrapped_transition['returns']), "Return is not saved correctly"
    # -------------------------- TEST ADD ---------------------------

    # Create a TransitionBatch and enviornment
    tb = TransitionBatch(max_size=2, transition_format=transition_format(), batch_size=1)
    env = EnviornmentTSP(th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32), max_nodes_per_graph=10)

    # Reset the environment and make step
    env.reset()
    action = th.tensor([0], dtype=th.long)
    state, reward, done, next_state = env.step(action)

    # Add a transition and test that is added correctly
    wrapped_transition = _wrap_transition(action, state, next_state, reward, done)
    tb.add(wrapped_transition)
    test_add(tb=tb, first=0, size=1, wrapped_transition=wrapped_transition, index=0)
    print("First Add test passed")

    # Add another transition
    action = th.tensor([1], dtype=th.long)
    state, reward, done, next_state = env.step(action.type(th.int32))
    wrapped_transition = _wrap_transition(action, state, next_state, reward, done)
    tb.add(wrapped_transition)
    test_add(tb=tb, first=0, size=2, wrapped_transition=wrapped_transition, index=1)
    print("Second Add test passed")

    # Add another transition ande test overflow
    action = th.tensor([2], dtype=th.long)
    state, reward, done, next_state = env.step(action.type(th.int32))
    wrapped_transition = _wrap_transition(action, state, next_state, reward, done)
    tb.add(wrapped_transition)
    test_add(tb=tb, first=1, size=2, wrapped_transition=wrapped_transition, index=0)
    print("Overflow handle tested")

    # -------------------------- TEST TRIM ---------------------------

    # Create a TransitionBatch and enviornment
    tb = TransitionBatch(max_size=2, transition_format=transition_format())
    env = EnviornmentTSP(th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32))

    # Reset the environment and make step
    env.reset()
    action = th.tensor([0], dtype=th.long)
    state, reward, done, next_state = env.step(action.type(th.int32))

    # Add a transition and test that is added correctly
    wrapped_transition = _wrap_transition(action, state, next_state, reward, done)
    tb.add(wrapped_transition)

    # trim the TransitionBatch and test that it is trimmed correctly
    tb.trim()
    assert tb.max_size == 1, "The max_size is not trimmed correctly"
    assert tb.size == 1, "The size is not trimmed correctly"
    assert tb.first == 0, "The first attribute is not trimmed correctly"
    print("Trim test passed")
    print("All tests passed")
    
def test_environment_tsp():
    """
    Test the TSP environment class.

    Args:
        None
    
    Returns:
        None    
    """

    def test_state(state: th.Tensor, 
                   current_city_value: int, 
                   first_city_value: int, 
                   previous_city_value: int, 
                   visited_cities: th.Tensor, 
                   cities: th.Tensor) -> None:
        """
        Test the state of the environment.

        Args:
            state (th.Tensor): The state of the environment.
            current_city_value (int): The value of the current city.
            first_city_value (int): The value of the first city.
            previous_city_value (int): The value of the previous city.
            visited_cities (th.Tensor): The value of the not visited cities tensor.
            cities (th.Tensor): The value of the cities tensor.
        
        Returns:
            None
        """
        # Dictionary test
        # assert state['current_cities'] == current_city_value, "The current city is not correct"
        # assert state['first_cities'] == first_city_value, "The first city is not correct"
        # assert state['previous_cities'] == previous_city_value, "The previous city is not correct"
        # assert th.all(state['visited_cities'] == visited_cities), "The visited cities tensor is not correct"
        # assert th.all(state['cities'] == cities), "The cities tensor is not correct"

        # Tensor test
        assert state[2] == current_city_value, "The current city is not correct"
        assert state[1] == first_city_value, "The first city is not correct"
        assert state[3] == previous_city_value, "The previous city is not correct"
        assert th.all(state[4:9] == visited_cities), "The visited cities tensor is not correct"
        assert th.all(state[14:24].view(-1, 2) == cities), "The cities tensor is not correct"

    cities = th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32)
    env = EnviornmentTSP(cities)

    state = env.reset()
    test_state(state, -1, -1, -1, th.tensor([0, 0, 0, 0, 0], dtype=th.bool), cities)
    print("Reset test passed")

    _, reward, done, next_state = env.step(0)
    test_state(next_state, 0, 0, -1, th.tensor([1, 0, 0, 0, 0], dtype=th.bool), cities)
    assert reward == 0, "The reward must be 0"
    assert done == False, "The episode is not done"
    print("Step 1 test passed")

    _, reward, done, next_state = env.step(1)
    test_state(next_state, 1, 0, 0, th.tensor([1, 1, 0, 0, 0], dtype=th.bool), cities)
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 2 test passed")
    
    _, reward, done, next_state = env.step(2)
    test_state(next_state, 2, 0, 1, th.tensor([1, 1, 1, 0, 0], dtype=th.bool), cities)
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 3 test passed")
    
    _, reward, done, next_state = env.step(3)
    test_state(next_state, 3, 0, 2, th.tensor([1, 1, 1, 1, 0], dtype=th.bool), cities)
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 4 test passed")
    
    _, reward, done, next_state = env.step(4)
    test_state(next_state, 4, 0, 3, th.tensor([1, 1, 1, 1, 1], dtype=th.bool), cities)
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == True, "The episode is done"
    print("Step 5 test passed")
    
    state = env.reset()
    test_state(state, -1, -1, -1, th.tensor([0, 0, 0, 0, 0], dtype=th.bool), cities)
    print("Reset test passed again")
    print("All tests passed")

def test_basic_network():
    """
    Test the BasicNetwork class.

    Args:
        None

    Returns:
        None
    """

    def get_batch(n_cities: th.Tensor, 
                current_city: th.Tensor, 
                first_city: th.Tensor, 
                previous_city: th.Tensor, 
                visited_cities: th.Tensor, 
                cities: th.Tensor,
                max_nodes_per_graph: int = 10,
                ) -> th.Tensor:
        """
        Returns the current state of the environment.
        
        Args:
            n_cities (int): The number of cities in the graph.
            current_city (int): The index of the current city.
            first_city (int): The index of the first city.
            previous_city (int): The index of the previous city.
            visited_cities (th.Tensor): A tensor indicating which cities have not been visited.
            cities (th.Tensor): A tensor containing the coordinates of the cities.
        
        Returns:
            th.Tensor: A tensor representing the state of the environment.
        """
        # Metadata
        metadata = th.cat((n_cities, current_city, first_city, previous_city), dim=1)

        # Flat cities tensor
        batch_size = n_cities.shape[0]
        cities = cities.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, -1)

        # Pad visited cities tensor
        if n_cities[0] < max_nodes_per_graph:
            padding = th.ones(batch_size, max_nodes_per_graph - n_cities[0])
            visited_cities = th.cat((visited_cities, padding), dim=1)

            padding = th.zeros(batch_size, (max_nodes_per_graph - n_cities[0])*2)
            cities = th.cat((cities, padding), dim=1)

        # Concatenate all tensors
        state = th.cat((metadata, visited_cities, cities), dim=1)
        return state
    
    def get_state_dict(first_city: th.Tensor, current_city: th.Tensor, visited_cities: th.Tensor, cities: th.Tensor) -> dict:
        """
        Returns the current state of the env in a dictionary.

        Args:
            first_city (int): The index of the first city.
            current_city (int): The index of the current city.
            visited_cities (th.Tensor): A tensor indicating which cities have not been visited.
            cities (th.Tensor): A tensor containing the coordinates of the cities.

        Returns:
            dict: A dictionary containing the state of the environment.
        """

        return {
            'first_cities': first_city,
            'current_cities': current_city,
            'visited_cities': visited_cities,
            'cities': cities
        }
    
    def test_output(pol_shape, start_mask, value_shape, value: th.Tensor, policy: th.Tensor) -> None:
        assert policy.shape == pol_shape, "The output tensor has the wrong shape"
        pol_shape_1 = policy.shape[1]
        batch_size = float(policy.shape[0])
        if pol_shape_1 > -start_mask: assert th.all(policy[0][start_mask:] == 0), f"The last {-start_mask} values of the output tensor are not 0"
        assert value.shape == value_shape, "The value tensor has the wrong shape"
        assert th.isclose(policy.sum(), th.tensor(batch_size)), "The sum of the policy tensor is not 1"
        
    max_nodes_per_graph = 10
    # Create a BasicNetwork object
    basic_network = BasicNetwork(max_nodes_per_graph = max_nodes_per_graph, node_dimension = 2, embedding_dimension = 4)

    # Get a state dict
    # current_city = th.tensor([-1], dtype=th.int32)
    # first_city = th.tensor([-1], dtype=th.int32)
    # cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32)
    # visited_cities = th.tensor([[1, 1, 1, 1, 1]], dtype=th.bool)
    # state = get_state_dict(first_city, current_city, visited_cities, cities)

    # Get a state tensor
    n_cities, current_city, first_city, previous_city = th.tensor([[5]]), th.tensor([[-1]]), th.tensor([[-1]]), th.tensor([[-1]])
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32)
    visited_cities = th.tensor([[0, 0, 0, 0, 0]], dtype=th.bool)
    state = get_batch(n_cities, current_city, first_city, previous_city, visited_cities, cities, max_nodes_per_graph)
    
    # Forward pass
    policy, value = basic_network(state)
    test_output((1, 10), -5, (1,), value, policy)
    print("Test 1 passed")

    # Get a state dict
    # current_city = th.tensor([0], dtype=th.int32)
    # first_city = th.tensor([0], dtype=th.int32)
    # cities = th.tensor([[1, 2], [3, 4], [5, 6]], dtype=th.float32)
    # visited_cities = th.tensor([[1, 0, 0]], dtype=th.bool)
    # state = get_state_dict(first_city, current_city, visited_cities, cities)

    # Get a state tensor
    n_cities, current_city, first_city, previous_city = th.tensor([[3]]), th.tensor([[0]]), th.tensor([[0]]), th.tensor([[0]])
    cities = th.tensor([[1, 2], [3, 4], [5, 6]], dtype=th.float32)
    visited_cities = th.tensor([[1, 0, 0]], dtype=th.bool)
    state = get_batch(n_cities, current_city, first_city, previous_city, visited_cities, cities, max_nodes_per_graph)

    # Forward pass
    policy, value = basic_network(state)
    test_output((1, 10), -7, (1,), value, policy)
    print("Test 2 passed")

    # Get a state dict
    # current_city = th.tensor([-1], dtype=th.int32)
    # first_city = th.tensor([-1], dtype=th.int32)
    # cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]], dtype=th.float32)
    # visited_cities = th.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=th.bool)
    # state = get_state_dict(first_city, current_city, visited_cities, cities)

    # Get a state tensor
    n_cities, current_city, first_city, previous_city = th.tensor([[10]]), th.tensor([[-1]]), th.tensor([[-1]]), th.tensor([[-1]])
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]], dtype=th.float32)
    visited_cities = th.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=th.bool)
    state = get_batch(n_cities, current_city, first_city, previous_city, visited_cities, cities, max_nodes_per_graph)

    # Forward pass
    policy, value = basic_network(state)

    # Output should have shape [10] and all values should be different from 0
    test_output((1, 10), -10, (1,), value, policy)
    print("Test 3 passed")

    # Get a state dict
    # current_city = th.tensor([-1], dtype=th.int32)
    # first_city = th.tensor([-1], dtype=th.int32)
    # cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [22, 22]], dtype=th.float32)
    # visited_cities = th.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=th.bool)
    # state = get_state_dict(first_city, current_city, visited_cities, cities)

    # Get a state tensor
    n_cities, current_city, first_city, previous_city = th.tensor([[11]]), th.tensor([[-1]]), th.tensor([[-1]]), th.tensor([[-1]])
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [22, 22]], dtype=th.float32)
    visited_cities = th.tensor([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]], dtype=th.bool)
    state = get_batch(n_cities, current_city, first_city, previous_city, visited_cities, cities, max_nodes_per_graph)

    # Forward pass should fail
    try:
        policy, value = basic_network(state)
    except AssertionError as e:
        # print(e)
        print("Test 4 passed")
    else:
        raise AssertionError("The forward pass should have failed")
    
    # Test batch of states inside the network
    # current_city = th.tensor([-1, 0], dtype=th.int32)
    # first_city = th.tensor([-1, 0], dtype=th.int32)
    # cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32)
    # visited_cities = th.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=th.bool)
    # batch = get_state_dict(first_city, current_city, visited_cities, cities)

    # Get a state tensor
    n_cities, current_city, first_city, previous_city = th.tensor([[5], [3]]), th.tensor([[-1], [0]]), th.tensor([[-1], [0]]), th.tensor([[-1], [0]])
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32)
    visited_cities = th.tensor([[0, 0, 0, 0, 0], [1, 0, 0, 0, 0]], dtype=th.bool)
    batch = get_batch(n_cities, current_city, first_city, previous_city, visited_cities, cities, max_nodes_per_graph)

    # Forward pass
    policy, value = basic_network(batch)
    test_output((2, 10), -5, (2,), value, policy)
    print("Batch test passed")

    print("All tests passed")

def test_generator():
    """
    Test the TSPGenerator class.
    
    Args:
        None
    
    Returns:
        None
    """
    # Test the generator
    generator = TSPGenerator()
    instance_tsp = generator.generate_instance(5)
    assert instance_tsp.shape == (5, 2)
    print("Instance generation test passed")

    batch_tsp = generator.generate_batch(2, 5)
    assert len(batch_tsp) == 2
    assert batch_tsp[0].shape == (5, 2)
    assert batch_tsp[1].shape == (5, 2)
    print("Batch generation test passed")

    batch_set_tsp = generator.generate_batch_set(2, [5, 10])
    assert len(batch_set_tsp) == 2
    assert len(batch_set_tsp[0]) == 2
    assert len(batch_set_tsp[1]) == 2
    assert batch_set_tsp[0][0].shape == (5, 2)
    assert batch_set_tsp[0][1].shape == (5, 2)
    assert batch_set_tsp[1][0].shape == (10, 2)
    assert batch_set_tsp[1][1].shape == (10, 2)
    print("Batch set generation test passed")
    print("All tests passed")

def test_ACController():
    """
    Test the ActorCriticController class.

    Args:
        None

    Returns:
        None
    """

    def get_batch(n_cities: th.Tensor, 
                current_city: th.Tensor, 
                first_city: th.Tensor, 
                previous_city: th.Tensor, 
                visited_cities: th.Tensor, 
                cities: th.Tensor,
                max_nodes_per_graph: int = 10,
                ) -> th.Tensor:
        """
        Returns the current state of the environment.
        
        Args:
            n_cities (int): The number of cities in the graph.
            current_city (int): The index of the current city.
            first_city (int): The index of the first city.
            previous_city (int): The index of the previous city.
            visited_cities (th.Tensor): A tensor indicating which cities have not been visited.
            cities (th.Tensor): A tensor containing the coordinates of the cities.
        
        Returns:
            th.Tensor: A tensor representing the state of the environment.
        """
        # Metadata
        metadata = th.cat((n_cities, current_city, first_city, previous_city), dim=1)

        # Flat cities tensor
        batch_size = n_cities.shape[0]
        cities = cities.unsqueeze(0).expand(batch_size, -1, -1).view(batch_size, -1)

        # Pad visited cities tensor
        if n_cities[0] < max_nodes_per_graph:
            padding = th.ones(batch_size, max_nodes_per_graph - n_cities[0])
            visited_cities = th.cat((visited_cities, padding), dim=1)

            padding = th.zeros(batch_size, (max_nodes_per_graph - n_cities[0])*2)
            cities = th.cat((cities, padding), dim=1)

        # Concatenate all tensors
        state = th.cat((metadata, visited_cities, cities), dim=1)
        return state
    
    def get_state_dict(first_city: th.Tensor, current_city: th.Tensor, visited_cities: th.Tensor, cities: th.Tensor) -> dict:
        """
        Returns the current state of the env in a dictionary.

        Args:
            first_city (int): The index of the first city.
            current_city (int): The index of the current city.
            visited_cities (th.Tensor): A tensor indicating which cities have not been visited.
            cities (th.Tensor): A tensor containing the coordinates of the cities.

        Returns:
            dict: A dictionary containing the state of the environment.
        """

        return {
            'first_cities': first_city,
            'current_cities': current_city,
            'visited_cities': visited_cities,
            'cities': cities
        }

    # Init network
    network = BasicNetwork(max_nodes_per_graph = 10, node_dimension = 2, embedding_dimension = 4)
    controller = ActorCriticController(network)

    # Test copy
    copy = controller.copy()
    assert copy.model == controller.model, "The model was not copied correctly"
    print("Copy test passed")

    # Test parameters
    params = controller.parameters()
    # iterate over the parameters
    for p1, p2 in zip(params, network.parameters()):
        assert th.all(p1 == p2), "The parameters are not the same"
    print("Parameters test passed")

    # Test choose_action    

    # Get a state dict
    # current_city = th.tensor([-1], dtype=th.int32)
    # first_city = th.tensor([-1], dtype=th.int32)    
    # visited_cities = th.tensor([[0, 0, 0, 0, 0]], dtype=th.bool)
    # cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32)
    # state = get_state_dict(first_city, current_city, visited_cities, cities)

    # Get a state tensor
    n_cities, current_city, first_city, previous_city = th.tensor([[5]]), th.tensor([[-1]]), th.tensor([[-1]]), th.tensor([[-1]])
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32)
    visited_cities = th.tensor([[0, 0, 0, 0, 0]], dtype=th.bool)
    state = get_batch(n_cities, current_city, first_city, previous_city, visited_cities, cities, max_nodes_per_graph=10)

    action = controller.choose_action(state)
    assert action.shape == (1,), "The action has the wrong shape"
    print("Choose action test passed")

    # Test probabilities
    probs = controller.probabilities(state)
    assert probs.shape == (1, 10), "The probabilities have the wrong shape"
    assert th.isclose(probs.sum(), th.tensor(1.0)), "The sum of the probabilities is not 1"
    print("Probabilities test passed")
    print("All tests passed")

def test_runner():
    """
    Test the Runner class.

    Args:
        None

    Returns:
        None
    """
    
    # Create a Controller
    network = BasicNetwork(max_nodes_per_graph = 10, node_dimension = 2, embedding_dimension = 4)
    controller = ActorCriticController(network)

    # Create an Environment
    cities = th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32)
    env = EnviornmentTSP(cities)

    # Create a Runner
    runner = Runner(controller, env)

    # Test run episode
    results = runner.run_episode()
    # print(results)
    # print(f"Actions: {results['buffer']['actions']}")
    assert results['episode_length'] == 5, "The episode has the wrong length    "
    assert results['env_steps'] == 5, "The environment has the wrong number of steps"
    print("Run episode test passed")

    # Test run 2 episodes
    results = runner.run(10)
    # print(results)
    # print(f"Actions: {results['buffer']['actions']}")
    assert results['episode_length'] == 5, "The episode has the wrong length"
    assert results['env_steps'] == 10, "The environment has the wrong number of steps"
    print("Run 2 episodes test passed")

    # # Test run a few steps
    results = runner.run(2)
    # print(results)
    # print(f"Actions: {results['buffer']['actions']}")
    assert results['episode_length'] == None, "The episode has the wrong length"
    assert results['env_steps'] == 2, "The environment has the wrong number of steps"
    print("Run 2 steps test passed")
    print("All tests passed")

def test_reinforce_learner():
    """
    Test the ReinforceLearner class.

    Args:
        None

    Returns:
        None
    """
    
    # Create network
    network = BasicNetwork(max_nodes_per_graph = 10, node_dimension = 2, embedding_dimension = 4)

    # Create controller
    controller = ActorCriticController(network)

    # Create optmiizer
    optimizer = th.optim.Adam(network.parameters(), lr=0.001)

    # Get initial parameters
    params = default_params()

    # Create ReinforceLearner
    learner = ReinforceLearner(network, controller, optimizer, params)

    # -------------------- TEST SET CONTROLLER ----------------------------
    controller2 = ActorCriticController(network)
    learner.set_controller(controller2)
    assert learner.controller == controller2, "The controller was not set correctly"
    print("Set controller test passed")

    # -------------------- TEST TRAIN -------------------------------------

    # Create a Runner
    cities = th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32)
    env = EnviornmentTSP(cities)
    runner = Runner(controller, env)

    # run an episode
    results = runner.run_episode()
    batch = results['buffer']

    # Train the learner
    learner.train(batch)
    print("Train test passed")


if __name__ == '__main__':
    print("Running tests...")
    print("---------- Testing TransitionBatch ----------")
    test_transition_batch()
    print("---------- Testing EnvironmentTSP ----------")
    test_environment_tsp()
    print("---------- Testing BasicNetwork ----------")
    test_basic_network()
    print("---------- Testing TSPGenerator ----------")
    test_generator()
    print("---------- Testing ACController ----------")
    test_ACController()
    print("---------- Testing Runner ----------")
    test_runner()
    print("---------- Testing ReinforceLearner ----------")
    test_reinforce_learner()
