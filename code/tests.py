import torch as th
from torch import tensor

from utils.transition_batch import TransitionBatch
from environments.environment_tsp import EnvironemntTSP
from networks.basic_network import BasicNetwork
from generators.tsp_generator import TSPGenerator
from controllers.ac_controller import ActorCriticController
from runners.runner import Runner
from params import default_params
from learners.reinforce_learner import ReinforceLearner


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
            'states': ((4 + 10 + 2 * 10,), th.float32),
            'next_states': ((4 + 10 + 2 * 10,), th.float32),
            'rewards': ((1,), th.float32),
            'dones': ((1,), th.bool),
            'returns': ((1,), th.float32)
        }

    def test_add(tb: TransitionBatch, first:int, size:int, wrapped_transition: dict, index:int) -> None:
        assert tb.first == first, "The first attribute is not correct"
        assert tb.size == size, "The size of the TransitionBatch is not correct"
        assert (tb.dict['states'][index] == wrapped_transition['states']).all(), "State is not saved correctly"
        assert (tb.dict['next_states'][index] == wrapped_transition['next_states']).all(), "Next state is not saved correctly"
        assert (tb.dict['actions'][index] == wrapped_transition['actions']).all(), "Action is not saved correctly"
        assert (tb.dict['rewards'][index] == wrapped_transition['rewards']).all(), "Reward is not saved correctly"
        assert (tb.dict['dones'][index] == wrapped_transition['dones']).all(), "Done is not saved correctly"
        assert (tb.dict['returns'][index] == wrapped_transition['returns']).all(), "Return is not saved correctly"

    # -------------------------- TEST ADD ---------------------------

    # Create a TransitionBatch and enviornment
    tb = TransitionBatch(max_size=2, transition_format=transition_format(), batch_size=1)
    env = EnvironemntTSP(th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32), max_nodes_per_graph=10)

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
    env = EnvironemntTSP(th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32), max_nodes_per_graph=10)

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
    cities = th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32)
    env = EnvironemntTSP(cities, max_nodes_per_graph=10)

    state = env.reset()
    assert state[0][0] == 5, "The number of cities is not 5"
    assert state[0][1] == -1, "The current city must be -1"
    assert state[0][2] == -1, "The first city must be -1"
    assert state[0][3] == -1, "The previous city must be -1"
    assert state[0][4:9].sum() == 5, "No cities must be visited"
    print("Reset test passed")

    _, reward, done, next_state = env.step(0)
    assert next_state[0][0] == 5, "The number of cities is not 5"
    assert next_state[0][1] == 0, "The current city must be 0"
    assert next_state[0][2] == 0, "The first city must be 0"
    assert next_state[0][3] == -1, "The previous city must be -1"
    assert next_state[0][4:9].sum() == 4, "One city must be visited"
    assert reward == 0, "The reward must be 0"
    assert done == False, "The episode is not done"
    print("Step 1 test passed")

    _, reward, done, next_state = env.step(1)
    assert next_state[0][0] == 5, "The number of cities is not 5"
    assert next_state[0][1] == 1, "The current city must be 1"
    assert next_state[0][2] == 0, "The first city must be 0"
    assert next_state[0][3] == 0, "The previous city must be 0"
    assert next_state[0][4:9].sum() == 3, "Two cities must be visited"
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 2 test passed")
    
    _, reward, done, next_state = env.step(2)
    assert next_state[0][0] == 5, "The number of cities is not 5"
    assert next_state[0][1] == 2, "The current city must be 2"
    assert next_state[0][2] == 0, "The first city must be 0"
    assert next_state[0][3] == 1, "The previous city must be 1"
    assert next_state[0][4:9].sum() == 2, "Three cities must be visited"
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 3 test passed")
    
    _, reward, done, next_state = env.step(3)
    assert next_state[0][0] == 5, "The number of cities is not 5"
    assert next_state[0][1] == 3, "The current city must be 3"
    assert next_state[0][2] == 0, "The first city must be 0"
    assert next_state[0][3] == 2, "The previous city must be 2"
    assert next_state[0][4:9].sum() == 1, "Four cities must be visited"
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 4 test passed")
    
    _, reward, done, next_state = env.step(4)
    assert next_state[0][0] == 5, "The number of cities is not 5"
    assert next_state[0][1] == 4, "The current city must be 4"
    assert next_state[0][2] == 0, "The first city must be 0"
    assert next_state[0][3] == 3, "The previous city must be 3"
    assert next_state[0][4:9].sum() == 0, "All cities must be visited"
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == True, "The episode is done"
    print("Step 5 test passed")
    
    state = env.reset()
    assert state[0][0] == 5, "The number of cities is not 5"
    assert state[0][1] == -1, "The current city must be -1"
    assert state[0][2] == -1, "The first city must be -1"
    assert state[0][3] == -1, "The previous city must be -1"
    assert state[0][4:9].sum() == 5, "No cities must be visited"
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

    def get_state(n_cities: int, current_city: int, first_city: int, previous_city: int, not_visited_cities: th.Tensor, cities: th.Tensor) -> th.Tensor:
        """
        Returns the current state of the environment.
        
        Args:
            n_cities (int): The number of cities in the graph.
            current_city (int): The index of the current city.
            first_city (int): The index of the first city.
            previous_city (int): The index of the previous city.
            not_visited_cities (th.Tensor): A tensor indicating which cities have not been visited.
            cities (th.Tensor): A tensor containing the coordinates of the cities.
        
        Returns:
            th.Tensor: A tensor representing the state of the environment.
        """
        info = th.Tensor([n_cities, current_city, first_city, previous_city])
        state = th.cat((info, not_visited_cities, cities.view(-1))).unsqueeze(0)
        return state
    
    # Create a BasicNetwork object
    basic_network = BasicNetwork(max_nodes_per_graph = 10, node_dimension = 2, embedding_dimension = 4)

    # Get a state tensor
    n_cities = 5
    current_city = -1
    first_city = -1
    previous_city = -1
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32)
    cities = th.cat((cities, th.zeros(5, 2)), dim=0)
    not_visited_cities = th.tensor([1, 1, 1, 1, 1], dtype=th.bool)
    not_visited_cities = th.cat((not_visited_cities, th.zeros(5, dtype=th.bool)), dim=0)
    state = get_state(n_cities, current_city, first_city, previous_city, not_visited_cities, cities)

    # Forward pass
    policy, value = basic_network(state)

    # Output should have shape [10] and 5 last values should be 0
    assert policy.shape == (10,), "The output tensor has the wrong shape"
    assert th.all(policy[-5:] == 0), "The last 5 values of the output tensor are not 0"
    assert value.shape == (), "The value tensor has the wrong shape"
    # Sum of all values in policy should be 1
    assert th.isclose(policy.sum(), th.tensor(1.0)), "The sum of the policy tensor is not 1"
    print("Test 1 passed")

    # Get a state tensor
    n_cities = 3
    current_city = 0
    first_city = 0
    previous_city = -1
    cities = th.tensor([[1, 2], [3, 4], [5, 6]], dtype=th.float32)
    cities = th.cat((cities, th.zeros(7, 2)), dim=0)
    not_visited_cities = th.tensor([0, 1, 1], dtype=th.bool)
    not_visited_cities = th.cat((not_visited_cities, th.zeros(7, dtype=th.bool)), dim=0)
    state = get_state(n_cities, current_city, first_city, previous_city, not_visited_cities, cities)

    # Forward pass
    policy, value = basic_network(state)

    # Output should have shape [10] and 7 last values should be 0. Also the first value should be 0
    assert policy.shape == (10,), "The output tensor has the wrong shape"
    assert th.all(policy[-7:] == 0), "The last 7 values of the output tensor are not 0"
    assert value.shape == (), "The value tensor has the wrong shape"
    # Sum of all values in policy should be 1
    assert th.isclose(policy.sum(), th.tensor(1.0)), "The sum of the policy tensor is not 1"
    print("Test 2 passed")

    n_cities = 10
    current_city = -1
    first_city = -1
    previous_city = -1
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]], dtype=th.float32)
    not_visited_cities = th.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=th.bool)
    state = get_state(n_cities, current_city, first_city, previous_city, not_visited_cities, cities)
    
    # Forward pass
    policy, value = basic_network(state)

    # Output should have shape [10] and all values should be different from 0
    assert policy.shape == (10,), "The output tensor has the wrong shape"
    assert th.all(policy != 0), "All values of the output tensor are 0"
    assert value.shape == (), "The value tensor has the wrong shape"
    # Sum of all values in policy should be 1
    assert th.isclose(policy.sum(), th.tensor(1.0)), "The sum of the policy tensor is not 1"
    print("Test 3 passed")

    n_cities = 11
    current_city = -1
    first_city = -1
    previous_city = -1
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [22, 22]], dtype=th.float32)
    not_visited_cities = th.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=th.bool)
    state = get_state(n_cities, current_city, first_city, previous_city, not_visited_cities, cities)

    # Forward pass should fail
    try:
        policy, value = basic_network(state)
    except AssertionError as e:
        # print(e)
        print("Test 4 passed")
    else:
        raise AssertionError("The forward pass should have failed")

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

    def get_state(n_cities: int, current_city: int, first_city: int, previous_city: int, not_visited_cities: th.Tensor, cities: th.Tensor) -> th.Tensor:
        """
        Returns the current state of the environment.
        
        Args:
            n_cities (int): The number of cities in the graph.
            current_city (int): The index of the current city.
            first_city (int): The index of the first city.
            previous_city (int): The index of the previous city.
            not_visited_cities (th.Tensor): A tensor indicating which cities have not been visited.
            cities (th.Tensor): A tensor containing the coordinates of the cities.
        
        Returns:
            th.Tensor: A tensor representing the state of the environment.
        """
        info = th.Tensor([n_cities, current_city, first_city, previous_city])
        state = th.cat((info, not_visited_cities, cities.view(-1))).unsqueeze(0)
        return state
    

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
    n_cities = 5
    current_city = -1
    first_city = -1
    previous_city = -1
    not_visited_cities = th.tensor([1, 1, 1, 1, 1], dtype=th.bool)
    not_visited_cities = th.cat((not_visited_cities, th.zeros(5, dtype=th.bool)), dim=0)
    cities = th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32)
    cities = th.cat((cities, th.zeros(5, 2)), dim=0)
    state = get_state(n_cities, current_city, first_city, previous_city, not_visited_cities, cities)

    action = controller.choose_action(state)
    assert action.shape == (), "The action has the wrong shape"
    print("Choose action test passed")

    # Test probabilities
    probs = controller.probabilities(state)
    assert probs.shape == (10,), "The probabilities have the wrong shape"
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
    env = EnvironemntTSP(cities, max_nodes_per_graph=10)

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
    env = EnvironemntTSP(cities, max_nodes_per_graph=10)
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
    
