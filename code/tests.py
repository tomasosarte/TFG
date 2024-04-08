import torch as th

from utils.transition_batch import TransitionBatch
from environments.environment_tsp import EnvironemntTSP
from networks.basic_network import BasicNetwork
from generators.tsp_generator import TSPGenerator
from controllers.ac_controller import ActorCriticController

def test_transition_batch():
    """
    Test the TransitionBatch class.

    Args:
        None

    Returns:
        None
    """

    def _wrap_transition(action, state, next_state, reward, done):
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
            'action': action,
            'state': state,
            'next_state': next_state,
            'reward': reward,
            'done': done,
            'return': th.zeros(1, dtype=th.float32)
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
            'action': ((1,), th.float32),
            'state': ((1,), dict),
            'next_state': ((1,), dict),
            'reward': ((1,), th.float32),
            'done': ((1,), th.bool),
            'return': ((1,), th.float32)
        }

    def test_add(tb: TransitionBatch, first:int, size:int, wrapped_transition: dict, index:int) -> None:
        assert tb.first == first, "The first attribute is not correct"
        assert tb.size == size, "The size of the TransitionBatch is not correct"
        assert tb.dict['state'][index] == wrapped_transition['state'], "State is not saved correctly"
        assert tb.dict['next_state'][index] == wrapped_transition['next_state'], "Next state is not saved correctly"
        assert tb.dict['action'][index] == wrapped_transition['action'], "Action is not saved correctly"
        assert tb.dict['reward'][index] == wrapped_transition['reward'], "Reward is not saved correctly"
        assert tb.dict['done'][index] == wrapped_transition['done'], "Done is not saved correctly"
        assert tb.dict['return'][index] == wrapped_transition['return'], "Return is not saved correctly"

    # -------------------------- TEST ADD ---------------------------

    # Create a TransitionBatch and enviornment
    tb = TransitionBatch(max_size=2, transition_format=transition_format(), batch_size=1)
    env = EnvironemntTSP(th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32))

    # Reset the environment and make step
    env.reset()
    action = th.tensor([0], dtype=th.float32)
    state, reward, done, next_state = env.step(action.type(th.int32))

    # Add a transition and test that is added correctly
    wrapped_transition = _wrap_transition(action, state, next_state, reward, done)
    tb.add(wrapped_transition)
    test_add(tb=tb, first=0, size=1, wrapped_transition=wrapped_transition, index=0)
    print("First Add test passed")

    # Add another transition
    action = th.tensor([1], dtype=th.float32)
    state, reward, done, next_state = env.step(action.type(th.int32))
    wrapped_transition = _wrap_transition(action, state, next_state, reward, done)
    tb.add(wrapped_transition)
    test_add(tb=tb, first=0, size=2, wrapped_transition=wrapped_transition, index=1)
    print("Second Add test passed")

    # Add another transition ande test overflow
    action = th.tensor([2], dtype=th.float32)
    state, reward, done, next_state = env.step(action.type(th.int32))
    wrapped_transition = _wrap_transition(action, state, next_state, reward, done)
    tb.add(wrapped_transition)
    test_add(tb=tb, first=1, size=2, wrapped_transition=wrapped_transition, index=0)
    print("Overflow handle tested")

    # -------------------------- TEST TRIM ---------------------------

    # Create a TransitionBatch and enviornment
    tb = TransitionBatch(max_size=2, transition_format=transition_format())
    env = EnvironemntTSP(th.tensor([[0, 0], [1, 1], [2, 2], [3, 3], [4, 4]], dtype=th.float32))

    # Reset the environment and make step
    env.reset()
    action = th.tensor([0], dtype=th.float32)
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
    env = EnvironemntTSP(cities)

    state = env.reset()
    assert state['current_city'] == None, "The current city must be None"
    assert state['first_city'] == None, "The first city must be None"
    assert state['previous_city'] == None, "The previous city must be None"
    assert state['not_visited_cities'].sum() == 5, "No cities must be visited"
    print("Reset test passed")

    _, reward, done, next_state = env.step(0)
    assert next_state['current_city'] == 0, "The current city must be 0"
    assert next_state['first_city'] == 0, "The first city must be 0"
    assert next_state['previous_city'] == None, "The previous city must be None"
    assert next_state['not_visited_cities'].sum() == 4, "One city must be visited"
    assert reward == 0, "The reward must be 0"
    assert done == False, "The episode is not done"
    print("Step 1 test passed")

    _, reward, done, next_state = env.step(1)
    assert next_state['current_city'] == 1, "The current city must be 1"
    assert next_state['first_city'] == 0, "The first city must be 0"
    assert next_state['previous_city'] == 0, "The previous city must be 0"
    assert next_state['not_visited_cities'].sum() == 3, "Two cities must be visited"
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 2 test passed")

    _, reward, done, next_state = env.step(2)
    assert next_state['current_city'] == 2,  "The current city must be 2"
    assert next_state['first_city'] == 0, "The first city must be 0"
    assert next_state['previous_city'] == 1, "The previous city must be 1"
    assert next_state['not_visited_cities'].sum() == 2, "Three cities must be visited"
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 3 test passed")

    _, reward, done, next_state = env.step(3)
    assert next_state['current_city'] == 3, "The current city must be 3"
    assert next_state['first_city'] == 0, "The first city must be 0"
    assert next_state['previous_city'] == 2, "The previous city must be 2"
    assert next_state['not_visited_cities'].sum() == 1, "Four cities must be visited"
    assert reward == -1.4142135, "The reward must be -1.4142135"
    assert done == False, "The episode is not done"
    print("Step 4 test passed")

    _, reward, done, next_state = env.step(4)
    assert next_state['current_city'] == 4, "The current city must be 4"
    assert next_state['first_city'] == 0, "The first city must be 0"
    assert next_state['previous_city'] == 3, "The previous city must be 3"
    assert next_state['not_visited_cities'].sum() == 0, "All cities must be visited"
    assert reward == -1.4142135381698608, "The reward must be -1.4142135381698608"
    assert done == True, "The episode is done"
    print("Step 5 test passed")

    state = env.reset()
    assert state['current_city'] == None, "The current city must be None"
    assert state['first_city'] == None, "The first city must be None"
    assert state['previous_city'] == None, "The previous city must be None"
    assert state['not_visited_cities'].sum() == 5, "No cities must be visited"
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
    # Create a BasicNetwork object
    basic_network = BasicNetwork(max_nodes_per_graph = 10, node_dimension = 2, embedding_dimension = 4)

    # Define a state dictionary
    state = {
        'first_city': None,
        'current_city': None,
        'cities': th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32),
        'not_visited_cities': th.tensor([1, 1, 1, 1, 1], dtype=th.bool)
    }

    # Forward pass
    policy, value = basic_network(state)

    # Output should have shape [10] and 5 last values should be 0
    assert policy.shape == (10,), "The output tensor has the wrong shape"
    assert th.all(policy[-5:] == 0), "The last 5 values of the output tensor are not 0"
    assert value.shape == (), "The value tensor has the wrong shape"
    # Sum of all values in policy should be 1
    assert th.isclose(policy.sum(), th.tensor(1.0)), "The sum of the policy tensor is not 1"
    print("Test 1 passed")

    # Define a state dictionary with less than 10 cities
    state = {
        'first_city': th.tensor([1, 2], dtype=th.float32),
        'current_city': th.tensor([1, 2], dtype=th.float32),
        'cities': th.tensor([[1, 2], [3, 4], [5, 6]], dtype=th.float32),
        'not_visited_cities': th.tensor([0, 1, 1], dtype=th.bool)
    }

    # Forward pass
    policy, value = basic_network(state)

    # Output should have shape [10] and 7 last values should be 0. Also the first value should be 0
    assert policy.shape == (10,), "The output tensor has the wrong shape"
    assert th.all(policy[-7:] == 0), "The last 7 values of the output tensor are not 0"
    assert value.shape == (), "The value tensor has the wrong shape"
    # Sum of all values in policy should be 1
    assert th.isclose(policy.sum(), th.tensor(1.0)), "The sum of the policy tensor is not 1"
    print("Test 2 passed")

    # Define a state dictionary with more than 10 cities
    state = {
        'first_city': None,
        'current_city': None,
        'cities': th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20]], dtype=th.float32),
        'not_visited_cities': th.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=th.bool)
    }

    
    # Forward pass
    policy, value = basic_network(state)

    # Output should have shape [10] and all values should be different from 0
    assert policy.shape == (10,), "The output tensor has the wrong shape"
    assert th.all(policy != 0), "All values of the output tensor are 0"
    assert value.shape == (), "The value tensor has the wrong shape"
    # Sum of all values in policy should be 1
    assert th.isclose(policy.sum(), th.tensor(1.0)), "The sum of the policy tensor is not 1"
    print("Test 3 passed")

    # Define a state dictionary with more than 10 cities
    state = {
        'first_city': None,
        'current_city': None,
        'cities': th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16], [17, 18], [19, 20], [22, 22]], dtype=th.float32),
        'not_visited_cities': th.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1], dtype=th.bool)
    }

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
    state = {
        'first_city': None,
        'current_city': None,
        'cities': th.tensor([[1, 2], [3, 4], [5, 6], [7, 8], [9, 10]], dtype=th.float32),
        'not_visited_cities': th.tensor([1, 1, 1, 1, 1], dtype=th.bool)
    }
    action = controller.choose_action(state)
    assert action.shape == (), "The action has the wrong shape"
    print("Choose action test passed")

if __name__ == '__main__':
    # print("Running tests...")
    # print("---------- Testing TransitionBatch ----------")
    # test_transition_batch()
    # print("---------- Testing EnvironmentTSP ----------")
    # test_environment_tsp()
    # print("---------- Testing BasicNetwork ----------")
    # test_basic_network()
    # print("---------- Testing TSPGenerator ----------")
    # test_generator()
    print("---------- Testing ACController ----------")
    test_ACController()
