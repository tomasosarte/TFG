def default_params():
    """ These are the default parameters used in the framework. """
    return {
            # Debugging outputs and plotting during training
            'plot_frequency': 10,             # plots a debug message avery n steps
            'plot_train_samples': True,       # whether the x-axis is env.steps (True) or episodes (False)
            'print_when_plot': True,          # prints debug message if True
            'print_dots': False,              # prints dots for every gradient update

            # Environment parameters
            'problem': 'tsp',                 # the problem the agent is learning to solve
            'max_episode_length': 1000,       # maximum number of steps in an episode

            # Runner parameters
            'multi_runner': False,            # uses multiple runners if True       
            'parallel_environments': 4,       # number of parallel runners  (only if multi_runner==True)     

            # Optimization parameters
            'gamma': 0.99,                    # discount factor gamma
            'lr': 5E-4,                       # learning rate of optimizer
            'grad_norm_clip': 1,              # gradent clipping if grad norm is larger than this value
            'batch_size': 0,                  # number of transitions in a mini-batch. If 0, the batch size it the legth of an episode

            # Exploration parameters
            'epsilon_start': 1,               # annealing starts at this epsilon
            'epsilon_finish': 1E-5,           # annealing stops at (and keeps) this epsilon
            'epsilon_anneal_time': 1E3,       # exploration anneals epsilon over these many steps

            # Actor-critic parameters  
            'value_loss_param': 0.1,          # governs the relative impact of the value relative to policy loss
            'offpolicy_iterations': 0,        # how many off-policy iterations are performed per training step
           }