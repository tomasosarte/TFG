import torch as th

def default_params():
    """ These are the default parameters used in the framework. """
    return {
            # Debugging outputs and plotting during training
            'plot_frequency': 10,             # plots a debug message avery n steps
            'plot_train_samples': True,       # whether the x-axis is env.steps (True) or episodes (False)
            'debug_messages': False,          # prints debug message if True
            'print_dots': False,              # prints dots for every gradient update

            # Environment parameters
            'env': 'tsp',                     # the problem the agent is learning to solve
            'max_episode_length': 1000,       # maximum number of steps in an episode
            'max_nodes_per_graph': 100,       # maximum number of nodes in a graph
            'node_dimension': 2,              # dimension of the node features
            'diff_sizes': False,              # whether the graphs have different sizes in training
            'diff_cities': False,             # whether the graphs have different cities in training
            'use_training_set': True,         # whether the training set is used if diff_cities==True
            'training_sizes': [20, 50, 100],  # sizes of the training set if diff_cities==True
            'num_train_instance_per_size': 1, # number of training instances per size if diff_cities==True
            
            # Runner parameters
            'max_episodes': 1000,             # experiment stops after this many episodes
            'max_steps': 10000000,            # experiment stops after this many steps
            'multi_runner': False,            # uses multiple runners if True       
            'parallel_environments': 4,       # number of parallel runners  (only if multi_runner==True)    

            # Exploration parameters
            'epsilon_anneal_time': 5E3,       # exploration anneals epsilon over these many steps
            'epsilon_finish': 1E-5,           # annealing stops at (and keeps) this epsilon 
                                              # epsilon_finish should be 0 for on-policy sampling, 
                                              # but pytorch sometimes produced NaN gradients if probabilities get 
                                              # too close to zero (because then ln(0)=-infty)
            'epsilon_start': 1,               # annealing starts at this epsilon
            'epsilon_decay': 'linear',        # either 'linear' or 'exponential'
            'entropy_weight': 0.1,            # weight of the entropy term in the loss
            'entropy_regularization': True,   # whether entropy regularization is used
            'use_epsilon_greedy': True,       # whether epsilon-greedy is used
            'decay_entropy': False,           # whether the entropy weight is decayed
            'entropy_weight_start': 0.1,      # initial entropy weight
            'entropy_weight_end': 0.01,       # final entropy weight
            'entropy_anneal_time': 1E6,       # annealing time of the entropy weight

            # Optimization parameters
            'lr': 5E-4,                       # learning rate of optimizer
            'gamma': 0.99,                    # discount factor gamma
            'batch_size': 100,                # number of transitions in a mini-batch. If 0, the batch size it the legth of an episode
            'grad_norm_clip': 1,              # gradient clipping if grad norm is larger than this value

            # Actor-critic parameters  
            'value_loss_param': 0.1,          # governs the relative impact of the value relative to policy loss
            'advantage_bias': True,           # whether the advantages have the value as bias
            'advantage_bootstrap': True,      # whether advantages use bootstrapping (alternatively: returns)
            'offpolicy_iterations': 0,        # how many off-policy iterations are performed
            'value_targets': 'returns',       # either 'returns' or 'td' as regression targets of the value function

            # PPO parameters
            'ppo_clipping': True,             # whether we use the PPO loss
            'ppo_clip_eps': 0.1,              # the epsilon for the PPO loss

            # Network parameters
            'embedding_dimension': 4,         # dimension of the node embeddings
            'encoder_layers': 6,              # number of layers in the encoder
            'model_dimension': 512,           # dimension of the model
            'dimension_k': 64,                # dimension of the key and query vectors
            'dimension_v': 64,                # dimension of the value vectors
            'num_heads': 4,                   # number of heads in the multihead attention
            'num_layers': 3,                  # number of layers in the model
            'normalization': 'batch',         # either 'batch' or 'layer' normalization
            'feed_forward_hidden': 512,       # dimension of the hidden layer in the feed forward network
            'embed_dim': 4,                   # dimension of the embeddings

            # Device
            'device': 'cpu',                  # device used for training
            'use_tqdm': False,                # whether to use tqdm for progress bars
            'final_plot': False,              # whether to plot the final results
            'wandb': False                    # whether to use wandb for logging
           }

def set_tsp_params(params, max_nodes_per_graph, embedding_dimension, max_episodes, episodes_in_batch):
    """ Sets the parameters for the TSP problem. """
    params['problem'] = 'tsp'
    params['max_nodes_per_graph'] = max_nodes_per_graph
    params['node_dimension'] = 2
    params['embedding_dimension'] = embedding_dimension
    params['max_episode_length'] = max_nodes_per_graph + 1
    params['batch_size'] = (max_nodes_per_graph + 1)*episodes_in_batch
    params['max_episodes'] = max_episodes
    params['max_steps'] = max_episodes * max_nodes_per_graph
    return params
