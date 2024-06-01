import torch as th
import wandb

from params import default_params

# ------------------------- ENVIRONMENTS ----------------------------
from environments.environment_tsp import EnviornmentTSP
# --------------------------------

# ------------------------- EXPERIMENTS -----------------------------
from experiments.actor_critic_experiment import ActorCriticExperiment
# -------------------------------------------------------------------

# -------------------------- MODELS ---------------------------------
from networks.more_basic_net import MoreBasicNetwork
from networks.new_transformer import NewTransformer
# -------------------------------------------------------------------

# ------------------------- CONTROLLERS -----------------------------
from controllers.ac_controller import ActorCriticController
from controllers.epsilon_greedy_controller import EpsilonGreedyController
# ------------------------------------------------------------------

# ------------------------- LEARNERS --------------------------------
from learners.reinforce_learner import ReinforceLearner
from learners.biased_reinforce_learner import BiasedReinforceLearner
from learners.actor_critic_learner import ActorCriticLearner
from learners.off_policy_actor_critic_learner import OffpolicyActorCriticLearner
from learners.ppo_learner import PPOLearner
# ------------------------------------------------------------------

from generators.tsp_generator import TSPGenerator
from solvers.gurobi_tsp import solve_tsp

# Get params
params = default_params()

max_nodes_per_graph = 10
rollouts_per_batch = 50
pct_epsilon_anneal_time = 0.75
max_episodes = 3000

# Debugging outputs and plotting during training
params['plot_frequency'] = None
params['plot_train_samples'] = False
params['debug_messages'] = True
# params['print_dots'] = False

# Environment parameters
params['env'] = 'tsp'
params['node_dimension'] = 2
params['max_nodes_per_graph'] = max_nodes_per_graph
params['max_episode_length'] = max_nodes_per_graph + 1
params['diff_sizes'] = True
params['diff_cities'] = True
params['use_training_set'] = False
params['training_sizes'] = [5, 6, 7, 8, 9, max_nodes_per_graph]
# params['num_train_instance_per_size'] = 10

# Runner parameters
params['max_episodes'] = max_episodes
params['max_steps'] = params['max_episodes'] * params['max_episode_length'] * rollouts_per_batch
params['multi_runner'] = False               
# params['parallel_environments'] = 2  

# Exploration parameters
params['epsilon_anneal_time'] =  pct_epsilon_anneal_time * params['max_steps']
params['epsilon_finish'] = 1E-5
params['epsilon_start'] = 1.0
params['epsilon_decay'] = "linear"
params['entropy_regularization'] = True
params['entropy_weight'] = 0.1

# Optimization parameters
params['lr'] = 5E-4
params['gamma'] = 0.99
params['batch_size'] = params['max_episode_length'] * rollouts_per_batch
params['grad_norm_clip'] = 1

# Actor-critic parameters
params['value_loss_param'] = 0.1
params['advantage_bias'] = True
params['advantage_bootstrap'] = True
params['offpolicy_iterations'] = 10
params['value_targets'] = 'td'

# PPO parameters
params['ppo_clipping'] = True
params['ppo_clip_eps'] = 0.1

# Network parameters
params['embedding_dimension'] = 4              
params['encoder_layers'] = 6                
params['model_dimension'] = 512      
params['dimension_k'] = 64
params['dimension_v'] = 64             
params['num_heads'] = 4               
params['num_layers'] = 3 
params['normalization'] = 'batch'
params['feed_forward_hidden'] = 512      
params['embed_dim'] = 4

# Device
# params['device'] = 'cpu'
params['use_tqdm'] = True
params['final_plot'] = False
params['wandb'] = True

print("Epsilon anneal time: ", params['epsilon_anneal_time'])
print("Total transitions: ", params['max_steps'])

params['device'] = "cuda" if th.cuda.is_available() else "cpu"
print("Device in use: ", params['device'])
th.device(params['device'])

if params['wandb']: wandb.init(project="tsp", config=params)

# Run experiment
model = NewTransformer(params=params)
env = EnviornmentTSP(params=params)
experiment = ActorCriticExperiment(params, model, env, PPOLearner(model=model, params=params))
episode_returns, episode_lengths, episode_losses, env_steps = experiment.run()

if params['wandb']: wandb.finish()

avg_gap = experiment.compare_vs_baseline_greedy_rollout(num_episodes=100)
print(f"Average gap: {avg_gap}%")