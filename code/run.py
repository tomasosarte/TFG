import torch as th
import wandb

from environments.environment_tsp import EnviornmentTSP
from networks.more_basic_net import MoreBasicNetwork
from params import default_params
from experiments.actor_critic_experiment import ActorCriticExperiment

# ------------------------- LEARNERS --------------------------------
from learners.reinforce_learner import ReinforceLearner
from learners.biased_reinforce_learner import BiasedReinforceLearner
from learners.actor_critic_learner import ActorCriticLearner
from learners.off_policy_actor_critic_learner import OffpolicyActorCriticLearner
from learners.ppo_learner import PPOLearner
# ------------------------------------------------------------------

from generators.tsp_generator import TSPGenerator
from exact_solvers.solver_tsp import solve_tsp

# Get params
params = default_params()

rollouts_per_batch = 50
pct_epsilon_anneal_time = 0.75
max_episodes = 750

max_nodes_per_graph = 10
params['problem'] = 'tsp'
params['node_dimension'] = 2
params['max_nodes_per_graph'] = max_nodes_per_graph
params['max_episode_length'] = max_nodes_per_graph + 1
params['max_episodes'] = max_episodes
params['batch_size'] = params['max_episode_length'] * rollouts_per_batch
params['max_steps'] = params['max_episodes'] * params['max_episode_length'] * rollouts_per_batch
params['epsilon_start'] = 0.9
params['epsilon_finish'] = 1E-5
params['epsilon_anneal_time'] =  pct_epsilon_anneal_time * params['max_steps']
params['lr'] = 5E-4
params['gamma'] = 0.99
params['entropy_regularization'] = True
params['entropy_weight'] = 0.1
params['plot_frequency'] = None
params['plot_train_samples'] = False
params['debug_messages'] = True
params['use_tqdm'] = True
params['final_plot'] = False
params['wandb'] = True

# params['device'] = 'cuda' if th.cuda.is_available() else 'cpu'
print("Device in use: ", params['device'])
th.set_default_device(params['device'])

# Get instance
instance = 0
num_nodes = 10
cities = th.load(f"training/tsp/size_{num_nodes}/instance_{instance}.pt") 
cities = cities.to(params['device'])

if params['wandb']: wandb.init(project="tsp", config=params)

# Run experiment
model = MoreBasicNetwork(params)
env = EnviornmentTSP(cities, params)
experiment = ActorCriticExperiment(params, model, env, ReinforceLearner(model, params))
episode_returns, episode_lengths, episode_losses, env_steps = experiment.run()

if params['wandb']: wandb.finish()