
## README for NP-Hard Problem Solving with Deep Reinforcement Learning

### Overview
This repository contains the code and experimental setups for the bachelor thesis titled **"Search & Analysis of New Heuristics for Solving NP-Hard Problems with Deep Reinforcement Learning"** by Tomàs Osarte Segura. The thesis explores the potential of Deep Reinforcement Learning (DRL) in developing heuristic solutions for NP-Hard (NPH) problems, offering insights into various DRL techniques and novel approaches to enhance their effectiveness.

### Repository Structure

Inside the tfg folder:

- **controllers/**: Contains the controller logic for managing the RL environment and agents.
- **environments/**: Definitions of various environments used for training and evaluation.
- **experiments/**: Configurations and scripts for setting up and running different experiments.
- **generators/**: Utilities for generating problem instances.
- **jupyter_experiments/**: Jupyter notebooks for interactive experimentation.
- **learners/**: Implementation of different learning algorithms.
- **Models/**: Pre-trained models and model definitions.
- **networks/**: Neural network architectures used in the experiments.
- **runners/**: Scripts to run training and evaluation sessions.
- **solvers/**: Solver implementations for comparison with DRL approaches.
- **training/**: Training scripts and configurations.
- **utils/**: Utility functions and classes.
- **Dockerfile**: Docker configuration for containerized setup.
- **pyproject.toml**: Configuration file for Poetry.
- **poetry.lock**: Lock file for Poetry dependencies.
- **README.md**: This file.

### Getting Started

#### Prerequisites
- Python 3.x
- Poetry

#### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/nphard-drl.git
   cd tfg
   ```
2. Install Poetry if you haven't already:
   ```bash
   pip install poetry
   ```
3. Install the dependencies and create the virtual environment:
   ```bash
   poetry install
   ```
4. Activate the virtual environment:
   ```bash
   poetry shell
   ```

### Usage
```bash
python3 run.py
```
Tune the parameters inside the folder as you see fit. Thos are explained in the `params.py`file.

### Experiments
Detailed experiments and how to replicate them can be found in the `jupyter_experiments` directory. Each experiment is a `.ipynb` file with a different purpose to test. really esasy to replicate.

### Acknowledgements
Special thanks to my thesis supervisor, Sergio Álvarez Napagao, for his invaluable guidance and support, and to Manuel Romero for his crucial assistance during the development and training stages.

### Contact
For any questions or further information, please contact Tomàs Osarte Segura at [tomasosarte@gmail.com].