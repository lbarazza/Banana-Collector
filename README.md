# Banana-Collector

## Project Details
This project solves the environment provided in the Udacity Reinforcement Learning Nanodegree Program. The system consists of an agent that has to collect as many yellow bananas as possible while avoiding blue bananas. More specifically, a reward of +1 is associated with every yellow banana the agent runs into, while a reward of -1 is given for any collision with a blue banana. The environment is considered solved when the avervage reward over the last 100 episodes reaches 13.
The action space of the environment has four actions (corresponding to going forward, backwards, turning right and left), while the state space has 37 dimensions which represent the presence of objects nearby the agent.

## Dependencies
This project is implemented using Python 3.6, PyTorch, NumPy and the UnityEnvironment package. For the plotting part of the project Matplotlib is used.
### Installation

To install all of the dependencies, first clone the Unity ML agents repository,

'''
git clone https://github.com/Unity-Technologies/ml-agents.git
git -C ml-agents checkout 0.4.0b
pip install ml-agents/python/.
'''

and then download all of the other requirements,

'''
pip install -r requirements.txt
'''

## Run the Code
To run the code in this repository download/clone the repository and run the main.py file.
