# Banana-Collector

## Project Details
This project solves the environment provided in the Udacity Reinforcement Learning Nanodegree Program. The system consists of an agent that has to collect as many yellow bananas as possible while avoiding blue bananas. More specifically, a reward of +1 is associated with every yellow banana the agent runs into, while a reward of -1 is given for any collision with a blue banana. The environment is considered solved when the avervage reward over the last 100 episodes reaches 13.
The action space of the environment has four actions (corresponding to going forward, backwards, turning right and left), while the state space has 37 dimensions which represent the presence of objects nearby the agent.

## Dependencies
This project is implemented using Python 3.6, PyTorch, NumPy and the UnityEnvironment package. For the plotting part of the project Matplotlib is used.
### Installation
Download the repository with

```
git clone https://github.com/lbarazza/Banana-Collector
```

To install the environment, download the version corresponding to your operating system

[Linux](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Linux.zip)

[Mac OSX](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana.app.zip)

[Windows (32-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86.zip)

[Windows (64-bit)](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P1/Banana/Banana_Windows_x86_64.zip)

and place the unzipped file in the root of the project folder.

Create a virtual environment to contain the all of the dependencies,

```
conda create --name test4 python=3.6 matplotlib
```

then install all the dependencies

```
pip install -r requirements.txt
```

## Run the Code
To run the code in this repository download/clone the repository and run the main.py file.
