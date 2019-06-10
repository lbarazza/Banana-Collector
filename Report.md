# Report

## Deep Q-Learning
Q-learning is a very popular algorithm for reinforcement learning, but it has various limitations. Just like its siblings, Sarsa and Expected Sarsa, it has difficulties in dealing with continous state spaces. One way to overcome this is to, discard the old tabular rapresentation of the Q-function and, instead, try to approximate it directly, which can be done through function approximators.
The use of neural networks as these approximators seems promising at first, except for the fact that they don't want to work with these temporal differnece learning methods we are using. If we directly apply them to the Q-learning algorithm, we find that this new learning method is extremely unstable. Because we are now using neural netowrks, when we update the Q-function at one state-action pair, the neural network is going to change the value of all the other state-action pairs nearby as well. This makes it hard for us to utilize current predictions for updating the network (temporal difference learning), because every time we update the netowrk our target changes as well, creating a sort of constantly changing target which is unstable.
One way to address this problem of constantly changing targets called Fixed Q-Targets, is to 'fix' the target at one point for a while and update it less frequently than the normal network.
Another way to stabilize the learning of the agent, even though it doesn't address the same problem describe above, is a technique called experience replay. In experience replay, instead of learning the current experience on policy, the agent stores the experiences it had in a replay buffer from which it then samples a minibatch uniformly at random for training. This allows for the harmful correlations between subsequent experiences to be broken and thus results in better learning.
This repository implements the Double DQN improvement of DQN, which changes the rule for calculating targets for updating the network by using a combination of both the local and the target network. In the comments there is also a raw implementation of prioritized experience replay (without importance sampling).

## Deep Q-Network
The deep Q-Network used is a 2-hidden layer network with each hidden layer having 64 neurons. It uses a ReLU activation function after each of the hidden layers as well as a batch normalization layer after the first hidden layer. The input layer is of size 37 as the state space and the output layer is of size 4 as the action space.

## Hyperparamters
An informal search over the hyperparaters has been conducted with the following results:

|     Hyperparamter             |      Value                      |
|-------------------------------|:-------------------------------:|
|    epsilon start              |          1.0                    |
|    epsilon end                |          0.1                    |
|    epsilon decay frames       |          100 000                |
|    gamma                      |          0.99                   |
|    target net update rate     |          5                      |
|    tau                        |          0.001                  |
|    replay buffer length       |          500 000                |
|    batch size                 |          256                    |
|    replay buffer start size   |          1000                   |

## Further Improvements
To further improve the algorithm some modifications can be made such as prioritized experience replay with importance sampling and a dueling network architecture. Other improvements could be Noisy DQN,  Distributional DQN and multi-step bootstrap targets (components of the Rainbow algorithm).






