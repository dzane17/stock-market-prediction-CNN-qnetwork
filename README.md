# Predicting the Stock Market Using Deep Q-Networks

### Comp_Sci 396/496 Winter 2021, Prof. Han Liu, Northwestern University
### By David Zane and Joshua Zhao


- Based on the Paper: [Global Stock Market Prediction Based on Stock Chart Images Using Deep Q-Network](https://arxiv.org/abs/1902.10948)
- Authors: Jinho Lee, Raehyun Kim, Yookyung Koh, and Jaewoo Kang


- Data that we used to train model in this repository is not uploaded due to licensing. 

<!---
```python
function test() { 
  console.log("look ma’, no spaces"); 
} 
```
--->

A large part of machine learning revolves around reinforcement learning. Generally, reinforcement learning is having some agent in an environment maximize its reward. This reward will be used to help the agent reach some goal, which could be virtually anything from winning a game of chess to driving a car safely. A famous example of reinforcement learning is AlphaGo, which was developed by DeepMind Technologies and beats even the best human Go players across the world.

Because of the wide range of application, reinforcement learning is found in many areas today, ranging from machine learning to game theory to statistics. Here, we will explore an application of Deep Q-Learning, a subfield within reinforcement learning based around deep learning.

Default Q-Learning follows reinforcement learning principles very closely. Essentially, it explores the environment and stores state-action and Q values in a Q-table using the bellman equation. Q-Learning works well and always finds an optimal policy when given a Markov decision process that is not infinite. However, having a very large (or non-finite) amount of state-action pairs will cause problems, such as an unrealistic Q-table size. Deep Q-Learning addresses this issue by instead approximating the Q-table using deep learning. Given an environment state, a deep network will approximate an action and Q-value pair.


## Overview of Paper

Predicting the stock market has long been an application of reinforcement learning and machine learning. The paper [Global Stock Market Prediction Based on Stock Chart Images Using Deep Q-Network](https://arxiv.org/abs/1902.10948) looks at the problem of predicting the stock market using Deep Q-Learning. Here, the authors train a convolutional neural network (CNN) as the approximator for the Q-table. 

The model was trained only using stock market data from the United States, and they found that they were still able to use the model and still obtain returns in the stock markets of other countries. This shows that stock market patterns can be observed as similar even in different countries. Furthermore, the stock markets of countries that have less historical data can still be predicted by models trained on stock markets that have more data and history, such as the United States.


<img src="https://static.wikia.nocookie.net/disney/images/7/74/Profile_-_Buzz_Lightyear.jpeg/revision/latest?cb=20190623020017" width="200" height="200">

