# Predicting the Stock Market Using Deep Q-Networks

### Comp_Sci 396/496 Winter 2021, Prof. Han Liu, Northwestern University
### By David Zane and Joshua Zhao


- Based on the Paper: [Global Stock Market Prediction Based on Stock Chart Images Using Deep Q-Network](https://arxiv.org/abs/1902.10948)
- Original GitHub Repository: [DQN-global-stock-market-prediction](https://github.com/lee-jinho/DQN-global-stock-market-prediction)
- Authors: Jinho Lee, Raehyun Kim, Yookyung Koh, and Jaewoo Kang


- Data that we used to train model in this repository is not uploaded due to licensing. 
- VIDEO PRESENTATION LINK: [YouTube](https://youtu.be/qd9x1VhY9mo)

<!---
```python
function test() { 
  console.log("look ma’, no spaces"); 
} 
```

<img src="https://static.wikia.nocookie.net/disney/images/7/74/Profile_-_Buzz_Lightyear.jpeg/revision/latest?cb=20190623020017" width="200" height="200">
--->

A large part of machine learning revolves around reinforcement learning. Reinforcement learning aims to have an agent in an environment reach a goal by maximizing a reward. The goal could be something like winning a game of chess and the reward function could be based on capturing pieces and checkmating. A famous example of reinforcement learning is AlphaGo, which was developed by DeepMind Technologies and beats even the best human Go players across the world.

Because of the wide range of application, reinforcement learning is found in many areas today, ranging from machine learning to game theory to statistics. Here, we will explore an application of Deep Q-Learning, a subfield within reinforcement learning based around deep learning.

Default Q-Learning follows reinforcement learning principles very closely. Essentially, it explores the environment and stores state-action and Q values in a Q-table using the bellman equation. Q-Learning works well and always finds an optimal policy when given a Markov decision process that is not infinite. However, having a very large (or non-finite) amount of state-action pairs will cause problems, such as an unrealistic Q-table size. Deep Q-Learning addresses this issue by instead approximating the Q-table using deep learning. Given an environment state, a deep network will approximate an action and Q-value pair.


## Overview of Paper

Predicting the stock market has long been an application of reinforcement learning and machine learning. The paper [Global Stock Market Prediction Based on Stock Chart Images Using Deep Q-Network](https://arxiv.org/abs/1902.10948) looks at the problem of predicting the stock market using Deep Q-Learning. Here, the authors train a convolutional neural network (CNN) as the approximator for the Q-table. 

The model was trained only using stock market data from the United States, and they found that they were still able to use the model and still obtain returns in the stock markets of other countries. This shows that stock market patterns can be observed as similar even in different countries. Furthermore, the stock markets of countries that have less historical data can still be predicted by models trained on stock markets that have more data and history, such as the United States.

## Datasets

#### Dataset used in paper
The data used for training and testing in the [paper](https://arxiv.org/abs/1902.10948) came from Yahoo Finance. The United States stock market data over 17 years was used to train the model. The model was then tested on stock market data for 30 countries other than the US over 12 years. Some companies in these countries were filtered out if they had no price data or had more than 25% days with zero volume in order to eliminate noise. Daily closing price and volume was used for all countries. 

#### Our dataset
The dataset we used included stock prices of companies in the United States. The stock price and volume for each company was given over 1-minute intervals. We did not use any stock market data from outside the United States, so we instead split training and testing sets using the companies. 

The dataset initially included 86 companies, but we first filtered out companies that only had data from September 5, 2018 to February 21, 2021, removing 22 companies. The remaining 64 companies had data spanning December 30, 2015 to February 21, 2021. Of these companies, 50 were used for training and 14 for testing.

For the data itself, we did not use all 1-minute increments. Instead, we chose to follow the paper and only used the final-minute closing stock price for each day. Total stock volume was also computed by summing up the stock volumes for each 1-minute interval during each day. For all companies, the number of days of data was adjusted to be the same. This number was chosen using the company with the least number of days of data, which ended up being 1293 days of data over the 5 years. 1292 days are used for the X input data, as the Y input data will need a final day to compute.

### Data processing and manipulation

#### Input X data
The convolutional neural network that was trained used inputs of size 32x32x1 of the stock market data over a 32 day period. For each 32-day period, the volume and price was min-max normalized between 0 and 15. In the 32x32x1 matrix, the top 15 rows were used for stock volume, and the bottom 15 were used for price. The two rows between the price and volume data in the matrix are left blank in order to help the network distinguish between price and volume. The image below shows what a 32x32x1 matrix would look like. The two empty middle rows are highlighted:

<img src="https://github.com/dzane17/stock-market-prediction-CNN-qnetwork/blob/main/src/Images_Readme/Sample_32x32x1.PNG" width="400" height="400">

Our data was given in the form of excel spreadsheets for each company. We first filtered these in excel to only contain the closing price, volume, and time for each 1-minute interval. The time was also converted to integer format in excel. The rest of our processing was done in Python and will be described below.

The python packages we used are as follows:

```python
import pandas as pd
import numpy as np
import os
import glob
import copy
```

After initial processing, the spreadsheet data can be loaded in using the pandas and glob packages. The following code loads in the the path for all .csv files in the given folder and then uses pandas to read the first file and print the first 5 rows of data.

```python
filenames = glob.glob("new_hist_price/*.csv")

PATH = filenames[0]
data = pd.read_csv (PATH)
df = pd.DataFrame(data, columns= ['v','c','t'])

print(df.head())
```

<img src="https://github.com/dzane17/stock-market-prediction-CNN-qnetwork/blob/main/src/Images_Readme/example_data_head.PNG" width="400" height="160">

We then convert the pandas dataframe to a numpy array and begin to create the convolutional data. Since we will be using 32x32x1 input data, with 1292 days of data, the data will have a total size of 32x32x(1292-31). We initialize this as an array of zeros.

```python
data = df.to_numpy()[:1292,:]
h = np.shape(data)[0]-31
conv_data = np.zeros((32,32,h))
```

For each 32x32x1 input slice, we then min-max normalize the volume and prices. We multiply the normalized data by 14 to vary the values across 0 and 14 (which is 15 different rows). These values are rounded to the nearest integer. The price values are then shifted by 17 (to be at the bottom of the matrix when added in). The resulting values are the coordinates of the convolutional network inputs that will be 1 instead of 0.

In order to change the correct values to 1 in the 32x32x1 inputs, we then loop over a range of 32, and using the min-max normalized values from before, we set values of the convolutional data to 1. 

The steps to generate the convolutional data are shown below.

```python
for i in range(h):
    vol_min = data[i:i+32, 0].min()
    vol_max = data[i:i+32, 0].max()
    cost_min = data[i:i+32, 1].min()
    cost_max = data[i:i+32, 1].max()

    temp_data = copy.deepcopy(data[i:i+32, :])
    temp_data[:,0] = (temp_data[:,0] - vol_min)/(vol_max - vol_min) * 14
    temp_data[:,1] = (temp_data[:,1] - cost_min)/(cost_max - cost_min) * 14

    temp_data = np.rint(temp_data)
    temp_data[:,0] = (temp_data[:,0] - 14) * -1
    temp_data[:,1] = ((temp_data[:,1] - 14) * -1) + 17

    for j in range(32):
        conv_data[int(temp_data[j,0]),j,i] = 1
        conv_data[int(temp_data[j,1]),j,i] = 1

conv_data = conv_data.astype(np.int)
```

The code reads in text files to use as training inputs, so we then save our data into text format. For the training inputs, companies are split using ‘F’, and the days for each company are split by ‘E’.

```python
with open('inputX_test.txt', 'w') as outfile:
    for i in range(np.shape(conv_data)[2]):
        cut_2d = conv_data[:,:,i]
        np.savetxt(outfile, cut_2d,fmt='%i')
        if i == (np.shape(conv_data)[2]-1):
            outfile.write('F' + "\n")
            break
        outfile.write('E' + "\n")
```

The previous code shows how to generate a text file of training data for a single company. For our training and testing data, we combined the data for all companies into a single text document. This was done by looping through the filenames in the first step. For the Input X data, 50 companies were in the text document for the training data and 14 companies were in the text document for the testing document.

#### Input Y data
For each input 32x32x1, we also need to compute the reward for actions taken on the day. This is computed through the equation below. Here, L<sup>c</sup><sub>t</sub> is the reward for company c on the current day t. Prc<sup>c</sup><sub>t</sub> is the closing price of company c on day t and Prc<sup>c</sup><sub>t+1</sub> is for the day following.

<img src="https://github.com/dzane17/stock-market-prediction-CNN-qnetwork/blob/main/src/Images_Readme/input_Y_equation.PNG" width="500" height="57">

In order to avoid outliers, if Prc<sup>c</sup><sub>t+1</sub> is bound between ± 20% of Prc<sup>c</sup><sub>t</sub>.

When loading in the data for input Y, we include an extra day compared to Input X. This is done similar to before. 

```python
filenames = glob.glob("new_hist_price/*.csv")

PATH = filenames[0]
data = pd.read_csv (PATH)
df = pd.DataFrame(data, columns= ['v','c','t'])
```

We then loop through and compute the Input Y Values.

```python
h = np.shape(data)[0]-32
y_vals = []
twenty_count = 0

for i in range(h):
    cur_cost = data[i+31,1]
    next_cost = data[i+32,1]
    
    if next_cost < cur_cost*0.8:
        next_cost = cur_cost*0.8
        twenty_count += 1
    elif next_cost > cur_cost*1.2:
        next_cost = cur_cost*1.2
        twenty_count += 1
        
    y_vals.append(100 * (next_cost-cur_cost)/cur_cost)

y_vals = np.array(y_vals)
```

Our final step is saving Input Y data into a text file. The values are saved as float values with four decimal places.

```python
with open('inputY_test.txt', 'w') as outfile:
    for i in range(np.shape(y_vals)[0]):
        np.savetxt(outfile, [y_vals[i]],fmt='%1.4f') 
```

Similar to Input X data, we combined all companies into the same text documents. For the Input Y data, 50 companies were in the text document for the training data and 14 companies were in the text document for the testing document.



## Model

We based our model on the [GitHub repository](https://github.com/lee-jinho/DQN-global-stock-market-prediction) provided in the original paper. Our main changes include converting from Python 2 to Python 3, updating TensorFlow version, and creating custom testing/experiment scripts.

The original repository was built on TensorFlow v1.2.0. We converted all scripts, functions, and layers to the latest release TensorFlow v2.4.1. This was required for running on Python 3 and in order to make use of the best available resources.

#### CNN
The input to the CNN model is a 32x32 array which represents the normalized price and volume graph of the stock over the previous 32 days. Output of the CNN is two vectors of length 3. The first vector represented by the letter rho is the action values for each action long, neutral, and short. The second vector is all 0’s except one element set to ‘1’ at the index of the maximum element in vector rho.

The diagram below shows the architecture and inputs of the CNN model. There are four hidden convolutional layers followed by two dense layers. Each convolution is followed by a ReLU and max pooling after the 2nd and 4th conv layers.

<img src="https://github.com/dzane17/stock-market-prediction-CNN-qnetwork/blob/main/src/Images_Readme/cnn_architecture.png" width="800">

#### Deep Q-Network
Since the input state to the q-learning algorithm is complex (graphical image), we use the CNN described above as a function approximator for the deep q-network. The CNN encodes the image into two vectors which are inserted into the q-learning formula. Our model employs two additional techniques- [experience replay](https://arxiv.org/pdf/1902.10948.pdf) and [parameter freezing](https://arxiv.org/pdf/1902.10948.pdf) - to ensure successful integration of the neural net approximator.

## Training

We used the following hyperparameters consistent with the original paper during training.

```python
############################################################################
maxiter         = 5000000       # maximum iteration number          
learning_rate   = 0.00001       # learning rate 
epsilon_min     = 0.1           # minimum epsilon  

W               = 32            # input matrix size 
M               = 1000          # memory buffer capacity 
B               = 10            # parameter theta  update interval                
C               = 1000          # parameter theta^* update interval ( TargetQ ) 
Gamma           = 0.99          # discount factor 
P               = 0             # transaction penalty while training.  0.05 (%) for training, 0 for testing 
Beta            = 32            # batch size
############################################################################
```

Our final model was trained for roughly 8hrs (800k iterations) on the MAGICS Lab cluster.

## Testing

We performed both market neutral portfolio and top/bottom K portfolio experiments from the original paper. The market neutral takes an equivalent long & short position everyday. The top/bottom K experiment only takes positions on the top K and bottom K companies. All test data spanned December 30, 2015 to February 21, 2021.

#### Market Neutral Portfolio Results
|  Num Companies  |  Annual Return  |  Total Return  |
|:---------------:|:---------------:|:--------------:|
|        14       |      20.1%      |     157.0%     |

#### Top/Bottom K Portfolio Results
| Num Companies | K* | Annual Return | Total Return |
|:-------------:|:--:|:-------------:|:------------:|
|       14      |  1 |     83.3%     |    2167.6%   |
|       14      |  2 |     53.1%     |    796.8%    |
|       14      |  3 |     30.6%     |    295.1%    |
|       14      |  4 |     22.9%     |    189.3%    |
|       14      |  5 |     18.8%     |    142.6%    |

*K represents the number of top/bottom companies used. This differs from the paper where K={5%,10%,20%} represents a percentage of companies in the dataset.

## Evaluation

#### vs US Total Market Returns
In both experiments, our model outperformed the total US stock market index for the same time period.

| Total US Market | S&P 500 | Market Neutral | Top/Bottom K |
|:---------------:|:-------:|:--------------:|:------------:|
|      97.5%      |  93.4%  |     157.0%     |    2167.6%   |

These results show that our deep q-network was able to effectively learn patterns in the price and volume chart data. It was able to apply knowledge from 50 companies in the training dataset to 14 new companies it had never seen before in the test dataset. The market neutral portfolio shows that the model can effectively signal [long, neutral, short] positions while the top/bottom k result shows that the model can also accurately predict the magnitude of a company's future gain/loss.

#### vs Paper Results
Our model performed similarly to or better than the original paper results.

In the market neutral experiment, the paper returned US annual gains of 20.64% for 2006-2010, 9.39% for 2010-2014, and 4.11% for 2014-2018. As mentioned above, our model returned an annual gain of 20.1% for Dec 2015 through Feb 2021. Since the time interval for our data does not precisely align with the paper’s data time interval we cannot directly compare the annual return values, but our model performs similarly to the best result from the paper for the market neutral test.

In the top/bottom k experiment the paper returned annual returns of 7.22% for 2006-2010, 21.8% for 2010-2014, and 11.25% for 2014-2018 when averaging results for K=[5,10,20]. The top individual result was 73.27% for 2010-2014 with K=5. As shown in the testing section above, our model achieved 83.3%, 53.1%, 30.6%, 22.9%, and 18.8% for K=1,2,3,4,5. Therefore, our results on the tested time interval meet or exceed those reported in the paper.

