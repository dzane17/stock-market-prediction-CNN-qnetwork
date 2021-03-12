# Predicting the Stock Market Using Deep Q-Networks

### Comp_Sci 396/496 Winter 2021, Prof. Han Liu, Northwestern University
### By David Zane and Joshua Zhao


- Based on the Paper: [Global Stock Market Prediction Based on Stock Chart Images Using Deep Q-Network](https://arxiv.org/abs/1902.10948)
- Authors: Jinho Lee, Raehyun Kim, Yookyung Koh, and Jaewoo Kang


- Data that we used to train model in this repository is not uploaded due to licensing. 
- VIDEO PRESENTATION LINK: [YouTube](https://www.youtube.com/)

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


