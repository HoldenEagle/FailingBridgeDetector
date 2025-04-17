# FailingBridgeDetector
Built a custom decision tree algorithm mechanism to detect patterns of failing bridges in Pittsburgh using the Pittsburgh bridges dataset from CMU's  Engineering Design Research Center.

The code can be found in both the jupyter notebook file as well as the python file. This custom decision tree algorithm is based off the ID3 algorithm for generating a decision tree, adding a
few extra pre pruning steps such as the max depth of a branch and the minimum samples split at a given node. I tested this tree on 25 datapoints, or about 26 percent of the data. 
There were 8 instances of failed bridges compared to the 88 other instances of non failed bridges. By using the proper pre pruning parameters, the decision tree built from our implementation of 
the ID3 algorithm was able to achieve 96 percent accuracy, while also detecting 2/2 of the failed bridges in the testing set. By using this custom algorithm, we can also display the tree and 
find the attributes and features that lead to bridges in Pittsburgh failing, allowing civil engineers more insight when designing future bridges in Pittsburgh and other parts of the country.

The txt file also describes the dataset. You can run everything through either file, and both files have comments explaining what each step entails.


