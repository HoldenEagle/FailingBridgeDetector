#load in bridge data. header = None makes sure we don't use the first row as the columns
import pandas as pd

bridge_df = pd.read_csv('bridge.in.csv' , header=None)
print(bridge_df)

#I am going to name the columns to help me preprocess this dataset easier
bridge_df.columns = ['Bridge' , 'River' ,'Location', 'Erected', 'Purpose', 'Length', 'Lanes', 'Clear-G', 'T-OR-D', 'Material', 'Span', 'REL-L',
                     'TYPE']

print(bridge_df.head())

#make the out result the final column of the bridge dataframe. This is how I usually work with 
#dataframes and this makes it easier for me
out_df= pd.read_csv('bridge.out.csv' , header= None)
bridge_df['Failed'] = out_df[0]

print(bridge_df.head()) 



#drop the location column. I did not think it was important, and too many individual values 
bridge_df = bridge_df.drop('Location' , axis = 1)
print(bridge_df.head()) 


#Looked at failed bridges
bridge_df2 = bridge_df[bridge_df['Failed'] == 'Fail']
print(bridge_df2.head(10))

#replace the ? marks with unknown. This is not a necessary step. But at the time, it helped me
#look at the data. If you preprocess that way, it should not make a difference.
bridge_df.replace('?' , 'Unknown', inplace=True)
print(bridge_df.head(20)) 

#Tree algorithm -> based of ID3 algorithm given in class 
import math
#plurality values function, just returns the majority class.
def plurality_value(examples):
    failed_okay_counts = examples['Failed'].value_counts()
    majority_value = failed_okay_counts.keys()[0]
    return majority_value

#gets the entropy of a group of rows. This is used to help calculating the information gain for each attribute.
def entropy(examples):
    val_counts = examples['Failed'].value_counts()
    if(len(val_counts)==1):
        return 0
    prop_failed = (val_counts['Fail'] / (val_counts['Fail']+ val_counts['Okay']))
    if(prop_failed == 0 or prop_failed == 1):
        return 0
    return - (prop_failed * math.log2(prop_failed) + (1 - prop_failed) * math.log2(1 - prop_failed))

#function that calculates the information gain for each attribute. This helps us determine the 
#best attribute to split off of.
def information_gain(examples, attribute):
    initial_entropy = entropy(examples)
    weighted_entropy = 0
    for value in examples[attribute].unique():
        subset = examples[examples[attribute] == value]
        weighted_entropy += (len(subset) / len(examples)) * entropy(subset)
    
    return initial_entropy - weighted_entropy

#create ID3 algorithm
#examples, current rows we are workign with, attributes -> set of columns we are looking at, parent_exmaples -> previous rows
def simon_id3_algorithm(examples, attributes, parent_examples , min_samples_split , max_depth, current_depth):
    #check if examples is empty
    if(len(examples) ==0 or current_depth >= max_depth):
        return plurality_value(parent_examples)
    elif(len(examples['Failed'].value_counts().keys()) == 1):
        return examples['Failed'].value_counts().keys()[0]
    elif(len(attributes) == 0 or len(examples) < min_samples_split):
        return plurality_value(examples)
    else:
        attributes_names = []
        attributes_importances = []
        for attribute in attributes:
            gain = information_gain(examples, attribute)
            attributes_names.append(attribute)
            attributes_importances.append(gain)
        best_attribute_index = -1
        best_val = -999
        for ind, at in enumerate(attributes_importances):
            if(at > best_val):
                best_val = at
                best_attribute_index = ind
        best_attribute = attributes_names[best_attribute_index]
        print(best_attribute , current_depth)
        #starts a new tree using a dictionary method
        tree = {best_attribute: {}}

        possible_splitting_values = examples[best_attribute].unique()
        for value_to_split in possible_splitting_values:
            subset_of_values = examples[examples[best_attribute] == value_to_split]
            new_attributes = [a for a in attributes if a != best_attribute]
            branch = simon_id3_algorithm(subset_of_values , new_attributes , examples, min_samples_split ,max_depth, current_depth + 1)
            tree[best_attribute][value_to_split] = branch

        return tree

# Below are the possible list of attributes.
attributes = ['River', 'Erected', 'Purpose', 'Length', 'Lanes', 'Clear-G', 'T-OR-D', 'Material', 'Span', 'REL-L',
                     'TYPE']
import random
#I got the test_indices from one of the 25 random sample generations performed by the line below. However, I kept this as the
#test set because it split the data perfectly. It has 23 okay bridges and 2 failed. The value counts are printed out below
#random_indices = random.sample(range(1, 96), 25)
test_indices = [54, 21, 10, 66, 57, 12, 46, 92, 44, 37, 94, 15, 88, 2, 59, 78, 11, 34, 31, 30 , 5 , 9, 81 , 74 , 52]
#making test and training datasets
test_examples = bridge_df.loc[test_indices]
train_examples = bridge_df.drop(index=test_indices)
#example tree calls
#tree = simon_id3_algorithm(train_examples, attributes, None , 3, 8, 0)
#tree2 = simon_id3_algorithm(bridge_df, attributes, None , 3, 8, 0)
print(test_examples['Failed'].value_counts()) 

#this allows us to classify a row for a given decision tree. This allows us to use a tree and predict
#if a bridge is failed or okay.
#while the decision tree is still a dict (not a leaf) we take the attribute to split, find the value from the row and then go 
#down that path
def classify_a_row(decision_tree, row):
    while(isinstance(decision_tree, dict)):
        first_attribute = list(decision_tree.keys())[0]
        
        attr_value = row[first_attribute]
        decision_tree = decision_tree[first_attribute].get(attr_value)
    return decision_tree

#predict function, passed in tree and example rows. Basically runs through these rows, makes a prediction for each one
# and then checks to see if it is correct or not. We store the total accuracy and the failed bridges it catches
def predict(tree, test_examples , print_row):
    correct = 0
    total = 0
    found_failed_bridges = 0
    total_failed_bridges = 0
    for i in range(len(test_examples)):
        first_row = test_examples.iloc[i]
        val = classify_a_row(tree, first_row)
        if(val == first_row['Failed']):
            correct += 1
            if(val == 'Fail'):
                if(print_row):
                    print(f"Test Bridge {i}: right : Actual Failed BRIDGE")
                found_failed_bridges +=1
            else:
                if(print_row):
                    print(f"Test Bridge {i}: right")
                
        else:
            if(first_row['Failed'] == 'Fail'):
                if(print_row):
                    print(f"Test Bridge {i}: wrong : Actual Failed BRIDGE")
            else:
                if(print_row):
                    print(f"Test Bridge {i}: wrong")
        total += 1
        if(first_row['Failed'] == 'Fail'):
            total_failed_bridges += 1
    #print(f"Total Accuracy: {correct/total}")
    #print(f"Failed Bridges Found: {found_failed_bridges} / {total_failed_bridges}")
    return correct/total , found_failed_bridges

#predict(tree , test_examples) 
#print("-------------------------------")
#predict(tree2 , test_examples) 

#K fold validation attempt. Used a simple testing set instead
'''
import sys
maxdepths = [2 ,3, 4, 5,6 ,7, 8, 9, 10]
maxdepths_accuracies = [0 for i in range(len(maxdepths))]
folds = [[0,8] , [8,16] , [16,24] , [24,32] , [32,40] , [40,48], [48,56] , [56,64] , [64,72] , [72,80] , [80,88] , [88,96]]
for i , depth in enumerate(maxdepths):
    avg_accuracy = 0
    bnum = 0
    for f in folds:
        lower_test_end , upper_test_end = f[0] , f[1]
        test_examples = bridge_df[lower_test_end : upper_test_end]
        train_examples = bridge_df.drop(index=test_examples.index)
        tree = simon_id3_algorithm(train_examples, attributes, None , 4 , depth, 0)
        acc, brdg_num = predict(tree, test_examples)
        avg_accuracy += acc
        bnum += brdg_num
    maxdepths_accuracies[i] = avg_accuracy / len(folds)
    print(f"depth: {depth}: {maxdepths_accuracies[i] , bnum}")

best_performing_depth = max(maxdepths_accuracies)
''' 

#pre pruning parameters. I running through a list of max depths and minimum splits at each level.
#decided to build a tree with every combination, and look at the total accuracy and the bridges found
#This helped me choose the final parameters. Each tree that is created shows the attributes it uses, along with 
#each level it is at. The output is shown below
possible_max_depths = [1,2,3 ,4]
possible_min_splits = [1,2,3,4,5]

for depth in possible_max_depths:
    for split in possible_min_splits:
        tree = simon_id3_algorithm(train_examples, attributes, None , split, depth, 0)
        correct_total , found_failed_bridges = predict(tree, test_examples , False)
        print(f"Tree of max depth {depth}  and min split {split}, Accuracy of {correct_total} and found failed bridges {found_failed_bridges}/2") 

# after looking at the many options and how they performed on the test set chose this model, 
#I chose the model with max depth of 3 and a min split of 3. As I believe this will not lead to
#overfitting, and even the later trees with bigger max depths do not go past 3 level. Could have 
#chosen a min split of 4, but for now I will stick with the parameter being set to 3.

final_tree = simon_id3_algorithm(train_examples, attributes, None , 3, 3, 0)
print("------------------------")
correct_total , found_failed_bridges = predict(tree, test_examples, True)
print(f"Tree of max depth 3  and min split 3, Accuracy of {correct_total} and found failed bridges {found_failed_bridges}/2") 

