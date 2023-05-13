#-------------------------------------------------------------------------
# AUTHOR: Pablo Duenas
# FILENAME: association_rule_mining.py
# SPECIFICATION: Using the Apriori algorithm to find frequent itemsets and association rules
# FOR: CS 4210- Assignment #5
# TIME SPENT: 2 hours
#-------------------------------------------------------------------------*/

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mlxtend.frequent_patterns import apriori, association_rules

#Use the command: "pip install mlxtend" on your terminal to install the mlxtend library

#read the dataset using pandas
df = pd.read_csv('retail_dataset.csv', sep=',')

#find the unique items all over the data and store them in the set below
itemset = set()
for i in range(0, len(df.columns)):
    items = (df[str(i)].unique())
    itemset = itemset.union(set(items))

#remove nan (empty) values by using:
itemset.remove(np.nan)

#To make use of the apriori module given by mlxtend library, we need to convert the dataset accordingly. Apriori module requires a
# dataframe that has either 0 and 1 or True and False as data.
#Example:

#Bread Wine Eggs
#1     0    1
#0     1    1
#1     1    1

#To do that, create a dictionary (labels) for each transaction, store the corresponding values for each item (e.g., {'Bread': 0, 'Milk': 1}) in that transaction,
#and when is completed, append the dictionary to the list encoded_vals below (this is done for each transaction)

encoded_vals = []
for index, row in df.iterrows():
    labels = {}
    for item in itemset:
        if str(item) in str(row):
            labels[str(item)] = 1
        else:
            labels[str(item)] = 0
    encoded_vals.append(labels)

#adding the populated list with multiple dictionaries to a data frame
ohe_df = pd.DataFrame(encoded_vals)

# convert to boolean values
ohe_df = ohe_df.astype('bool')

#calling the apriori algorithm informing some parameters
freq_items = apriori(ohe_df, min_support=0.2, use_colnames=True, verbose=1)
rules = association_rules(freq_items, metric="confidence", min_threshold=0.6)

#iterate the rules data frame and print the apriori algorithm results by using the following format:
for index, row in rules.iterrows():
    antecedent = set(row['antecedents'])
    consequent = set(row['consequents'])
    support = row['support']
    confidence = row['confidence']
    
    # check that denominator is not zero
    if support > 0:
        support_count = sum(ohe_df[list(antecedent)].apply(lambda x: all(x), axis=1))
        prior = support_count / len(encoded_vals)
        gain = (100 * (confidence - prior) / prior)
        print(', '.join(antecedent), "->", ', '.join(consequent))
        print("Support: " + str(support))
        print("Confidence: " + str(confidence))
        print("Prior: " + str(prior))
        print("Gain in Confidence: " + str(gain))
        print()


#Finally, plot support x confidence
plt.scatter(rules['support'], rules['confidence'], alpha=0.5)
plt.xlabel('support')
plt.ylabel('confidence')
plt.title('Support vs Confidence')
plt.show()
