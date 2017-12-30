# Author: Ben Greenawald

# Translating the Python code and implementing Random Forest and XGBoost

# Import Statement

import pandas as pd
import scipy.sparse as sp
import numpy as np
import sys
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import f1_score

def readData(groupName, base_dir, train=True, colLen = None):

    if train:
        print("Reading in train features for " + groupName)
        filename = "{0}/{1}TrainFeatures.txt".format(groupName, groupName)
    else:
        print("Reading in test features for " + groupName)
        filename = "{0}/{1}TestFeatures.txt".format(groupName, groupName)

    with open(base_dir + filename, "r") as Features:
        rows = np.array([int(x.strip()) for x in Features.readline().strip().split(",")])
        cols = np.array([int(x.strip()) for x in Features.readline().strip().split(",")])
        vals = np.array([float(x.strip()) for x in Features.readline().strip().split(", ")])
        Features.close()

    
    row_len = max(rows) + 1
    
    if colLen:
        col_len = colLen
    else:
        col_len = max(cols) + 1

    features = pd.DataFrame(sp.coo_matrix((vals, (rows, cols)), shape=(row_len, col_len)).toarray())
    del rows, cols, vals

    if train:
        print("Reading in train labels for " + groupName)
        filename = "{0}/{1}TrainLabels.txt".format(groupName, groupName)
    else:
        print("Reading in test labels for " + groupName)
        filename = "{0}/{1}TestLabels.txt".format(groupName, groupName)

    # Read in the labels
    with open(base_dir + filename, "r") as Labels:
        labels = np.array([int(float(x.strip())) for x in Labels.readline().strip().split(", ")])
        Labels.close()

    return((features, labels))

# Load in the data
base_dir = "C:/Users/Ben/Documents/Capstone/Models/exported-features/"
group = "CNN"
features, response = readData(group, base_dir)
print("Train feature shape: " + str(features.shape))
print("Train label length: " + str(len(response)))

# Build the classifier
print("Building Classifier")
clf = RandomForestClassifier(max_depth=4, random_state=0, n_estimators=1000)
clf.fit(features, response)

# Read in the test data
test_features, test_response = readData(group, base_dir, train=False, colLen = features.shape[1])
print("Test feature shape: " + str(test_features.shape))
print("Test label length: " + str(len(test_response)))
preds = clf.predict(test_features)
print(preds)

# Evaluate the results
print(sum(preds == test_response)/len(preds))