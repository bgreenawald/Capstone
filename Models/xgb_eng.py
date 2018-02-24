# Author: Ben Greenawald

import preprocess_eng as pr
import xgboost as xgb
from sklearn.metrics import f1_score
from datetime import date
import numpy as np
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import pandas as pd
import sys

# base_dir = "C:\\Users\\bgree\\Documents\\capstone\\Eng\\exported-features\\"
# group = "yasir-qadhi"
# features, response = pr.readData(group, base_dir)
# print("Train feature shape: " + str(features.shape))
# print("Train label length: " + str(len(response)))

# ros = RandomOverSampler(random_state=0)
# features, response = ros.fit_sample(features, response)
# print(sorted(Counter(response).items()))
# xg_train = xgb.DMatrix(features, label = response)
# del features, response
# n_folds = 3
# early_stopping = 50
# params = {'max_depth': 10, 'n_estimators':1000,
# 'learning_rate':0.1, 'objective': 'binary:logistic', 'subsample':0.7}
# cv = xgb.cv(params, xg_train, nfold=n_folds, early_stopping_rounds=early_stopping, verbose_eval=1)
# print(cv)
# sys.exit()

'''
Depth: [6,8,10]
Estimators: [100,500,1000]
Learning rate: [0.1,0.5,1]
Depth: 6, Learning Rate:1 , NTrees: 100      0.025215
Depth: 8, Learning Rate:1 , NTrees: 100      0.0206537
Depth: 10, Learning Rate:1 , NTrees: 100     0.0184997
Depth: 10, Learning Rate:1 , NTrees: 1000    0.0184997
Depth: 10, Learning Rate:0.5 , NTrees: 1000   0.0183723
Depth: 10, Learning Rate:0.1 , NTrees: 1000   0.039407

'''
# From the results, we see that depth is much more important
# than the number of estimators.

# Load in the data
base_dir = "C:\\Users\\bgree\\Documents\\capstone\\Eng\\exported-features\\"

# If there is a test argument, write it to file
if len(sys.argv) > 1:
    # Evaluate the results
    with open("C:\\Users\\bgree/Documents/capstone/Results/XGBoost/results-{0}.txt".format(str(date.today())), "a+") as file:
        file.write(sys.argv[1] + "\n\n")
        file.close()

eng_base = "C:\\Users\\bgree/Documents/capstone/Eng/exported-features/"

for group in pr.groups:
    features, response = pr.readData(group, base_dir)
    print("Train feature shape: " + str(features.shape))
    print("Train label length: " + str(len(response)))

    # If eng base, read in the file list
    if eng_base:
        with open(eng_base + group + "/fileList.txt", "r") as file:
            files = [x for x in file.readlines()]
            file.close()

    ros = RandomOverSampler(random_state=0)
    features, response = ros.fit_sample(features, response)
    print(sorted(Counter(response).items()))

    features = pd.DataFrame(features)
    response = np.array(response)

    xg_train = xgb.DMatrix(features, label = response)
    params = {'max_depth': 10, 'nthread':6, 'n_estimators':1000,
    'learning_rate':0.5, 'objective': 'binary:logistic'}
    bst = xgb.train(params, xg_train, verbose_eval = 0)

    # Read in the test data
    test_features, test_response = pr.readData(group, base_dir, train=False, colLen = features.shape[1])
    print("Test feature shape: " + str(test_features.shape))
    print("Test label length: " + str(len(test_response)))

    transformed_features = xgb.DMatrix(test_features)

    preds1 = bst.predict(transformed_features)
    preds = np.array(preds1)
    preds[preds >= 0.5] = 1
    preds[preds < 0.5] = 0

    # Write the probabilites to a file
    probs = ""
    for index, prob in enumerate(preds1):
        probs += files[index].split("\\")[-1].strip() + ", " + str(prob) + "\n"
    with open(eng_base + group + "/{0}_XGB_eng_preds.txt".format(group), "w+") as file:
        file.write(probs)
        file.close()

    print(preds)
    print(sum(preds == test_response)/len(preds))
    print(f1_score(test_response, preds, pos_label=test_response[0]))
    # Evaluate the results
    with open("C:\\Users\\bgree/Documents/capstone/Results/XGBoost/results-{0}.txt".format(str(date.today())), "a+") as file:
        file.write(group + "\n")
        file.write("Accurary: " + str(sum(preds == test_response)/len(preds)) + "\n")
        file.write("F1-Score: " + str(f1_score(test_response, preds, pos_label=test_response[0])) + "\n\n")
        file.close()
