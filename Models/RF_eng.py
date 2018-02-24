# Author: Ben Greenawald

import preprocess_eng as pr
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from datetime import date
from collections import Counter
import sys

# Group used: Hamas



### Hyperparameter optimization using grid search
# base_dir = "C:\\Users\\bgree\\Documents\\capstone\\Eng\\exported-features\\"
# group = "yasir-qadhi"
# features, response = pr.readData(group, base_dir)
# print("Train feature shape: " + str(features.shape))
# print("Train label length: " + str(len(response)))

# ros = RandomOverSampler(random_state=0)
# features, response = ros.fit_sample(features, response)
# print(sorted(Counter(response).items()))
# rf = RandomForestClassifier(random_state=0, n_jobs=6)
# params = {'max_depth': [8, 10], 'n_estimators':[1000, 2000]}
# clf = GridSearchCV(rf, params, verbose=1)
# clf.fit(features, response)
# for params, mean_score, scores in clf.grid_scores_:
#     print("%0.3f (+/-%0.03f) for %r"
#             % (mean_score, scores.std() / 2, params))
# print()
# print(clf.best_params_)
# sys.exit()

"""
Results
0.854 (+/-0.030) for {'max_depth': 2, 'n_estimators': 500}
0.856 (+/-0.030) for {'max_depth': 2, 'n_estimators': 1000}
0.857 (+/-0.030) for {'max_depth': 2, 'n_estimators': 3000}
0.888 (+/-0.025) for {'max_depth': 4, 'n_estimators': 500}
0.889 (+/-0.025) for {'max_depth': 4, 'n_estimators': 1000}
0.891 (+/-0.025) for {'max_depth': 4, 'n_estimators': 3000}
0.913 (+/-0.021) for {'max_depth': 6, 'n_estimators': 500}
0.913 (+/-0.021) for {'max_depth': 6, 'n_estimators': 1000}
0.914 (+/-0.021) for {'max_depth': 6, 'n_estimators': 3000}
0.931 (+/-0.017) for {'max_depth': 8, 'n_estimators': 3000}
0.931 (+/-0.017) for {'max_depth': 8, 'n_estimators': 5000}
0.943 (+/-0.014) for {'max_depth': 10, 'n_estimators': 3000}
0.943 (+/-0.014) for {'max_depth': 10, 'n_estimators': 5000}
0.930 (+/-0.018) for {'max_depth': 8, 'n_estimators': 1000}
0.931 (+/-0.018) for {'max_depth': 8, 'n_estimators': 2000}
0.941 (+/-0.015) for {'max_depth': 10, 'n_estimators': 1000}
0.942 (+/-0.014) for {'max_depth': 10, 'n_estimators': 2000}
"""
# From the results, we see that depth is much more important
# than the number of estimators.

# pr.evaluateGridSearch(clf)

# Load in the data
base_dir = "C:\\Users\\bgree\\Documents\\capstone\\Eng\\exported-features\\"

# If there is a test argument, write it to file
if len(sys.argv) > 1:
    # Evaluate the results
    with open("C:\\Users\\bgree/Documents/capstone/Results/RandomForest/results-{0}.txt".format(str(date.today())), "a+") as file:
        file.write(sys.argv[1] + "\n\n")
        file.close()

eng_base = "C:\\Users\\bgree/Documents/capstone/Eng/exported-features/"

for group in pr.groups:
    features, response = pr.readData(group, base_dir)
    print("Train feature shape: " + str(features.shape))
    print("Train label length: " + str(len(response)))

    ros = RandomOverSampler(random_state=0)
    features, response = ros.fit_sample(features, response)
    print(sorted(Counter(response).items()))

    # If eng base, read in the file list
    if eng_base:
        with open(eng_base + group + "/fileList.txt", "r") as file:
            files = [x for x in file.readlines()]
            file.close()

    # Make classifier using the best parameters, increase depth
    rf = RandomForestClassifier(random_state=0, max_depth=10,
        n_estimators=1000, n_jobs=6)

    print("Building Classifier")
    rf.fit(features, response)

    # Read in the test data
    test_features, test_response = pr.readData(group, base_dir, train=False, colLen = features.shape[1])
    print("Test feature shape: " + str(test_features.shape))
    print("Test label length: " + str(len(test_response)))
    preds = rf.predict(test_features)
    preds_prob = [x[1] for x in rf.predict_proba(test_features)]

    # Write the probabilites to a file
    probs = ""
    for index, prob in enumerate(preds_prob):
        probs += files[index].split("\\")[-1].strip() + ", " + str(prob) + "\n"
    with open(eng_base + group + "/{0}_RF_eng_preds.txt".format(group), "w+") as file:
        file.write(probs)
        file.close()

    print(preds)
    print(sum(preds == test_response)/len(preds))
    print(f1_score(test_response, preds, pos_label=test_response[0]))
    # Evaluate the results
    with open("C:\\Users\\bgree/Documents/capstone/Results/RandomForest/results-{0}.txt".format(str(date.today())), "a+") as file:
        file.write(group + "\n")
        file.write("Accurary: " + str(sum(preds == test_response)/len(preds)) + "\n")
        file.write("F1-Score: " + str(f1_score(test_response, preds, pos_label=test_response[0])) + "\n\n")
        file.close()

    # Clean up the environment
    del features, response, test_features, test_response