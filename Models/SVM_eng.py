# Author: Ben Greenawald

import preprocess_eng as pr
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.model_selection import GridSearchCV
from datetime import date
from imblearn.over_sampling import RandomOverSampler
from collections import Counter
import sys

# Find best parameters using Hamas
def main():
    # base_dir = "C:\\Users\\bgree\\Documents\\capstone\\Eng\\exported-features\\"
    # group = "yasir-qadhi"
    # features, response = pr.readData(group, base_dir)
    # print("Train feature shape: " + str(features.shape))
    # print("Train label length: " + str(len(response)))

    # ros = RandomOverSampler(random_state=0)
    # features, response = ros.fit_sample(features, response)
    # print(sorted(Counter(response).items()))
    # rf = SVC(random_state=0)
    # params = {'kernel': ['rbf', 'sigmoid'], 'C':[100]}
    # clf = GridSearchCV(rf, params, verbose=3, n_jobs=4)
    # clf.fit(features, response)
    # for params, mean_score, scores in clf.grid_scores_:
    #     print("%0.3f (+/-%0.03f) for %r"
    #             % (mean_score, scores.std() / 2, params))
    # print()
    # print(clf.best_params_)
    # sys.exit()

    """
    Results
    0.981 (+/-0.005) for {'C': 10, 'kernel': 'rbf'}
    0.979 (+/-0.005) for {'C': 10, 'kernel': 'sigmoid'}
    0.982 (+/-0.005) for {'C': 100, 'kernel': 'rbf'}
    0.982 (+/-0.005) for {'C': 100, 'kernel': 'sigmoid'}
    0.982 (+/-0.005) for {'C': 1000, 'kernel': 'rbf'}
    0.982 (+/-0.005) for {'C': 1000, 'kernel': 'sigmoid'}
    """

    # pr.evaluateGridSearch(clf)


    # From the results we see that the result are pretty much
    # the same across the board, but sigmoud seems slightly
    # better

    # Load in the data
    base_dir = "C:\\Users\\bgree\\Documents\\capstone\\Eng\\exported-features\\"

    # If there is a test argument, write it to file
    if len(sys.argv) > 1:
        # Evaluate the results
        with open("C:\\Users\\bgree/Documents/capstone/Results/SVM/results-{0}.txt".format(str(date.today())), "a+") as file:
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

        # Make classifier using the best parameters, increase depth
        sv = SVC(random_state=0, kernel='sigmoid', C = 100, verbose=True)

        print("Building Classifier")
        sv.fit(features, response)

        # Read in the test data
        test_features, test_response = pr.readData(group, base_dir, train=False, colLen = features.shape[1])
        print("Test feature shape: " + str(test_features.shape))
        print("Test label length: " + str(len(test_response)))
        preds = sv.predict(test_features)

        print(preds)
        print(sum(preds == test_response)/len(preds))
        print(f1_score(test_response, preds, pos_label=test_response[0]))
        # Evaluate the results
        with open("C:\\Users\\bgree/Documents/capstone/Results/SVM/results-{0}.txt".format(str(date.today())), "a+") as file:
            file.write(group + "\n")
            file.write("Accurary: " + str(sum(preds == test_response)/len(preds)) + "\n")
            file.write("F1-Score: " + str(f1_score(test_response, preds, pos_label=test_response[0])) + "\n\n")
            file.close()

if __name__=="__main__":
    main()