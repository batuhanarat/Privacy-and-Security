import sys
import random

import numpy as np
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns


from collections import Counter

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors



###############################################################################
############################# Label Flipping ##################################
###############################################################################
def attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p):
    """
    Performs a label flipping attack on the training data.

    Parameters:
    X_train: Training features
    X_test: Testing features
    y_train: Training labels
    y_test: Testing labels
    model_type: Type of model ('DT', 'LR', 'SVC')
    p: Proportion of labels to flip

    Returns:
    Accuracy of the model trained on the modified dataset
    """
    avg_acc= 0
    for trial in range(100):
        index_count = int(len(X_train) * p)
        selected_indexes = random.sample(range(len(X_train)), index_count)
        y_train_flipped = [1 - y_train[j] if j in selected_indexes else y_train[j] for j in range(len(y_train))]

        if(model_type == "DT"):
            myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
            myDEC.fit(X_train, y_train_flipped)
            DEC_predict = myDEC.predict(X_test)
            avg_acc += accuracy_score(y_test, DEC_predict)
        elif(model_type == "LR"):
            myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
            myLR.fit(X_train, y_train_flipped)
            LR_predict = myLR.predict(X_test)
            avg_acc += accuracy_score(y_test, LR_predict)
        elif(model_type == "SVC"):
            mySVC = SVC(C=0.5, kernel='poly', random_state=0)
            mySVC.fit(X_train, y_train_flipped)
            SVC_predict = mySVC.predict(X_test)
            avg_acc += accuracy_score(y_test, SVC_predict)
        else:
            print("Something is wrong.")

    avg_acc = avg_acc/100
    # TODO: You need to implement this function!
    # Implementation of label flipping attack
    return avg_acc


###############################################################################
########################### Label Flipping Defense ############################
###############################################################################

def label_flipping_defense(X_train, y_train, p):
    """
    Performs a label flipping attack, applies outlier detection, and evaluates the effectiveness of outlier detection.

    Parameters:
    X_train: Training features
    y_train: Training labels
    p: Proportion of labels to flip

    Prints:
    A message indicating how many of the flipped data points were detected as outliers
    """
    # TODO: You need to implement this function!
    # Perform the attack, then the defense, then print the outcome
    ##Performing attack
    index_count = int(len(X_train) * p)
    selected_indexes = random.sample(range(len(X_train)), index_count)
    y_train_flipped = [1 - y_train[j] if j in selected_indexes else y_train[j] for j in range(len(y_train))]
    ##Performing defence
    neighbors = NearestNeighbors(n_neighbors=15, algorithm='ball_tree').fit(X_train)
    distances, indices = neighbors.kneighbors(X_train)

    successful = 0
    for i in range(len(indices)):
        count_of_0 = 0
        count_of_1 = 0

        for j in range(15):
            if y_train_flipped[indices[i][j]] == 0:
                count_of_0= count_of_0+1
            else:
                count_of_1 = count_of_1 +1

        # Check the label that dominates in the neighborhood
        dominant_label = 0 if count_of_0 > count_of_1 else 1

        # Check if the dominant label is flipped and above the threshold
        if dominant_label == 0 and count_of_0 > 8 and y_train_flipped[indices[i][0]] == 1  and indices[i][0] in selected_indexes:
            successful += 1
        elif dominant_label == 1 and count_of_1 > 8 and y_train_flipped[indices[i][0]] == 0 and indices[i][0] in selected_indexes:
            successful += 1

    print(f"Out of {index_count} flipped data points, {successful} were correctly identified.")


###############################################################################
############################# Evasion Attack ##################################
###############################################################################
def evade_model(trained_model, actual_example):
    """
    Attempts to create an adversarial example that evades detection.

    Parameters:
    trained_model: The machine learning model to evade
    actual_example: The original example to be modified

    Returns:
    modified_example: An example crafted to evade the trained model
    """
    actual_class = trained_model.predict([actual_example])[0]
    modified_example = copy.deepcopy(actual_example)
    maximum_list = [6.8248,12.9516]
    minimum_list = [-7.0421, -13.7731]
    pred_class = trained_model.predict([modified_example])[0]
    while pred_class == actual_class:
        for i in range(2):
            if actual_class == 1:
                if modified_example[i] < maximum_list[i] and (i == 0 or i == 1):
                    modified_example[i] += 0.1
                    pred_class = trained_model.predict([modified_example])[0]
            elif actual_class == 0:
                if modified_example[i] > minimum_list[i] and (i == 0 or i == 1):
                    modified_example[i] -= 0.1
                    pred_class = trained_model.predict([modified_example])[0]

    return modified_example


def calc_perturbation(actual_example, adversarial_example):
    """
    Calculates the perturbation added to the original example.

    Parameters:
    actual_example: The original example
    adversarial_example: The modified (adversarial) example

    Returns:
    The average perturbation across all features
    """
    # You do not need to modify this function.
    if len(actual_example) != len(adversarial_example):
        print("Number of features is different, cannot calculate perturbation amount.")
        return -999
    else:
        tot = 0.0
        for i in range(len(actual_example)):
            tot = tot + abs(actual_example[i] - adversarial_example[i])
        return tot / len(actual_example)


###############################################################################
########################## Transferability ####################################
###############################################################################

def evaluate_transferability(DTmodel, LRmodel, SVCmodel, actual_examples):
    """
    Evaluates the transferability of adversarial examples.

    Parameters:
    DTmodel: Decision Tree model
    LRmodel: Logistic Regression model
    SVCmodel: Support Vector Classifier model
    actual_examples: Examples to test for transferability

    Returns:
    Transferability metrics or outcomes
    """
    # TODO: You need to implement this function!
    # Implementation of transferability evaluation
    # Initialize counters
    dt_lr = 0
    dt_svc = 0
    lr_dt = 0
    lr_svc = 0
    svc_lr = 0
    svc_dt = 0

    DTadvExamples = [evade_model(DTmodel, i) for i in actual_examples]
    LRadvExamples = [evade_model(LRmodel, i) for i in actual_examples]
    SVCadvExamples = [evade_model(SVCmodel, i) for i in actual_examples]

    for i in DTadvExamples:
        dt_lr += sum([DTmodel.predict([i])[0] == LRmodel.predict([i])[0]])
        dt_svc += sum([DTmodel.predict([i])[0] == SVCmodel.predict([i])[0]])

    for i in LRadvExamples:
        lr_dt += sum([LRmodel.predict([i])[0] == DTmodel.predict([i])[0]])
        lr_svc += sum([LRmodel.predict([i])[0] == SVCmodel.predict([i])[0]])

    for i in SVCadvExamples:
        svc_lr += sum([SVCmodel.predict([i])[0] == LRmodel.predict([i])[0]])
        svc_dt += sum([SVCmodel.predict([i])[0] == DTmodel.predict([i])[0]])

    # Extract the counts for each combination


    print("Out of 40 adversarial examples crafted to evade DT :")
    print(f"-> {dt_lr} of them transfer to LR.")
    print(f"-> {dt_svc} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade LR :")
    print(f"-> {lr_dt} of them transfer to DT.")
    print(f"-> {lr_svc} of them transfer to SVC.")

    print("Out of 40 adversarial examples crafted to evade SVC :")
    print(f"-> {svc_lr} of them transfer to DT.")
    print(f"-> {svc_dt} of them transfer to LR.")



###############################################################################
############################### Main ##########################################
###############################################################################

## DO NOT MODIFY CODE BELOW THIS LINE. FEATURES, TRAIN/TEST SPLIT SIZES, ETC. SHOULD STAY THIS WAY. ##
## JUST COMMENT OR UNCOMMENT PARTS YOU NEED. ##
def main():
    data_filename = "BankNote_Authentication.csv"
    features = ["variance", "skewness", "curtosis", "entropy"]

    df = pd.read_csv(data_filename)
    df = df.dropna(axis=0, how='any')
    y = df["class"].values
    y = LabelEncoder().fit_transform(y)
    X = df[features].values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=0)


    # Raw model accuracies:
    print("#" * 50)
    print("Raw model accuracies:")

    # Model 1: Decision Tree
    myDEC = DecisionTreeClassifier(max_depth=5, random_state=0)
    myDEC.fit(X_train, y_train)
    DEC_predict = myDEC.predict(X_test)
    print('Accuracy of decision tree: ' + str(accuracy_score(y_test, DEC_predict)))

    # Model 2: Logistic Regression
    myLR = LogisticRegression(penalty='l2', tol=0.001, C=0.1, max_iter=100)
    myLR.fit(X_train, y_train)
    LR_predict = myLR.predict(X_test)
    print('Accuracy of logistic regression: ' + str(accuracy_score(y_test, LR_predict)))

    # Model 3: Support Vector Classifier
    mySVC = SVC(C=0.5, kernel='poly', random_state=0)
    mySVC.fit(X_train, y_train)
    SVC_predict = mySVC.predict(X_test)
    print('Accuracy of SVC: ' + str(accuracy_score(y_test, SVC_predict)))

    # Label flipping attack executions:
    print("#"*50)
    print("Label flipping attack executions:")
    model_types = ["DT", "LR", "SVC"]
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for model_type in model_types:
        for p in p_vals:
            acc = attack_label_flipping(X_train, X_test, y_train, y_test, model_type, p)
            print("Accuracy of poisoned", model_type, str(p), ":", acc)

    # Label flipping defense executions:
    print("#" * 50)
    print("Label flipping defense executions:")
    p_vals = [0.05, 0.10, 0.20, 0.40]
    for p in p_vals:
        print("Results with p=", str(p), ":")
        label_flipping_defense(X_train, y_train, p)

    #Correlation matrix and heat map
    correlation_matrix = df.corr()
    print(correlation_matrix)
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
    plt.show()

    # Find maximum value for the 'variance' column
    max_variance = df['variance'].max()

    # Find minimum value for the 'variance' column
    min_variance = df['variance'].min()

    # Find maximum value for the 'skewness' column
    max_skewness = df['skewness'].max()

    # Find minimum value for the 'skewness' column
    min_skewness = df['skewness'].min()



    print(max_variance)
    print(min_variance)

    print(max_skewness)
    print(min_skewness)

    # Evasion attack executions:
    print("#"*50)
    print("Evasion attack executions:")
    trained_models = [myDEC, myLR, mySVC]
    model_types = ["DT", "LR", "SVC"]
    num_examples = 40
    for a, trained_model in enumerate(trained_models):
        total_perturb = 0.0
        for i in range(num_examples):
            actual_example = X_test[i]
            adversarial_example = evade_model(trained_model, actual_example)
            if trained_model.predict([actual_example])[0] == trained_model.predict([adversarial_example])[0]:
                print("Evasion attack not successful! Check function: evade_model.")
            perturbation_amount = calc_perturbation(actual_example, adversarial_example)
            total_perturb = total_perturb + perturbation_amount
        print("Avg perturbation for evasion attack using", model_types[a], ":", total_perturb / num_examples)

    # Transferability of evasion attacks:
    print("#"*50)
    print("Transferability of evasion attacks:")
    trained_models = [myDEC, myLR, mySVC]
    num_examples = 40
    evaluate_transferability(myDEC, myLR, mySVC, X_test[0:num_examples])



if __name__ == "__main__":
    main()


