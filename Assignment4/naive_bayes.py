#-------------------------------------------------------------------------
# AUTHOR: Brandon Diep
# FILENAME: naive_bayes.py
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #4
# TIME SPENT: 50 mins
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

s_values = [0.1, 0.001, 0.0001, 0.00001, 0.000001, 0.0000001, 0.00000001, 0.000000001, 0.0000000001]

#reading the training data
#--> add your Python code here
train_data = pd.read_csv('weather_training.csv')
X_training = np.array(train_data.values)[:, 1:-1].astype('f')
y_training = np.array(train_data.values)[:, -1].astype('f')

#update the training class values according to the discretization (11 values only)
#--> add your Python code here
y_training = np.digitize(y_training, classes, right=True)

#reading the test data
#--> add your Python code here
test_data = pd.read_csv('weather_test.csv')
X_test = np.array(test_data.values)[:, 1:-1].astype('f')
y_test = np.array(test_data.values)[:, -1].astype('f')

#update the test class values according to the discretization (11 values only)
#--> add your Python code here
y_test = np.digitize(y_test, classes, right=True)


highest_accuracy = 0

#loop over the hyperparameter value (s)
#--> add your Python code here

for s in s_values:

    #fitting the naive_bayes to the data
    clf = GaussianNB(var_smoothing=s)
    clf = clf.fit(X_training, y_training)

    correct = 0


    #make the naive_bayes prediction for each test sample and start computing its accuracy
    #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values
    #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
    #--> add your Python code here
    for x_testSample, y_testSample in zip(X_test, y_test):
        predicted_value = clf.predict([x_testSample])[0]
        diff = 100 * abs(predicted_value - y_testSample) / y_testSample

        if diff <= 15:
            correct += 1

    accuracy = correct / len(y_test)

    # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
    # with the KNN hyperparameters. Example: "Highest Naive Bayes accuracy so far: 0.32, Parameters: s=0.1
    # --> add your Python code here
    if accuracy > highest_accuracy:
        highest_accuracy = accuracy
        print(f"Highest Naive Bayes accuracy so far: {highest_accuracy}")
        print(f"Parameters: s={s}")



