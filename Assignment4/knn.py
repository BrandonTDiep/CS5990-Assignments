#-------------------------------------------------------------------------
# AUTHOR: Brandon Diep
# FILENAME: knn.py
# SPECIFICATION: description of the program
# FOR: CS 5990- Assignment #4
# TIME SPENT: 1hr 30 mins
#-----------------------------------------------------------*/

#importing some Python libraries
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
import numpy as np

#11 classes after discretization
classes = [i for i in range(-22, 40, 6)]

#defining the hyperparameter values of KNN
k_values = [i for i in range(1, 20)]
p_values = [1, 2]
w_values = ['uniform', 'distance']

#reading the training data
#reading the test data
#hint: to convert values to float while reading them -> np.array(df.values)[:,-1].astype('f')
train_data = pd.read_csv('weather_training.csv')
test_data = pd.read_csv('weather_test.csv')

X_training = np.array(train_data.values)[:, 1:-1].astype('f')
X_test = np.array(test_data.values)[:, 1:-1].astype('f')

y_training = np.array(train_data.values)[:, -1].astype('f')
y_training = np.digitize(y_training, classes, right=True)

y_test = np.array(test_data.values)[:, -1].astype('f')
y_test = np.digitize(y_test, classes, right=True)

highest_accuracy = 0


#loop over the hyperparameter values (k, p, and w) ok KNN
#--> add your Python code here
for k in k_values:
    for p in p_values:
        for w in w_values:

            #fitting the knn to the data
            clf = KNeighborsClassifier(n_neighbors=k, p=p, weights=w)
            clf = clf.fit(X_training, y_training)

            correct = 0

            #make the KNN prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously, use zip()
            #Example. for (x_testSample, y_testSample) in zip(X_test, y_test):
            #to make a prediction do: clf.predict([x_testSample])
            #the prediction should be considered correct if the output value is [-15%,+15%] distant from the real output values.
            #to calculate the % difference between the prediction and the real output values use: 100*(|predicted_value - real_value|)/real_value))
            #--> add your Python code here
            for x_testSample, y_testSample in zip(X_test, y_test):
                predicted_value = clf.predict([x_testSample])[0]

                diff = 100 * abs(predicted_value - y_testSample) / y_testSample

                if diff <= 15:
                    correct += 1

            accuracy = correct / len(y_test)
            #check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
            #with the KNN hyperparameters. Example: "Highest KNN accuracy so far: 0.92, Parameters: k=1, p=2, w= 'uniform'"
            #--> add your Python code here
            if accuracy > highest_accuracy:
                highest_accuracy = accuracy
                print(f"Highest KNN accuracy so far: {highest_accuracy}")
                print(f"Parameters: k={k}, p={p}, w={w}")






