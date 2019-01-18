import numpy as np
from sklearn import datasets 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#read in the data
iris = datasets.load_iris()
"""
print(iris.data)

print(iris.target)

print(iris.target_names)
"""
#My hard coded algorithm
class HardCodedClassifier():
    def __init__(self):
        pass
    
    def fit(self, data, target):
        return "This data is well fitted."
    
    def predict(self, X_test):
        array = []
        for instance in range(len(X_test)):
            array.append(0)
        return array

#data
X = iris.data
y = iris.target

#training the data
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = .7, test_size = 0.3, random_state = 50, stratify=y, shuffle = True)

"""
print("X_train")
print(X_train)
print()
print("X_test")
print(X_test)
print()
print("y_train")
print(y_train)
print()
print("y_test")
print(y_test)
print()
"""
#function to obtain accuracy
def accuracy_test(data_test, prediction, target_test):
    count = 0
    for i in range(len(data_test)):
        if prediction[i] == target_test[i]:
            count += 1
    accuracy = float(count) / len(data_test) * 100
    return accuracy

#Gaussian performance
classifier = GaussianNB()
classifier.fit(X_train, y_train)
targets_predicted = classifier.predict(X_test)
gaccuracy = accuracy_test(X_test, targets_predicted, y_test)

#My hard coded performance
my_classifier = HardCodedClassifier()
my_classifier.fit(X_train, y_train)
my_prediction = my_classifier.predict(X_test)
myaccuracy = accuracy_test(X_test, my_prediction, y_test)

#conclusion
print("My hard coded algorithm proves to be {:.2f}% accurate\nwhile the Gaussian algorith proves to be {:.2f}% accurate".format(myaccuracy, gaccuracy))
