import numpy as np
from math import sqrt
from operator import itemgetter
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split

#reading in the data
iris = datasets.load_iris()

#setting the data and target arrays
x = iris.data
y = iris.target

#splitting the train and test data
train_data, test_data, train_target, test_target = train_test_split(x, y, train_size = .7, test_size = 0.3, random_state = 48, shuffle = True)

#My hard coded nearest neighbor classifier
class KNNClassifier():
    def __init__(self, k):
        self.k = k
    
    def fit(self, train_data, train_target):
        self.train_data = train_data
        self.train_target = train_target
    
    def predict(self, test_data):
        self.test_data = test_data
        self.prediction = []
        self.distance = []
                
        for row in range(len(self.test_data)):
            for i in range(len(self.train_data)):
                self.d = sqrt(((self.train_data[i][0] - self.test_data[row][0])**2 + (self.train_data[i][1] - self.test_data[row][1])**2 + (self.train_data[i][2] - self.test_data[row][2])**2 + (self.train_data[i][3] - self.test_data[row][3])**2))
                self.distance.append(self.d)
                self.class_distance = list(zip(self.distance, self.train_target))
                self.class_distance.sort(key=itemgetter(0), reverse=False)
                self.nearest = self.class_distance[0:self.k]
            self.count0 = 0
            self.count1 = 0
            self.count2 = 0
            for item in range(self.k):
                if self.nearest[item][1] == 0:
                    self.count0 += 1
                elif self.nearest[item][1] == 1:
                    self.count1 += 1
                else:
                    self.count2 += 1
            if self.count0 > self.count1 and self.count0 > self.count2:
                self.prediction.append(0)
            elif self.count1 > self.count0 and self.count1 > self.count2:
                self.prediction.append(1)
            elif self.count2 > self.count0 and self.count2 > self.count1:
                self.prediction.append(2)
            else:
                self.prediction.append(self.nearest[0][1])
            self.distance = []
        self.prediction = np.array(self.prediction)
        return self.prediction


#training with my model
myknn = KNNClassifier(6)
myknn.fit(train_data, train_target)
myprediction = myknn.predict(test_data)


tally = 0
for i in range(len(test_data)):
    if myprediction[i] == test_target[i]:
        tally += 1
myaccuracy = float(tally) / len(test_data) * 100

#training with sklearn model
knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(train_data, train_target)
theirprediction = knn.predict(test_data)


count = 0
for i in range(len(test_data)):
    if theirprediction[i] == test_target[i]:
        count += 1
accuracy = float(count) / len(test_data) * 100

print()
print("Their Accuracy: {:.2f}%".format(accuracy))
print()
print("Their classifier predicted\n {} out of {} right.".format(count, len(test_target)))
print()
print("My Accuracy: {:.2f}%".format(myaccuracy))
print()
print("My classifier predicted\n {} out of {} right.".format(tally, len(test_target)))
