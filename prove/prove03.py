import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing

#Reading in the data and setting it up
mpg_header = ["mpg", "cyl", "disp", "hp", "wt", "acc", "year", "origin", "car_name"]
car_header = ["buying", "maint", "doors", "persons", "lug_boot", "safety", "target"]
car = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/car/car.data", header = None, names=car_header, index_col = False)
mpg = pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data-original", sep="\s+", header=None, names=mpg_header)
student = pd.read_csv("C:/Users/james/Desktop/Professional/current classes/CS450/data/student/student-mat.csv", sep=";")

#The Car data set

#Change the columns to be numeric
replace_car_values = {"target": {"unacc":0, "acc":1, "good":2, "vgood":3},
                  "buying": {"low":0, "med":1, "high":2, "vhigh":3},
                  "maint": {"low":0, "med":1, "high":2, "vhigh":3},
                  "doors": {"2":2, "3":3, "4":4, "5more":5},
                  "persons": {"2":2, "4":4, "more":5},
                  "lug_boot": {"small":0, "med":1, "big":2},
                  "safety": {"low":0, "med":1, "high":2}}

car.replace(replace_car_values, inplace=True)

car_target = car["target"]
car_data = car.drop("target", axis=1).values

car_target = np.asarray(car_target)

car_train_data, car_test_data, car_train_target, car_test_target = train_test_split(car_data, car_target, train_size = .7, test_size = 0.3, random_state = 48, shuffle = True)

knn = KNeighborsClassifier(n_neighbors=10)
knn.fit(car_train_data, car_train_target)
car_prediction = knn.predict(car_test_data)

count = 0
for i in range(len(car_test_data)):
    if prediction[i] == car_test_target[i]:
        count += 1
accuracy = float(count) / len(car_test_data) * 100
"""
print()
print("Accuracy: {:.2f}%".format(accuracy))
print()
print("The classifier predicted\n {} out of {} right.".format(count, len(car_test_target)))
print()
"""

#The MPG Data Set

stats = mpg.describe()
#print(mpg.isnull().sum())
#print(mpg.shape)
mpg.dropna(inplace=True)
#print(mpg.shape)
mpg_target = mpg["mpg"]
mpg_target = np.asarray(mpg_target)
mpg_data = mpg.iloc[:, 1:7]

mpg_train_data, mpg_test_data, mpg_train_target, mpg_test_target = train_test_split(mpg_data, mpg_target, train_size = .7, test_size = 0.3, random_state = 48, shuffle = True)

std_scale = preprocessing.StandardScaler().fit(mpg_train_data)
mpg_train_data_std = std_scale.transform(mpg_train_data)
mpg_test_data_std = std_scale.transform(mpg_test_data)
print()
for i in range(3,9):
    regr = KNeighborsRegressor(n_neighbors=i)
    regr.fit(mpg_train_data_std, mpg_train_target)
    mpg_prediction = regr.predict(mpg_test_data_std)
    accuracy = regr.score(mpg_test_data_std, mpg_test_target)
    #print("When k = {} Accuracy: {:.2f}%".format(i, accuracy * 100))
    
#The Student Math Data
    
binary = {"school" : {"GP":1, "MS":0},
          "sex" : {"M":1, "F":0},
          "address" : {"R":1, "U":0},
          "famsize" : {"GT3":1, "LE3":0},
          "Pstatus" : {"T":1, "A":0},
          "schoolsup" : {"yes":1, "no":0},
          "famsup" : {"yes":1, "no":0},
          "paid" : {"yes":1, "no":0},
          "activities" : {"yes":1, "no":0},
          "nursery" : {"yes":1, "no":0},
          "higher" : {"yes":1, "no":0},
          "internet" : {"yes":1, "no":0},
          "romantic" : {"yes":1, "no":0}}

student.replace(binary, inplace=True)

student_target = np.asarray(student["G3"])
student_data = student.iloc[:, 0:32]
student_data = pd.get_dummies(student_data, columns=["Mjob", "Fjob", "reason", "guardian"], dtype = float)

student_train_data, student_test_data, student_train_target, student_test_target = train_test_split(student_data, student_target, train_size = .7, test_size = 0.3, random_state = 48, shuffle = True)

std_scale = preprocessing.StandardScaler().fit(student_train_data)
student_train_data_std = std_scale.transform(student_train_data)
student_test_data_std = std_scale.transform(student_test_data)
print()
for i in range(3,9):
    regr = KNeighborsRegressor(n_neighbors=i)
    regr.fit(student_train_data_std, student_train_target)
    s_prediction = regr.predict(student_test_data_std)
    accuracy = regr.score(student_test_data_std, student_test_target)
    print("When k = {} Accuracy: {:.2f}%".format(i, accuracy * 100))
    
    
    
    