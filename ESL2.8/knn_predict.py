import numpy as np
from sklearn.neighbors import KNeighborsClassifier as KNClassifier
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score as accuracy
from matplotlib import pyplot as plt

train_data = np.loadtxt('training_data')
train_df = train_data[np.where((train_data[:, 0] == 2) | (train_data[:, 0] == 3))]

test_data = np.loadtxt('test_data')
test_df = test_data[np.where((test_data[:, 0] == 2) | (test_data[:, 0] == 3))]

X_train, X_test = train_df[:, 1:], test_df[:, 1:]
y_train, y_test = train_df[:, 0].reshape(-1), test_df[:, 0].reshape(-1)

k_list = range(1,16)
classifiers = []
for k in k_list:
    classifier = KNClassifier(k, n_jobs = -1)
    classifier.fit(X_train, y_train)
    classifiers.append(classifier)

accs_train = []
accs_test = []
for i in range(len(k_list)):
    y_train_predict = classifiers[i].predict(X_train)
    y_test_predict = classifiers[i].predict(X_test)
    accs_train.append(accuracy(y_train_predict, y_train))
    accs_test.append(accuracy(y_test_predict, y_test))
    #print("(Train, Test) accuracy with k = {k}: ({acc_train}, {acc_test})".format(
    #   k = k_list[i], acc_train = round(accs_train[i],4), acc_test = round(accs_test[i],4)))


lin_model = LinearRegression()
lin_model.fit(X_train, y_train)
linear_train_acc = accuracy(y_train, lin_model.predict(X_train).round())
linear_test_acc = accuracy(y_test, lin_model.predict(X_test).round())

print("Linear (Train, Test) accuracy: ({acc_train}, {acc_test})".format(
       acc_train = round(linear_train_acc,4), acc_test = round(linear_test_acc,4)))


plt.plot(k_list, accs_train, color = 'red', label = 'Train')
plt.plot(k_list, accs_test, color = 'blue', label = 'Test')
plt.axhline(linear_train_acc, color = 'green', label = 'Linear', ls = '--')
plt.axhline(linear_test_acc, color = 'green', ls = '--')

plt.xlabel('k')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
