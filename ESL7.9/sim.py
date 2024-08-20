import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold

import itertools


#Load and prepare data, split into train and test:

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)

data = pd.read_csv('data.csv')
train_data = data[data.train == 'T']
test_data = data[data.train == 'F']

X_train, y_train = train_data.drop(columns = ['Unnamed: 0', 'lpsa', 'train']), train_data.lpsa
X_test, y_test = test_data.drop(columns = ['Unnamed: 0', 'lpsa', 'train']), test_data.lpsa

scaler = StandardScaler()

X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train.values), columns= X_train.columns)
X_test_scaled = pd.DataFrame(scaler.fit_transform(X_test.values), columns= X_train.columns)


#k-subset selection:
def fitModel(model, X, y, feature_set):
    X_select = X[list(feature_set)]
    new_model = model
    new_model.fit(X_select,y)
    return mean_squared_error(new_model.predict(X_select), y)

def kSubsetSelection(model, X, y, k):
    models_scores_list = []
    for feature_subset in itertools.combinations(X.columns, k):
        new_score = fitModel(model, X, y, feature_subset)
        models_scores_list.append([feature_subset, new_score])
    best = sorted(models_scores_list, key = lambda x: x[1])[0]
    return best[0], best[1]

linear_model = LinearRegression()
k_subset_features = {}
k_subset_scores = {}
for k in range(1, 9):
    bst_features, bst_score = kSubsetSelection(linear_model, X_train_scaled, y_train, k)
    k_subset_features[k] = list(bst_features)
    k_subset_scores[k] = bst_score
#print(k_subset_features)


#5 and 10-fold CV scores:
scores_CV_5 = []
scores_CV_10 = []

def kBestSubsetCV(model, X, y, k, n):
    kf = KFold(n_splits= n, shuffle=True, random_state=42)
    fold_scores = []
    for train_index, val_index in kf.split(X):
        subset_scores = []
        for feature_subset in itertools.combinations(X.columns, k):
            X_sel = X[list(feature_subset)]
            X_train_fold, X_val, y_train_fold, y_val = X_sel.iloc[train_index], X_sel.iloc[val_index], y.iloc[train_index], y.iloc[val_index]
            lin_mod = model
            lin_mod.fit(X_train_fold, y_train_fold)
            subset_scores.append(mean_squared_error(lin_mod.predict(X_val), y_val))
        fold_scores.append(min(subset_scores))
    return np.mean(fold_scores)

for k in range(1,9):
    scores_CV_5.append(kBestSubsetCV(linear_model, X_train, y_train, k, 5))
    scores_CV_10.append(kBestSubsetCV(linear_model, X_train, y_train, k, 10))
    #scores_CV_5.append(-np.mean(cross_val_score(estimator = linear_model, X = X_train_scaled[k_features], y = y_train, cv = 5, scoring = 'neg_mean_squared_error')))
    #scores_CV_10.append(-np.mean(cross_val_score(estimator = linear_model, X = X_train_scaled[k_features], y = y_train, cv = 10, scoring = 'neg_mean_squared_error')))

print(scores_CV_5)

#AIC and BIC scores:

N = X_train.shape[0]
p = 8

#First get irreducible error estimate:
full_model = LinearRegression()
full_model.fit(X_train_scaled, y_train)
y_variance = N*mean_squared_error(full_model.predict(X_train_scaled), y_train)/(N-p-1)

#AIC and BIC Computations:
AIC_scores = []
BIC_scores = []
for k in range(1,9):
    AIC_scores.append(k_subset_scores[k] + 2*k*y_variance/N)
    BIC_scores.append(k_subset_scores[k] + np.log(N)*k*y_variance/N)


#Get test errors:
test_scores = []
for k in range(1,9):
    k_model = LinearRegression()
    k_model.fit(X_train_scaled[k_subset_features[k]], y_train)
    test_scores.append(mean_squared_error(k_model.predict(X_test_scaled[k_subset_features[k]]), y_test))


#Display results:

ks = range(1,9)
#plt.plot(ks, test_scores, '--o', color = 'purple', label = 'Test')
plt.plot(ks, scores_CV_5, '--o', color = 'green', label = '5-fold CV')
plt.plot(ks, scores_CV_10, '--o', color = 'blue', label = '10-fold CV')
plt.plot(ks, AIC_scores, '--o', color = 'red', label = 'AIC')
plt.plot(ks, BIC_scores, '--o', color = 'orange', label = 'BIC')
plt.legend()
plt.xlabel('Subset size')
plt.ylabel('Prediction error')
plt.show()

