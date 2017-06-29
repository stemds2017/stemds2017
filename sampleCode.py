# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 11:11:52 2017

@author: Francis Parisi

Sample Code for basic stats and functionality

import packages and define functions upfront
"""
import numpy as np
import scipy as sp
import statsmodels as sm
import statsmodels.formula.api as smf
import statsmodels.tools.tools as smt
import statsmodels.regression.linear_model as lm
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from matplotlib.colors import ListedColormap
from sklearn.linear_model import LogisticRegression


def lin_regplot(X, y, model):
    plt.scatter(X, y, c='blue')
    plt.plot(X, model.predict(X), color='red')
    return None

def plot_decision_regions(X, y, classifier,
    test_idx=None, resolution=0.02):

    # setup marker generator and color map 
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
    
    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
        np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    # plot all samples
    X_test, y_test = X[test_idx, :], y[test_idx]
    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
            alpha=0.8, c=cmap(idx),
            marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(X_test[:, 0], X_test[:, 1], c='',
            alpha=1.0, linewidth=1, marker='o',
            s=55, label='test set')


"""
Working with numpy and scipy -- basic stats
"""
# np.  gives biased var estimator, but allows you to cahnage using ddof
# sp.describe is unbiased
# create an array -- one dimensional ==> vector
x = np.array([1, 2, 3, 4, 5])
sp.stats.describe(x) # displays summary statistics
np.var(x)

# display the mean
# either use the mean() method with object x
# or call it using np explicity and pasing x as a parameter
x.mean()
np.mean(x)

# standard deviation
print('%.4f' % x.std())
print('%.4f' % np.std(x))

# let's take another look
y = (x-x.mean())**2
z = (y.sum()/len(x))**(0.5)
print('%.4f' % z)

"""
notice np.std() divides by n by default change the parameter ddof=1 
to divide by n-1 for an unbiased standard deviation
"""
print('%.4f' % x.std(ddof=1))
print('%.4f' % np.std(x, ddof=1))

# check it
w = (y.sum()/(len(x)-1))**(0.5)
print('%.4f' % w)

np.median(x)  

# create a two dimensional array
# summary stats for the entire array
x2 = np.array([[1, 2, 3, 4, 6], [7, 8, 10, 9, 12]])
print(np.mean(x2), np.median(x2), '%.4f' % np.std(x2, ddof=1), sep='\t') 

# summary stats column wise
np.set_printoptions(precision=4) # set precision for array output
print(np.mean(x2, axis=0), np.median(x2, axis=0), np.std(x2, ddof=1, axis=0), sep='\t')

# row wise -- print formatting retained
print(np.mean(x2, axis=1), np.median(x2, axis=1), np.std(x2, ddof=1, axis=1), sep='\t')

# we can calculate the correlation in several ways as well
# np.corrcoef rreturns an np array
np.corrcoef(x2)

# calculate Pearson's rho and the p-value
# the p-value suggest the corelation is statistically significant
rho, pstat = sp.stats.pearsonr(x2[0, ], x2[1, ])
print('rho = %.4f' % rho, 'p-val = %.4f' % pstat)

"""
working with pandas
"""
# read data, create a DataFrame and display the first few observations
bpdat = pd.read_csv("C:\\Users\\Francis\\Google Drive\\Data\\___PACE\\STEM-Summer2017\\STEM\\bplong.csv")
bpdat.head()

# summarize by diferent factors
pd.set_option('precision', 2)
bpdat["bp"].groupby(bpdat["sex"]).mean()
bpdat["bp"].groupby(bpdat["agegrp"]).mean()
bpdat["bp"].groupby(bpdat["sex"]).std()

# make a histogram
hist, bin_edges = np.histogram(bpdat["bp"], density=True)
plt.hist(bpdat["bp"], bins='auto')
plt.show()

# compare sample means
tstat, pval = sp.stats.ttest_ind(x2[0], x2[1])
print('t-stat = %.3f ' % tstat, 'p-value = %.3f' % pval)

# linear regression
Y = np.array([1, 3, 4, 5, 2, 3, 4, 8, 12, 14])
X = np.array([1, 2, 3, 4, 8, 6, 7, 9, 4, 17])
X = smt.add_constant(X)

model = lm.OLS(Y, X)
results = model.fit()
results.summary()

# need this -> from sklearn.linear_model import LinearRegression
model3 = LinearRegression()
X = np.array([1, 2, 3, 4, 8, 6, 7, 9, 4, 17]) # recreate without adding the constant; scikit adds it
x1 = pd.DataFrame(X)
y1 = pd.DataFrame(Y)
model3.fit(x1, y1)
print('beta1 = %.4f' % model3.coef_, 'beta0 = %.4f' % model3.intercept_)

url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/housing/housing.data'
df = pd.read_csv(url, header=None, sep='\s+')
df.columns = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD',
              'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.head()

sns.set(style='whitegrid', context='notebook')
# use sns.reset_orig() to reset matplotlib settings
cols = ['LSTAT', 'INDUS', 'NOX', 'RM', 'MEDV']
sns.pairplot(df[cols], size=2.5)
plt.show()

slr = LinearRegression()
XX = df[['RM']].values
y = df['MEDV'].values
slr.fit(XX, y)
print('Slope: %.3f' % slr.coef_[0], 'Intercept: %.3f' % slr.intercept_)

# plot the results
lin_regplot(XX, y, slr)
plt.xlabel('Average number of rooms [RM]')
plt.ylabel('Price in$1000\'s [MEDV]')
plt.show()

"""
Logistic Regression
"""
# need this -> from sklearn import datasets, plus more (see header)
iris = datasets.load_iris() # load the data on iris plants
W = iris.data[:, [2, 3]]
V = iris.target

# create a training data set and a test set
# standardize the x-values
X_train, X_test, y_train, y_test = train_test_split(W, V, test_size=0.3, random_state=0)
X_train_std = StandardScaler().fit_transform(X_train)
X_test_std = StandardScaler().fit_transform(X_test)
X_combined_std = np.vstack((X_train_std, X_test_std))
y_combined = np.hstack((y_train, y_test))

# fit the model
lr = LogisticRegression(C=1000.0, random_state=0)
lr.fit(X_train_std, y_train)

# Plot results
plot_decision_regions(X_combined_std, y_combined, classifier=lr, test_idx=range(105,150))
plt.xlabel('petal length (standardized)')
plt.ylabel('petal width (standardized)')
plt.legend(loc='upper left')
plt.show()
# test data are not circled! Something's not working in plot_decision_regions()

lr.predict_proba(X_test_std[0, :].reshape(1, -1))
