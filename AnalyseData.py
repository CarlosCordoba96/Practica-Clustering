# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:11:38 2019

@author: Carlos
"""

import matplotlib.pyplot as plt
import numpy
import itertools
import seaborn as sns
import pandas as pd

df = pd.read_csv("clusters.csv")

cm1=df[df.group==-1]
print("\n Cluster    -  -1\n")
print(cm1.describe())
#cm1.mean().plot(kind='bar',title="c-1")

c0=df[df.group==0]
print("\n Cluster    -  0\n")
print(c0.describe())
#c0.mean().plot(kind='bar',title="c0")

from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
datanorm = min_max_scaler.fit_transform(c0)


#1.2. Principal Component Analysis
from sklearn.decomposition import PCA
estimator = PCA (n_components = 4)
X_pca = estimator.fit_transform(datanorm)
import numpy
print(estimator.explained_variance_ratio_) 
x=0
for i in estimator.explained_variance_ratio_:
    x=x+i
print(x)

#PCA
import matplotlib.pyplot as plt
numbers = numpy.arange(len(X_pca))
fig, ax = plt.subplots()
for i in range(len(X_pca)):
    plt.plot(X_pca[i][0], X_pca[i][1],marker='.',color='k') 
plt.xlim(-1, 1)
plt.ylim(-1,1.5)
ax.grid(True)
plt.show()



c1=df[df.group==1]
#c1.mean().plot(kind='bar',title="c1")
print("\n Cluster    -  1\n")
print(c1.describe())
c2=df[df.group==2]
#c2.mean().plot(kind='bar',title="c2")
print("\n Cluster    -  2\n")
print(c2.describe())
c3=df[df.group==3]
#c3.mean().plot(kind='bar',title="c3")
print("\n Cluster    -  3\n")
print(c3.describe())