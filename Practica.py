# -*- coding: utf-8 -*-
"""
Editor de Spyder

Este es un archivo temporal.
"""

import matplotlib.pyplot as plt
import numpy
import itertools
import seaborn as sns
import pandas as pd


def male(x):
    if x=='yes':
        return 1
    if x=='no':
        return 0
 
def ethnicity(x):
    if x=='native':
        return 0
    if x=='western':
        return 1
    else:
        return 2

    
df = pd.read_csv("nts_data.csv")
df['male']=df['male'].apply(male)
df['ethnicity']=df['ethnicity'].apply(ethnicity)
df['mode_main']=df['mode_main'].map({'walk':0,'car':1,'bike':2,'pt':3})
df['education']=df['education'].map({'lower':0,'middle':0.5,'higher':1})
df['income']=df['income'].map({'less20':0,'20to40':0.5,'more40':1})
df['license']=df['license'].map({'yes':1,'no':0})
df['weekend']=df['weekend'].map({'yes':1,'no':0})

df.drop("mode_main",axis=1,inplace=True)
df=df[:20000]
from sklearn import preprocessing 
min_max_scaler = preprocessing.MinMaxScaler()
datanorm = min_max_scaler.fit_transform(df)


#1.2. Principal Component Analysis
from sklearn.decomposition import PCA
estimator = PCA (n_components = 4)
X_pca = estimator.fit_transform(datanorm)
minPts=500


# 2.1 Parametrization
import sklearn.neighbors	
dist = sklearn.neighbors.DistanceMetric.get_metric('euclidean')
matsim = dist.pairwise(X_pca)

from sklearn.neighbors import kneighbors_graph
A = kneighbors_graph(X_pca, minPts, include_self=False)
Ar = A.toarray()

seq = []
for i,s in enumerate(X_pca):
    for j in range(len(X_pca)):
        if Ar[i][j] != 0:
            seq.append(matsim[i][j])
            
seq.sort()
plt.plot(seq)
plt.show()

# 2.2 DBSCAN Execution
from sklearn.cluster import DBSCAN
import numpy

pos_eps=[0.2,0.25,0.30,0.35,0.40,0.45,0.5,0.55]
results=[]
for opt in pos_eps:
    print("\nCon eps: {} ".format(opt))
    db = DBSCAN(eps=opt, min_samples=minPts).fit(X_pca)
    core_samples_mask = numpy.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    
    print ('Number of clusters %d' % n_clusters_)
    
    
    # 4. plot
    import numpy
    colors = numpy.array([x for x in 'bgrcmykbgrcmykbgrcmykbgrcmyk'])
    colors = numpy.hstack([colors] * 20)
    numbers = numpy.arange(len(X_pca))
    fig, ax = plt.subplots()
    for i in range(len(X_pca)):
        plt.text(X_pca[i][0], X_pca[i][1], numbers[i], color=colors[labels[i]]) 
    plt.xlim(-1, 4)
    plt.ylim(-0.2, 1)
    ax.grid(True)
    fig.tight_layout()
    plt.show()
    
    
    
    # 6. characterization
    from sklearn import metrics
    n_clusters_ = len(set(labels)) #- (1 if -1 in labels else 0)
    print('Estimated number of clusters: %d' % n_clusters_)
    print("Silhouette Coefficient: %0.3f"
          % metrics.silhouette_score(datanorm, labels))
    df['group'] = labels
    res = df.groupby(('group')).mean()
    results.append(res)


#


#
#from sklearn.decomposition import PCA
#estimator = PCA (n_components = 2)
#X_pca = estimator.fit_transform(datanorm)
#import numpy
#print(estimator.explained_variance_ratio_) 
#x=0
#for i in estimator.explained_variance_ratio_:
#    x=x+i
#print(x)
#pd.DataFrame(numpy.matrix.transpose(estimator.components_), columns=['PC-1', 'PC-2'], index=df.columns)
#
#
#import matplotlib.pyplot as plt
#numbers = numpy.arange(len(X_pca))
#fig, ax = plt.subplots()
#for i in range(len(X_pca)):
#    plt.text(X_pca[i][0], X_pca[i][1],'.') 
#plt.xlim(-1, 1)
#plt.ylim(-1,1.5)
#ax.grid(True)
#plt.show()