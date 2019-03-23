# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 11:34:55 2019

@author: Carlos
"""

import matplotlib.pyplot as plt
import numpy as np
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

import seaborn as sns

f, ax = plt.subplots(figsize=(10, 8))
corr = df.corr()
sns.heatmap(corr, mask=np.zeros_like(corr, dtype=np.bool), cmap=sns.diverging_palette(220, 10, as_cmap=True),
            square=True, ax=ax)

walk=df[df.mode_main==0]
walk.drop("mode_main",axis=1,inplace=True)
car=df[df.mode_main==1]
car.drop("mode_main",axis=1,inplace=True)
bike=df[df.mode_main==2]
bike.drop("mode_main",axis=1,inplace=True)
pt=df[df.mode_main==3]
pt.drop("mode_main",axis=1,inplace=True)

print("WALK")
print(walk.weekend.mean())
print(walk.cars.mean())
print(walk.diversity.mean())
#walk.mean().plot(kind='bar',title="Walk")
print("CAR")
print(car.weekend.mean())
print(car.cars.mean())
print(car.diversity.mean())
#car.mean().plot(kind='bar',title="Car")
print("BIKE")
print(bike.weekend.mean())
print(bike.cars.mean())
print(bike.diversity.mean())
#bike.mean().plot(kind='bar',title="bike")
print("PT")
print(pt.weekend.mean())
print(pt.cars.mean())
print(pt.diversity.mean())
#pt.mean().plot(kind='bar',title="Pt")