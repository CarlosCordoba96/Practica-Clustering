# -*- coding: utf-8 -*-
"""
Created on Sat Mar 23 17:11:38 2019

@author: Carlos
"""
def analyse(df):
    print("MODE_MAIN:")
    print(df.groupby(('mode_main')).size())
    print(df.groupby(('male')).size())
    print(df.groupby(('education')).size())
    print(df.groupby(('income')).size())
    print(df.groupby(('license')).size())
    print(df.groupby(('weekend')).size())
    
import matplotlib.pyplot as plt
import numpy
import itertools
import seaborn as sns
import pandas as pd
#Cargamos el fichero clusters
df = pd.read_csv("clusters.csv")

#Cogemos el dataset que pertenezcan al grupo -1 y realizamos un estudio descriptivo de las variables
cm1=df[df.group==-1]
print("\n Cluster    -  -1\n")
cm1desc=cm1.describe()
analyse(cm1)
#creamos la gráfica de la media de dicho cluster
cm1.mean().plot(kind='bar',title="c-1")
#cogemos dataset que pertenezcan al grupo 0 y realizar estudio
c0=df[df.group==0]
print("\n Cluster    -  0\n")
c0desc=c0.describe()
analyse(c0)
#c0.mean().plot(kind='bar',title="c0")

#cogemos dataset que pertenezcan al grupo 1 y realizar estudio
c1=df[df.group==1]
#c1.mean().plot(kind='bar',title="c1")
print("\n Cluster    -  1\n")
c1desc=c1.describe()
analyse(c1)


#cogemos dataset que pertenezcan al grupo 2 y realizar estudio
c2=df[df.group==2]
#c2.mean().plot(kind='bar',title="c2")
print("\n Cluster    -  2\n")
c2desc=c2.describe()
analyse(c2)

#cogemos dataset que pertenezcan al grupo 3 y realizar estudio
c3=df[df.group==3]
#c3.mean().plot(kind='bar',title="c3")
print("\n Cluster    -  3\n")
c3desc=c3.describe()
analyse(c3)