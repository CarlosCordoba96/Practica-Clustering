# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 09:51:11 2019

@author: Carlos
"""

import matplotlib.pyplot as plt
import numpy
import itertools
import seaborn as sns
import pandas as pd
from scipy.stats import kruskal

def evaluar(stat,p):
    alpha = 0.05
    if p > alpha:
        print('Same distributions (fail to reject H0)')
    else:
        print('Different distributions (reject H0)')


df = pd.read_csv("clusters.csv")

cm1=df[df.group==-1]
c0=df[df.group==0]
c1=df[df.group==1]
c2=df[df.group==2]
c3=df[df.group==3]

variable_selected="bicycles"
stat, p = kruskal(c0[variable_selected].as_matrix(), c1[variable_selected].as_matrix(), c2[variable_selected].as_matrix(), c3[variable_selected].as_matrix())
print(stat)
print(p)
evaluar(stat,p)

