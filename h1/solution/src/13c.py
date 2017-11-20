#Homework 1.3 c)
#without readin_values and draw function
import numpy as np
from math import pi,sin,sqrt
import matplotlib.pyplot as plt
from sympy import *
from sklearn.metrics import mean_squared_error

def calculate_phi(features):
    feature_list = []
    x = Symbol('x')
    for i in range(0, features): 
        phi_i = [sin(x*pow(2,i))]
        feature_list.append(phi_i)
    feature_list=np.asarray(feature_list)
    return feature_list;
    
def calculate_lls(feature_list):
    x = Symbol('x')
    A = np.zeros([len(x_train),len(feature_list)])
    for p in range(0,len(feature_list)):
        for j in range(0,len(x_train)):
            tmp=np.ndarray.tolist(feature_list[p])[0]
            tmp=float(tmp.subs({x:x_train[j]}))
            A[j,p]=tmp
    tmp3=np.linalg.inv(np.dot(A.T,A))
    tmp2=np.dot(A.T,y_train)
    x_res=(np.dot(tmp3,tmp2.T))
    f=np.dot(x_res,feature_list)[0]
    xx = np.arange(0, 6, 0.01)
    yy=[]
    for u in range(0,600):
        tmp4 = f.subs({x:u*0.01})
        yy.append(tmp4)
    return[xx,yy];
    
readin_values()
drawc()
