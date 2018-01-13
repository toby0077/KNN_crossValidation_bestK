# -*- coding: utf-8 -*-
"""
Created on Sun Jan  7 23:13:36 2018

@author: Administrator
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd 
#导入数据预处理，包括标准化处理或正则处理
from sklearn import preprocessing
#样本平均测试，评分更加
from sklearn.cross_validation import cross_val_score
 
from sklearn import datasets
#导入knn分类器
from sklearn.neighbors import KNeighborsClassifier
 
#excel文件名
fileName="data.xlsx"
#读取excel
df=pd.read_excel(fileName)
# data为Excel前几列数据
x=df[df.columns[:4]]
#标签为Excel最后一列数据
y=df[df.columns[-1:]]
 
#把dataframe 格式转换为阵列
x=np.array(x)
y=np.array(y)
#数据预处理，否则计算出错
y=[i[0] for i in y]
y=np.array(y)
 
 
#标准化X数据，缩小不同维度数据差异，有时候正则化后准确率降低
#print("data is normalized")
#x_normal=preprocessing.scale(x)
 

k_range=range(1,31)
k_score=[]
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    score_knn=cross_val_score(knn,x,y,cv=5,scoring='accuracy')
    #添加平均值
    k_score.append(score_knn.mean())
    
#print("cross value knn score:",score_knn.mean())

#绘图
plt.plot(k_range,k_score)
plt.title("KNN crossValide Test ")
plt.xlabel("K value of KNN ")
plt.ylabel("Cross-validated accuracy")
plt.show()


dict_K_score=dict(zip(k_score,k_range))
max_proprobility=max(dict_K_score)
best_K=dict_K_score[max_proprobility]
print("best K is:",best_K)
print("max_proprobility is:",max_proprobility)









