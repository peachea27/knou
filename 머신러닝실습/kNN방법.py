import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
import seaborn as sns #시각화
import matplotlib.pyplot as plt #시각화

#데이터 불러오기
iris = sns.load_dataset('iris')
print(iris.shape)
iris.head()
#(150, 5)

#자료의 분할
##설명변수와 타겟변수를 따로 저장한 후, 각각을 학습자료/ 검증자료/ 시험자료로 분할(6:2:2)
X=iris[['sepal_length','sepal_width','petal_length','petal_width']]
y=iris.iloc[:,4]
# 6:4로 train과 test를 나눔
X_train, X_test, y_train, y_test = \
    train_test_split(X,y,test_size=0.4,random_state=24)
#test자료를 validation과 test로 다시 나눔
X_val, X_test, y_val, y_test = \
    train_test_split(X_test,y_test, test_size=0.5, random_state=24)

#표준화과정
##표준화를 하기 위해 학습자료의 평균과 분산을 구한다.
##이를 기반으로 모든 자료의 표준화를 수행

scaler = StandardScaler()
scaler.fit(X_train) #설명변수 X_train의 평균과 분산 계산
X_train = pd.DataFrame(scaler.transform(X_train))
X_val = pd.DataFrame(scaler.transform(X_val))
X_test= pd.DataFrame(scaler.transform(X_test))

# k의 선택과정
##다양한 k값에 따라, 학습자료로 kNN방법을 적용하고 검증자료에서 정분류율을 확인하여 이를 저장
k_list = range(1,31)
acc_list = []
for k in k_list:
    model = KNeighborsClassifier(n_neighbors = k) #k= 1부터 30까지 
    model.fit(X_train, y_train) #값 저장
    y_pred = model.predict(X_val)
    acc_rate = (y_pred == y_val).mean() # 둘이 같은 것의 평균 : 정분류율
    acc_list.append(acc_rate) #k가 1일때의 정분류율부터 30일때의 정분류율까지 append

k_acc_df = pd.DataFrame({'k' :k_list, 'Accuracy' : acc_list})

# 최적의 k에 대한 kNN방법 적용 및 결과
## 앞서 선택된 k=15에 대해, 학습자료로 kNN방법 적용
## 시험자료에 대해 위의 kNN모형에 대입하여 정분류율 확인
knn_model = KNeighborsClassifier(n_neighbors = 15)
knn_model.fit(X_train, y_train)
y_pred = knn_model.predict(X_test) #시험자료를 적합한 모형에 넣어서 predict
acc_rate = (y_pred == y_test).mean() #에측값과 실제타겟값이 일치하는게 얼마나 있는가 그 비율
print(acc_rate)
#0.9333333333333333