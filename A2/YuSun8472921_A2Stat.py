from scipy.stats import friedmanchisquare
from scipy import stats
from scipy.stats import friedmanchisquare

import numpy as np

n1 = np.array([0.9932,0.9930,0.9943,0.9946,0.9884,0.9950,0.9904,0.8122,0.9312,0.8326])
n2 = np.array([0.9278,0.9737,0.9772,0.9791,0.9440,0.9758,0.9290,0.6885,0.8388,0.5288])
n3 = np.array([0.6918,0.9645,0.9967,0.9932,0.9919,0.9966,0.9974,0.9602,0.8238,0.8489])
n4 = np.array([0.7042,0.9540,0.9788,0.9802,0.9676,0.9706,0.9103,0.5538,0.7250,0.7804])
n5 = np.array([0.8536,0.9834,0.9848,0.9800,0.9651,0.9799,0.9743,0.5635,0.6604,0.5792])
n6 = np.array([0.5983,0.9699,0.9936,0.9967,0.9856,0.9940,0.9868,0.5919,0.7832,0.8011])

# nn=[n1, n2, n3, n4, n5 , n6]

# for i in range(6):
#      print(nn[i].mean())

# r=""
# for i in range(6):
#     for j in range(i+1,6):
#         temp = stats.ttest_rel(nn[i],nn[j])
#         r =r +str(np.around(temp[1],4)) + " & "
# print(r)

# res=[]
# for i in range(6):
#     for j in range(i+1,6):
#         res.append(np.around(nn[i]-nn[j],4))
        
# # print(res)
# res=np.array(res)

# for i in range(10):
#     temp = res[:,i]
#     for j in temp:
#         print(str(j)+" & ", end = ' ')
#     print(" ")

# for i in range(15):
#     temp = res[i,:]
#     print(str(np.around(temp.mean(),4))+" & ", end = '')
# print(" ")

# for i in range(15):
#     temp = res[i,:]   
#     print(str(np.around(temp.std(),4))+" & ", end = '')
# print(" ")

# n11=np.array([0.9923,0.9927,0.9938,0.9935,0.9896,0.9947,0.9909,0.8128,0.9382,0.8325])
# n12=np.array([0.9930,0.9929,0.9944,0.9941,0.9887,0.9950,0.9902,0.8126,0.9349,0.8348])
# n31=np.array([0.8886,0.9495,0.9630,0.9650,0.9670,0.9670,0.9735,0.9423,0.8452,0.8131])
# n32=np.array([0.8074,0.9689,0.9909,0.9890,0.9806,0.9929,0.9926,0.9267,0.8197,0.8450])

# nn1=[n1,n11,n12,n3,n31,n32]
# for i in range(6):
#      print(nn1[i].mean())

# for i in range(6):
#      print(nn1[i].std())
# r=''

# print (np.around(nn1[1]-nn1[0],4))
# print (np.around(nn1[2]-nn1[0],4))
# print (np.around(nn1[4]-nn1[3],4))
# print (np.around(nn1[5]-nn1[3],4))

# print (np.around(nn1[1]-nn1[0],4).mean())
# print (np.around(nn1[2]-nn1[0],4).mean())
# print (np.around(nn1[4]-nn1[3],4).mean())
# print (np.around(nn1[5]-nn1[3],4).mean())

# print (np.around(nn1[1]-nn1[0],4).std())
# print (np.around(nn1[2]-nn1[0],4).std())
# print (np.around(nn1[4]-nn1[3],4).std())
# print (np.around(nn1[5]-nn1[3],4).std())

# print(stats.ttest_rel(nn1[1],nn1[0]))
# print(stats.ttest_rel(nn1[2],nn1[0]))
# print(stats.ttest_rel(nn1[4],nn1[3]))
# print(stats.ttest_rel(nn1[5],nn1[3]))

data1 = [1,2]
data2 = [1,2]
data3 = [2,1]
data4 = [1,2]
data5 = [2,1]
data6 = [1,2]
data7 = [2,1]
data8 = [1,2]

data=[data1,data2,data3,data4,data5,data6,data7,data8]
data=np.array(data)

r1=data.mean()
n=4
k=3
t1=n* np.sum((data.mean(axis=0)-r1)* (data.mean(axis=0)-r1))
t2=(1/(n*(k-1)))*np.sum((data-r1) * (data-r1))

print(t1/t2)
stat, p = friedmanchisquare(data1, data2, data3,data4)
print(stat)
print(p)

from Orange.evaluation import compute_CD,graph_ranks
import matplotlib.pyplot as plt
names = ["SVM", "KNN", "RF" ]
avranks =  [2.5, 1.5, 1.75]
cd = compute_CD(avranks, 4) #tested on 4 datasets
# print(cd)
graph_ranks(avranks, names, cd=cd, width=6, textspace=1.5)
plt.show()