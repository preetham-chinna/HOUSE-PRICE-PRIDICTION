# HOUSE-PRICE-PRIDICTION
USING REGRESSION ALGORITHM


%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
cd
/Users/preetham
pwd
'/Users/preetham'
x=pd.read_csv('Downloads/headbrain.csv')
​
x
Gender	Age Range	Head Size(cm^3)	Brain Weight(grams)
0	1	1	4512	1530
1	1	1	3738	1297
2	1	1	4261	1335
3	1	1	3777	1282
4	1	1	4177	1590
5	1	1	3585	1300
6	1	1	3785	1400
7	1	1	3559	1255
8	1	1	3613	1355
9	1	1	3982	1375
10	1	1	3443	1340
11	1	1	3993	1380
12	1	1	3640	1355
13	1	1	4208	1522
14	1	1	3832	1208
15	1	1	3876	1405
16	1	1	3497	1358
17	1	1	3466	1292
18	1	1	3095	1340
19	1	1	4424	1400
20	1	1	3878	1357
21	1	1	4046	1287
22	1	1	3804	1275
23	1	1	3710	1270
24	1	1	4747	1635
25	1	1	4423	1505
26	1	1	4036	1490
27	1	1	4022	1485
28	1	1	3454	1310
29	1	1	4175	1420
...	...	...	...	...
207	2	2	3995	1296
208	2	2	3318	1175
209	2	2	2720	955
210	2	2	2937	1070
211	2	2	3580	1320
212	2	2	2939	1060
213	2	2	2989	1130
214	2	2	3586	1250
215	2	2	3156	1225
216	2	2	3246	1180
217	2	2	3170	1178
218	2	2	3268	1142
219	2	2	3389	1130
220	2	2	3381	1185
221	2	2	2864	1012
222	2	2	3740	1280
223	2	2	3479	1103
224	2	2	3647	1408
225	2	2	3716	1300
226	2	2	3284	1246
227	2	2	4204	1380
228	2	2	3735	1350
229	2	2	3218	1060
230	2	2	3685	1350
231	2	2	3704	1220
232	2	2	3214	1110
233	2	2	3394	1215
234	2	2	3233	1104
235	2	2	3352	1170
236	2	2	3391	1120
237 rows × 4 columns

plt.rcParams['figure.figsize']=(20.0,10.0)
data=pd.read_csv("Downloads/headbrain.csv")
print(data.shape)
data.head()
X=data['Head Size(cm^3)'].values
Y=data['Brain Weight(grams)'].values
mean_x=np.mean(X)
mean_y=np.mean(Y)
n=len(X)
numer=0
denom=0
for i in range (n):
    numer+=(X[i]-mean_x)*(Y[i]-mean_y)
    denom+=(X[i]-mean_x)**2
    b1=numer/denom
    b0=mean_y-(b1*mean_x)
print(b1,b0)
(237, 4)
0.26342933948939945 325.57342104944223
​
max_X=np.max(X)+100
min_X=np.min(X)-100
x=np.linspace(min_X,max_X,1000)
y=b0+b1*x
plt.plot(x,y,color='#58b970',label='Regline')
plt.scatter(X,Y,color='#ef5423',label='Scatplt')
plt.xlabel('HeadSize')
plt.ylabel('BrainHeight')
plt.legend()
plt.show()

ss_t=0
ss_r=0
for i in range (n):
    y_pred=b0+b1*X[i]
    ss_t+=(Y[i]-mean_y)**2
    ss_r+=(Y[i]-y_pred)**2
r2=1-(ss_r/ss_t)
print(r2)
0.6393117199570003
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
X=X.reshape((n,1))
reg=LinearRegression()
reg=reg.fit(X,Y)
Y_pred=reg.predict(X)
mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
r2_score=reg.score(X,Y)
print(rmse)
print(r2_score)
72.1206213783709
0.639311719957
​
​
​
​
