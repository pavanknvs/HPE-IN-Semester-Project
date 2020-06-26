import numpy as np
import math
import csv
#no of atoms is 32000
lj  = np.arange(32000*4,dtype=float).reshape(32000,4)
for i in range(32000):
    for j in range(4):
        lj[i][j] =0



i=0
with open('/content/final.csv','rt')as f:
  data = csv.reader(f)
  for words in data:
             lj[i][0] =  float(words[0])
             lj[i][1] = float(words[1])
             lj[i][2] = float(words[2])
             i=i+1


rc=5

for i in range(32000):
    tf=0
    for j in range(32000):
        if i!=j :
            r= (lj[i][0]-lj[j][0])*(lj[i][0]-lj[j][0]) + (lj[i][1]-lj[j][1])*(lj[i][1]-lj[j][1]) +(lj[i][2]-lj[j][2])*(lj[i][2]-lj[j][2])
            if r < rc*rc :
                f = 4*(pow((1/math.sqrt(r)),12)-pow((1/math.sqrt(r)),6))
                tf=tf+f
    lj[i][3] = round(tf,3)
    print(i,lj[i][3])

np.savetxt('/content/LJPotentialDataset.csv',lj,delimiter=',',fmt='%1.3f')