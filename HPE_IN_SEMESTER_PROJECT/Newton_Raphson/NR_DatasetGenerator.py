import random
import numpy as np

epsilon=1e-5
x1=0
g=0
def newtons_method( x0, a, b, c):
    j=0
    if df(x0,a,b,c)!=0:
      x1=x0-f(x0,a,b,c)/df(x0,a,b,c)
      while abs(x1-x0) > epsilon  and df(x0,a,b,c)!=0:
          x0=x1
          x1 = x0 - f(x0,a,b,c)/df(x0,a,b,c)
          j=j+1

      if abs(x1 - x0) <= epsilon and abs(x1)>=1 :
        a2.append(round(a,2))
        b2.append(round(b,2))
        c2.append(round(c,2))
        res.append(round(x1,3))
        return 1
    return 0

def f(x, a, b, c):
    return a*(x**2)+b*x+c
def df(x, a, b, c):
    return 2*a*x+b
size=115712
train_x= np.arange(4*size,dtype=np.float32).reshape(size,4)

res=[]
a2=[]
b2=[]
c2=[]
n=0

for i in range(0,9000000):
  a = random.randint(-100,100)
  b = random.randint(-100,100)
  c = random.randint(-100,100)

  x0 = 100
  if((b*b-4*a*c)<=0 or a==0 or b==0 or c==0) :
    continue


  if(n==size+1):
      break

  if newtons_method( x0, a, b, c)==1:
      n = n + 1
print ('a  b  c   res')
for i in range(0,n-1):
  print (i+1,a2[i],b2[i],c2[i],res[i])
  train_x[i][0]=a2[i]
  train_x[i][1] = b2[i]
  train_x[i][2] = c2[i]
  train_x[i][3]=res[i]

np.savetxt('E:/hpnn/newtonraphson_dataset.csv', train_x,delimiter=',',fmt='%1.3f')

