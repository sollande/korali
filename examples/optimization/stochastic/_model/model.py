#!/usr/bin/env python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1-d problem
def model(p):
  x = p["Parameters"]
  V = 0
  dim = 3

  vref = pd.read_csv('MOTUS_ref_data.csv')
  t= vref.t

  for i in range(dim):
    V = V + x[i]*(np.sin(2*np.pi/x[i+dim]*t+x[i+2*dim])+1)


  fig, ax = plt.subplots()
  ax.plot(t,vref.vx,c='r',label='Reference profile')
  ax.plot(t,V,c='b',label='Estimated profile')
  ax.set(title='Estimated vs Reference velocity profiles, Obj_f ='+str(-np.sqrt(np.sum((V-vref.vx.values)**2))))
  fig.set_size_inches(10,5)
  plt.xlabel('Time [s]')
  plt.ylabel('Horizontal velocity [m/s]')
  plt.legend()
  plt.savefig('figs/testfig_gen'+str(p["Current Generation"]).zfill(3) + '_sample' + str(p["Sample Id"]).zfill(4)+'.png')
  plt.close()
  p["F(x)"] = -np.sqrt(np.sum((V-vref.vx.values)**2))

# multi dimensional problem (sphere)
def negative_sphere(p):
    x = p["Parameters"]
    dim = len(x)
    res = 0.
    grad = [0.]*dim
    for i in range(dim):
        res += x[i]**2
        grad[i] = -x[i]

    p["F(x)"] = -0.5*res
    p["Gradient"] = grad

# multi dimensional problem (rosenbrock)
def negative_rosenbrock(p):
    x = p["Parameters"]
    dim = len(x)
    res = 0.
    grad = [0.]*dim
    for i in range(dim-1):
        res += 100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
        grad[i] += 2.*(1-x[i]) + 200.*(x[i+1]-x[i]**2)*2*x[i]
        grad[i+1] -= 200.*(x[i+1]-x[i]**2)

    p["F(x)"] = -res
    p["Gradient"] = grad

# multi dimensional problem (ackley)
def negative_ackley(p):
    x = p["Parameters"]
    a = 20.
    b = 0.2
    c = 2.*np.pi
    dim = len(x)

    sum1 = 0.
    sum2 = 0.
    for i in range(dim):
        sum1 += x[i]*x[i]
        sum2 += np.cos(c*x[i])

    sum1 /= dim
    sum2 /= dim
    r1 = a*np.exp(-b*np.sqrt(sum1))
    r2 = np.exp(sum2)

    p["F(x)"] = r1 + r2 - a - np.exp(1)
    
    grad = [0.]*dim
    for i in range(dim):
      grad[i] = r1*-1*b*0.5/np.sqrt(sum1)*1.0/dim*2.0*x[i]
      grad[i] -= r2*1.0/dim*np.sin(c*x[i])*c

    p["Gradient"] = grad
