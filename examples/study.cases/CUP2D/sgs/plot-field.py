import argparse
import numpy as np
import matplotlib.pyplot as plt

####### Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', help='NPZ file with flow field.', required=True, type=str)
parser.add_argument('--L', help='Length of field.', required=False, type=float, default=2.*np.pi)
args = parser.parse_args()

L = args.L
data = np.load(args.filename)

u = data['u']
v = data['v']

print(u.shape)
print(v.shape)
print(u[:,-1])
print(u[:,0])
print(u[:,1])
print(u[:5,:5])

dx = L/u.shape[0]
dy = L/v.shape[0]

um = np.roll(u, 1, axis=1)
dudy = (u-um)/dy

vm = np.roll(v, 1, axis=0)
dvdx = (u-vm)/dx

curl = dvdx - dudy

xaxis = L/np.arange(1,u.shape[0]+1)
yaxis = L/np.arange(1,v.shape[0]+1)
    
umax = max(u.max(), v.max())
umin = min(u.min(), v.min())
ulevels = np.linspace(umin, umax, 50)

fig1Name="field.png"

fig1, axs1 = plt.subplots(1, 3, sharex='col', sharey='col', figsize=(10,10))
axs1[0].contourf(xaxis, yaxis, u) #, ulevels)
axs1[0].set_aspect('equal', adjustable='box')

axs1[1].contourf(xaxis, yaxis, v) #, ulevels)
axs1[1].set_aspect('equal', adjustable='box')

axs1[2].contourf(xaxis, yaxis, curl) #, ulevels)
axs1[2].set_aspect('equal', adjustable='box')

fig1.savefig(fig1Name)


