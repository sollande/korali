import argparse
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt

####### Parsing arguments
parser = argparse.ArgumentParser()
parser.add_argument('--filename', help='NPZ file with flow field.', required=True, type=str)
parser.add_argument('--output', help='Name of plot.', required=False, type=str, default='field.png')
parser.add_argument('--nfilter', help='Freq filter.', required=True, type=int)
parser.add_argument('--L', help='Length of field.', required=False, type=float, default=2.*np.pi)
args = parser.parse_args()

L = args.L
Nf = args.nfilter
data = np.load(args.filename)

u = data['u']
v = data['v']

N = u.shape[0]

dx = L/N
dy = L/N

um = np.roll(u, 1, axis=1)
dudy = (u-um)/dy

vm = np.roll(v, 1, axis=0)
dvdx = (v-vm)/dx

curl = dvdx - dudy

Fu = np.fft.fft2(u)
Fv = np.fft.fft2(v)

FreqCompRows = np.fft.fftfreq(Fu.shape[0], L / (2*np.pi*N))
FreqCompCols = np.fft.fftfreq(Fu.shape[1], L / (2*np.pi*N))

idxH = np.abs(FreqCompCols) > Nf//2
idxH2d = np.ix_(idxH, idxH)

Fuh = Fu
Fvh = Fv

Fuh[idxH2d] = 0
Fvh[idxH2d] = 0

uh = np.real(np.fft.ifft2(Fuh))
vh = np.real(np.fft.ifft2(Fvh))

xaxis = np.arange(0,L,dx)
yaxis = np.arange(0,L,dx)

fuh = interpolate.interp2d(xaxis, yaxis, uh, kind='cubic')
fvh = interpolate.interp2d(xaxis, yaxis, vh, kind='cubic')

uhm = np.roll(uh, 1, axis=1)
duhdy = (uh-uhm)/dy

vhm = np.roll(vh, 1, axis=0)
dvhdx = (vh-vhm)/dx

curlh = dvhdx - duhdy

X, Y = np.meshgrid(xaxis, yaxis, indexing='ij')
    
umax = max(u.max(), v.max())
umin = min(u.min(), v.min())
ulevels = np.linspace(umin, umax, 20)

fig1, axs1 = plt.subplots(2, 3, sharex='col', sharey='col', figsize=(10,10))
axs1[0,0].contourf(X, Y, u, ulevels)
axs1[0,0].set_aspect('equal', adjustable='box')

axs1[0,1].contourf(X, Y, v, ulevels)
axs1[0,1].set_aspect('equal', adjustable='box')

axs1[0,2].contourf(X, Y, curl)#, ulevels)
axs1[0,2].set_aspect('equal', adjustable='box')

Xh, Yh = np.meshgrid(xaxis, yaxis, indexing='ij')

axs1[1,0].contourf(Xh, Yh, uh, ulevels)
axs1[1,0].set_aspect('equal', adjustable='box')

axs1[1,1].contourf(Xh, Yh, vh, ulevels)
axs1[1,1].set_aspect('equal', adjustable='box')

axs1[1,2].contourf(Xh, Yh, curlh) #, ulevels)
axs1[1,2].set_aspect('equal', adjustable='box')

fig1.savefig(args.output)


