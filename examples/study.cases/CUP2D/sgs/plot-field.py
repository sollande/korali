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

dx = L/u.shape[0]
dy = L/v.shape[0]

um = np.roll(u, 1, axis=1)
dudy = (u-um)/dy

vm = np.roll(v, 1, axis=0)
dvdx = (u-vm)/dx

curl = dvdx - dudy

print(u.shape)
print(v.shape)
