import numpy as np
import matplotlib.pyplot as plt

# define size of domain
Lx = 4
Ly = 4

# margins at boundary
marginLeft        = 0.6
marginRight       = 1
yMargin           = 0.5

# define diamond parameters
dx = 0.3
dy = 0.2

# compute number of fish per direction
Nx = int(( Lx - marginLeft - marginRight ) / dx) + 1
NyOdd = int(( Ly - 2*yMargin ) / dy) + 1
NyEven = NyOdd-1

# create initialPosition
print("std::vector<std::vector<double>> initialPositions{{")

x0     = marginLeft
y0odd  = yMargin
y0even = yMargin+dy/2
N = 0
plt.xlim([0,Lx])
plt.ylim([0,Ly])
for i in range(Nx):
	if i % 2 == 0:
		y0 = y0even
		Ny = NyEven
	else:
		y0 = y0odd
		Ny = NyOdd
	for j in range(Ny):
		x = x0+i*dx
		y = y0+j*dy
		plt.plot(x,y,"D")

		if( (i == Nx-1) and (j == Ny-1) ):
			print("{","{:.2f}, {:.2f}".format( x, y ),"}")
		else:
			print("{","{:.2f}, {:.2f}".format( x, y ),"},")

		N = N+1

print("}};")

print("This are initial condition for {} fish".format(N))

plt.show()