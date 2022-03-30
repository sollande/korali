import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy.signal import savgol_filter
from scipy.ndimage.filters import uniform_filter1d

def plotQuick():
	root = "/scratch/snx3000/pweber/korali/testhalfDisk/_trainingResults/sample00003460/"

	velData = np.loadtxt(root+"velocity_1.dat", skiprows=1)
	plt.plot(velData[:,0], velData[:,2], label='x-coordinate')
	plt.plot(velData[:,0], velData[:,3], label='y-coordinate')
	plt.ylim([0.8,1.4])
	plt.legend()
	plt.show(block=False)

	plt.plot(velData[:,0], velData[:,7], label='x-velocity')
	plt.plot(velData[:,0], velData[:,8], label='y-velocity')
	plt.ylim([-1,1])
	plt.show()

def plotEfficiency():
	rootControlled = "/scratch/snx3000/mchatzim/CUP2D/School_Control_09Fish/"
	rootUncontrolled = "/scratch/snx3000/mchatzim/CUP2D/School_NoControl_09Fish/"
	rootSingle = "/scratch/snx3000/mchatzim/CUP2D/single-fish/"

	row = 9

	powDataSingle = np.loadtxt(rootSingle+"powerValues_0.dat", skiprows=1)
	powDataSingleAverage = powDataSingle[:,row]
	powDataSingleSmoothned = savgol_filter(powDataSingleAverage, 51, 3)
	plt.plot(powDataSingle[:,0], powDataSingleSmoothned, linewidth=1, label='single fish')
	powDataSingleAveraged = uniform_filter1d(powDataSingleAverage, size=1000)
	plt.plot(powDataSingle[:,0], powDataSingleAveraged, "--", linewidth=2, label='single fish, average')

	numSwimmersUncontrolled = 9
	powDataUncontrolled = np.loadtxt(rootUncontrolled+"powerValues_0.dat", skiprows=1)
	powDataUncontrolledAverage = np.zeros(powDataUncontrolled[:,row].shape)
	for i in range(0,numSwimmersUncontrolled):
		powDataUncontrolled = np.loadtxt(rootUncontrolled+"powerValues_{}.dat".format(i), skiprows=1)
		powDataUncontrolledAverage = powDataUncontrolledAverage + powDataUncontrolled[:,row]
	powDataUncontrolledSmoothened = savgol_filter(powDataUncontrolledAverage, 51, 3)
	plt.plot(powDataUncontrolled[:,0], powDataUncontrolledSmoothened/numSwimmersUncontrolled, linewidth=1, label='9 fish, uncontrolled')
	powDataUncontrolledAveraged = uniform_filter1d(powDataUncontrolledAverage, size=1000)
	plt.plot(powDataUncontrolled[:,0], powDataUncontrolledAveraged/numSwimmersUncontrolled, "--", linewidth=2, label='9 fish, uncontrolled, average')

	powDataControlled = np.loadtxt(rootControlled+"powerValues_0.dat", skiprows=1)
	powDataAverage = np.zeros(powDataControlled[:,row].shape)
	for i in range(0,9):
		powDataControlled = np.loadtxt(rootControlled+"powerValues_{}.dat".format(i), skiprows=1)
		powDataAverage = powDataAverage + powDataControlled[:,row]
	powDataAverageSmoothened = savgol_filter(powDataAverage, 51, 3)
	plt.plot(powDataControlled[:,0], powDataAverageSmoothened/9, linewidth=1, label='9 fish, controlled')
	powDataAverageAveraged = uniform_filter1d(powDataAverage, size=1000)
	plt.plot(powDataControlled[:,0], powDataAverageAveraged/9, "--", linewidth=2, label='9 fish, controlled, average')

	plt.xlim([0,30])
	plt.legend()
	plt.show()

def plotCoM():
	# path = "/scratch/snx3000/pweber/korali/swarm9-testing/_testingResults/sample000/"
	path = "./"
	
	# Load data, set up figure, axis, and plot element we want to animate
	fig = plt.figure()
	fig.patch.set_alpha(0.)
	ax = plt.axes(xlim=(0.4, 2),ylim=( 0.6, 1.4 ) )
	ax.set_axis_off()
	lines = []
	data = []
	for i in range(0,9):
		data.append( np.loadtxt(path+"velocity_{}.dat".format(i), skiprows=1) )
		lines.append(ax.plot([], [], lw=2, color="C{}".format(i))[0])

	# initialization function: plot the background of each frame
	def init():
	  for line in lines:
	    line.set_data([], [])
	  return lines

	# animation function.  This is called sequentially
	def animate(i):
	  plt.title("Center of Mass, t={}".format(data[0][i,0]))
	  for j in range(0,9):
	  	lines[j].set_data(data[j][:i,2], data[j][:i,3])
	  return lines

	# call the animator.  blit=True means only re-draw the parts that have changed.
	anim = animation.FuncAnimation(fig, animate, init_func=init,
	                               frames=len(data[0][:,0]), interval=1, blit=True)

	# save the animation as an mp4
	anim.save('CoM_animation_fish.mp4', fps=200, extra_args=['-vcodec', 'libx264'])

if __name__ == '__main__':
	# plotCoM()
	plotEfficiency()