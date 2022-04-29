#!/usr/bin/env python3

import cubismup2d as cup2d
import numpy as np
import sys
import os


################################### ATTENTION ###################################
# Make sure this computation corresponds to the one in run-kolmogorov-flow.py!! #
#################################################################################
class ComputeSpectralLoss(cup2d.Operator):
    def __init__(self, sim, stepsPerAction, pathToGroundtruth):
        super().__init__(sim)
        self.stepsPerAction = stepsPerAction

        # Get number of frequency components
        numGridpoints = self.sim.cells[0]
        if numGridpoints%2 == 0:
            self.Nfreq = numGridpoints//2
        else:
            self.Nfreq = (numGridpoints+1)//2

        # Load reference spectrum 
        data = np.loadtxt(pathToGroundtruth)
        self.referenceSpectrum = data[1,:]
        self.referenceVariance = data[2,:]

        # Container for old deviation and reward
        self.curDeviation = 0.0
        self.oldDeviation = 0.0
        self.reward = 0.0
        self.isTerminated = 0.0

    def __call__(self, dt: float):
        data: cup2d.SimulationData = self.sim.data

        if data._nsteps%self.stepsPerAction > 0:
            return

        # Get the whole field as a large uniform matrix
        # Note that the order of axes is [y, x], not [x, y]!
        vel = data.vel.to_uniform()
        N = vel.shape[0]
        # print("Field:", vel, vel.shape)

        # Separate Field into x- and y-velocity and change order of axis
        u = vel[:,:,0].transpose()
        v = vel[:,:,1].transpose()
        # print("Velocities:", u, v, u.shape, v.shape)

        # Perform Fourier Transform on Fields
        Fu = np.fft.fft2(u)
        Fv = np.fft.fft2(v)
        # print("Transformed Velocities:", Fu, Fv, Fu.shape, Fv.shape )

        # Compute Energy
        # For real numbers the fourier transform is symmetric, so only half of the spectrum needed
        factor = 1 / ( 2 * N * N )
        energy = factor * np.real( np.conj(Fu)*Fu + np.conj(Fv)*Fv )
        energy = energy[:self.Nfreq,:self.Nfreq]
        energy = energy.flatten()
        # print("Computed Energies:", energy, energy.shape )

        # Get Wavenumbers
        h = 2*np.pi/N
        freq = np.fft.fftfreq(N,h)[:self.Nfreq]
        # print(freq, freq.shape)

        # Create Flattened Vector with absolute values for Wavenumbers
        kx, ky = np.meshgrid(freq, freq)
        k = np.sqrt(kx**2 + ky**2)
        k = k.flatten()

        # Perform (k+dk)-wise integration
        averagedEnergySpectrum = []
        # dk = k[1]
        # wavenumbers = np.arange(0,k[-1]+dk,dk)
        # for i, _k in enumerate(wavenumbers[:-1]):
        for i, _k in enumerate(freq[:-1]):
            # Get indices of entries that are in this shell
            # next_k  = wavenumbers[i+1]
            next_k  = freq[i+1]
            mid_k   = _k + (next_k - _k)/2
            indices = (_k <= k) & (k < next_k)

            # Compute mean and variance
            mean     = np.mean(   energy[indices] ) / mid_k

            # Append result
            averagedEnergySpectrum.append(mean)

        deviation = ( averagedEnergySpectrum - self.referenceSpectrum[:self.Nfreq-1] ) / self.referenceSpectrum[:self.Nfreq-1]

        # Store reward
        self.oldDeviation = self.curDeviation
        self.curDeviation = np.linalg.norm( deviation, ord=2 )

        # Deviation from Mean Spectrum
        if np.isfinite(self.oldDeviation - self.curDeviation):
            self.reward = self.oldDeviation - self.curDeviation
            self.isTerminated = False
        else:
            self.reward = -1
            self.isTerminated = True

        #### Log-likelihood for Gaussian ####
        #####################################
        # logLikelihood = deviation*deviation / ( 2* self.referenceVariance) - 1/2*np.log(np.mean(self.referenceVariance))

        # self.reward = np.mean(logLikelihood)

def runEnvironment(s, env, numblocks, stepsPerAction, pathToGroundtruth):
    # Save rundir
    curdir = os.getcwd()

    # Create output directory (if it does not exist already)
    outputDir = "_trainingResults/simulationData/"

    if not os.path.isdir(outputDir):
        os.mkdir(outputDir)

    # Go to output directory
    os.chdir(outputDir)

    # Initialize Simulation
    # Set smagorinskyCoeff to something non-zero to enable the SGS
    if env == "rectangle":
        sim = cup2d.Simulation(cells=(numblocks*16, numblocks*8), nlevels=1, start_level=0, extent=4, tdump=0.0, smagorinskyCoeff=0.1, mute_all=True, output_dir="./" )
        rectangle = cup2d.Rectangle(sim, a=0.2, b=0.2, center=(0.5, 0.5), vel=(0.2, 0.0), fixed=True, forced=True)
        sim.add_shape(rectangle)
        print("TODO")
        sys.exit()

    if env == "kolmogorovFlow":
        sim = cup2d.Simulation( cells=(numblocks*8, numblocks*8), nlevels=1,
                        start_level=0, extent=2.0*np.pi,
                        tdump=args.tdump, dumpCs=args.dumpCs, ic="random",
                        BCx="periodic", BCy="periodic",
                        forcingC=4, forcingW=4, nu=0.05,
                        bForcing=1, output_dir=output_dir,
                        cuda=False, smagorinskyCoeff=args.Cs )

    sim.init()
    spectralLoss = ComputeSpectralLoss(sim, stepsPerAction, pathToGroundtruth)
    sim.insert_operator(spectralLoss, after='advDiffSGS')
   
    # Accessing fields
    data: cup2d.SimulationData = sim.data

    # Simulate transient period
    sim.simulate(tend=10.)

     # Get Initial State
    states = []
    for velBlock in data.vel.blocks:
        velData = velBlock.data
        flowVelFlatten = velData.flatten()
        states.append(flowVelFlatten.tolist())
    s["State"] = states

    step = 0
    terminal = False

    while not terminal and step < 1000:
        # Compute Action
        s.update()
        actions = s["Action"]
        for i, CsBlock in enumerate(data.Cs.blocks):
            action = actions[i]
            Cs = np.reshape(action, (8,8))
            CsBlock = Cs

        # Simulate for given number of steps
        sim.simulate(nsteps=stepsPerAction)

        # Record reward and termination
        reward = spectralLoss.reward
        terminal = spectralLoss.isTerminated

        states = []
        rewards = []
        for velBlock in data.vel.blocks:
            velData = velBlock.data
            flowVelFlatten = velData.flatten()
            states.append(flowVelFlatten.tolist())
            rewards.append(reward)
        s["State"] = states
        print(rewards)
        s["Reward"] = rewards

        # Advancing step counter
        step = step + 1

    # Setting termination status
    if terminal:
        s["Termination"] = "Terminal"
    else:
        s["Termination"] = "Truncated"

    # Go back to base-directory and reset cout
    os.chdir(curdir)
