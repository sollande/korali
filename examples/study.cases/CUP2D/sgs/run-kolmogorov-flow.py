#!/usr/bin/env python3

"""Kolmogorov Flow with a custom operator to compute the energy spectrum.
Output files are stored in output/."""

import cubismup2d as cup2d
import numpy as np
import argparse
import os

class CustomOperator(cup2d.Operator):

    def __init__(self, sim):
        super().__init__(sim)

        # Get number of frequency components
        numGridpoints = self.sim.cells[0]
        if numGridpoints%2 == 0:
            self.Nfreq = numGridpoints//2
        else:
            self.Nfreq = (numGridpoints+1)//2

        # Allocate Memory for energy spectrum and its square
        self.energySpectrum  = np.zeros((self.Nfreq,self.Nfreq))
        self.energySpectrumSq = np.zeros((self.Nfreq,self.Nfreq))
         
        self.timeStart = 10.0
        self.timeEnd = 50.0

        self.dumpIdx = 0
        self.dumpTimes = [1., 10., 20., 30., 40., 50., np.inf]
        
        # Flag to stop averaging
        self.done = False

    def __call__(self, dt: float):
        data: cup2d.SimulationData = self.sim.data
        
        # Skip transient region
        if (data.time > self.timeStart) and (not self.done):
            # Get the whole field as a large uniform matrix
            # Note that the order of axes is [y, x], not [x, y]!
            vel = data.vel.to_uniform()
            pres = data.pres.to_uniform()
            N = vel.shape[0]
            # print("Field:", vel, vel.shape)

            # Separate Field into x- and y-velocity and change order of axis
            u = vel[:,:,0].transpose()
            v = vel[:,:,1].transpose()
            p = pres[:,0].transpose()
            
            # print("Velocities:", u, v, u.shape, v.shape)
            
            # Dump velocity field
            if (data.time > self.dumpTimes[self.dumpIdx]):
                np.savez("Field_N={}_Cs={}_T={}.npz".format(N,data.smagorinskyCoeff,self.dumpTimes[self.dumpIdx]), u=u, v=v, p=p)
                self.dumpIdx += 1

            # Perform Fourier Transform on Fields
            Fu = np.fft.fft2(u)
            Fv = np.fft.fft2(v)
            # print("Transformed Velocities:", Fu, Fv, Fu.shape, Fv.shape )

            # Compute Energy
            # For real numbers the fourier transform is symmetric, so only half of the spectrum needed
            factor = 1 / ( 2 * N * N )
            energy = factor * np.real( np.conj(Fu)*Fu + np.conj(Fv)*Fv )
            energy = energy[:self.Nfreq,:self.Nfreq]
            # print("Computed Energies:", energy, energy.shape )

            # Compute temporal average of energy
            # Attention: dt is not constant!
            self.energySpectrum   += energy*dt
            self.energySpectrumSq += energy*energy*dt

            # Finalize Spectrum
            if (data.time >= self.timeEnd):
                self.done = True

                # Divide by Integration-Horizon and flatten for further processing
                averageEnergy = self.energySpectrum/(timeEnd-timeStart)
                averageEnergy = averageEnergy.flatten()
                # print("Average Energies Squared:", averageEnergySq, averageEnergySq.shape )
                averageEnergySq = self.energySpectrumSq/(timeEnd-timeStart)
                averageEnergySq = averageEnergySq.flatten()
                # print("Average Energies Squared:", averageEnergySq, averageEnergySq.shape )

                # Get Wavenumbers
                h = 2*np.pi/N
                freq = np.fft.fftfreq(N,h)[:self.Nfreq]
                # print(freq, freq.shape)

                # Create Flattened Vector with absolute values for Wavenumbers
                kx, ky = np.meshgrid(freq, freq)
                k = np.sqrt(kx**2 + ky**2)
                k = k.flatten()

                # Perform (k+dk)-wise integration
                midWavenumbers         = []
                averagedEnergySpectrum = []
                varianceEnergySpectrum = []
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
                    mean     = np.mean(   averageEnergy[indices] ) / mid_k
                    variance = np.mean( averageEnergySq[indices] - averageEnergy[indices]*averageEnergy[indices] ) / mid_k

                    # Append result
                    midWavenumbers.append(mid_k)
                    averagedEnergySpectrum.append(mean)
                    varianceEnergySpectrum.append(variance)

                #### Save Energy Spectrum
                np.savetxt("Energy_N={}_Cs={}.out".format(N,data.smagorinskyCoeff), [midWavenumbers,averagedEnergySpectrum,varianceEnergySpectrum])

parser = argparse.ArgumentParser()
parser.add_argument('--N', help='Number of Gridpoints per Dimension.', required=False, type=int, default=64)
parser.add_argument('--Cs', help='Smagorinsky Model constant Cs', required=False, type=float, default=0.0)
parser.add_argument('--runname', help='Name of the run.', required=False, type=str, default="/_CUP_results")
parser.add_argument('--tdump', help='Dump frequency.', required=False, type=float, default=0.)
args = parser.parse_args()

if os.environ.get('SCRATCH') != "":
    output_dir = os.environ.get('SCRATCH') + args.runname
else:
    output_dir = arg.runname

sim = cup2d.Simulation( cells=(args.N, args.N), nlevels=1,
                        start_level=0, extent=2.0*np.pi,
                        tdump=args.tdump, ic="random",
                        bForcing=1, output_dir=output_dir,
                        cuda=True, smagorinskyCoeff=args.Cs )
sim.init()
if args.Cs == 0:
    sim.insert_operator(CustomOperator(sim), after='advDiff')
else:
    sim.insert_operator(CustomOperator(sim), after='advDiffSGS')
sim.simulate(tend=50.1)
