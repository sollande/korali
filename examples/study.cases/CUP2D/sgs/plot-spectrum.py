import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns
# sns.set_theme()
# sns.set_style("whitegrid")
# sns.set(rc={"xtick.minor.visible" : True, "ytick.minor.visible" : True})

#### load data
pathSCRATCH = "./"
runname = [ f"Energy_N={N}_Cs=0.0.out" for N in 2**np.arange(3,9) ]

#### plot data
for i, run in enumerate(runname):
	data = np.loadtxt(pathSCRATCH+run)
	freq = data[0,:]
	energy = data[1,:] / (2**(4+i))**2
	plt.loglog(freq, energy, label="N={}".format(2**(4+i)))

#### plot theoretical spectrum
wavenumbers = np.arange(freq[5], freq[-1], 0.15915494309189535)
wavenumbersKolmogorov = np.arange(0, freq[5], 0.15915494309189535)
plt.loglog(wavenumbersKolmogorov, 5*10**-2*wavenumbersKolmogorov**(-5/3), 'k-', label="$\\propto k^{-5/3}$")
plt.loglog(wavenumbers, 5*10**-2*wavenumbers**(-3)*np.log(wavenumbers)**(1/3), 'k--', label="$\\propto k^{-3}\log(k)^{-1/3}$")

#### adjust plotting parameteres
plt.tick_params(axis='both', which='minor')
plt.xlabel("Wavenumber $k$")
plt.ylabel("Energy $E(k)$")
plt.legend()
plt.rcParams["figure.figsize"] = (12,4)
plt.tight_layout()
plt.show()
# plt.savefig("KFspectrum.eps", dpi=300)
