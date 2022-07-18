#!/usr/bin/env python3
import os
import sys
import subprocess
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def model(p):

 x = p["Parameters"] 
 
 #SourceFolderName = "/users/dsolland/aphros-dev/sim/sim45_fans/"
 DestinationFolderName = "Results/"
 
 # Copy the 'model' folder into a temporary directory
 #if os.path.exists( DestinationFolderName ):
 # shutil.rmtree( DestinationFolderName)
 #shutil.copytree( SourceFolderName, DestinationFolderName )

 CurrentDirectory = os.getcwd()

 print(CurrentDirectory)

 # Move inside the temporary directory
 try:
  os.chdir( DestinationFolderName )
 except OSError as e:
  print("I/O error(" + str(e.errno) + "): " + e.strerror )
  print("The folder " + DestinationFolderName + " is missing")
  sys.exit(1)

 dim = 9 ### Set dimension
 
 # Storing base parameter file
 configFile='add.conf'
 if os.path.exists( configFile ):
  os.remove( configFile)
 with open(configFile, 'w') as f:
  f.write('set int ma 3\n')
  f.write('set int na 3\n')
  f.write('set double rho1 1.225\n')
  f.write('set double mu1 0.00001802\n')
  f.write('set vect x 0.5 0.5 0.5\n')
  f.write('set string dumplist \n')
  f.write('set string outfile Velocity.txt\n')
  f.write('set double tmax 10\n')
  f.write('set double xm 1\n')
  f.write('set string multiple Y\n')
  f.write('set int report_step_every 50\n')
  f.write('set int stat_step_every 50\n')
  f.write('set int dumppoly 0\n')
  f.write('set int dumppolymarch 0\n')
  
  f.write('set string amp')
  for i in range(dim):
    f.write(' '+str(x[i]))
  f.write('\n')
  f.write('set string pe')
  for i in range(dim,2*dim):
    f.write(' '+str(x[i]))
  f.write('\n')
  f.write('set string ph')
  for i in range(2*dim,3*dim):
    f.write(' '+str(x[i]))
  f.write('\n')


 # Run Aphros for this sample
 shutil.copyfile("../_model/run.sh",'run.sh')
 subprocess.call(["bash","run.sh"])
 

 # Loading results from file
 resultFile = 'Velocity.txt'
 try:
  with open(resultFile) as f:
   resultContent = f.read()
   print(resultFile)
 except IOError:
  print("[Korali] Error: Could not load result file: " + resultFile)
  exit(1)

 # Printing resultContent
 Results = pd.read_csv(resultFile,sep=",",names=['t','x','y','z','vx','vy','vz'])
 vm_calc = Results.loc[(Results['x']==1)&(Results['y']==0.5)&(Results['z']==0.5)&(Results['t']>=5)].reset_index().drop('index',axis=1)
 
 # Point to point fitting
 RefResultfile = "../MOTUS_ref_data.csv"
 vm_ref = pd.read_csv(RefResultfile)

 tref = vm_ref.t
 vmref = vm_ref.vx
 t = vm_calc.t
 vm = vm_calc.vx
 vmi = np.interp(tref, t, vm)

 # Slope fitting
 sref = pd.read_csv("../MOTUS_slope_data.csv")

 ts = sref.t
 vs = sref.vx
 vsi = np.interp(ts, t, vm)

 aref = np.sign((vs.values[1:]-vs.values[:-1]))
 a = np.sign((vsi[1:]-vsi[:-1]))
 

 print('ok')

 v_plot = Results.loc[(Results['x']==1)&(Results['y']==0.5)&(Results['z']==0.5)].reset_index().drop('index',axis=1)

 fig, ax = plt.subplots(1,2)
 ax[0].plot(tref,vmref,c='r',lw=2,label='Reference profile')
 ax[0].plot(tref,vmi,c='g',lw=2,label='Estimated profile')
 ax[0].plot(v_plot.t,v_plot.vx,'b--',lw=1,label='Full 10s Estimated profile')
 ax[0].set(title='Estimated vs Reference velocity profiles, Obj_f ='+str(-np.sqrt(np.sum((vmref.values-vmi)**2))))
 ax[0].set_xlabel('Time [s]')
 ax[0].set_ylabel('Horizontal velocity [m/s]')
 ax[0].legend()
 
 ax[1].quiver(sref.t.values[:-1],np.linspace(0,0,len(sref)-1),np.linspace(0,0,len(sref)-1),np.sign((sref.vx.values[1:]-sref.vx.values[:-1])),color='r',label='Reference')
 ax[1].quiver(sref.t.values[:-1],np.linspace(0,0,len(sref)-1),np.linspace(0,0,len(sref)-1),np.sign((vsi[1:]-vsi[:-1])),color='b',label='Estimated')
 ax[1].set(title='Signs, Obj_f ='+str(-np.sum(np.abs(aref-a))))
 ax[1].set_xlabel('Time [s]')
 ax[1].set_ylabel('Slope')
 ax[1].set_ylim(-1.5*np.abs(np.max(a)),1.5*np.abs(np.max(a)))
 ax[1].legend()
 
 fig.suptitle('Total Cost (fit +  0.05 x sign) = '+str(-(np.sqrt(np.sum((vmref.values-vmi)**2))+0.05*np.sum(np.abs(aref-a)))), fontsize=15)
 fig.set_size_inches(15,7)

 plt.savefig('../figs/aphrosfig_gen'+str(p["Current Generation"]).zfill(3) + '_sample' + str(p["Sample Id"]).zfill(4)+'.png')
 plt.close()

 with open('../params.txt', 'a') as g:
  g.write('GEN :'+str(p["Current Generation"])+' SAMPLE :'+str(p["Sample Id"])+'\n')
  g.write('parameters : ')
  for i in range(3*dim):
    g.write(' '+str(x[i]))
  g.write('\n')

 with open('../COSTS.txt', 'a') as g:
  g.write(str(p["Current Generation"])+','+str(p["Sample Id"])+','+str(-(np.sqrt(np.sum((vmref.values-vmi)**2))+0.05*np.sum(np.abs(aref-a))))+'\n')


 # Assigning objective function value
 p["F(x)"] = -(np.sqrt(np.sum((vmref.values-vmi)**2))+0.05*np.sum(np.abs(aref-a)))

 # Move back to the base directory
 os.remove(resultFile)
 os.chdir( CurrentDirectory )
 