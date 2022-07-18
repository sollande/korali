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
 
 SourceFolderName = "/users/dsolland/aphros-dev/sim/sim45_fans/"
 DestinationFolderName = "Results_gen"+str(p["Current Generation"]).zfill(3) + '_sample' + str(p["Sample Id"]).zfill(4)+'/'
 
 # Copy the 'model' folder into a temporary directory
 if os.path.exists( DestinationFolderName ):
  shutil.rmtree( DestinationFolderName)
 shutil.copytree( SourceFolderName, DestinationFolderName )

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
  f.write('set int report_step_every 10\n')
  f.write('set int stat_step_every 10\n')
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


 print(x[0],x[1],x[2],x[3])

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
 
 # Reference Results
 RefResultfile = "../MOTUS_ref_data.csv"
 vm_ref = pd.read_csv(RefResultfile)

 tref = vm_ref.t
 vmref = vm_ref.vx
 
 t = vm_calc.t
 vm = vm_calc.vx

 vmi = np.interp(tref, t, vm)

 aref = (vmref.values[1:]-vmref.values[:-1])/(tref.values[1:]-tref.values[:-1])
 a = (vmi[1:]-vmi[:-1])/(tref.values[1:]-tref.values[:-1])

 fig, ax = plt.subplots()
 ax.plot(aref,c='r',label='Reference profile')
 ax.plot(a,c='b',label='Estimated profile')
 ax.set(title='Estimated vs Reference velocity profiles slopes, Obj_f ='+str(-np.sqrt(np.sum((aref-a)**2))))
 fig.set_size_inches(10,5)
 plt.xlabel('Step')
 plt.ylabel('Horizontal velocity [m/s]')
 plt.legend()
 plt.savefig('../figs/aphrosfig_gen'+str(p["Current Generation"]).zfill(3) + '_sample' + str(p["Sample Id"]).zfill(4)+'.png')
 plt.close()

 with open('../params.txt', 'a') as g:
  g.write('GEN :'+str(p["Current Generation"])+' SAMPLE :'+str(p["Sample Id"])+'\n')
  g.write('parameters : ')
  for i in range(3*dim):
    g.write(' '+str(x[i]))
  g.write('\n')


 # Assigning objective function value
 p["F(x)"] = -np.sqrt(np.sum((aref-a)**2))

 # Move back to the base directory
 os.chdir( CurrentDirectory )
 shutil.rmtree( DestinationFolderName)
