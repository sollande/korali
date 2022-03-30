Study Case: Single Swimmer
==========================
This is a first test case with a single 3D fish. The fish is described by the StefanFish class (see _deps/CUP-3D/Obstacles/Stefanfish{.cpp,.h}).

The **reward** is the negative distance of the fish from a given point: -sqrt ( (x-xt)^2 + (y-yt)^2 ). No z coordinates are used at the moment.

The **state** is a 13-dimensional vector for this case: State = (dx,dy,dz,q0,q1,q2,q3,u,v,w,omega0,omega1,omega2). It contains the fish displacement, orientation (quaternions), translational velocity and angular velocity.

The **action** is given by modifying the baseline curvature :math:`C_0` and the swimming period :math:`T`.

The CUP3D commit that corresponds to this state is: d1636b2a0a18d9da66ae6690cf71d104674663e4

Settings
--------
The setup is described in the setting.sh file. The resolution is coarse and is only used for testing.

Running the code
----------------
You can submitt a job to the batch-system using

.. code-block:: bash

	./sbatch-run-vracer-swimmer.sh RUNNAME
