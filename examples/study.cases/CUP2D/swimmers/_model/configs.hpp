//  Korali environment for CubismUP-2D
//  Copyright (c) 2020 CSE-Lab, ETH Zurich, Switzerland.


// General Simulation Options
std::string OPTIONS = "-bpdx 4 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 2 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

/* OBSTACLES */

// TASK 0
std::string OBJECTShalfDisk = "halfDisk angle=10 xpos=0.6 bForced=1 bFixed=1 xvel=0.15 tAccel=5 radius=";

// TASK 1
std::string OBJECTSnaca = "NACA L=0.12 xpos=0.6 angle=0 fixedCenterDist=0.299412 bFixed=1 xvel=0.15 Apitch=13.15 tAccel=5 Fpitch=";

// TASK 2, 3 & SWARM cases
std::string OBJECTSstefanfish = "stefanfish T=1 xpos=0.6 bFixed=1 pid=1 L=";

// TASK 4
std::string OBJECTSwaterturbine = "waterturbine semiAxisX=0.05 semiAxisY=0.017 xpos=0.4 bForced=1 bFixed=1 xvel=0.2 angvel=-0.79 tAccel=0 ";

// PERIODIC FISH
std::string AGENT_periodic = " \n\
stefanfish L=0.2 T=1 bFixed=1 ";

// dx=0.25, dy=0.5
// std::string OPTIONS_periodic = "-bpdx 2 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 5 -Ctol 0.01 -extent 1 -CFL 0.5 -poissonTol 5e-6 -poissonTolRel 0 -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

// Ballo of 100 fish
// std::string OPTIONS_periodic = " -bpdx 16 -bpdy 16 -levelMax 7 -levelStart 4 -Rtol 5.0 -Ctol 0.01 -extent 8 -CFL 0.5 -poissonTol 1e-6 -poissonTolRel 0.0 bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0.1 -nu 0.00004 -tend 0 ";

// dx=0.3, dy=0.2 (same as other diamond school)
// std::string OPTIONS_periodic = "-bpdx 3 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 0.6 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

std::string OPTIONS_periodic = "-bpdx 6 -bpdy 2 -levelMax 7 -levelStart 4 -Rtol 2 -Ctol 1 -extent 1.2 -CFL 0.4 -poissonTol 1e-5 -poissonTolRel 0  -bMeanConstraint 1 -bAdaptChiGradient 0 -tdump 0 -nu 0.00004 -tend 0 -muteAll 0 -verbose 1";

/* AGENTS */

// SINGLE AGENT
std::string AGENT = " \n\
stefanfish L=0.2 T=1";

std::string AGENTPOSX  = " xpos=";
std::string AGENTPOSY  = " ypos=";
std::string AGENTANGLE = " angle=";

// COLUMN OF FISH
// std::vector<std::vector<double>> initialPositions{{
// 	{0.90, 0.50},
// 	{1.20, 0.50},
// 	{1.50, 0.50}
// }};

//For Task 5
// PERIODIC FISH
// dx=0.25, dy=0.5
//std::vector<std::vector<double>> initialPositions{{
//  {0.25, 0.25},
//  {0.25, 0.75},
//  {0.50, 0.50},
//  {0.75, 0.25},
//  {0.75, 0.75}
// }};

// dx=0.3, dy=0.2 (fundamental building block diamond school)
// std::vector<std::vector<double>> initialPositions{{
// 	{0.15, 0.10},
// 	{0.45, 0.30}
// }};

std::vector<std::vector<double>> initialPositions{{
	{0.15, 0.10},
	{0.45, 0.30},
	{0.75, 0.10},
	{1.05, 0.30}
}};

// DIAMOND-SHAPED SCHOOLS

// 4 SWIMMERS
// small domain
// std::vector<std::vector<double>> initialPositions{{
// 	{0.90, 0.40},
// 	{0.90, 0.60},
// 	{1.20, 0.5}
// }};

// large domain
// std::vector<std::vector<double>> initialPositions{{
// 	{0.90, 0.90},
// 	{0.90, 1.10},
//  {1.20, 1.00}
// }};

// 9 SWIMMERS
// std::vector<std::vector<double>> initialPositions{{
// 	{0.90, 0.90},
// 	{0.90, 1.10},
// 	{1.20, 0.80},
// 	{1.20, 1.00},
// 	{1.20, 1.20},
// 	{1.50, 0.90},
// 	{1.50, 1.10},
// 	{1.80, 1.00}
// }};

// 16 SWIMMERS
// std::vector<std::vector<double>> initialPositions{{
// 	{0.90, 0.90},
// 	{0.90, 1.10},
// 	{1.20, 0.80},
// 	{1.20, 1.00},
// 	{1.20, 1.20},
// 	{1.50, 0.70},
// 	{1.50, 0.90},
// 	{1.50, 1.10},
// 	{1.50, 1.30},
// 	{1.80, 0.80},
// 	{1.80, 1.00},
// 	{1.80, 1.20},
// 	{2.10, 0.90},
// 	{2.10, 1.10},
// 	{2.40, 1.00}
// }};

// 25 SWIMMERS
// std::vector<std::vector<double>> initialPositions{{
// 	{0.90, 0.90},
// 	{0.90, 1.10},
// 	{1.20, 0.80},
// 	{1.20, 1.00},
// 	{1.20, 1.20},
// 	{1.50, 0.70},
// 	{1.50, 0.90},
// 	{1.50, 1.10},
// 	{1.50, 1.30},
// 	{1.80, 0.60},
// 	{1.80, 0.80},
// 	{1.80, 1.00},
// 	{1.80, 1.20},
// 	{1.80, 1.40},
// 	{2.10, 0.70},
// 	{2.10, 0.90},
// 	{2.10, 1.10},
// 	{2.10, 1.30},
// 	{2.40, 0.80},
// 	{2.40, 1.00},
// 	{2.40, 1.20},
// 	{2.70, 0.90},
// 	{2.70, 1.10},
// 	{3.00, 1.00},
// }};

#if 0
// 100 randomly placed fish
std::vector<std::vector<double>> initialPositions{{
	{2.382004, 3.623185},
	{4.492665, 4.346672},
	{3.888276, 4.337652},
	{4.059618, 3.881329},
	{4.785671, 4.310653},
	{4.454930, 4.256848},
	{4.795963, 4.669287},
	{5.047658, 3.776380},
	{2.988607, 4.382494},
	{2.841151, 4.519938},
	{3.631999, 3.611386},
	{3.930931, 3.969094},
	{2.354015, 3.787728},
	{4.474645, 4.964190},
	{3.776405, 4.219282},
	{4.734181, 3.120898},
	{5.451896, 4.336683},
	{5.498319, 4.486452},
	{4.033965, 3.098028},
	{4.882495, 4.896022},
	{5.743458, 4.108994},
	{5.007251, 3.384703},
	{2.764340, 3.676606},
	{2.894300, 3.362114},
	{3.039031, 4.248457},
	{3.685477, 4.855835},
	{5.652807, 4.423518},
	{4.251111, 4.644463},
	{3.515858, 4.574749},
	{3.189585, 3.112053},
	{3.325217, 3.429223},
	{5.810343, 3.880926},
	{2.829667, 4.762703},
	{4.683723, 4.111146},
	{4.075525, 4.247647},
	{2.995062, 3.912092},
	{4.439059, 3.882292},
	{4.605828, 3.402649},
	{4.300441, 3.150732},
	{4.735501, 3.695729},
	{5.790124, 4.164168},
	{5.808758, 4.502667},
	{2.562544, 4.016675},
	{2.718508, 4.301910},
	{4.618455, 3.563410},
	{3.914379, 3.665516},
	{5.139635, 4.246255},
	{4.660738, 3.222516},
	{4.411698, 3.307833},
	{5.780759, 3.650236},
	{4.870673, 3.436120},
	{4.078643, 4.132021},
	{3.722937, 4.618553},
	{3.128401, 3.682643},
	{3.473944, 3.291317},
	{2.420277, 4.338499},
	{2.993664, 4.623239},
	{4.959942, 3.524807},
	{3.741828, 3.181983},
	{3.122557, 3.258320},
	{4.606730, 3.479996},
	{3.927589, 4.796025},
	{5.092669, 4.421770},
	{5.719427, 4.319545},
	{2.838068, 3.515611},
	{3.691272, 3.344118},
	{4.511941, 4.471027},
	{2.170768, 4.014809},
	{4.890903, 3.323351},
	{3.320274, 3.497844},
	{4.979951, 3.608831},
	{3.092363, 4.552406},
	{3.688515, 4.374185},
	{3.279909, 3.825602},
	{3.727029, 3.810748},
	{4.794859, 4.546813},
	{3.788048, 4.694894},
	{5.504343, 3.516487},
	{4.525584, 4.724472},
	{4.780760, 4.841053},
	{3.243665, 4.497109},
	{3.534119, 4.103876},
	{3.673774, 4.159058},
	{4.744055, 3.995615},
	{5.751819, 3.991367},
	{2.820627, 3.959096},
	{4.078105, 3.767721},
	{2.900309, 4.036438},
	{3.780057, 4.075798},
	{2.664449, 3.452309},
	{3.175674, 3.737877},
	{4.429360, 4.610346},
	{3.434311, 3.220398},
	{2.394176, 4.063930},
	{4.527076, 4.881313},
	{2.709827, 3.625707},
	{3.300981, 3.174080},
	{2.628217, 4.581500},
	{2.624361, 4.648163},
	{4.627735, 3.952156}
}};
#endif
