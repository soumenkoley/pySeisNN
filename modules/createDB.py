#!/usr/bin/env python
# coding: utf-8

# In[64]:


import numpy as np
from scipy.io import savemat
from pysurf96 import surf96
from modules import configLoader
from modules import validateInputs
import matplotlib.pyplot as plt
import subprocess
import os
import shutil
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

def main():
    # load the configuration
    config = configLoader.load_config("configParse.ini");
    validateInputs.validateInputs(config);
    
    # prepare all Qseis input files
    #splitAll = createFinalInputs(config)
    
    #print('Cores = ' + str(config.cpuCores))
    
    # run Qseis for all depth and receiver locations
    #testRuns = splitAll[0:2,:];
    #runMultiQseis(testRuns, config.inputPath, config.qseisExe, nWorkers = config.cpuCores)

    #createFullDataBase("configParse.ini");

def createFullDataBase(configFile="configParse.ini", testRuns=None):
    config = load_config(configFile)
    validateInputs.validateInputs(config)
    splitAll = createFinalInputs(config)

    print(f"Prepared {splitAll.shape[0]} total runs")
    print(f"Cores available: {config.cpuCores}")

    if testRuns is not None:
        runsToExec = splitAll[testRuns, :]
        print(f"Running test subset: {len(runsToExec)} runs")
    else:
        runsToExec = splitAll

    results = runMultiQseis(runsToExec, config.inputPath, config.qseisExe, nWorkers = config.cpuCoresQseis)
    print("Database generation complete")
    return results

def createFullDataBaseTD(zList, configFile="configParse.ini", testRuns=None):
    config = load_config(configFile)
    validateInputs.validateInputs(config)
    splitAll = createFinalInputsTD(config,zList)

    print(f"Prepared {splitAll.shape[0]} total runs")
    print(f"Cores available: {config.cpuCores}")

    if testRuns is not None:
        runsToExec = splitAll[testRuns, :]
        print(f"Running test subset: {len(runsToExec)} runs")
    else:
        runsToExec = splitAll

    results = runMultiQseis(runsToExec, config.inputPath, config.qseisExe, nWorkers = config.cpuCoresQseis)
    print("Database generation complete")
    return results
    
def runSingleQseis(layerName, inputPath, qseisExe):
    """Run one QSEIS instance in its own directory.
    Used for the parallel execution
    inputs: layerName: string of type "Layer0Run0" or similar
    inputPath: path where qseis inputs exist, check config file input_folder
    qseisExe: path to Qsesis executable, specified in config file
    outputs: saved in inputPath/Layer*Run*/fz-2.tz, fz-2.tr, fh-2.tr, fh-2.tz, fh-2.tt
    """
    runDir = os.path.join(inputPath, layerName)
    inputFileName = f"{layerName}.dat"
    os.makedirs(runDir, exist_ok=True)

    # Copy input file into that folder
    src = os.path.join(inputPath, inputFileName)
    dst = os.path.join(runDir, inputFileName)
    shutil.copy(src, dst)

    worker_id = os.getpid()
    start_time = time.strftime("%H:%M:%S")
    print(f"[Worker {worker_id}] Starting job {layerName} at {start_time}")

    t0 = time.time()
    # Run QSEIS process
    result = subprocess.run(
        [qseisExe],
        cwd=runDir,
        input=f"{inputFileName}\n",
        capture_output=True,
        text=True
    )

    dt = time.time() - t0
    end_time = time.strftime("%H:%M:%S")

    print(f"[Worker {worker_id}] Finished job {layerName} at {end_time} ({dt/60:.1f} min, rc={result.returncode})")
    
    return {
        "layerName": layerName,
        "returncode": result.returncode,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }

def runMultiQseis(splitAll, inputPath, qseisExe, nWorkers=1):
    """
    Run all QSEIS input files either sequentially or in parallel.
    nWorkers = 1 for sequential (for testing)
    nWorkers > 1 for parallel using ProcessPoolExecutor
    default workers set 1
    executes runSingleQseis in parallel
    """
    jobs = []
    for row in splitAll:
        layerName = f"Layer{int(row[0])}Run{int(row[1])}"
        jobs.append(layerName)

    print(f"Preparing to run {len(jobs)} QSEIS jobs with {nWorkers} worker(s)...")

    results = []

    if nWorkers == 1:
        # Sequential mode
        for layerName in jobs:
            print(f"Running {layerName} ...")
            res = runSingleQseis(layerName, inputPath, qseisExe)
            if res["returncode"] == 0:
                print(f"Completed {layerName}")
            else:
                print(f"Error in {layerName}:\n{res['stderr']}")
            results.append(res)

    else:
        # Parallel mode
        with ProcessPoolExecutor(max_workers=nWorkers) as executor:
            futures = {
                executor.submit(runSingleQseis, layerName, inputPath, qseisExe): layerName
                for layerName in jobs
            }

            for future in as_completed(futures):
                layerName = futures[future]
                try:
                    res = future.result()
                    if res["returncode"] == 0:
                        print(f"Completed {layerName}")
                    else:
                        print(f"Error in {layerName}:\n{res['stderr']}")
                    results.append(res)
                except Exception as e:
                    print(f"Exception in {layerName}: {e}")

    print("All QSEIS runs finished.")
    return results

    
def runSerialQseis(splitAll,fInpPath,qseisExe):
    # performs a single Qseis run
    nRuns = len(splitAll[:,0]); # total number of Qseis runs

    # loop over all runs, sequential at the moment
    for runNo in range(0,nRuns):
        layerName = 'Layer' + str(int(splitAll[runNo,0])) + 'Run' + str(int(splitAll[runNo,1]))
        print(f" Running QSEIS for {layerName}...")
        inputFileName = layerName + '.dat'
        
        # Each run happens inside its input folder
        runDir = os.path.join(fInpPath, layerName)

        # make the folder in which Qseis will run
        os.makedirs(runDir, exist_ok=True)

       # Copy input file into that folder
        inputPath = os.path.join(fInpPath, inputFileName)
        shutil.copy(inputPath, os.path.join(runDir, inputFileName))

        # Run QSEIS from inside that folder
        process = subprocess.Popen(
            [qseisExe],
            cwd=runDir,               # must be in folder where .dat is
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Send input file name (simulate typing)
        stdout, stderr = process.communicate(f"{inputFileName}\n")

        if process.returncode != 0:
            print(f"Error running {inputFileName}:\n{stderr}")
            continue

        print(f" Completed {inputFileName}")

def createFinalInputsTD(config,zList):
    """
    computes the grid size, and number of receiver points along X and Z(depth) direction
    and then generates the input files necessary for QSEIS run
    different from the original definition of createFinalInputs as it generates the input files
    only for two depths defined by zList
    """
    [depths,nX,nY,gridSize,sr] = getNXDP(config);
    
    print('Layer depths = '+ str(depths))
    print('Number of points per layer along Z = ' + str(nY))
    print('Spacing in each layer in meters = ' + str(gridSize))
    
    [zAll,nXAll, srAll] = getRecInfoTD(depths,nX,nY,gridSize,sr,zList);
    
    cleanInputFiles(config.inputPath,config.delFlag);
    
    splitAll = createAllInp(config,zAll,nXAll,srAll)
    #print(splitAll)
    
    # save splitAll for later use
    splitDic = {"splitAll": splitAll}
    fSavePath = config.inputPath + 'splitAll.mat'
    savemat(fSavePath, splitDic);
    
    return splitAll
    
def createFinalInputs(config):
    """
    computes the grid size, and number of receiver points along X and Z(depth) direction
    and then generates the input files necessary for QSEIS run
    """
    [depths,nX,nY,gridSize,sr] = getNXDP(config);
    
    print('Layer depths = '+ str(depths))
    print('Number of points per layer along Z = ' + str(nY))
    print('Spacing in each layer in meters = ' + str(gridSize))
    
    [zAll,nXAll, srAll] = getRecInfo(depths,nX,nY,gridSize,sr);
    
    cleanInputFiles(config.inputPath,config.delFlag);
    
    splitAll = createAllInp(config,zAll,nXAll,srAll)
    #print(splitAll)
    
    # save splitAll for later use
    splitDic = {"splitAll": splitAll}
    fSavePath = config.inputPath + 'splitAll.mat'
    savemat(fSavePath, splitDic);
    
    return splitAll
    
def cleanInputFiles(qseisFold,delFlag):
    for item in os.listdir(qseisFold):
        path = os.path.join(qseisFold, item)
    
        # Delete .dat input files
        if os.path.isfile(path) and path.endswith(".dat"):
            os.remove(path)
            #print(f"Removed file: {path}")
    
        # Delete subfolders (previous output folders)
        elif os.path.isdir(path):
            if(delFlag=="yes"):
                shutil.rmtree(path)
                print("QSEIS output folders from past runs deleted!")
            else:
                print("QSEIS output folders from past runs not deleted!")
            #print(f"Removed folder: {path}")

    print("Qseis Inputs cleaned, parent folder intact.");
    
def createAllInp(config, zAll, nXAll, srAll):
    # this loops over all zAll and creates all the input files necessary for QSEIS
    print('Creating all input files ...');
    splitAll = [];
    for i in range(0,len(zAll)):
        splits,nSplits = createQseisInp(config,zAll[i],nXAll[i], srAll[i], i);
        for j in range(0,nSplits):
            splitTemp = [];
            splitTemp.extend([int(i), int(j), zAll[i], splits[j,0], splits[j,1], int(splits[j,2])]);
            splitAll.append(splitTemp);
    splitAllArr = np.array(splitAll);
    print('All input files created!');
    return splitAllArr;
    
def createQseisInp(config,zAll,nXAll, srAll, layerNo):
    # this function copies the template file, edits it with the placeholders, and then creates a
    # new input file, splits are made if the number of receivers exceed maxRec
    # first perform the split
    
    #print(nXAll)
    splits,nSplits = splitRunCont(10,config.xMaxGF,int(nXAll),config.maxRec);
    velBlock,nLines = formatVelModel(config.thickness, config.vP, config.vS, config.rho, config.qP, config.qS)
    for i in range(0,nSplits):
        outFile = config.inputPath + 'Layer' + str(layerNo) + 'Run' + str(i) + '.dat';
        # copy the template
        shutil.copy(config.templateFile, outFile);
        with open(outFile, "r") as f:
            content = f.read();
        receiver_depth = zAll/1000  # depth in km of receiver
        n_points = int(splits[i,2])         # example number of points
        d1 = splits[i,0]/1000            # start distance [km]
        dn = splits[i,1]/1000               # end distance [km]
        srN = srAll
        tmax = config.tMax
        nsamp = config.nSamp

        content = content.format(
            velocity_model=velBlock,
            n_lines=nLines,
            receiver_depth=receiver_depth,
            n_points=n_points,
            d1=d1,
            dn=dn,
            tmax=tmax,
            nsamp=nsamp,
            srN=srN
        )
        with open(outFile, "w") as f:
            f.write(content)
     
    return splits, nSplits
    
def formatVelModel(thickness, vP, vS, rho, Qp, Qs):
    """
    Creates the QSEIS velocity model text block.
    Inputs in meters, m/s, kg/m^3.
    Returns a string suitable for direct insertion into the template.
    The half-space (last layer) is only printed once (top).
    """
    # Convert to QSEIS units (km, km/s, g/cm^3)
    thickness_km = thickness / 1000.0
    vP_km = vP / 1000.0
    vS_km = vS / 1000.0
    rho_gcc = rho / 1000.0

    # Compute interface depths
    z_interfaces = np.insert(np.cumsum(thickness_km[:-1]), 0, 0.0)
    n_layers = len(thickness_km)

    lines = []
    #lines.append("  The layered model of source site:")
    #lines.append("    no   z(km)    vp(km/s)   vs(km/s)  ro(g/cm^3)    qp       qs")

    n_lines = 0  # counter for number of printed rows

    for i in range(n_layers):
        z_top = z_interfaces[i]
        if i < n_layers - 1:  # finite layer: print top and bottom
            z_bot = z_interfaces[i] + thickness_km[i]
            for z in [z_top, z_bot]:
                lines.append(
                    f"{n_lines+1:3d}{z:10.3f}{vP_km[i]:10.4f}{vS_km[i]:10.4f}"
                    f"{rho_gcc[i]:10.4f}{Qp[i]:8.1f}{Qs[i]:7.1f}"
                )
                n_lines += 1
        else:  # half-space: print only top
            lines.append(
                f"{n_lines+1:3d}{z_top:10.3f}{vP_km[i]:10.4f}{vS_km[i]:10.4f}"
                f"{rho_gcc[i]:10.4f}{Qp[i]:8.1f}{Qs[i]:7.1f}"
            )
            n_lines += 1

    # Prepend line count
    #header = f"  {n_lines}"
    full_block = "\n".join(lines)

    return full_block, n_lines
    
def splitRunCont(dMin, dMax, nX, maxPoints):
    """
    Split distance range into sub-runs while keeping points continuous.
    Returns list of (d1, dn, n_points) tuples.
    """
    if nX <= maxPoints:
        # No split needed
        nSplits = 1
        splits = np.zeros((1,3))
        splits[0,0] = dMin; splits[0,1] = dMax; splits[0,2] = nX;
        return splits,nSplits;
        
    nSplits = int(np.ceil(nX / maxPoints))
    pointsPerSplit = np.ceil(nX / nSplits).astype(int)
    
    # Create all distances
    allDistances = np.linspace(dMin, dMax, nX)
    
    splits = np.zeros((nSplits,3))
    startIdx = 0
    for i in range(nSplits):
        endIdx = min(startIdx + pointsPerSplit, nX)
        splitDistances = allDistances[startIdx:endIdx]
        splits[i,0] = splitDistances[0]; splits[i,1] = splitDistances[-1];splits[i,2] = len(splitDistances)
        startIdx = endIdx
    
    return splits,nSplits
    
def getRecInfo(depths,nX,nY,gridSize,sr):
    # this function was written to return the receiver depth, receiver spacing, number of receiver points
    # for each run of QSEIS, these values would be written into the input file for QSEIS
    """
    Discretize layers by grid spacing, include interfaces,
    and attach nX values (upper layer value at interfaces).

    Parameters
    ----------
    depths : array-like
        Layer interface depths (len N+1)
    gridSize : array-like
        Grid spacing per layer (len N)
    nX : array-like
        nX per layer (len N)

    Returns
    -------
    zAll : np.ndarray
        Discretized depth values including interfaces
    nXAll : np.ndarray
        Corresponding nX values for each depth
    """
    zAll = []
    nXAll = []
    srAll = []
    
    for i in range(len(nX)):
        zTop, zBot = depths[i], depths[i + 1]
        #dz = gridSize[i]
        dz = (zBot-zTop)/nY[i]

        # Discretize current layer
        zLayer = np.arange(zTop, zBot, dz)

        # treat the top layer with both top and bottom in it
        if(i==0):
            zLayer = np.append(zLayer, zBot)
        else:
            zLayer = np.append(zLayer[1:],zBot)
            
        # Include bottom interface only if it's the last layer (bug fix Oct 29, 2025)
        # (or will not be repeated in the next one)
        #if i == len(nX) - 1:
        #    zLayer = np.append(zLayer, zBot)

        zAll.append(zLayer)
        nXAll.append(np.full_like(zLayer, nX[i], dtype=float))
        srAll.append(np.full_like(zLayer, sr[i], dtype=float ))

    # Concatenate
    zAll = np.concatenate(zAll)
    nXAll = np.concatenate(nXAll)
    srAll = np.concatenate(srAll)

    return zAll, nXAll, srAll

def getRecInfoTD(depths,nX,nY,gridSize,sr,zList):
    # this function was written to return the receiver depth, receiver spacing, number of receiver points
    # for each run of QSEIS, these values would be written into the input file for QSEIS
    # slightly different from getRecInfo which was originally written for full NN database generation
    # modified for using in the stochastic search case
    """
    Discretize layers by grid spacing, include interfaces,
    and attach nX values (upper layer value at interfaces).

    Parameters
    ----------
    depths : array-like
        Layer interface depths (len N+1)
    gridSize : array-like
        Grid spacing per layer (len N)
    nX : array-like
        nX per layer (len N)

    Returns
    -------
    zAll : np.ndarray
        Discretized depth values including interfaces
    nXAll : np.ndarray
        Corresponding nX values for each depth
    """
    zAll = []
    nXAll = []
    srAll = []

    # loop over zList
    for i in range(len(zList)):
        zInd = np.where(depths<=zList[i])[0][-1]
        
        zAll.append(zList[i])
        nXAll.append(nX[zInd])
        srAll.append(sr[zInd])

    return zAll, nXAll, srAll
    
def getNXDP(config):
    """
    this function returns the number of depth and horizontal points per layer
    it will be at these depth points and horizontal points that the database generation happens
    note that the interface depths have not been included
    
    """
    # main code
    # call getGridSize
    gridSize,sr = getGridSize(config);
    
    # use these grid sizes and the layer depths to get the number of points
    lenThick = len(config.thickness);
    depths = np.cumsum(config.thickness[0:(lenThick-1)]);
    depths = np.insert(depths,0,0.0,axis=0);
    
    # find the index to insert zMax
    zMaxInd = np.where(depths<config.zMaxGF)[0]
    
    newDepths = np.append(depths[0:(zMaxInd[-1]+1)],config.zMaxGF);
    lenNewDepth = len(newDepths);
    nPtsX = np.zeros((lenNewDepth-1,),dtype=int);
    nPtsY = np.zeros((lenNewDepth-1,),dtype=int);
    
    for i in range(0,(lenNewDepth-1)):
        nPt = round((newDepths[i+1]-newDepths[i])/gridSize[i]);
        nPtsY[i] = int(max(nPt,3));
        nPt = round(config.xMaxGF/gridSize[i]);
        nPtsX[i] = int(nPt);
    return newDepths,nPtsX,nPtsY,gridSize,sr;
    
def lgwtPoints(N, a, b):
    """
    generate points within an interval based on Gauss-Legendre quadrature rule
    # N is the number of points in the interval (a,b)
    
    """
    N = N-1;
    N1 = N+1;
    N2 = N+2;

    xu = np.linspace(-1,1,N1);

    # Initial guess
    y = np.cos((2*np.arange(N1) + 1)*np.pi/(2*N + 2)) + \
        0.27 / N1*np.sin(np.pi*xu*N/N2);

    y0 = 2.0;
    L = np.zeros((N1,N2));
    Lp = np.zeros(N1);

    # Iterate with Newton-Raphson until convergence
    while np.max(np.abs(y-y0)) > np.finfo(float).eps:
        y0 = y.copy();

        L[:,0] = 1.0;
        L[:,1] = y;

        for k in range(1, N1):
            L[:,k+1] = ((2*k+1)*y*L[:,k] - k*L[:,k-1])/(k+1);

        Lp = N2*(L[:,N1-1]-y*L[:,N1]) / (1-y**2);

        y = y0 - L[:,N1]/Lp

    # Map from [-1, 1] to [a, b]
    x = (a*(1-y) + b*(1+y)) / 2

    # Compute the weights
    w = (b-a)/((1-y**2)*Lp**2)*(N2/N1)**2

    # Sort x and reorder weights
    idx = np.argsort(x)
    x = x[idx]
    w = w[idx]

    return x, w;


def getGridSize(config):
    """
    the function returns the grid size per layer by comparing the S-wave velocities and
    the frequency dependent Rayleigh wave velocities
    thickness is the thickness of each layer in meters,set last layer thickness to 0 for halfspace
    vP, vS are the elastic P-wave and S-wave velocities of each layer in m/s, including half space
    rho is the density of each layer in kg/m^3
    fMin, fMax are the minimum ana maximum frequency of the NN band
    lambdaFrac is between 0 and 1, denoting the sensitivity of Rayleigh waves in depth
    if lambdaFrac=1/3, then sensitivity of Rayliegh wave is upto that depth at that frequency
    lambdaRes is the number of points per lambda to be resolved using the grid
    typical values of lambdaRes is between 4 and 6, increasing it will only increase the computational
    and the memory load
    
    """
    # main code
    [freqMin, vRMin, vSMin] = getMinVelFreq(config);

    sr = np.zeros(len(vSMin),);
    for i in range(0,len(vSMin)):
        if(vSMin[i]<1000):
            sr[i] = 8.0;
        elif vSMin[i]>=1000 and vSMin[i]<2000:
            sr[i] = 12.0;
        elif vSMin[i]>=2000 and vSMin[i]<3000:
            sr[i] = 20.0;
        elif vSMin[i]>=3000:
            sr[i] = 24.0;
        
    lambdaRMin = vRMin/freqMin*1000; # in meters
    
    lambdaSMin = config.vS/config.fMax; # in meters
    
    lambdaMin = np.minimum(lambdaRMin,lambdaSMin);

    gridSize = lambdaMin/config.lambdaRes;
    
    return gridSize, sr;
    
def getMinVelFreq(config):
    """
    # thickness is the thickness of each layer in meters,set last layer thickness to 0 for halfspace
    # vP, vS are the elastic P-wave and S-wave velocities of each layer in m/s, including half space
    # rho is the density of each layer in kg/m^3
    # fMin, fMax are the minimum ana maximum frequency of the NN band
    # lambdaFrac is between 0 and 1, denoting the sensitivity of Rayleigh waves in depth
    # if lambdaFrac=1/3, then sensitivity of Rayliegh wave is upto that depth at that frequency
    """
    # main code
    # convert everything to units suitable for surf96, CPS.330 code
    thick = config.thickness/1000; # converted to kilometers
    vP = config.vP/1000; vS = config.vS/1000; # converted to km/s
    rho = config.rho/1000; # converted to gm/cc
    
    # Periods we are interested in
    freqs = np.arange(config.fMin,config.fMax,0.2);
    
    periods = 1/freqs;

    vDispRay = surf96(thick,vP,vS,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    #print(vDispRay)
    # surf96 can fail if it is Love wave for homogeneous half-space for
    # put a check
    homoFlag = 1
    for i in range(0,(len(vS)-1)):
        if((vS[i+1]-vS[i])!=0):
            homoFlag = 0
    if not homoFlag:
        vDispLove = surf96(thick,vP,vS,rho,periods,wave="love",mode=1,velocity="phase",flat_earth=True)
    else:
        vDispLove = vDispRay
    #print(vDispLove)
    
    # get the minimum of the Rayleigh and Love wave phase velocity
    vMinRayLove = np.minimum(vDispRay, vDispLove);
    
    #print(vMinRayLove)
    
    lambdaVal = vMinRayLove/freqs;
    lambdaValBy3 = lambdaVal*config.lambdaFrac;
    depths = np.cumsum(thick);
    depths = np.insert(depths,0,0.0,axis=0);
    vMin = np.zeros((len(depths)-1),);
    freqMin = np.zeros((len(depths)-1),);
    vSMin = np.zeros((len(depths)-1),);
    
    for depthNo in range(0,(len(depths)-1)):
        fInd = np.where((lambdaValBy3<=depths[depthNo+1]) & (lambdaValBy3>=depths[depthNo]))[0]
        if(len(fInd)>0):
            vMin[depthNo] = vMinRayLove[fInd[-1]]
            freqMin[depthNo] = freqs[fInd[-1]];
            if(vMin[depthNo]>vS[depthNo]):
                vMin[depthNo] = vS[depthNo];
        else:
            vMin[depthNo] = vS[depthNo];
            freqMin[depthNo] = config.fMax;
        vSMin[depthNo] = vS[depthNo]*1000; # convert back to m/s
        
    return freqMin, vMin, vSMin;
        
if __name__ == "__main__":
    main()

