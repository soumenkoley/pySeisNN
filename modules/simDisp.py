#!/usr/bin/env python
# coding: utf-8

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
from scipy.special import j0
from scipy.special import jv
from joblib import Parallel, delayed
from math import ceil
import time
import uuid
import os, sys
from pysurf96 import surf96
from modules import configLoader
from modules import createDB
from modules import gfLoader
from modules import validateInputs
from modules import plotUtils
from modules import NNCompute
import matplotlib.pyplot as plt

def main():
    # load the config file
    configFile="configParseAttnTest.ini"
    config = configLoader.load_config(configFile)
    validateInputs.validateInputs(config)
    
    splitAll = createDB.createFinalInputs(config)

    print(f"Prepared {splitAll.shape[0]} total runs")
    print(f"Cores available: {config.cpuCoresQseis}")

    results = createDB.runMultiQseis(splitAll, config.inputPath, config.qseisExe, nWorkers = config.cpuCoresQseis)
    print("Database generation complete")
    
    # flag for pure Rayleigh wavefield in homogeneous half-space
    ifRay = config.ifRay

    # flag for plane P and S waves in homogeneous half space
    ifBody = config.ifBody

    # flag for wavefield simulation
    ifFullField = config.ifFullField

    saveHV = config.saveHV

    thickness = config.thickness # units in meters
    vs = config.vS # units in m/s
    vp = config.vP # units in m/s
    rho = config.rho # units in kg/m^3
    
    fMin = config.fMin; fMax = config.fMax; df = config.df;# units in Hz
    
    # given a cuboidal domain should be greater than sqrt of 
    xMaxGF = config.xMaxGF # maximum horizontal offset upto which displacements will be used
    zMaxGF = config.zMaxGF # maximum depth upto which displacements will be used
    
    xMin = -config.xExtent; xMax = config.xExtent # minimum and maximum of the simulation domain in X-direction (EW)
    yMin = -config.yExtent; yMax = config.yExtent # maximum and minimum of the simulation domain in Y-direction (NS)
    zMax = config.zExtent # maximum and minimum of the simulation domain in Z-direction (depth)

    # check if the GFs are generated up to correct hoizontal offsets
    xDiag = (xMax-xMin)*np.sqrt(2); yDiag = (yMax-yMin)*np.sqrt(2);
    if(xMaxGF<xDiag or xMaxGF<yDiag):
        raise ValueError("xMaxGF must be >= sqrt(2)x(xMax-xMin) and sqrt(2)x(yMax-yMin)")

    if(zMaxGF<zMax):
        raise ValueError("zMaxGF must be greater >= zMax")
        
    
    # some other inputs for simDisp
    tMax = config.tMax; nSamp = config.nSamp
    
    # specify the folder where you want to write all input files, should have rw access
    # fInpPath is the path where the Green's function database exists
    fInpPath = config.inputPath #like "/data/gravwav/koley/QseisInpN/"
    #fInpPath = "/data/gravwav/koley/SALVUSOut/"

    # outDispPath should have rw access and is used for saving temp displacementy files
    outDispPath = config.outDispPath # like "/data/gravwav/koley/OutDisp/"

    # GF components to be used
    components = ['fh-2.tz', 'fh-2.tr', 'fh-2.tt', 'fz-2.tz', 'fz-2.tr']

    # splitAll is necessary so load it
    nRea = config.nRea # number of realizations

    # this file is created at the time of database generation
    splitFileName = fInpPath + 'splitAll.mat'
    splitMat = loadmat(splitFileName)
    splitAll = splitMat['splitAll']

    # source distribution parameters
    R1 = config.R1; R2 = config.R2
    nSrc = config.nSrc
    srcDistri = config.srcDistri
    scaleVH = config.scaleVH
    
    # computation strategy
    computeStrategy = config.computeStrategy

    # frequency axis to be used by simDisp
    freqOut, idxFreq, _ = getFreqGrid(tMax, nSamp, fMin, fMax, df)
    nFreq = len(freqOut)

    # in case only Rayleigh waves in homogeneous half space is used, Rayleigh wave phase velocity is needed
    if(ifRay):
        periods = 1/freqOut
        vR = surf96(thickness,vp,vs,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    else:
        vR = None
    # two depth points for scaling, displacement saved per realization
    zTar = np.array((0.0, 250.0,))
    gridX = np.array((0.0,))
    gridY = np.array((0.0,))

    #recDistAB = np.sqrt((gridX[0]-gridX[1])**2 + (gridY[0]-gridY[1])**2 + (zTar[0]-zTar[1])**2)
    
    # nFreqx3 for the three percentiles
    attnZObs = np.zeros((nFreq,3), dtype=np.complex128)
    attnXObs = np.zeros((nFreq,3), dtype=np.complex128)
    attnYObs = np.zeros((nFreq,3), dtype=np.complex128)
    attnZSim = np.zeros((nFreq,), dtype=np.complex128)
    attnXSim = np.zeros((nFreq,), dtype=np.complex128)
    attnYSim = np.zeros((nFreq,), dtype=np.complex128)
    dispTotalAllReaSurf = np.zeros((nFreq,1,3), dtype=np.complex128)
    dispTotalAllReaDepth = np.zeros((nFreq,1,3), dtype=np.complex128)
    
    for reaNo in range(0,nRea):
        print('Doing realization = ' +str(reaNo))
        # get the source distribution per realization
        xSrc, ySrc, azSrc, srcMeta =  genAmbSrc(nSrc , mode = srcDistri, R1 = R1, R2 = R2, xMin=xMin, xMax=xMax, yMin=yMin,
                                           yMax=yMax, randomPhase=True, freqDependent=True, nFreq=nFreq, decoupledHV=config.decoupledHV,
                                           scaleVH = scaleVH)
        
        #srcMeta["ampV"] *= 0.0
        #ySrc = np.zeros((nSrc,))

        # in case the body wave in homogeneous half-space flag is turned on
        if(ifBody):
            metaBody = genBodySrc(nSrc, nFreq, eP = 1/3, eS = 2/3)
        else:
            metaBody=None
        
        dispTotalSurf, outDir, fOut = computeFullDispN(zTar[0], gridX, gridY, xSrc, ySrc, azSrc, srcMeta, idxFreq, freqOut, fMin, fMax,
                     outDispPath, splitAll, xMaxGF, fInpPath, components, nCPU=1, nChunk=20000, minVel=100.0,
                     computeStrategy=computeStrategy, saveHV=saveHV)
        
        dispTotalDepth, outDir, fOut = computeFullDispN(zTar[1], gridX, gridY, xSrc, ySrc, azSrc, srcMeta, idxFreq, freqOut, fMin, fMax,
                     outDispPath, splitAll, xMaxGF, fInpPath, components, nCPU=1, nChunk=20000, minVel=100.0,
                     computeStrategy=computeStrategy, saveHV=saveHV)
        # populate the cross correlation matrices
        
        dispTotalAllReaSurf += (np.abs(dispTotalSurf))**2
        dispTotalAllReaDepth += (np.abs(dispTotalDepth))**2

    dispTotalAllReaSurf = np.sqrt(dispTotalAllReaSurf/nRea)
    dispTotalAllReaDepth = np.sqrt(dispTotalAllReaDepth/nRea)
    
          
    attnXSim += (dispTotalAllReaSurf[:,0,0]/dispTotalAllReaDepth[:,0,0])
    attnYSim += (dispTotalAllReaSurf[:,0,1]/dispTotalAllReaDepth[:,0,1])
    attnZSim += (dispTotalAllReaSurf[:,0,2]/dispTotalAllReaDepth[:,0,2])

    attnSim = np.stack((attnXSim,attnYSim,attnZSim),axis=-1)
    print(np.shape(attnSim))    
    # load Terziet Attenuation
    terzAttn = loadmat(os.path.join(config.siteASDPath,'attnModel.mat'))
    for i in range(0,3):
        attnXObs[:,i] = np.interp(freqOut,terzAttn['attnE'][:,0],terzAttn['attnE'][:,i+1])
        attnYObs[:,i] = np.interp(freqOut,terzAttn['attnN'][:,0],terzAttn['attnN'][:,i+1])
        attnZObs[:,i] = np.interp(freqOut,terzAttn['attnZ'][:,0],terzAttn['attnZ'][:,i+1])
    
    figAttn, axAttn = plotUtils.plotAttn(freqOut, attnSim, attnXObs, attnYObs, attnZObs)
    
def computeFullDispN(zTar, gridX, gridY, xSrc, ySrc, azSrc, srcMeta, idxFreq, freqOut, fMin, fMax,
                     outDispPath, splitAll, xMaxGF, fInpPath, components, nCPU=4, nChunk=20000, minVel=100.0,
                     computeStrategy="threading_shared", saveHV=False):
    """
    Compute the full displacement field (H-force + V-force) at one depth slice zTar
    on the receiver grid defined by (gridX, gridY).

    This function:
      1. Loads interpolated Green's functions for depth zTar.
      2. Runs simFixedDepth_partition_receivers to compute dispHForce, dispVForce
         as memmapped arrays on disk or on RAM.
      3. Creates a new memmap dispTotal.dat = dispHForce + dispVForce (component-wise).
      4. Returns (dispTotal_memmap, outDir, freqOut).

    Parameters
    ----------
    zTar : float
        Target depth [meters] (positive down).
    gridX, gridY : 1D arrays
        Receiver coordinates (flattened, same length nRec).
    xSrc, ySrc, azSrc, srcMeta :
        Source parameters from genAmbSrc (or similar).
    idxFreq, freqOut :
        Frequency indices and values to use (from getFreqGrid).
    fMin, fMax : float
        Frequency band [Hz].
    outDispPath : str
        Base output directory for displacement files.
    splitAll, xMaxGF, fInpPath, components :
        Green's function / database config.
    nCPU : int
        Number of workers for simFixedDepth_partition_receiversN.
    nChunk : int
        Receiver chunk size per worker.
    minVel : float
        Minimum velocity for GF interpolation (passed to gfLoader).
    computeStrategy : default set to threading_shared
    saveHV : default set False, if true returns the displacement field
        for vertical and horizontal forces separately
    

    Returns
    -------
    Always returns:
        dispTotal, outDir, freqOut_used
    """

    # 1) output folder for this depth
    outDir = makeDepthDispfolder(outDispPath, zTar)
    os.makedirs(outDir, exist_ok=True)

    # 2) load Green's functions for this depth
    xxW, tVec, distVec = gfLoader.getInterpolatedGF(
        splitAll, zTar, 0.01, xMaxGF, fInpPath, components, minVel=minVel
    )

    # 3) compute displacement at this depth
    result = simFixedDepth_partition_receiversN(xxW=xxW, tVec=tVec, distVec=distVec,xGrid=gridX,
                                                yGrid=gridY, xSrc=xSrc, ySrc=ySrc, azSrc=azSrc,
                                                srcMeta=srcMeta, fMin=fMin, fMax=fMax, chunk_size=nChunk,
                                                outDir=outDir, n_workers=nCPU, idxFreq=idxFreq, freqOut=freqOut,
                                                saveHV=saveHV,computeStrategy = computeStrategy)

    # 4) unpack depending on mode, but always return dispTotal
    if saveHV:
        # result = (dispH, dispV, dispTotal, freqOut_used)
        _, _, dispTotal, freqOut_used, shmPaths = result
    else:
        # result = (dispTotal, freqOut_used)
        dispTotal, freqOut_used, shmPaths = result

    return dispTotal, outDir, freqOut_used, shmPaths

def computeFullDispF(zTar, gridX, gridY, itmAC, xSrc, ySrc, azSrc, srcMeta,idxFreq, freqOut, outDispPath,
                     splitAllNew, xMaxGF, fInpPath, nCPU=4, nChunk=20000, computeStrategy="loky_sharedmem",
                     saveHV=False, reduceNN=False, nnMode=None, nnMeta=None):
    """
    FFT-domain analogue of computeFullDispN.

    If reduceNN=False:
        returns displacement at one fixed depth.
    If reduceNN=True:
        returns directly reduced NN contribution for the provided geometry.
    """

    # 1) output folder
    outDir = makeDepthDispfolder(outDispPath, zTar)
    #os.makedirs(outDir, exist_ok=True)

    # 2) load FFT-domain Green's functions for this depth
    xxWFFT, distVec = gfLoader.getInterpolatedGFFFT(
        splitAllNew, zTar, 0.01, xMaxGF, fInpPath
    )

    # 3) compute field / reduced NN
    result = simFixedDepth_partition_receiversF(xxWFFT=xxWFFT, distVec=distVec, xGrid=gridX, yGrid=gridY, itmAC=itmAC,
                                                zTar=zTar, xSrc=xSrc, ySrc=ySrc, azSrc=azSrc, srcMeta=srcMeta,
                                                chunk_size=nChunk, outDir=outDir, n_workers=nCPU, freqOut=freqOut,
                                                saveHV=saveHV, computeStrategy=computeStrategy, reduceNN=reduceNN,
                                                nnMode=nnMode, nnMeta=nnMeta)

    # 4) modify the returns depending on which mode the code runs
    if reduceNN:
        I_total, freqOut_used, shmPaths = result
        return I_total, outDir, freqOut_used, shmPaths

    if saveHV:
        dispH, dispV, dispTotal, freqOut_used, shmPaths = result
        return dispH, dispV, dispTotal, outDir, freqOut_used, shmPaths

    # legacy usage
    dispTotal, freqOut_used, shmPaths = result
    return dispTotal, outDir, freqOut_used, shmPaths


def assembleSurfDeepDispAllRea(outReaPath,zList, freqOut, nRea):
    """
    to be run at the end of all realizations
    compute the rms of surface and deep displacements
    
    """
    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointAllRea = np.zeros((nFreq,3,zLen))
    attnAllRea = np.zeros((nFreq,3))
    
    for reaNo in range(0,nRea):
        sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
        data = np.load(os.path.join(outReaPath,sName));
        dispPointFull = data["dispPointFull"]
        dispPointAllRea = dispPointAllRea + dispPointFull**2

    dispPointAllRea = np.sqrt(dispPointAllRea/nRea)
    attnAllRea = dispPointAllRea[:,:,0]/dispPointAllRea[:,:,1]
    freqOut = data["freqOut"];
    sName = 'fullDisp.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointAllRea=dispPointAllRea, freqOut=freqOut, attnAllRea=attnAllRea)
    return dispPointAllRea, attnAllRea, freqOut

def assembleSurfDeepDispAllReaMultiPoint(outReaPath,itmAll, zList, freqOut, nRea):
    """
    to be run at the end of all realizations
    compute the rms of surface and deep displacements
    
    """
    nFreq = len(freqOut)
    zLen = len(zList)
    nRec = len(itmAll[:,0])
    
    dispPointAllRea = np.zeros((nFreq,nRec,3,zLen))
    attnAllRea = np.zeros((nFreq,nRec,3))
    
    for reaNo in range(0,nRea):
        sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
        data = np.load(os.path.join(outReaPath,sName));
        dispPointFull = data["dispPointFull"]
        dispPointAllRea = dispPointAllRea + dispPointFull**2

    dispPointAllRea = np.sqrt(dispPointAllRea/nRea)
    attnAllRea = dispPointAllRea[:,:,:,0]/dispPointAllRea[:,:,:,1]
    freqOut = data["freqOut"]
    sName = 'fullDisp.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointAllRea=dispPointAllRea, freqOut=freqOut, attnAllRea=attnAllRea)
    return dispPointAllRea, attnAllRea, freqOut
  
def getSurfDeepDispPerRea(zList, xSrc, ySrc, azSrc, srcMeta, xMaxGF, splitAll, fMin, fMax, outReaPath,
                    fInpPath, components, reaNo, idxFreq = None, freqOut=None, nCPU = 4):
    """
    this script will be run before every realization
    for a particular source distribution which is fixed per realization
    it will compute the displacement at point (0,0) on the surface and at depth
    specified by zList
    the grid will be generated within and will be a small one because we are only interested
    at getting the displacement at one point (0,0)
    
    """
    # generate the small grid, private variables not to be changed
    nx, ny = 11, 11
    x = np.linspace(-50, 50, nx)
    y = np.linspace(-50, 50, ny)
    
    gridXMat, gridYMat = np.meshgrid(x, y, indexing='ij')  # shape (nx, ny)
    # make them flat
    gridX = gridXMat.ravel(); gridY = gridYMat.ravel();

    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointFull = np.zeros((nFreq,3,zLen))
    
    for zNo, zVal in enumerate(zList):
        # make outDispDir
        outDir = makeDepthDispfolder(outReaPath, zVal)
        # load the green's funtion for the depth
        xxW, tVec, distVec = gfLoader.getInterpolatedGF(splitAll, zVal, 0.01, xMaxGF, fInpPath,
                                                            components, minVel=100.0)
        dispHForce, dispVForce, freqOut = simFixedDepth_partition_receivers(xxW, tVec, distVec, gridX, gridY, xSrc, ySrc, azSrc,
                                                srcMeta, fMin, fMax, 20000, outDir, n_workers=nCPU, idxFreq = idxFreq, freqOut=freqOut)
        dispPointFull[:,:,zNo] = getDispPerRea(outDir,0.0,0.0)

    # save this in the rea path, to be read in again after all realizations have been done
    # to scale the outputNN
    sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointFull=dispPointFull, freqOut=freqOut)
    return dispPointFull, freqOut
    
def getSurfDeepDispPerReaMultiPoint(zList, itmAll, xSrc, ySrc, azSrc, srcMeta, xMaxGF, splitAll, fMin, fMax, outReaPath,
                    fInpPath, components, reaNo, idxFreq = None, freqOut=None, nCPU = 4):
    """
    this script will be run before every realization
    for a particular source distribution which is fixed per realization
    it will compute the displacement at desired points on the surface and at depth
    specified by zList, and itmAll
    the grid will be generated within and will be a small one because we are only interested
    at getting the displacement at desired points
    
    """
    
    # make them flat
    gridX = np.asarray(itmAll[:,0]); gridY = np.asarray(itmAll[:,1])
    nRec = len(gridX)

    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointFull = np.zeros((nFreq,nRec,3,zLen))
    
    for zNo, zVal in enumerate(zList):
        # make outDispDir
        outDir = makeDepthDispfolder(outReaPath, zVal)
        # load the green's funtion for the depth
        xxW, tVec, distVec = gfLoader.getInterpolatedGF(splitAll, zVal, 0.01, xMaxGF, fInpPath,
                                                            components, minVel=100.0)
        dispHForce, dispVForce, freqOut = simFixedDepth_partition_receivers(xxW, tVec, distVec, gridX, gridY, xSrc, ySrc, azSrc,
                                                srcMeta, fMin, fMax, 20000, outDir, n_workers=nCPU, idxFreq = idxFreq, freqOut=freqOut)
        dispPointFull[:,:,:,zNo] = np.abs(dispHForce + dispVForce)

    # save this in the rea path, to be read in again after all realizations have been done
    # to scale the outputNN
    sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointFull=dispPointFull, freqOut=freqOut)
    return dispPointFull, freqOut

def getDispPerRea(outDir,xP,yP):
    # first load the all the receievr grid
    data = np.load(os.path.join(outDir, "receiverGrid.npz"));
    
    xGrid = data["xGrid"];
    yGrid = data["yGrid"];
    freqOut = data["freqOut"];
    nFreq = data["nFreq"];
    nRec = len(xGrid);
    
    # load the displacement file
    dispHForce = np.memmap(
        os.path.join(outDir,"dispHForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    dispVForce = np.memmap(
        os.path.join(outDir,"dispVForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    
    rec_index, distRec = find_receiver_index(xP, yP, xGrid, yGrid)  # example receiver index
    dispPointFull = np.abs(dispHForce[:, rec_index, :] + dispVForce[:, rec_index, :]) # shape (nFreq, 3)
    #dispPointFull = np.abs(dispVForce[:, rec_index, :]) # shape (nFreq, 3)
    return dispPointFull

def qseisSourceWavelet(tMax, nSamp, waveletDuration=4.0):
    """
    Generate the QSEIS default source wavelet (sw_wavelet = 1):

        w(t) = (2/tau) * sin^2(pi*t/tau),   0 < t < tau
             = 0,                           otherwise

    Parameters
    ----------
    tMax : float
        QSEIS time-window length [s].

    nSamp : int
        Number of QSEIS time samples.

    waveletDuration : float, optional
        QSEIS wavelet duration in units of time samples.
        Default is 4.0, corresponding to the input

            4.0  1

    Returns
    -------
    t : ndarray
        Time vector [s], shape (nSamp,).

    w : ndarray
        QSEIS source wavelet on the time grid.

    tau : float
        Physical wavelet duration [s].

    dt : float
        Time sampling interval [s].
    """

    dt = tMax / nSamp
    tau = waveletDuration * dt
    t = np.arange(nSamp) * dt
    w = np.zeros_like(t)
    mask = (t > 0.0) & (t < tau)
    w[mask] = (2.0 / tau) * np.sin(np.pi * t[mask] / tau)**2

    return t, w, tau, dt

def qseisWaveletResponse(freqOut, tMax, nSamp, waveletDurationSamples=4.0):
    """
    Fourier response of the QSEIS default source wavelet

        w(t) = (2/tau) sin^2(pi*t/tau),   0 < t < tau

    evaluated exactly at freqOut.
    """

    dt = tMax / nSamp
    tau = waveletDurationSamples * dt

    nu = freqOut * tau

    W = np.exp(-1j * np.pi * nu) * (
        np.sinc(nu)
        + 0.5 * np.sinc(nu - 1.0)
        + 0.5 * np.sinc(nu + 1.0)
    )

    return W, tau
    
def makeDepthDispfolder(outDispPath, zTarget):
    """
    Create (if needed) and return the output folder for a given receiver depth.

    Example:
        base_dir = '/data/.../OutDisp/'
        zTarget = 6.25
        -> returns '/data/.../OutDisp/Depth6p25/'
    """
    # Format depth safely for filenames
    depth_folder = f"Depth{zTarget:.2f}".replace(".", "p")

    # Combine into full path
    outDir = os.path.join(outDispPath, depth_folder)

    # Ensure directory exists
    os.makedirs(outDir, exist_ok=True)

    #print(f"[simDisp] Output directory for z={zTarget:.2f} m - {outDir}")
    return outDir

def worker_procN(worker_id, rec_start, rec_end, xxWFFT, distVec, idx, xGrid, yGrid, xSrc, ySrc, azSrc, srcMeta,
                nFreq, chunk_size, outDir, dispHFile, dispVFile, freqOut):
    """
    Worker that computes contributions for a disjoint receiver slice.
    Each worker loops over all sources, but only writes to its assigned receiver indices.
    """

    start_time = time.time()

    nRec = len(xGrid)
    if nRec == 0:
        raise ValueError("simFixedDepth_partition_receivers: nRec == 0 (empty receiver grid)")
    
    # old version
    # compute receiver slice for this worker (balanced, contiguous)
    #base = np.array_split(np.arange(nRec), n_workers)
    #rec_inds = base[worker_id]            # this is an array of receiver indices for this worker
    #print(f"[Worker {worker_id}] Assigned receiver indices: {rec_inds[0]}–{rec_inds[-1]} "
    #      f"(total {len(rec_inds)})")

    # new version
    print(f"[Worker {worker_id}] Assigned receiver indices: {rec_start}–{rec_end-1} "
      f"(total {rec_end - rec_start})")
    
    # unchanged from last version
    # Create memmap views for the whole file (each worker only writes its indices)
    dispH = np.memmap(dispHFile, dtype=np.complex128, mode='r+', shape=(nFreq, nRec, 3))
    dispV = np.memmap(dispVFile, dtype=np.complex128, mode='r+', shape=(nFreq, nRec, 3))

    nSrc = len(xSrc)
    xxWFFT_sel = xxWFFT[idx, :, :]  # shape (nFreq, nx, ncomp)


    for srcNo in range(nSrc):
        xS = xSrc[srcNo]; yS = ySrc[srcNo]; azS = azSrc[srcNo]
        #phaseFactor = ampSrc[srcNo]*np.exp(-1j * phiSrc[srcNo]);
        #phaseFactor = ampSrc[srcNo, :][:, np.newaxis, np.newaxis] * \
        #          np.exp(-1j * phiSrc[srcNo, :][:, np.newaxis, np.newaxis])
        
        phaseH = srcMeta["ampH"][srcNo, :][:, None, None] * np.exp(-1j * srcMeta["phiH"][srcNo, :][:, None, None])
        phaseV = srcMeta["ampV"][srcNo, :][:, None, None] * np.exp(-1j * srcMeta["phiV"][srcNo, :][:, None, None])
        
        for j0 in range(rec_start, rec_end, chunk_size):
            j1 = min(j0 + chunk_size, rec_end)
            dx = xGrid[j0:j1] - xS
            dy = yGrid[j0:j1] - yS
            distGrid = np.sqrt(dx * dx + dy * dy)

            azGridV = np.arctan2(dx, dy)         # for vertical source (no rotation)
            azGridH = azGridV - azS   # for horizontal source rotation
                
            # interpolation indices
            distGrid = np.clip(distGrid, distVec[0], distVec[-1]);
            idx_hi = np.searchsorted(distVec, distGrid, side='right')
            idx_hi = np.clip(idx_hi, 1, len(distVec) - 1)
            idx_lo = idx_hi - 1

            w = (distGrid - distVec[idx_lo]) / (distVec[idx_hi] - distVec[idx_lo])
            w = w[np.newaxis, :, np.newaxis]

            G_lo = xxWFFT_sel[:, idx_lo, :]
            G_hi = xxWFFT_sel[:, idx_hi, :]

            xxInterpH = ((1 - w) * G_lo + w * G_hi) * phaseH
            xxInterpV = ((1 - w) * G_lo + w * G_hi) * phaseV
            
            sinH = np.sin(azGridH)[np.newaxis, :]
            cosH = np.cos(azGridH)[np.newaxis, :]
            sinV = np.sin(azGridV)[np.newaxis, :]
            cosV = np.cos(azGridV)[np.newaxis, :]
                    
            dispH[:, j0:j1, 0] += xxInterpH[:, :, 1] * sinH + (-xxInterpH[:, :, 2]) * cosH
            dispH[:, j0:j1, 1] += xxInterpH[:, :, 1] * cosH - (-xxInterpH[:, :, 2]) * sinH
            dispH[:, j0:j1, 2] += -xxInterpH[:, :, 0] # note negative bevause of QSEIS convention

            dispV[:, j0:j1, 0] += xxInterpV[:, :, 4] * sinV
            dispV[:, j0:j1, 1] += xxInterpV[:, :, 4] * cosV
            dispV[:, j0:j1, 2] += -xxInterpV[:, :, 3]

        # --- progress print every 100 sources ---
        #if (srcNo + 1) % 100 == 0 or (srcNo + 1) == nSrc:
        #    elapsed = time.time() - last_print_time
        #    total_elapsed = time.time() - start_time
        #    print(f"[Worker {worker_id}] Processed {srcNo + 1}/{nSrc} sources "
        #          f"({100*(srcNo+1)/nSrc:.1f}%) | "
        #          f"Elapsed since last: {elapsed:.2f}s | Total: {total_elapsed:.1f}s")
        #    last_print_time = time.time()

    # updated flush to the end of all sources
    dispH.flush()
    dispV.flush()
    total_time = time.time() - start_time
    
    print(f"[Worker {worker_id}] Finished in {total_time:.2f} s")

    return True

def worker_proc_loky_sharedout(worker_id, rec_start, rec_end, xxWFFT_sel, distVec, xGrid, yGrid,
                               xSrc, ySrc, azSrc, phaseVAll, phaseHAll, nFreq, chunk_size, dispTotal_path,
                               disp_shape, disp_dtype_str, saveHV=False, dispH_path=None, dispV_path=None):
    
    nRec_w = rec_end - rec_start
    if nRec_w <= 0:
        return True

    disp_dtype = np.dtype(disp_dtype_str)

    # Reopen shared output arrays inside this worker process
    dispTotal = np.memmap(
        dispTotal_path, dtype=disp_dtype, mode="r+", shape=tuple(disp_shape)
    )

    if saveHV:
        dispH = np.memmap(
            dispH_path, dtype=disp_dtype, mode="r+", shape=tuple(disp_shape)
        )
        dispV = np.memmap(
            dispV_path, dtype=disp_dtype, mode="r+", shape=tuple(disp_shape)
        )
    else:
        dispH = dispV = None

    # Local accumulation in normal RAM
    localTotal = np.zeros((nFreq, nRec_w, 3), dtype=disp_dtype)
    # commenting this part out, was used for correlation tensor analysis
    #localPtr = np.zeros((nFreq,), dtype=np.float64)
    #localPtt = np.zeros((nFreq,), dtype=np.float64)
    
    if saveHV:
        localH = np.zeros((nFreq, nRec_w, 3), dtype=disp_dtype)
        localV = np.zeros((nFreq, nRec_w, 3), dtype=disp_dtype)
    else:
        localH = localV = None

    nSrc = len(xSrc)

    for srcNo in range(nSrc):
        xS = xSrc[srcNo]
        yS = ySrc[srcNo]
        azS = azSrc[srcNo]

        phaseH = phaseHAll[srcNo, :][:, None, None]
        phaseV = phaseVAll[srcNo, :][:, None, None]
        sS = np.sin(azS)
        cS = np.cos(azS)

        for j0 in range(rec_start, rec_end, chunk_size):
            j1 = min(j0 + chunk_size, rec_end)

            # Local indices into the worker-owned arrays
            jj0 = j0 - rec_start
            jj1 = j1 - rec_start

            dx = xGrid[j0:j1] - xS
            dy = yGrid[j0:j1] - yS
            dist = np.sqrt(dx * dx + dy * dy)
            dist_safe = np.maximum(dist, 1e-12)

            sinV = (dx / dist_safe)[None, :]
            cosV = (dy / dist_safe)[None, :]
            sinH = sinV * cS - cosV * sS
            cosH = cosV * cS + sinV * sS

            dist_clip = np.clip(dist_safe, distVec[0], distVec[-1])
            idx_hi = np.searchsorted(distVec, dist_clip, side="right")
            idx_hi = np.clip(idx_hi, 1, len(distVec) - 1)
            idx_lo = idx_hi - 1

            den = distVec[idx_hi] - distVec[idx_lo]
            den_safe = np.maximum(den, 1e-12)
            w = (dist_clip - distVec[idx_lo]) / den_safe
            w = np.clip(w, 0.0, 1.0)
            w = w[None, :, None]

            G_lo = xxWFFT_sel[:, idx_lo, :]
            G_hi = xxWFFT_sel[:, idx_hi, :]
            G_interp = (1 - w) * G_lo + w * G_hi

            xxInterpH = G_interp * phaseH
            xxInterpV = G_interp * phaseV

            Hx = xxInterpH[:, :, 1] * sinH + (-xxInterpH[:, :, 2]) * cosH
            Hy = xxInterpH[:, :, 1] * cosH - (-xxInterpH[:, :, 2]) * sinH
            Hz = -xxInterpH[:, :, 0]

            Vx = xxInterpV[:, :, 4] * sinV
            Vy = xxInterpV[:, :, 4] * cosV
            Vz = -xxInterpV[:, :, 3]

            localTotal[:, jj0:jj1, 0] += (Hx + Vx)
            localTotal[:, jj0:jj1, 1] += (Hy + Vy)
            localTotal[:, jj0:jj1, 2] += (Hz + Vz)

            # commenting this part was used for correlation tensor analysis
            #localPtr += np.sum(np.abs(xxInterpH[:, :, 1])**2, axis=1)
            #localPtt += np.sum(np.abs(xxInterpH[:, :, 2])**2, axis=1)
            
            #localPtr += np.sum(np.abs(Hx)**2, axis=1)
            #localPtt += np.sum(np.abs(Hy)**2, axis=1)

            if saveHV:
                localH[:, jj0:jj1, 0] += Hx
                localH[:, jj0:jj1, 1] += Hy
                localH[:, jj0:jj1, 2] += Hz

                localV[:, jj0:jj1, 0] += Vx
                localV[:, jj0:jj1, 1] += Vy
                localV[:, jj0:jj1, 2] += Vz

    # One final write to shared output
    dispTotal[:, rec_start:rec_end, :] = localTotal
    dispTotal.flush()

    if saveHV:
        dispH[:, rec_start:rec_end, :] = localH
        dispV[:, rec_start:rec_end, :] = localV
        dispH.flush()
        dispV.flush()

    # not returning localPtt and local Ptr anymore
    return True

def worker_proc_SP(xxWFFT_sel, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, phaseVAll, phaseHAll, nFreq):
    
    # does the same thing as worker_proc_loky_sharedout but is
    # tailored for single point usage, especially useful when parallelizing multiple
    # points that span different depths

    # Local accumulation in normal RAM
    localTotal = np.zeros((nFreq, 1, 3), dtype=np.complex128)
    
    nSrc = len(xSrc)
    xGrid = np.atleast_1d(xGrid)
    yGrid = np.atleast_1d(yGrid)

    for srcNo in range(nSrc):
        xS = xSrc[srcNo]
        yS = ySrc[srcNo]
        azS = azSrc[srcNo]

        phaseH = phaseHAll[srcNo, :][:, None, None]
        phaseV = phaseVAll[srcNo, :][:, None, None]
        sS = np.sin(azS)
        cS = np.cos(azS)

        dx = xGrid - xS
        dy = yGrid - yS
        dist = np.sqrt(dx * dx + dy * dy)
        dist_safe = np.maximum(dist, 1e-12)

        sinV = (dx / dist_safe)[None, :]
        cosV = (dy / dist_safe)[None, :]
        sinH = sinV * cS - cosV * sS
        cosH = cosV * cS + sinV * sS

        dist_clip = np.clip(dist_safe, distVec[0], distVec[-1])
        idx_hi = np.searchsorted(distVec, dist_clip, side="right")
        idx_hi = np.clip(idx_hi, 1, len(distVec) - 1)
        idx_lo = idx_hi - 1

        den = distVec[idx_hi] - distVec[idx_lo]
        den_safe = np.maximum(den, 1e-12)
        w = (dist_clip - distVec[idx_lo]) / den_safe
        w = np.clip(w, 0.0, 1.0)
        w = w[None, :, None]

        G_lo = xxWFFT_sel[:, idx_lo, :]
        G_hi = xxWFFT_sel[:, idx_hi, :]
        G_interp = (1 - w) * G_lo + w * G_hi

        xxInterpH = G_interp * phaseH
        xxInterpV = G_interp * phaseV

        Hx = xxInterpH[:, :, 1] * sinH + (-xxInterpH[:, :, 2]) * cosH
        Hy = xxInterpH[:, :, 1] * cosH - (-xxInterpH[:, :, 2]) * sinH
        Hz = -xxInterpH[:, :, 0]

        Vx = xxInterpV[:, :, 4] * sinV
        Vy = xxInterpV[:, :, 4] * cosV
        Vz = -xxInterpV[:, :, 3]

        localTotal[:,0:1,0] += (Hx + Vx)
        localTotal[:,0:1,1] += (Hy + Vy)
        localTotal[:,0:1, 2] += (Hz + Vz)

    # not returning localPtt and local Ptr anymore
    return localTotal

def worker_procF(worker_id, rec_start, rec_end, xxWFFT_sel, distVec, xGrid, yGrid, itmAC, xSrc, ySrc, azSrc,
                 phaseVAll, phaseHAll, nFreq, chunk_size, saveHV=False, reduceNN=False, nnMode=None,
                 nnMeta=None):
    """
    FFT-domain worker for one contiguous receiver slice.

    Two modes
    ---------
    1) Displacement mode:
         reduceNN=False
         returns local displacement arrays for this worker slice
    2) NN reduction mode:
         reduceNN=True
         computes displacement locally and immediately reduces to
         partial NN contribution for this worker slice

    Parameters
    ----------
    worker_id : int
    rec_start, rec_end : int
        Receiver slice owned by this worker.
    xxWFFT_sel : (nFreq, nDist, nComp) complex
        Frequency-domain GFs already restricted to the desired frequencies.
    distVec : (nDist,) float
        GF distance axis.
    xGrid, yGrid : (nRec,) float
        Flattened receiver coordinates for this depth slice.
    xSrc, ySrc, azSrc : (nSrc,) float
        Source coordinates and azimuths.
    phaseVAll, phaseHAll : (nSrc, nFreq) complex
        Precomputed source phase/amplitude factors.
    nFreq : int
    chunk_size : int
    saveHV : bool
        Only supported in displacement mode.
    reduceNN : bool
        If True, return reduced NN chunk instead of displacement.
    nnMode : str or None
        'volume', 'vert_surface', 'hor_surface'
    nnMeta : dict or None
        Receiver-aligned geometry metadata for reduction mode.

    Returns
    -------
    If reduceNN=False and saveHV=False:
        localTotal : (nFreq, nRec_w, 3)
    If reduceNN=False and saveHV=True:
        (localH, localV, localTotal)
    If reduceNN=True:
        I_chunk : (nFreq, 3)
    """
    nRec_w = rec_end - rec_start
    if nRec_w <= 0:
        if reduceNN:
            return np.zeros((nFreq, 3), dtype=np.complex128)
        if saveHV:
            z = np.zeros((nFreq, 0, 3), dtype=np.complex128)
            return z.copy(), z.copy(), z.copy()
        return np.zeros((nFreq, 0, 3), dtype=np.complex128)

    if reduceNN and saveHV:
        raise ValueError("saveHV=True is not supported when reduceNN=True")

    # local accumulation only in RAM
    localTotal = np.zeros((nFreq, nRec_w, 3), dtype=np.complex128)
    
    if saveHV:
        localH = np.zeros((nFreq, nRec_w, 3), dtype=np.complex128)
        localV = np.zeros((nFreq, nRec_w, 3), dtype=np.complex128)
    else:
        localH = None
        localV = None

    nSrc = len(xSrc)

    # worker-owned receiver coordinates
    xRec_w = xGrid[rec_start:rec_end]
    yRec_w = yGrid[rec_start:rec_end]

    for srcNo in range(nSrc):
        xS = xSrc[srcNo]
        yS = ySrc[srcNo]
        azS = azSrc[srcNo]

        # shape: (nFreq, 1, 1)
        phaseH = phaseHAll[srcNo, :][:, None, None]
        phaseV = phaseVAll[srcNo, :][:, None, None]

        # chunk within this worker-owned slice
        for loc0 in range(0, nRec_w, chunk_size):
            loc1 = min(loc0 + chunk_size, nRec_w)

            dx = xRec_w[loc0:loc1] - xS
            dy = yRec_w[loc0:loc1] - yS
            distGrid = np.sqrt(dx * dx + dy * dy)

            # azimuth conventions kept consistent with your existing worker
            azGridV = np.arctan2(dx, dy)
            azGridH = azGridV - azS

            sinH = np.sin(azGridH)[None, :]
            cosH = np.cos(azGridH)[None, :]
            sinV = np.sin(azGridV)[None, :]
            cosV = np.cos(azGridV)[None, :]

            # clamp distances to GF range
            distGrid = np.clip(distGrid, distVec[0], distVec[-1])

            idx_hi = np.searchsorted(distVec, distGrid, side="right")
            idx_hi = np.clip(idx_hi, 1, len(distVec) - 1)
            idx_lo = idx_hi - 1

            denom = distVec[idx_hi] - distVec[idx_lo]
            w = (distGrid - distVec[idx_lo]) / denom
            w = w[None, :, None]  # shape (1, nChunk, 1)

            G_lo = xxWFFT_sel[:, idx_lo, :]   # (nFreq, nChunk, nComp)
            G_hi = xxWFFT_sel[:, idx_hi, :]   # (nFreq, nChunk, nComp)
            xxInterp = (1.0 - w) * G_lo + w * G_hi

            xxInterpH = xxInterp * phaseH
            xxInterpV = xxInterp * phaseV

            # same component mapping as your current FFT worker
            Hx = xxInterpH[:, :, 1] * sinH + (-xxInterpH[:, :, 2]) * cosH
            Hy = xxInterpH[:, :, 1] * cosH - (-xxInterpH[:, :, 2]) * sinH
            Hz = -xxInterpH[:, :, 0]

            Vx = xxInterpV[:, :, 4] * sinV
            Vy = xxInterpV[:, :, 4] * cosV
            Vz = -xxInterpV[:, :, 3]

            localTotal[:, loc0:loc1, 0] += (Hx + Vx)
            localTotal[:, loc0:loc1, 1] += (Hy + Vy)
            localTotal[:, loc0:loc1, 2] += (Hz + Vz)

            if saveHV:
                localH[:, loc0:loc1, 0] += Hx
                localH[:, loc0:loc1, 1] += Hy
                localH[:, loc0:loc1, 2] += Hz

                localV[:, loc0:loc1, 0] += Vx
                localV[:, loc0:loc1, 1] += Vy
                localV[:, loc0:loc1, 2] += Vz

    # displacement-only mode
    if not reduceNN:
        if saveHV:
            return localH, localV, localTotal
        return localTotal

    # NN-reduction mode
    if nnMode is None:
        raise ValueError("reduceNN=True but nnMode is None")
    if nnMeta is None:
        raise ValueError("reduceNN=True but nnMeta is None")

    nnMeta_local = slice_nn_meta(nnMeta, rec_start, rec_end)
    return reduce_nn_chunk(localTotal, nnMode, nnMeta_local, itmAC)

def slice_nn_meta(nnMeta, rec_start, rec_end):
    """
    Slice only per-receiver arrays; keep scalar/shared fields unchanged.
    Convention:
      - arrays with length nRec are sliced on axis 0
      - everything else is passed through unchanged
    """
    if nnMeta is None:
        return None

    nnMeta_local = {}
    nloc = rec_end - rec_start

    for key, val in nnMeta.items():
        if isinstance(val, np.ndarray):
            if val.ndim >= 1 and val.shape[0] >= rec_end:
                nnMeta_local[key] = val[rec_start:rec_end].copy()
            else:
                nnMeta_local[key] = val
        else:
            nnMeta_local[key] = val

    return nnMeta_local


def reduce_nn_chunk(localTotal, nnMode, nnMeta_local, itmAC):
    """
    Reduce one worker-local displacement chunk to one partial NN contribution.

    Parameters
    ----------
    localTotal : (nFreq, nRec_w, 3) complex
        Total displacement for this worker slice.
    nnMode : str
        One of: 'volume', 'vert_surface', 'hor_surface'
    nnMeta_local : dict
        Chunk-local geometry/weight metadata.

    Returns
    -------
    I_chunk : (nFreq, 3) complex
    """
    if nnMode == "volume":
        
        return NNCompute.getVolNN_chunk_multiTM(localTotal, nnMeta_local["rVec"],
                                        nnMeta_local["rho"], nnMeta_local["dV"], itmAC)

    elif nnMode == "vert_surface":
        
        return NNCompute.computeVertSurfNN_multiTM(localTotal, nnMeta_local["rVec"],
                                           nnMeta_local["n_hat"], nnMeta_local["dS"], itmAC)

    elif nnMode == "hor_surface":
        
        return NNCompute.computeVertSurfNN_multiTM(localTotal,nnMeta_local["rVec"],
                                           nnMeta_local["n_hat"], nnMeta_local["dS"], itmAC)

    else:
        raise ValueError(f"Unknown nnMode={nnMode!r}")


def simFixedDepth_partition_receivers(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, srcMeta,
                                      fMin, fMax, chunk_size, outDir, n_workers=None, idxFreq=None, freqOut=None):
    """
    Partition receivers across workers (no per-worker big files).
    Each worker loops over all sources but writes only to its receiver slice.
    legacy version: much slower, mostly used on small receiver grids
    """

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)

    nt, nx, ncomp = xxW.shape
    nRec = len(xGrid)
    nSrc = len(xSrc)

    if nRec == 0:
        print("[simDisp] No receivers - returning empty displacement.")
        # Return empty memmaps or None; whichever your caller expects
        return None, None, freqOut

    # cap workers globally
    n_workers = min(n_workers, nRec)
    if n_workers < 1:
        n_workers = 1
    
    xxWFFT = np.fft.rfft(xxW, axis=0)
    
    if idxFreq is None or freqOut is None:
        dt = tVec[1] - tVec[0]
        freqUse = np.fft.rfftfreq(xxW.shape[0], dt)
        idxFreq = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
        freqOut = freqUse[idxFreq]
    nFreq = len(idxFreq);

    os.makedirs(outDir, exist_ok=True)
    dispHFile = os.path.join(outDir, "dispHForce.dat")
    dispVFile = os.path.join(outDir, "dispVForce.dat")

    dispH = np.memmap(dispHFile, dtype=np.complex128, mode='w+', shape=(nFreq, nRec, 3))
    dispV = np.memmap(dispVFile, dtype=np.complex128, mode='w+', shape=(nFreq, nRec, 3))
    dispH[:] = 0
    dispV[:] = 0
    dispH.flush(); dispV.flush()

    # decide on effective number of workers to use
    n_workers_eff = choose_workers(nRec, n_workers, min_rec_per_worker=150)
    
    # update on Feb 06, 2026 for cleaner chunking
    splits = np.linspace(0, nRec, n_workers_eff + 1, dtype=int)
    worker_ranges = [(splits[i], splits[i+1]) for i in range(n_workers_eff)]

    print(f"Starting parallel processing with {n_workers_eff} workers...")
    start = time.time()

    Parallel(n_jobs=n_workers_eff, backend='loky')(
        delayed(worker_procN)(
            worker_id, rec_start, rec_end,
            xxWFFT, distVec, idxFreq, xGrid, yGrid,
            xSrc, ySrc, azSrc, srcMeta,
            nFreq, chunk_size, outDir, dispHFile, dispVFile, freqOut
        )
        for worker_id, (rec_start, rec_end) in enumerate(worker_ranges)
    )

    print(f"All workers finished in {time.time() - start:.2f} s")

    dispH_final = np.memmap(dispHFile, dtype=np.complex128, mode='r', shape=(nFreq, nRec, 3))
    dispV_final = np.memmap(dispVFile, dtype=np.complex128, mode='r', shape=(nFreq, nRec, 3))

    np.savez(os.path.join(outDir, "receiverGrid.npz"), xGrid=xGrid, yGrid=yGrid, nFreq=nFreq, freqOut=freqOut)

    return dispH_final, dispV_final, freqOut

def simFixedDepth_partition_receiversN(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, srcMeta,
                                       fMin, fMax, chunk_size, outDir, n_workers=None, idxFreq=None, freqOut=None,
                                       saveHV=False, computeStrategy="loky_sharedmem"):
    """
    Computes displacement at one depth, partitioning receivers across workers.
    expects xxW to be the time domain Green's function

    Returns:
      - if save_HV=False: (dispTotal, freqOut)
      - if save_HV=True:  (dispH, dispV, dispTotal, freqOut)

    disp* are either ndarrays (threading_shared) or memmaps (merged output).
    """

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)

    nt, nx, ncomp = xxW.shape
    nRec = len(xGrid)
    nSrc = len(xSrc)
    if nRec == 0:
        raise ValueError("No receivers in grid.")
    os.makedirs(outDir, exist_ok=True)

    # FFT once (shared by all workers in either strategy)
    xxWFFT = np.fft.rfft(xxW, axis=0)

    # Choose freqs
    if idxFreq is None or freqOut is None:
        dt = tVec[1] - tVec[0]
        freqUse = np.fft.rfftfreq(xxW.shape[0], dt)
        idxFreq = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
        freqOut = freqUse[idxFreq]
    nFreq = len(idxFreq)

    # Decide worker count + contiguous ranges
    n_workers_eff = choose_workers(nRec, n_workers, min_rec_per_worker=150)
    splits = np.linspace(0, nRec, n_workers_eff + 1, dtype=int)
    worker_ranges = [(splits[i], splits[i+1]) for i in range(n_workers_eff)]

    print(f"[simDisp] nRec={nRec}, nSrc={nSrc}, nFreq={nFreq}, workers={n_workers_eff}, strategy={computeStrategy}")

    xxWFFT_sel = xxWFFT[idxFreq, :, :]  # (nFreq, nx, ncomp)
    phaseH_all = srcMeta["ampH"] * np.exp(-1j * srcMeta["phiH"])
    phaseV_all = srcMeta["ampV"] * np.exp(-1j * srcMeta["phiV"])
    
    # -------- Strategy: loky + shared output memmaps in /dev/shm --------
    if computeStrategy == "loky_sharedmem":
        
        shm_dir = "/dev/shm"
        if not os.path.isdir(shm_dir):
            raise RuntimeError("/dev/shm not available on this system.")

        tag = uuid.uuid4().hex
        disp_shape = (nFreq, nRec, 3)
        disp_dtype = np.complex128
        disp_dtype_str = np.dtype(disp_dtype).str

        dispTotal_path = os.path.join(shm_dir, f"dispTotal_{tag}.dat")
        dispTotal = np.memmap(
            dispTotal_path, dtype=disp_dtype, mode="w+", shape=disp_shape
        )
        dispTotal[:] = 0
        dispTotal.flush()

        if saveHV:
            dispH_path = os.path.join(shm_dir, f"dispH_{tag}.dat")
            dispV_path = os.path.join(shm_dir, f"dispV_{tag}.dat")

            dispH = np.memmap(
                dispH_path, dtype=disp_dtype, mode="w+", shape=disp_shape
            )
            dispV = np.memmap(
                dispV_path, dtype=disp_dtype, mode="w+", shape=disp_shape
            )
            dispH[:] = 0
            dispV[:] = 0
            dispH.flush()
            dispV.flush()
        else:
            dispH = dispV = None
            dispH_path = dispV_path = None

        t0 = time.time()
        Parallel(n_jobs=n_workers_eff, backend="loky")(
            delayed(worker_proc_loky_sharedout)(
                worker_id, rec_start, rec_end,
                xxWFFT_sel, distVec, xGrid, yGrid,
                xSrc, ySrc, azSrc, phaseV_all,phaseH_all,
                nFreq, chunk_size,
                dispTotal_path, disp_shape, disp_dtype_str,
                saveHV, dispH_path, dispV_path
            )
            for worker_id, (rec_start, rec_end) in enumerate(worker_ranges)
        )
        # print(f"[simDisp] loky_sharedmem done in {time.time()-t0:.1f}s")

        dispTotal.flush()
        if saveHV:
            dispH.flush()
            dispV.flush()

        np.savez(
            os.path.join(outDir, "receiverGrid.npz"),
            xGrid=xGrid, yGrid=yGrid, nFreq=nFreq, freqOut=freqOut
        )

        dispTotal_r = np.memmap(
            dispTotal_path, dtype=disp_dtype, mode="r", shape=disp_shape
        )

        if saveHV:
            dispH_r = np.memmap(
                dispH_path, dtype=disp_dtype, mode="r", shape=disp_shape
            )
            dispV_r = np.memmap(
                dispV_path, dtype=disp_dtype, mode="r", shape=disp_shape
            )
            return dispH_r, dispV_r, dispTotal_r, freqOut, [dispTotal_path, dispH_path, dispV_path]

        return dispTotal_r, freqOut, [dispTotal_path]
    
    else:
        raise ValueError("strategy must be 'loky_sharedmem'")


def simFixedDepth_partition_receiversF(xxWFFT, distVec, xGrid, yGrid, itmAC, zTar, xSrc, ySrc, azSrc, srcMeta,
                                       chunk_size, outDir=None, n_workers=None, freqOut=None, saveHV=False,
                                       computeStrategy="loky_sharedmem", reduceNN=False, nnMode=None, nnMeta=None):
    """
    Computes displacement at one depth, partitioning receivers across workers.

    Modes
    -----
    1) Displacement mode (legacy):
         reduceNN=False
         Returns displacement arrays/memmaps, depending on computeStrategy.

    2) NN-reduction mode (new):
         reduceNN=True
         Computes displacement in each worker and immediately reduces it
         to partial NN contributions. No writing to /dev/shm.

    Parameters
    ----------
    xxWFFT : np.ndarray
        Shape (nFreq, nDist, nComp), FFT-domain Green's functions for this zTar.
    distVec : np.ndarray
        Shape (nDist,), horizontal GF distance axis.
    xGrid, yGrid : np.ndarray
        Flattened receiver coordinates at fixed depth zTar.
    zTar : float
        Receiver depth for this call.
    xSrc, ySrc, azSrc : np.ndarray
        Source coordinates and azimuth.
    srcMeta : dict
        Must contain ampH, phiH, ampV, phiV with shape (nSrc, nFreq).
    chunk_size : int
        Inner chunking within each worker.
    outDir : str or None
        Needed only for legacy displacement mode.
    n_workers : int or None
    freqOut : np.ndarray or None
    saveHV : bool
        Only supported in displacement mode.
    computeStrategy : str
        "loky_sharedmem" for legacy displacement mode.
        "loky_reduce" for new reduction mode.
    reduceNN : bool
        If True, use worker_procF and return summed NN contribution.
    nnMode : str or None
        "volume", "vert_surface", or "hor_surface"
    nnMeta : dict or None
        Per-receiver geometry metadata aligned with xGrid/yGrid ordering.

    Returns
    -------
    If reduceNN=False and saveHV=False:
        dispTotal_r, freqOut, shmPaths
    If reduceNN=False and saveHV=True:
        dispH_r, dispV_r, dispTotal_r, freqOut, shmPaths
    If reduceNN=True:
        I_total, freqOut, None
    """

    if n_workers is None:
        n_workers = min(os.cpu_count() or 1, 32)

    nFreq, nDist, nComp = xxWFFT.shape
    nRec = len(xGrid)
    nSrc = len(xSrc)
    itmAC = np.atleast_2d(itmAC)
    nTM = len(itmAC[:,0])

    if nRec == 0:
        raise ValueError("No receivers in grid.")

    if len(yGrid) != nRec:
        raise ValueError("xGrid and yGrid must have the same length.")

    if reduceNN and saveHV:
        raise ValueError("saveHV=True is not supported when reduceNN=True")

    if reduceNN:
        if nnMode is None:
            raise ValueError("reduceNN=True but nnMode is None")
        if nnMeta is None:
            raise ValueError("reduceNN=True but nnMeta is None")
    else:
        if outDir is None:
            raise ValueError("outDir must be provided in displacement mode")
        #os.makedirs(outDir, exist_ok=True)

    # Decide worker count + contiguous ranges
    n_workers_eff = choose_workers(nRec, n_workers, min_rec_per_worker=150)
    splits = np.linspace(0, nRec, n_workers_eff + 1, dtype=int)
    worker_ranges = [(splits[i], splits[i + 1]) for i in range(n_workers_eff)]

    #print(f"[simDisp] zTar={zTar}, nRec={nRec}, nSrc={nSrc}, nFreq={nFreq}, "
    #      f"workers={n_workers_eff}, strategy={computeStrategy}, reduceNN={reduceNN}")

    # Precompute source phase factors once
    phaseH_all = srcMeta["ampH"] * np.exp(-1j * srcMeta["phiH"])
    phaseV_all = srcMeta["ampV"] * np.exp(-1j * srcMeta["phiV"])

    # ------------------------------------------------------------------
    # NEW MODE: compute displacement in worker and reduce immediately
    # ------------------------------------------------------------------
    if reduceNN:
        # In reduction mode we do not use /dev/shm and do not write files.
        # Each worker returns a partial NN contribution of shape (nFreq, 3).

        results = Parallel(n_jobs=n_workers_eff, backend="loky")(
            delayed(worker_procF)(worker_id, rec_start, rec_end,xxWFFT, distVec, xGrid, yGrid, itmAC,
                                  xSrc, ySrc, azSrc, phaseV_all, phaseH_all, nFreq, chunk_size,
                                  saveHV=False, reduceNN=True, nnMode=nnMode,nnMeta=nnMeta)
            for worker_id, (rec_start, rec_end) in enumerate(worker_ranges)
        )

        I_total = np.zeros((nFreq, 3, nTM), dtype=np.complex128)
        for res in results:
            I_total += res

        return I_total, freqOut, None

    # ------------------------------------------------------------------
    # LEGACY MODE: displacement first, shared output in /dev/shm
    # ------------------------------------------------------------------
    if computeStrategy == "loky_sharedmem":
        shm_dir = "/dev/shm"
        if not os.path.isdir(shm_dir):
            raise RuntimeError("/dev/shm not available on this system.")

        tag = uuid.uuid4().hex
        disp_shape = (nFreq, nRec, 3)
        disp_dtype = np.complex128
        disp_dtype_str = np.dtype(disp_dtype).str

        dispTotal_path = os.path.join(shm_dir, f"dispTotal_{tag}.dat")
        dispTotal = np.memmap(
            dispTotal_path, dtype=disp_dtype, mode="w+", shape=disp_shape
        )
        dispTotal[:] = 0
        dispTotal.flush()

        if saveHV:
            dispH_path = os.path.join(shm_dir, f"dispH_{tag}.dat")
            dispV_path = os.path.join(shm_dir, f"dispV_{tag}.dat")

            dispH = np.memmap(
                dispH_path, dtype=disp_dtype, mode="w+", shape=disp_shape
            )
            dispV = np.memmap(
                dispV_path, dtype=disp_dtype, mode="w+", shape=disp_shape
            )
            dispH[:] = 0
            dispV[:] = 0
            dispH.flush()
            dispV.flush()
        else:
            dispH = dispV = None
            dispH_path = dispV_path = None

            # not returning worker_results anymore, set back to return True for success
            Parallel(n_jobs=n_workers_eff, backend="loky")(
                delayed(worker_proc_loky_sharedout)(
                    worker_id, rec_start, rec_end,
                    xxWFFT, distVec, xGrid, yGrid,
                    xSrc, ySrc, azSrc, phaseV_all, phaseH_all,
                    nFreq, chunk_size,
                    dispTotal_path, disp_shape, disp_dtype_str,
                    saveHV, dispH_path, dispV_path
                    )
                    for worker_id, (rec_start, rec_end) in enumerate(worker_ranges)
                    )
            # no correlation coefficient accumulation anymore
            #localPtr = np.zeros((nFreq,), dtype=np.float64)
            #localPtt = np.zeros((nFreq,), dtype=np.float64)

            # no need to unpack the correlation weights
            #for ptr_w, ptt_w in worker_results:
            #    localPtr += ptr_w
            #    localPtt += ptt_w

        dispTotal.flush()
        dispTotal_r = np.memmap(
            dispTotal_path, dtype=disp_dtype, mode="r", shape=disp_shape
        )

        if saveHV:
            dispH.flush()
            dispV.flush()

            dispH_r = np.memmap(
                dispH_path, dtype=disp_dtype, mode="r", shape=disp_shape
            )
            dispV_r = np.memmap(
                dispV_path, dtype=disp_dtype, mode="r", shape=disp_shape
            )
            os.makedirs(outDir, exist_ok=True)
            # keep the grid sidecar only in displacement mode
            np.savez(
                os.path.join(outDir, "receiverGrid.npz"),
                xGrid=xGrid, yGrid=yGrid, zTar=zTar, nFreq=nFreq, freqOut=freqOut
            )

            return dispH_r, dispV_r, dispTotal_r, freqOut, [dispTotal_path, dispH_path, dispV_path]

        #np.savez(
        #    os.path.join(outDir, "receiverGrid.npz"),
        #    xGrid=xGrid, yGrid=yGrid, zTar=zTar, nFreq=nFreq, freqOut=freqOut
        #)

        # return set to the old version
        return dispTotal_r, freqOut, [dispTotal_path]

    else:
        raise ValueError("computeStrategy must be 'loky_sharedmem' in displacement mode, "
                         "or set reduceNN=True for direct NN reduction")

def choose_workers(nRec, n_workers_user=None, min_rec_per_worker=150):
    hw = os.cpu_count() or 1

    #print('total cpus ' + str(hw))
    if n_workers_user is None:
        n_workers_user = hw

    # hard caps
    n_workers = min(n_workers_user, hw)

    # avoid empty slices
    n_workers = min(n_workers, nRec)

    # avoid too-fine partitioning
    n_workers_by_rec = max(1, nRec // min_rec_per_worker)
    n_workers = min(n_workers, n_workers_by_rec if n_workers_by_rec > 0 else 1)

    return max(1, n_workers)

def find_receiver_index(x1, y1, xGrid, yGrid, xVec=None, yVec=None):
    """
    Find the receiver index closest to a given (x1, y1) location.

    Parameters
    ----------
    x1, y1 : float
        Target coordinates of interest.
    xGrid, yGrid : np.ndarray
        Flattened receiver coordinate arrays (1D, same length).
    xVec, yVec : np.ndarray, optional
        1D coordinate vectors used to generate the grid (optional).
        If provided, index computation is analytic and faster.

    Returns
    -------
    rec_index : int
        Index of the nearest receiver in flattened arrays.
    distance : float
        Euclidean distance between target and receiver location.
    """

    # --- Case 1: structured grid provided ---
    if xVec is not None and yVec is not None:
        nx = len(xVec)
        ix = np.argmin(np.abs(xVec - x1))
        iy = np.argmin(np.abs(yVec - y1))
        rec_index = iy * nx + ix
        distance = np.sqrt((xVec[ix] - x1)**2 + (yVec[iy] - y1)**2)
        return rec_index, distance

    # --- Case 2: general case, arbitrary receiver layout ---
    dist = np.sqrt((xGrid - x1)**2 + (yGrid - y1)**2)
    rec_index = np.argmin(dist)
    #print('RecInd = ' + str(rec_index) + ' Dist = ' + str(dist[rec_index]));
    #print('Recpoint x = ' + str(xGrid[rec_index]) + ' y = ' + str(yGrid[rec_index]));
    
    return rec_index, dist[rec_index]
    
def checkSurfCC(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, srcMeta, applyRot, thickness, vP, vS, rho, fMin, fMax, idxFreq=None, freqOut=None):
    """
    simply implements computeSurfCC for the two points specified as xGrid, yGrid
    additionally computes the bessel function for the Rayleigh and Love waes
    a plot is created to check if they match
    must match in all components if applyRot is set to 0,
    only the Z component will match if applyRot is set to 1.
    """
    ccRealZ, ccRealX, ccRealY, freqUse = computeSurfCC(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, srcMeta, applyRot, fMin, fMax, idxFreq, freqOut);
    recDist = yGrid[1] - yGrid[0];
    j0R, j0L, fJ = getBessel(thickness,vP,vS,rho,fMin,fMax,recDist)
    
    plotUtils.plotCC(ccRealZ, ccRealX, ccRealY, freqUse, j0R, j0L, fJ);
    
def computeSurfCC(xxW, tVec, distVec, xGrid, yGrid, xSrc, ySrc, azSrc, srcMeta, applyRot, fMin, fMax, idxFreq=None, freqOut=None):
    """
    Simulate displacement for many sources (single-depth case) in a fully vectorized way
    then compute the correlation between two points on the surface
    this is a serial version of the code and meant to work for just two points
    so xGrid and yGrid have length 2
    in case xGrid and yGrid have much larger length, the first two entries are considered
    
    Parameters
    ----------
    xxW : np.ndarray
        Green's function matrix of shape (nt, nx, ncomp)
    tVec : np.ndarray
        Time vector
    distVec : np.ndarray
        Distance samples for xxW
    xGrid, yGrid : np.ndarray
        Receiver coordinates (ideally of length 2)
    xSrc, ySrc : np.ndarray
        Source x, y positions (1D arrays of length nSrc)
    applyRot can be 0 or 1
    if applyRot is 0, then radial and transverse components are not rotated, in that case
    theoretical bessel functions for Love and Rayleigh waves must match simulation
    if applyRot is 1, then new X and Y components will not match the theoretical Bessel function
    but the Z component will match, same as for applyRot = 0, Z component is invariant
    
    Returns
    -------
    ccRealZ, ccRealX, ccRealY : np.ndarray
        Simulated frequency domain normalized cross-correlation between the two points
        freqUse: frequency vector corresponding to the cross-correlation
    """
    nt, nx, ncomp = xxW.shape
    nRec = len(xGrid)
    nSrc = len(xSrc)
    
    # Flatten receiver grids if they are 2D
    xGridFlat = np.ravel(xGrid)
    yGridFlat = np.ravel(yGrid)

    dt = tVec[1]-tVec[0];
    freqUse = np.fft.rfftfreq(nt,dt);
    nFreq = len(freqUse);
    
    # perform FFT and perform entire operation frequency domain
    xxWFFT = np.fft.rfft(xxW,axis=0);

    if idxFreq is None or freqOut is None:
        dt = tVec[1] - tVec[0]
        freqUse = np.fft.rfftfreq(xxW.shape[0], dt)
        idxFreq = np.where((freqUse >= fMin) & (freqUse <= fMax))[0]
        freqOut = freqUse[idxFreq]
        
    nFreq = len(idxFreq);
    
    ccAAZ = np.zeros((nFreq,))
    ccBBZ = np.zeros((nFreq,))
    ccABZ = np.zeros((nFreq,))

    ccAAX = np.zeros((nFreq,))
    ccBBX = np.zeros((nFreq,))
    ccABX = np.zeros((nFreq,))

    ccAAY = np.zeros((nFreq,))
    ccBBY = np.zeros((nFreq,))
    ccABY = np.zeros((nFreq,))
    
    for srcNo in range(nSrc):
        # 1) Compute distance and azimuth from current source to all receivers
        
        phaseH = srcMeta["ampH"][srcNo, :][:, None, None] * np.exp(-1j * srcMeta["phiH"][srcNo, :][:, None, None])
        phaseV = srcMeta["ampV"][srcNo, :][:, None, None] * np.exp(-1j * srcMeta["phiV"][srcNo, :][:, None, None])
        dx = xGridFlat - xSrc[srcNo]
        dy = yGridFlat - ySrc[srcNo]
        distGrid = np.sqrt(dx**2 + dy**2)
        azGridV = np.arctan2(dx, dy)  # QSEIS convention: North = +x
        # now each horizontal source has a azimuth so do that correction
        azGridH = azGridV - azSrc[srcNo];
        
        # 2) Interpolate GF along horizontal distance for all components
        xxInterpH = np.zeros((nFreq, nRec, ncomp),dtype='complex')
        xxInterpV = np.zeros((nFreq, nRec, ncomp),dtype='complex')
        
        for icomp in range(ncomp):
            f = interp1d(distVec, xxWFFT[idxFreq, :, icomp], kind='linear', bounds_error=False, fill_value=0.0, axis=1)
            #xxInterp[:, :, icomp] = f(distGrid)*phaseFactor
            xxInterpH[:,:,icomp] = f(distGrid)*phaseH
            xxInterpV[:,:,icomp] = f(distGrid)*phaseV
        # 3) Rotate and sum contributions
        # Components: [fh-2.tz, fh-2.tr, fh-2.tt, fz-2.tz, fz-2.tr]

        if(applyRot):
            uHZ = -xxInterpH[:,:,0];
            uHX = xxInterpH[:, :, 1] * np.sin(azGridH) + xxInterpH[:, :, 2] * np.cos(azGridH)  # X
            uHY = xxInterpH[:, :, 1] * np.cos(azGridH) - xxInterpH[:, :, 2] * np.sin(azGridH)  # Y
        else:
            uHZ = -xxInterpH[:,:,0];
            uHX = xxInterpH[:, :, 2];
            uHY = xxInterpH[:, :, 1];

        # populate the cross-correlation
        ccAAZ = ccAAZ + uHZ[:,0]*np.conjugate(uHZ[:,0]);
        ccBBZ = ccBBZ + uHZ[:,1]*np.conjugate(uHZ[:,1]);
        ccABZ = ccABZ + uHZ[:,0]*np.conjugate(uHZ[:,1]);

        ccAAX = ccAAX + uHX[:,0]*np.conjugate(uHX[:,0]);
        ccBBX = ccBBX + uHX[:,1]*np.conjugate(uHX[:,1]);
        ccABX = ccABX + uHX[:,0]*np.conjugate(uHX[:,1]);
        
        ccAAY = ccAAY + uHY[:,0]*np.conjugate(uHY[:,0]);
        ccBBY = ccBBY + uHY[:,1]*np.conjugate(uHY[:,1]);
        ccABY = ccABY + uHY[:,0]*np.conjugate(uHY[:,1]);

    # compute the normalized cross-correlation
    ccRealZ = np.real(ccABZ/np.sqrt(ccAAZ*ccBBZ));
    ccRealX = np.real(ccABX/np.sqrt(ccAAX*ccBBX));
    ccRealY = np.real(ccABY/np.sqrt(ccAAY*ccBBY));
    
    return ccRealZ, ccRealX, ccRealY, freqOut
    
def getBessel(thickness,vP,vS,rho,fMin,fMax,recDist,love=True):
    """
    computes the bessel function J0(2*pi*d*f/v)
    thickness, vP, vS, and rho are used for calculating the love and Rayleigh
    wave phase velocities
    computation is done in the frequency band fMin to fMax
    recDist is the distance between the two receivers in meters
    """
    freqs = np.arange(fMin,fMax,0.1);
    
    periods = 1/freqs;

    vDispRay = surf96(thickness,vP,vS,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    #print(freqs)
    #print(vDispRay)
    if(love):
        vDispLove = surf96(thickness,vP,vS,rho,periods,wave="love",mode=1,velocity="phase",flat_earth=True)
        j0L = j0(2*np.pi*recDist*freqs/vDispLove);
    else:
        j0L = np.zeros((len(freqs),));
    #print(vDispLove)
    j0R = j0(2*np.pi*recDist*freqs/vDispRay);
    #j0L = j0(2*np.pi*recDist*freqs/vDispLove);

    return j0R, j0L, freqs

def getBesselN(thickness,vP,vS,rho,fMin,fMax,recDist,n,love=True):
    """
    computes the bessel function J0(2*pi*d*f/v)
    thickness, vP, vS, and rho are used for calculating the love and Rayleigh
    wave phase velocities
    computation is done in the frequency band fMin to fMax
    recDist is the distance between the two receivers in meters
    """
    freqs = np.arange(fMin,fMax,0.1);
    
    periods = 1/freqs;

    vDispRay = surf96(thickness,vP,vS,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    #print(freqs)
    #print(vDispRay)
    if(love):
        vDispLove = surf96(thickness,vP,vS,rho,periods,wave="love",mode=1,velocity="phase",flat_earth=True)
        jnL = jv(n,2*np.pi*recDist*freqs/vDispLove);
    else:
        jnL = np.zeros((len(freqs),))
    #print(vDispLove)
    jnR = jv(n,2*np.pi*recDist*freqs/vDispRay);
    #jnL = jv(n,2*np.pi*recDist*freqs/vDispLove);

    return jnR, jnL, freqs

def computeRayleighDisp(z, xVec, yVec, nSrc, freqVec, theta, phase0, amp, vR, vp, vs):
    """
    Ambient-noise-like Rayleigh displacement field in frequency domain.

    Convention:
      - z positive up
      - free surface at z=0, half-space occupies z <= 0
      - Rayleigh plane waves propagate horizontally with random azimuth

    Parameters
    ----------
    z : float
        Depth coordinate (must be <= 0 for points inside half-space).
    xVec, yVec : (nGrid,) array_like
        Grid coordinates.
    nSrc : int
        Number of random plane-wave sources.
    freqVec : (nFreq,) array_like
        Frequencies (Hz).
    vR : float or (nFreq,) array_like
        Rayleigh phase velocity (m/s).
    vp, vs : float
        P and S velocities (m/s) for eigenfunction decay.
    theta: (nSrc,) array_like
        random angles of propagation
    phase0: (nSrc,) array_like
        random initial source phases
    amp: (nSrc,) array_like
        random source amplitude between (0,1)

    Returns
    -------
    U : (nFreq, nGrid, 3) complex ndarray
        Components ordered as [Uz, Ux, Uy].
    """

    #x = np.asarray(xVec, dtype=float)
    #y = np.asarray(yVec, dtype=float)

    x = np.atleast_1d(xVec).astype(float)
    y = np.atleast_1d(yVec).astype(float)
    freq = np.asarray(freqVec, dtype=float)

    nG = x.size
    nF = freq.size

    # Ensure vR is (nF,)
    vR = np.asarray(vR, dtype=float)
    if vR.ndim == 0:
        vR = np.full(nF, float(vR))
    elif vR.shape != (nF,):
        raise ValueError("vR must be scalar or have shape (nFreq,)")

    if z > 0:
        raise ValueError("For half-space z<=0 with z positive up, provide z <= 0.")

    omega = 2*np.pi*freq                              # (nF,)
    k = omega / vR                                     # (nF,)

    # Depth decay rates (positive)
    # Rayleigh condition implies vR < vs < vp, so these stay real.
    alpha = k * np.sqrt(1.0 - (vR / vp)**2)           # (nF,)
    beta  = k * np.sqrt(1.0 - (vR / vs)**2)           # (nF,)

    # B/A ratio from free-surface traction
    ratio = -2j * k * alpha / (k**2 + beta**2)        # (nF,)

    # Depth factors at your z (<=0)
    Ea = np.exp(alpha * z)                             # (nF,) decays with depth
    Eb = np.exp(beta  * z)                             # (nF,)

    # Random directions
    #theta = rng.uniform(0.0, 2*np.pi, size=nSrc)      # (nSrc,)
    ct = np.cos(theta)
    st = np.sin(theta)

    #phase0 = rng.uniform(0.0, 2*np.pi, size=(nSrc, nF))
    A = amp * np.exp(1j * phase0)                     # (nSrc,nF) complex

    # Compute B
    B = A * ratio[None, :]                             # (nSrc,nF)

    # Rayleigh eigenfunctions at depth z
    Ur = 1j * k[None, :] * A * Ea[None, :] + beta[None, :] * B * Eb[None, :]   # (nSrc,nF)
    Uz = alpha[None, :] * A * Ea[None, :] - 1j * k[None, :] * B * Eb[None, :]  # (nSrc,nF)

    # Horizontal coordinate along propagation for each source and grid point
    s = ct[:, None] * x[None, :] + st[:, None] * y[None, :]                    # (nSrc,nG)

    # Accumulate without forming giant (nSrc,nF,nG) arrays if you want:
    # Here we do a wave loop to reduce memory; still vectorized over (nF,nG).
    Ux = np.zeros((nF, nG), dtype=np.complex128)
    Uy = np.zeros((nF, nG), dtype=np.complex128)
    UZ = np.zeros((nF, nG), dtype=np.complex128)

    for m in range(nSrc):
        ph = np.exp(1j * k[:, None] * s[m][None, :])   # (nF,nG)
        Ux += (Ur[m][:, None] * ct[m]) * ph
        Uy += (Ur[m][:, None] * st[m]) * ph
        UZ += (Uz[m][:, None]) * ph

    U = np.stack([Ux, Uy, UZ], axis=-1)                # (nF,nG,3)

    return U

def rayleigh_transfer_function(freq, vR, vp, vs, z):
    """
    Theoretical Rayleigh-wave depth transfer function in a homogeneous half-space.

    Convention:
      - e^{-i ω t}
      - z positive upward, free surface at z=0, half-space z<=0
      - z must be <= 0 (e.g., z=-250 for 250 m depth)

    Parameters
    ----------
    freq : (nF,) array_like
        Frequency in Hz.
    vR : float or (nF,) array_like
        Rayleigh phase velocity (m/s).
    vp, vs : float
        P and S velocity (m/s).
    z : float
        Depth coordinate (<=0).

    Returns
    -------
    Th : (nF,) ndarray
        Horizontal transfer function |Ur(z)|/|Ur(0)| (applies to Ux and Uy).
    Tz : (nF,) ndarray
        Vertical transfer function |Uz(z)|/|Uz(0)|.
    """
    freq = np.asarray(freq, dtype=float)
    omega = 2*np.pi*freq

    vR = np.asarray(vR, dtype=float)
    if vR.ndim == 0:
        vR = np.full_like(freq, float(vR))
    elif vR.shape != freq.shape:
        raise ValueError("vR must be scalar or same shape as freq.")

    if z > 0:
        raise ValueError("Use z<=0 for depth in z-positive-up convention.")

    # Avoid omega=0 division issues (if freq includes 0)
    tiny = 1e-30
    k = omega / (vR + tiny)

    alpha = k * np.sqrt(1.0 - (vR / vp)**2)
    beta  = k * np.sqrt(1.0 - (vR / vs)**2)

    r = -2j * k * alpha / (k**2 + beta**2 + tiny)   # B/A

    Ea = np.exp(alpha * z)   # decays since z<=0
    Eb = np.exp(beta  * z)

    Ur_z = 1j*k*Ea + beta*r*Eb
    Uz_z = alpha*Ea - 1j*k*r*Eb

    Ur_0 = 1j*k + beta*r
    Uz_0 = alpha - 1j*k*r

    Th = np.abs(Ur_z) / (np.abs(Ur_0) + tiny)
    Tz = np.abs(Uz_z) / (np.abs(Uz_0) + tiny)

    return Th, Tz

def getSurfDeepRayDispPerRea(zList, azSrc, phiSrc, ampSrc, outReaPath, reaNo, vR, vP, vS, freqOut=None):
    """
    this script will be run before every realization
    for a particular source distribution which is fixed per realization
    it will compute the Rayleigh wave displacement at point (0,0) on the surface and at depth
    specified by zList
    
    """
    nSrc = len(azSrc)
    gridX = np.array((0.0,)); gridY = np.array((0.0,))

    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointFull = np.zeros((nFreq,3,zLen))
    
    for zNo, zVal in enumerate(zList):
        # run computeRayleighDisp
        if(zVal>0):
            zValUse = -zVal
        else:
            zValUse = zVal

        U = computeRayleighDisp(zValUse, gridX, gridY, nSrc, freqOut, azSrc, phiSrc, ampSrc, vR, vP, vS)

        dispPointFull[:,:,zNo] = np.abs(U[:,0,:])

    # save this in the rea path, to be read in again after all realizations have been done
    # to scale the outputNN
    sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointFull=dispPointFull, freqOut=freqOut)
    return dispPointFull, freqOut

def getSurfDeepRayDispPerReaMultiPoint(zList, itmAll, azSrc, phiSrc, ampSrc, outReaPath, reaNo, vR, vP, vS, freqOut=None):
    """
    this script will be run before every realization
    for a particular source distribution which is fixed per realization
    it will compute the Rayleigh wave displacement at point (0,0) on the surface and at depth
    specified by zList
    
    """
    nSrc = len(azSrc)
    gridX = np.asarray(itmAll[:,0]); gridY = np.asarray(itmAll[:,1])

    nRec = len(gridX)
    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointFull = np.zeros((nFreq,nRec,3,zLen))
    
    for zNo, zVal in enumerate(zList):
        # run computeRayleighDisp
        if(zVal>0):
            zValUse = -zVal
        else:
            zValUse = zVal

        U = computeRayleighDisp(zValUse, gridX, gridY, nSrc, freqOut, azSrc, phiSrc, ampSrc, vR, vP, vS)

        dispPointFull[:,:,:,zNo] = np.abs(U[:,:,:])

    # save this in the rea path, to be read in again after all realizations have been done
    # to scale the outputNN
    sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointFull=dispPointFull, freqOut=freqOut)
    return dispPointFull, freqOut

def computeFullBodyDisp(z, xVec, yVec, nSrc, freqVec, vp, vs, meta):
    """
    Superpose random plane P-waves and S-waves in a homogeneous medium (frequency domain).

    Convention:
      - time dependence e^{-i ω t}
      - coordinates (x,y,z) with z positive up
      - returns displacement components ordered as [Ux, Uy, Uz]

    Parameters
    ----------
    z : float
        Single depth coordinate for all grid points (e.g., z=-250 for 250 m depth).
    xVec, yVec : array_like
        Grid coordinates (nGrid,). Can be scalars or arrays.
    nSrc : int
        Number of plane P-wave sources and S-wave sources: nSrc each.
    freqVec : (nFreq,) array_like
        Frequencies [Hz].
    vp, vs : float
        P and S speeds [m/s].
    meta is a dictionary of plane wave propagation parameters

    Returns
    -------
    U : (nFreq, nGrid, 3) complex ndarray
        Displacement field: [:,:,0]=Uz, [:,:,1]=Ux, [:,:,2]=Uy

    """
    x = np.atleast_1d(xVec).astype(float)
    y = np.atleast_1d(yVec).astype(float)
    z = float(z)

    freq = np.asarray(freqVec, dtype=float)
    nF = freq.size
    nG = x.size

    omega = 2*np.pi*freq  # (nF,)
    kP = omega / vp
    kS = omega / vs

    # Allocate output: [Uz, Ux, Uy]
    UZ = np.zeros((nF, nG), dtype=np.complex128)
    UX = np.zeros((nF, nG), dtype=np.complex128)
    UY = np.zeros((nF, nG), dtype=np.complex128)

    # unpack meta
    thetaP = meta["thetaP"]
    muP = meta["muP"]
    ampP = meta["ampP"]
    phaseP = meta["phaseP"]

    thetaS = meta["thetaS"]
    muS = meta["muS"]
    psiS = meta["psiS"]
    ampS = meta["ampS"]
    phaseS = meta["phaseS"]

    # Loop over P-wave sources, vectorize over (nF,nG)
    uP = rand_dirs(thetaP, muP)      # (nSrc,3)
    CP = amps_phases(ampP, phaseP)  # (nSrc,nF) or (nP,1)

    for m in range(nSrc):
        # dot(uhat, r) for all grid points
        s = uP[m, 0]*x + uP[m, 1]*y + uP[m, 2]*z     # (nG,)

        # phase term exp(i k s)
        ph = np.exp(1j * kP[:, None] * s[None, :])    # (nF,nG)

        Cm = CP[m]                                    # (nF,) or (1,)
        if Cm.shape[0] == 1:
            Cm = np.repeat(Cm, nF)

        ph *= Cm[:, None]                             # (nF,nG)

        # Polarization = uhat for P
        UX += ph * uP[m, 0]
        UY += ph * uP[m, 1]
        UZ += ph * uP[m, 2]

    # loop over S-wave sources
    uS = rand_dirs(thetaS, muS)
    polS = s_polarizations(uS, psiS)                # (nS,3)
    CS = amps_phases(ampS, phaseS)

    for m in range(nSrc):
        s = uS[m, 0]*x + uS[m, 1]*y + uS[m, 2]*z      # (nG,)
        ph = np.exp(1j * kS[:, None] * s[None, :])     # (nF,nG)

        Cm = CS[m]
        if Cm.shape[0] == 1:
            Cm = np.repeat(Cm, nF)

        ph *= Cm[:, None]

        # Polarization ⟂ uhat, random mix angle psi
        UX += ph * polS[m, 0]
        UY += ph * polS[m, 1]
        UZ += ph * polS[m, 2]

    U = np.stack([UX, UY, UZ], axis=-1)  # (nF,nG,3), ordered [Ux,Uy,Uz]
    return U

def computeFullBodyDisp_parallel(z, xVec, yVec, nSrc, freqVec, vp, vs, meta,
                                 n_workers=4, min_rec_per_worker=150):

    x = np.atleast_1d(xVec).astype(float)
    y = np.atleast_1d(yVec).astype(float)
    z = float(z)

    freq = np.asarray(freqVec, dtype=float)
    nF = freq.size
    nRec = x.size

    # choose effective workers
    n_workers_eff = choose_workers(nRec, n_workers, min_rec_per_worker=min_rec_per_worker)

    # split receiver indices into contiguous chunks
    rec_ranges = []
    for inds in np.array_split(np.arange(nRec), n_workers_eff):
        if len(inds) == 0:
            continue
        rec_ranges.append((inds[0], inds[-1] + 1))

    results = Parallel(n_jobs=n_workers_eff, backend="loky")(
        delayed(body_worker)(
            rec_start, rec_end, z, x, y, freq, vp, vs, meta
        )
        for rec_start, rec_end in rec_ranges
    )

    U = np.zeros((nF, nRec, 3), dtype=np.complex128)
    for rec_start, rec_end, U_chunk in results:
        U[:, rec_start:rec_end, :] = U_chunk

    return U

def body_worker(rec_start, rec_end, z, x, y, freq, vp, vs, meta):
    xw = x[rec_start:rec_end]
    yw = y[rec_start:rec_end]

    nF = freq.size
    nG = xw.size

    omega = 2 * np.pi * freq
    kP = omega / vp
    kS = omega / vs

    UX = np.zeros((nF, nG), dtype=np.complex128)
    UY = np.zeros((nF, nG), dtype=np.complex128)
    UZ = np.zeros((nF, nG), dtype=np.complex128)

    # ---- unpack meta ----
    thetaP = meta["thetaP"]
    muP = meta["muP"]
    ampP = meta["ampP"]
    phaseP = meta["phaseP"]

    thetaS = meta["thetaS"]
    muS = meta["muS"]
    psiS = meta["psiS"]
    ampS = meta["ampS"]
    phaseS = meta["phaseS"]

    # ---- P waves ----
    uP = rand_dirs(thetaP, muP)         # (nSrc, 3)
    CP = amps_phases(ampP, phaseP)      # (nSrc, nF)

    for m in range(uP.shape[0]):
        s = uP[m, 0] * xw + uP[m, 1] * yw + uP[m, 2] * z
        ph = np.exp(1j * kP[:, None] * s[None, :])

        Cm = CP[m]
        if Cm.shape[0] == 1:
            Cm = np.repeat(Cm, nF)

        ph *= Cm[:, None]

        UX += ph * uP[m, 0]
        UY += ph * uP[m, 1]
        UZ += ph * uP[m, 2]

    # ---- S waves ----
    uS = rand_dirs(thetaS, muS)
    polS = s_polarizations(uS, psiS)
    CS = amps_phases(ampS, phaseS)

    for m in range(uS.shape[0]):
        s = uS[m, 0] * xw + uS[m, 1] * yw + uS[m, 2] * z
        ph = np.exp(1j * kS[:, None] * s[None, :])

        Cm = CS[m]
        if Cm.shape[0] == 1:
            Cm = np.repeat(Cm, nF)

        ph *= Cm[:, None]

        UX += ph * polS[m, 0]
        UY += ph * polS[m, 1]
        UZ += ph * polS[m, 2]

    return rec_start, rec_end, np.stack([UX, UY, UZ], axis=-1)

def getSurfDeepBodyDispPerRea(zList, meta, outReaPath, reaNo, vP, vS, freqOut=None):
    """
    this script will be run before every realization
    for a particular source distribution which is fixed per realization
    it will compute the Rayleigh wave displacement at point (0,0) on the surface and at depth
    specified by zList
    z, xVec, yVec, nSrc, freqVec, vp, vs, meta
    """
    
    nSrc = len(meta["thetaP"])
    gridX = np.array((0.0,)); gridY = np.array((0.0,))

    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointFull = np.zeros((nFreq,3,zLen))
    
    for zNo, zVal in enumerate(zList):
        # run computeFullBodyDisp

        U = computeFullBodyDisp(zVal, gridX, gridY, nSrc, freqOut, vP, vS, meta)

        dispPointFull[:,:,zNo] = np.abs(U[:,0,:])

    # save this in the rea path, to be read in again after all realizations have been done
    # to scale the outputNN
    sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointFull=dispPointFull, freqOut=freqOut)
    return dispPointFull, freqOut

def getSurfDeepBodyDispPerReaMultiPoint(zList, itmAll, meta, outReaPath, reaNo, vP, vS, freqOut=None):
    """
    this script will be run before every realization
    for a particular source distribution which is fixed per realization
    it will compute the Rayleigh wave displacement at point desired points on the surface and at depth
    specified by zList
    z, itmAll, xVec, yVec, nSrc, freqVec, vp, vs, meta
    """
    
    nSrc = len(meta["thetaP"])
    gridX = np.asarray(itmAll[:,0]); gridY = np.asarray(itmAll[:,1])
    nRec = len(gridX)
    nFreq = len(freqOut)
    zLen = len(zList)
    
    dispPointFull = np.zeros((nFreq,nRec,3,zLen))
    
    for zNo, zVal in enumerate(zList):
        # run computeFullBodyDisp

        U = computeFullBodyDisp(zVal, gridX, gridY, nSrc, freqOut, vP, vS, meta)

        dispPointFull[:,:,:,zNo] = np.abs(U[:,:,:])

    # save this in the rea path, to be read in again after all realizations have been done
    # to scale the outputNN
    sName = 'surfDeepDispRea' + str(reaNo) + '.npz'
    np.savez(os.path.join(outReaPath, sName), dispPointFull=dispPointFull, freqOut=freqOut)
    return dispPointFull, freqOut

def genAmbSrc(nSrc, mode="ring", R1=2000.0, R2=2500.0, xMin=-1000.0, xMax=1000.0,
              yMin=-1000.0, yMax=1000.0, randomPhase=True, freqDependent=True, nFreq=None,
              decoupledHV=True, scaleVH = 1.0):
    """
    Generate ambient noise sources, each with a random amplitude and phase per frequency bin.
    operates in three modes: ring, internal, inDisk
    ring = ring of sources outside the sensor array defined by inner and out radius of R1 and R2
    internal = sources randomly distributed within the receiver grid
    inDisk = sources distributed within a disk of radius R1 and R2, uniform area distribution
    xMin, xMax, yMin, yMax are the simulation domain applicabe to mode=internal
    randomPhase and frequencyDependent set to True sd default
    nFreq = number of frequency bins to be analyzed (generated from getFreqGrid)
    
    Returns:
      xSrc, ySrc, azSrc, srcMeta
    """
    
    if nFreq is None:
        raise ValueError("genAmbSrc: nFreq must be provided")

    rng = np.random.default_rng()

    # positions
    if mode == "ring":
        theta = rng.uniform(0, 2*np.pi, nSrc)
        r = rng.uniform(R1, R2, nSrc)
        xSrc, ySrc = r*np.cos(theta), r*np.sin(theta)
    elif mode == "internal":
        xSrc = rng.uniform(xMin, xMax, nSrc)
        ySrc = rng.uniform(yMin, yMax, nSrc)
    elif mode == "inDisk":
        theta = rng.uniform(0, 2*np.pi, nSrc)
        r = np.sqrt(rng.uniform(R1**2, R2**2, nSrc))
        xSrc = r*np.cos(theta); ySrc = r*np.sin(theta)
    else:
        raise ValueError("mode must be 'ring', 'internal', or 'inDisk'")

    azSrc = rng.uniform(0, 2*np.pi, nSrc)

    def make_phi_amp():
        if randomPhase:
            phi = 2*np.pi * rng.random((nSrc, nFreq))
        else:
            phi = np.zeros((nSrc, nFreq))
        if freqDependent:
            amp = rng.random((nSrc, nFreq))
        else:
            amp = rng.random((nSrc, 1)) * np.ones((nSrc, nFreq))
        return phi, amp

    if not decoupledHV:
        phiH, ampH = make_phi_amp()
        phiV = phiH.copy()
        ampV = ampH.copy()
        ampV *= scaleVH
        srcMeta = {"phiH": phiH, "ampH": ampH,"phiV": phiV, "ampV": ampV}
        
        return xSrc, ySrc, azSrc, srcMeta

    phiH, ampH = make_phi_amp()
    phiV, ampV = make_phi_amp()
    ampV *= scaleVH

    srcMeta = {"phiH": phiH, "ampH": ampH,"phiV": phiV, "ampV": ampV}
    return xSrc, ySrc, azSrc, srcMeta


def genAmbSrcTheta(nSrc, mode="ring", R1=2000.0, R2=2500.0, ang1 = 0, ang2 =2*np.pi, xMin=-1000.0, xMax=1000.0,
              yMin=-1000.0, yMax=1000.0, randomPhase=True, freqDependent=True, nFreq=None,
              decoupledHV=True, scaleVH = 1.0):
    """
    Generate ambient noise sources, each with a random amplitude and phase per frequency bin.
    operates in three modes: ring, internal, inDisk
    ring = ring of sources outside the sensor array defined by inner and out radius of R1 and R2
    internal = sources randomly distributed within the receiver grid
    inDisk = sources distributed within a disk of radius R1 and R2, uniform area distribution
    xMin, xMax, yMin, yMax are the simulation domain applicabe to mode=internal
    randomPhase and frequencyDependent set to True sd default
    nFreq = number of frequency bins to be analyzed (generated from getFreqGrid)
    
    Returns:
      xSrc, ySrc, azSrc, srcMeta
    """
    
    if nFreq is None:
        raise ValueError("genAmbSrc: nFreq must be provided")

    rng = np.random.default_rng()

    # positions
    if mode == "ring":
        theta = rng.uniform(ang1, ang2, nSrc)
        r = rng.uniform(R1, R2, nSrc)
        xSrc, ySrc = r*np.cos(theta), r*np.sin(theta)
    elif mode == "internal":
        xSrc = rng.uniform(xMin, xMax, nSrc)
        ySrc = rng.uniform(yMin, yMax, nSrc)
    elif mode == "inDisk":
        theta = rng.uniform(ang1, ang2, nSrc)
        r = np.sqrt(rng.uniform(R1**2, R2**2, nSrc))
        xSrc = r*np.cos(theta); ySrc = r*np.sin(theta)
    else:
        raise ValueError("mode must be 'ring', 'internal', or 'inDisk'")

    azSrc = rng.uniform(0, 2*np.pi, nSrc)

    def make_phi_amp():
        if randomPhase:
            phi = 2*np.pi * rng.random((nSrc, nFreq))
        else:
            phi = np.zeros((nSrc, nFreq))
        if freqDependent:
            amp = rng.random((nSrc, nFreq))
        else:
            amp = rng.random((nSrc, 1)) * np.ones((nSrc, nFreq))
        return phi, amp

    if not decoupledHV:
        phiH, ampH = make_phi_amp()
        phiV = phiH.copy()
        ampV = ampH.copy()
        ampV *= scaleVH
        srcMeta = {"phiH": phiH, "ampH": ampH,"phiV": phiV, "ampV": ampV}
        
        return xSrc, ySrc, azSrc, srcMeta

    phiH, ampH = make_phi_amp()
    phiV, ampV = make_phi_amp()
    ampV *= scaleVH

    srcMeta = {"phiH": phiH, "ampH": ampH,"phiV": phiV, "ampV": ampV}
    return xSrc, ySrc, azSrc, srcMeta

def genBodySrc(nSrc, nFreq, eP = 1/3, eS = 2/3):
    meta = {}

    scaleP = np.sqrt(eP); scaleS = np.sqrt(eS)
    # --- P waves ---
    thetaP = np.random.uniform(0.0, 2*np.pi, size=nSrc)
    muP = np.random.uniform(-1.0, 1.0, size=nSrc)          # cos(phi)
    ampP = scaleP*np.random.uniform(0.0, 1.0, size=(nSrc, nFreq))
    phaseP = np.random.uniform(0.0, 2*np.pi, size=(nSrc, nFreq))

    thetaS = np.random.uniform(0.0, 2*np.pi, size=nSrc)
    muS = np.random.uniform(-1.0, 1.0, size=nSrc)          # cos(phi)
    ampS = scaleS*np.random.uniform(0.0, 1.0, size=(nSrc, nFreq))
    phaseS = np.random.uniform(0.0, 2*np.pi, size=(nSrc, nFreq))

    psiS = np.random.uniform(0.0, 2*np.pi, size=nSrc)

    meta["thetaP"] = thetaP
    meta["muP"] = muP
    meta["ampP"] = ampP
    meta["phaseP"] = phaseP

    # --- S waves ---    
    #uS, thS, phS = rand_dirs(nSrc)
    #polS, psiS = s_polarizations(uS)                # (nS,3)
    #CS, ampS, phaseS = amps_phases(nSrc, nFreq)
    meta["thetaS"] = thetaS
    meta["muS"] = muS
    meta["psiS"] = psiS
    meta["ampS"] = ampS
    meta["phaseS"] = phaseS

    return meta

def rand_dirs(theta, mu):
        """
        Random isotropic directions on the unit sphere.
        theta ~ U(0,2pi), mu=cos(phi)~U(-1,1)
        Returns uhat: (n,3), theta, phi.
        """
        sinphi = np.sqrt(1.0 - mu**2)
        ux = sinphi * np.cos(theta)
        uy = sinphi * np.sin(theta)
        uz = mu
        uhat = np.stack([ux, uy, uz], axis=1)         # (n,3)
        #phi = np.arccos(mu)
        return uhat

def amps_phases(amp, phase):
        """
        Returns complex source spectrum C: (nSrc,nFreq) if freqDependent else (n,1).
        """
        C = amp * np.exp(1j * phase)
        return C

def s_polarizations(uhat, psi):
        """
        Build random S-wave polarization vectors perpendicular to uhat.
        Uses a random 'mix angle' psi in the transverse plane.

        Returns pol: (n,3), psi: (n,)
        """
        n = uhat.shape[0]
        #psi = np.random.uniform(0.0, 2*np.pi, size=n)

        # Pick a helper vector not parallel to uhat to build a basis
        a = np.tile(np.array([0.0, 0.0, 1.0]), (n, 1))
        # If uhat is too close to z-axis, switch helper to y-axis
        near = np.abs(uhat[:, 2]) > 0.9
        a[near] = np.array([0.0, 1.0, 0.0])

        e1 = np.cross(uhat, a)
        e1 /= np.linalg.norm(e1, axis=1, keepdims=True)

        e2 = np.cross(uhat, e1)  # already unit-length if uhat,e1 are unit + orthogonal

        pol = np.cos(psi)[:, None] * e1 + np.sin(psi)[:, None] * e2
        return pol

def getFreqGrid(tTot, nSamp, fMin, fMax, df):
    """
    Compute the FFT frequency grid and select bins within [fMin, fMax]
    subsampled to ~df_target.

    Returns
    -------
    freqOut : ndarray
        Selected frequency bins (Hz)
    idx : ndarray
        Indices of those bins in the native FFT grid
    df_native : float
        Native FFT frequency spacing
    """

    dt = tTot / nSamp
    freqUse = np.fft.rfftfreq(nSamp, dt)
    print(freqUse[0:5])
    df_native = freqUse[1] - freqUse[0]
    step = max(1, int(round(df / df_native)))
    idx = np.where((freqUse >= fMin) & (freqUse <= fMax))[0][::step]
    freqOut = freqUse[idx]
    return freqOut, idx, df_native
    
def computeDistAzi(xGrid, yGrid, xSrc, ySrc):
    """
    Compute distance and azimuth from a source to all grid points.

    Parameters
    ----------
    xGrid : np.ndarray
        2D array of x-coordinates of the grid (shape: ny x nx)
    yGrid : np.ndarray
        2D array of y-coordinates of the grid (shape: ny x nx)
    xSrc : float
        x-coordinate of the source
    ySrc : float
        y-coordinate of the source

    Returns
    -------
    distVec : np.ndarray
        1D array of distances from source to all grid points (flattened)
    azimuthVec : np.ndarray
        1D array of azimuths (radians) from source to all grid points (flattened)
    """
    dx = xGrid - xSrc
    dy = yGrid - ySrc

    distVec = np.sqrt(dx**2 + dy**2).ravel()        # Flattened distance vector
    # note that this is dx/dy because QSEIS operates like that, North is +x
    azimuthVec = np.arctan2(dx, dy).ravel()        # Flattened azimuth vector

    return distVec, azimuthVec

def cleanup_sharedmem(shm_paths):
    for p in shm_paths:
        if p and os.path.exists(p):
            os.remove(p)
    return True
    
if __name__ == "__main__":
    main()