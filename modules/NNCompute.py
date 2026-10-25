#!/usr/bin/env python
# coding: utf-8

import numpy as np
import math
import os
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
from scipy.io import savemat
from scipy.io import loadmat
import scipy.constants as const
from pysurf96 import surf96
from modules import NNGeometryN
from modules import simDisp
from modules import plotUtils

def main():
    # don't use config yet, just hard code for now, can be implemented anytime
    # test the calculation engine first with just a P-wave propagating along +x
    # Define the velocity model in km and km/s
    thickness = np.array([10000.0, 0.0]) # units in meters
    vs = np.array([2000, 2500.0]) # units in m/s
    vp = np.array([4000, 5000.0]) # units in m/s
    rho = np.array([2465.0,2606.0]) # units in kg/m^3
    rhoAir = 1.2; # density of air
    fMin = 2.0; fMax = 8.0; df = 0.05;# units in Hz
    lambdaFrac = 1/3; # fraction
    lambdaRes = 6; #must be greater than 4
    xMaxGF = 5000.0;# maximum horizontal offset upto which displacements will be used
    zMaxGF = 5000.0; # maximum depth upto which displacements will be used
    maxRec = 500; # same value of number of receivers that qseis can handle in one go check qsglobal.h
    
    xMin = -2000.0; xMax = 2000.0; # minimum and maximum of the simulation domain in X-direction (EW)
    yMin = -2000.0; yMax = 2000.0; # maximum and minimum of the simulation domain in Y-direction (NS)
    zMin = 0.0; zMax = 4000.0; # maximum and minimum of the simulation domain in Z-direction (depth)
    domXYBounds = (xMin,xMax,yMin,yMax);
    cubeC = 250.0; rCavity = 20.0;
    cubeS = 2*rCavity;
    cubeTop = cubeC-cubeS; cubeBot = cubeC+cubeS;
    
    G = const.G
    # some other inputs for simDisp
    maxRec = 500; # same value of number of receivers that qseis can handle in one go check qsglobal.h
    tMax = 40; nSamp = 2048;
    # specify the folder where you want to write all input files, should have rw access
    fInpPath = "/data/gravwav/koley/QseisInpN/"
    #fInpPath = "/data/gravwav/koley/SALVUSOut/"
    outDispPath = "/data/gravwav/koley/OutDisp/"
    outDispRea = "/data/gravwav/koley/OutDispRea/"
    components = ['fh-2.tz', 'fh-2.tr', 'fh-2.tt', 'fz-2.tz', 'fz-2.tr']

    # splitAll is necessary so load it
    nRea = 10; # number of realizations
    splitFileName = fInpPath + 'splitAll.mat';
    splitMat = loadmat(splitFileName);
    splitAll = splitMat['splitAll'];

    R1 = 0; R2 = 2500;
    nSrc = 100;
    srcDistri = "internal"
    # frequency axis to be used by simDisp
    freqOut, idxFreq, df_native = simDisp.getFreqGrid(tMax, nSamp, fMin, fMax, df);
    nFreq = len(freqOut)

    # two depth points for scaling, displacement saved per realization
    zList = [0.0,250.0]
    
    # just make sure zMax never coincides with an actual horizontal interface
    # bug to be fixed later if needed
    thetaW = 90; phiW = 0; # theta and phi for a plane wave travelling along +x

    # generate the grid size required
    gridSize = getGridSize(thickness,vp, vs, rho, fMin, fMax, lambdaFrac, lambdaRes);
    print('GridSize = ' + str(gridSize[0]))
    
    print('grid size in each layer = ' + str(gridSize));
    layers = []

    # Create and add multiple layers
    lenThick = len(thickness);
    depths = np.cumsum(thickness[0:(lenThick-1)]);
    depths = np.insert(depths,0,0.0,axis=0);
    
    # find the index to insert zMax
    zMaxInd = np.where(depths<zMax)[0]
    
    newDepths = np.append(depths[0:(zMaxInd[-1]+1)],zMax);
    lenNewDepth = len(newDepths);
    #print(newDepths);
    for i in range(0,(lenNewDepth-1)):
        layers.append(NNGeometryN.Layer(xMin=xMin, xMax=xMax, yMin= yMin, yMax = yMax, zTop=newDepths[i], zBot=newDepths[i+1], vP=vp[i], vS=vs[i], rho=rho[i]));

    print('Total number of layers = ' + str(len(layers)));
    nLayers = len(layers)
    
    for reaNo in range(0,nRea):
        print('Doing realization = ' +str(reaNo))
        # get the source distribution per realization
        xSrc, ySrc, azSrc, phiSrc, ampSrc =  simDisp.genAmbSrc(nSrc , mode = srcDistri, R1 = R1, R2 = R2, xMin=xMin, xMax=xMax, yMin=yMin,
                                           yMax=yMax, randomPhase=True, freqDependent=True, nFreq=nFreq)

        # create and save the displacement field on surface and depth for scaling afterwards
        simDisp.getSurfDeepDispPerRea(zList, xSrc, ySrc, azSrc, phiSrc, ampSrc, xMaxGF, splitAll, fMin, fMax, outDispRea,
                    fInpPath, components, reaNo, idxFreq = idxFreq, freqOut=freqOut)
        # preallocate per realization
        IBlockTot = np.zeros((nFreq,3),dtype=np.complex128)
        IVertFaceTot  = np.zeros((nFreq,3),dtype=np.complex128)
        IHorFaceTot = np.zeros((nFreq,3),dtype=np.complex128)
        ITot = np.zeros((nFreq,3),dtype=np.complex128)
        
        for layerNo, layer in enumerate(layers):
            layer = layers[layerNo]

            layer.updateCubeInteraction(cubeTop, cubeBot)
            layer.generateBlocks(xMin, xMax, yMin, yMax, cubeC, cubeS, domXYBounds)

            #freqOut = np.linspace(fMin,fMax,60);
            #nFreq = len(freqOut)
            
            print(layer)
            blkNo = 1
            for blk in layer.blocks:
                #print(blk);
                print('Block = '+str(blkNo))
                print('Integrating volume  NN block in layer' + str(layerNo))
        
                IBlock = runVolNNComputeBlock(blk, gridSize[layerNo], freqOut, cubeC, rCavity, xSrc, ySrc, azSrc,
                                              phiSrc, ampSrc, idxFreq, fMin, fMax, outDispPath, splitAll, xMaxGF,
                                              fInpPath, components, useSimDisp=True, nCPU=4, nChunk=20000,
                                              thetaW=thetaW, phiW=phiW)

                IBlockTot = IBlockTot + IBlock;

                # now compute the surface NN from the vertical faces of the block which form the outer boundary
                for f in blk.verticalFaces:
                    if(f.isBoundary):
                        print('Integrating vertical surface NN block in layer' + str(layerNo))
                        print('vertical boundary face found')
                        # is an outer boundary
                        IVertFace = runVertSurfNNCompute(f, gridSize[layerNo], freqOut, cubeC, xSrc, ySrc, azSrc,
                                                         phiSrc, ampSrc, idxFreq, fMin, fMax, outDispPath, splitAll,
                                                         xMaxGF, fInpPath, components, nCPUDisp = nCPUDisp, nCPUNN=nCPUNN,
                                                         nChunk = 20000, useSimDisp=True, vP_for_test=blk.vP, thetaW=thetaW, phiW=phiW)
                        
                        IVertFaceTot = IVertFaceTot + IVertFace
        
                
                # now compute the surface NN from the horizontal interfaces at the very end of layers
                print('Integrating horizontal surface NN block in layer' + str(layerNo))
                IHorFace = handleHorSurfNN(layer, layers, blk, gridSize[layerNo], nLayers, layerNo, rhoAir, freqOut, 
                                           cubeC, xSrc, ySrc, azSrc, phiSrc, ampSrc, idxFreq, fMin, fMax, outDispPath,
                                           splitAll, xMaxGF, fInpPath, components, nCPU = 4, useSimDisp=True,
                                           vP_for_test=blk.vP, thetaW=thetaW, phiW=phiW, chunk_size=5000, rCavity=rCavity)
                
                IHorFaceTot = IHorFaceTot + IHorFace
                
                blkNo = blkNo+1
    
        IVertFaceTot = IVertFaceTot*G
        IHorFaceTot = IHorFaceTot*G
        IBlockTot = IBlockTot*G
        ITot = (IBlockTot-IVertFaceTot-IHorFaceTot)
        # save the NN per realization
        nnFName = 'NNFullRea' + str(reaNo) + '.npz'
        np.savez(os.path.join(outDispRea, nnFName), IVertFaceTot = IVertFaceTot, IHorFaceTot = IHorFaceTot,
                 IBlockTot = IBlockTot, ITot = ITot, freqOut=freqOut)

    dispPointAllRea, attnAllRea, freqDisp = simDisp.assembleSurfDeepDispAllRea(outDispRea, zList, freqOut, nRea)
    nnAllRea, fNN = assembleNNAllRea(outDispRea,freqOut, nRea)

    # scale NN such that the surface displacement is 1
    nnAllRea = nnAllRea/dispPointAllRea[:,:,0]
    
    figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut, dispPointAllRea[:,:,0], 'Surface', 'b', fig=None, axs=None, 
                                         quantity="ASD")
    figASD, axASD = plotUtils.plotPSDDeepSurfMulti(freqOut, dispPointAllRea[:,:,1], 'Deep', 'r', fig=figASD, axs=axASD, 
                                         quantity="ASD")

    # plot the surface to deep attenuation
    figAttn, axAttn = plotUtils.plotPSDDeepSurfMulti(freqOut, attnAllRea, 'Surf-Deep', 'b', fig=None, axs=None, 
                                         quantity="Attn")
    # load the Terziet attenuation model
    terzAttn = loadmat('/data/gravwav/koley/TerzModel/attnModel.mat');
    attnZ = terzAttn['attnZ']
    attnE = terzAttn['attnE']
    attnN = terzAttn['attnN']

    axAttn[0].plot(attnZ[:,0],attnZ[:,2],'r')
    axAttn[1].plot(attnZ[:,0],attnE[:,2],'r')
    axAttn[2].plot(attnZ[:,0],attnN[:,2],'r')

    # plot the NN acceleration ASD
    figNN, axNN = plotUtils.plotPSDDeepSurfMulti(fNN, nnAllRea, 'NNFull', 'b', fig=None, axs=None, 
                                         quantity="NN acceleration")
    
    fig,axs = plt.subplots(1, 3, figsize=(9,3))
    axs[0].plot(freqOut,np.abs(ITot[:,0]),'b', label = "Simulated")
    axs[0].plot(freqOut,np.abs(IBlockTot[:,0]),'m', label = "Simulated") 
    axs[0].plot(freqOut,np.abs(IVertFaceTot[:,0]),'r', label = "Simulated")
    axs[0].plot(freqOut,np.abs(IHorFaceTot[:,0]),'k', label = "Simulated")
    axs[0].plot(freqOut,np.ones((nFreq,))*8*np.pi/3*G*2800,'g',label = 'Theoretical')
    
    #axs[0].plot(fJ,j0R,'r', label = "Theoretical J_0(2*pi*f*d/v_R(f))");
    axs[0].set_xlim([0,8]);
    axs[0].set_xlabel('Frequency (Hz)')
    axs[0].set_ylabel('NN_x');
    axs[0].set_title('X component');
    
    axs[1].plot(freqOut,np.abs(ITot[:,1]),'b', label = "Simulated")
    axs[1].plot(freqOut,np.abs(IBlockTot[:,1]),'m', label = "Simulated") 
    axs[1].plot(freqOut,np.abs(IVertFaceTot[:,1]),'r', label = "Simulated")
    axs[1].plot(freqOut,np.abs(IHorFaceTot[:,1]),'k', label = "Simulated")
    axs[1].plot(freqOut,np.ones((nFreq,))*8*np.pi/3*G*2800,'g',label = 'Theoretical')
    
    axs[1].set_xlim([0,8]);
    axs[1].set_ylabel('NN_y');
    axs[1].set_xlabel('Frequency (Hz)')
    axs[1].set_title('Y component');
    
    axs[2].plot(freqOut,np.abs(ITot[:,2]),'b', label = "Simulated")
    axs[2].plot(freqOut,np.abs(IBlockTot[:,2]),'m', label = "Simulated") 
    axs[2].plot(freqOut,np.abs(IVertFaceTot[:,2]),'r', label = "Simulated")
    axs[2].plot(freqOut,np.abs(IHorFaceTot[:,2]),'k', label = "Simulated")
    axs[2].plot(freqOut,np.ones((nFreq,))*8*np.pi/3*G*2800,'g',label = 'Theoretical')
    
    axs[2].set_xlim([0,8]);
    axs[2].set_ylabel('NN_z');
    axs[2].set_xlabel('Frequency (Hz)')
    axs[2].set_title('Z component');

def assembleNNAllReaAllTM(outReaPath,freqOut, nRea, nTM):
    """
    to be run at the end of all realizations
    compute the rms of surface and deep displacements
    
    """
    nFreq = len(freqOut)
    
    nnTotAllRea = np.zeros((nFreq,3,nTM))
    nnVolAllRea = np.zeros((nFreq,3,nTM))
    nnSurfAllRea = np.zeros((nFreq,3,nTM))

    for reaNo in range(0,nRea):
        sName = 'NNFullRea' + str(reaNo) + '.npz'
        data = np.load(os.path.join(outReaPath,sName));
        nnTot = np.abs(data["ITot"])
        nnVol = np.abs(data["IBlockTot"])
        nnSurf = np.abs(data["IVertFaceTot"]+data['IHorFaceTot'])
        nnTotAllRea = nnTotAllRea + nnTot**2
        nnVolAllRea = nnVolAllRea + nnVol**2
        nnSurfAllRea = nnSurfAllRea + nnSurf**2

    nnTotAllRea = np.sqrt(nnTotAllRea/nRea)
    nnVolAllRea = np.sqrt(nnVolAllRea/nRea)
    nnSurfAllRea = np.sqrt(nnSurfAllRea/nRea)

    freqOut = data["freqOut"]
    
    return nnTotAllRea, nnVolAllRea, nnSurfAllRea, freqOut  

def assembleNNAllRea(outReaPath,freqOut, nRea):
    """
    to be run at the end of all realizations
    compute the rms of surface and deep displacements
    
    """
    nFreq = len(freqOut)
    
    nnTotAllRea = np.zeros((nFreq,3))
    nnVolAllRea = np.zeros((nFreq,3))
    nnSurfAllRea = np.zeros((nFreq,3))

    for reaNo in range(0,nRea):
        sName = 'NNFullRea' + str(reaNo) + '.npz'
        data = np.load(os.path.join(outReaPath,sName));
        nnTot = np.abs(data["ITot"])
        nnVol = np.abs(data["IBlockTot"])
        nnSurf = np.abs(data["IVertFaceTot"]+data['IHorFaceTot'])
        nnTotAllRea = nnTotAllRea + nnTot**2
        nnVolAllRea = nnVolAllRea + nnVol**2
        nnSurfAllRea = nnSurfAllRea + nnSurf**2

    nnTotAllRea = np.sqrt(nnTotAllRea/nRea)
    nnVolAllRea = np.sqrt(nnVolAllRea/nRea)
    nnSurfAllRea = np.sqrt(nnSurfAllRea/nRea)

    freqOut = data["freqOut"]
    
    return nnTotAllRea, nnVolAllRea, nnSurfAllRea, freqOut  

def handleHorSurfNN(layer, layers, block, gridSize, nLayers, i, rhoAir, freqOut,
                    allCavities, itmAC, xSrc, ySrc, azSrc, srcMeta,
                    idxFreq, fMin, fMax, outDispPath, splitAll, xMaxGF, fInpPath, components, vR, nCPUDisp=4,
                    nCPUNN=4, useSimDisp=False, ifRay = 0, ifBody = 0, vP_for_test=4000, vS_for_test = 2000,
                    metaBody = None, chunk_size=20000, computeStrategy="threading_shared",
                    saveHV=False,reduceNN=True,fullSpace=False):
    
    nFreq = len(freqOut)
    itmAC = np.atleast_2d(itmAC)
    nTM = len(itmAC)
    IHorFaceTot = np.zeros((nFreq,3,nTM),dtype=np.complex128)
    internalFlag = 0
    for f in block.horizontalFaces:
        if f.isBoundary:
            print('Horizontal face found')
            z_face = f.position
            
            # --- Determine which kind of face this is ---
            if np.isclose(z_face, layer.zTop):
                # Top of current layer
                if i == 0:
                    # Topmost layer → surface (air)
                    
                    if not fullSpace:
                        print('Applying free surface condition, using rhoAir for the topmost layer')
                        rho_eff = rhoAir
                    else:
                        print('Using full space, using layer rho for the topmost layer')
                        rho_eff = layer.rho
                    # this needs to be switched on for free surface
                    #rho_eff = layer.rho
                    internalFlag = 0
                else:
                    internalFlag = 1
                    continue  # skip internal top faces
                    
            elif np.isclose(z_face, layer.zBot):
                # Bottom of current layer
                if i < nLayers - 1:
                    #print('I am in midlayer situation, taking bottom surface of layer')
                    # Internal interface → difference term handled implicitly
                    rho_eff = layers[i + 1].rho - layer.rho
                    internalFlag = 1
                else:
                    #print('I am at the very bottom')
                    # Bottommost layer → outer boundary
                    rho_eff = layer.rho
                    internalFlag = 0
            else:
                continue  # shouldn't happen

            # --- Compute the horizontal surface NN for the outer bottom ot top face ---
            #print('rho_eff = '+str(rho_eff))
            if(not internalFlag):
                print('Actually computing NN for the horizontal surface')
                IHorFace = runHorSurfNNCompute(f, block.spaceType, gridSize, freqOut, allCavities,
                                               itmAC, rho_eff, xSrc, ySrc, azSrc, srcMeta, idxFreq, fMin, fMax, outDispPath, splitAll,
                                               xMaxGF, fInpPath, components, vR, nCPUDisp=nCPUDisp, nCPUNN=nCPUNN,
                                               useSimDisp=useSimDisp, ifRay = ifRay, ifBody = ifBody, vP_for_test = vP_for_test,
                                               vS_for_test = vS_for_test, metaBody = metaBody, chunk_size=chunk_size,
                                               computeStrategy=computeStrategy, saveHV=saveHV,reduceNN=reduceNN)
            
                #
                IHorFaceTot = IHorFaceTot + IHorFace
    
    return IHorFaceTot


def runHorSurfNNCompute(face, spaceType, gridSize, freqOut, allCavities,
                        itmAC, rho, xSrc, ySrc, azSrc, srcMeta, idxFreq, 
                        fMin, fMax, outDispPath, splitAllNew, xMaxGF, fInpPath, components, vR, nCPUDisp=4, nCPUNN=4,
                        useSimDisp=False, ifRay = 0, ifBody = 0, vP_for_test = 4000, vS_for_test = 2000, metaBody = None,
                        chunk_size=20000, computeStrategy="threading_shared",saveHV=False, reduceNN=True):
    """
    Compute NN from ONE horizontal interface (z = const), parallel over (x,y).

    Assumptions
    -----------
    - displacements for ALL (x,y) on that plane can be obtained in one go
      either from simDisp (recommended) or we synthesize a plane wave.
    """
    zFace = face.position # zFace is positive
    zFaceNeg = -zFace
    
    if(spaceType == 'lgwt'):
        #print('I am here!')
        totX = (face.xLim[1]-face.xLim[0])
        totY = (face.yLim[1]-face.yLim[0])
        
        nX = max(3,int(np.round(totX/gridSize)))
        nY = max(3,int(np.round(totY/gridSize)))
        #print('nX = '+str(nX) + 'nY = ' + str(nY))    
        gridX, gridXW = lgwtPoints(nX, face.xLim[0], face.xLim[1])
        gridY, gridYW = lgwtPoints(nY, face.yLim[0], face.yLim[1])
        #zlgwt = -np.abs(zlgwt)

        X, Y = np.meshgrid(gridX, gridY, indexing="ij")
        WX, WY = np.meshgrid(gridXW, gridYW, indexing="ij")

        X_flat = X.ravel()
        Y_flat = Y.ravel()
        dS = (WX * WY).ravel()    # area weights
    else:
        print('Uniform spacing')
        # the case of the block enclosing the cavity, no lgwt sampliing here
        # adaptive uniform sampling to be done here
        
        xy_result = generate_adaptive_xy_pointcloud(cavities=allCavities,x_min=face.xLim[0],x_max=face.xLim[1],
                                                    y_min=face.yLim[0], y_max=face.yLim[1], dx_inside=1.0,
                                                    dy_inside=1.0, dx_near=1.0, dy_near=1.0, dx_mid=2.0,
                                                    dy_mid=2.0, dx_far=5.0, dy_far=5.0, near_frac=1.0/3.0, mid_frac=2.0/3.0,
                                                    near_cap=20.0, mid_cap=20.0, gap_tol=10.0)
        
        X_flat = xy_result["gridX_flat"]
        Y_flat = xy_result["gridY_flat"]
        dS = xy_result["gridW_flat"]
    
    zArr = np.full_like(X_flat, zFaceNeg, dtype=float)

    # remove cavity footprint if this horizontal plane cuts it
    if spaceType == 'uniform':

        mask = mask_points_outside_all_cavities(X_flat, Y_flat, zFaceNeg, allCavities)
        X_flat = X_flat[mask]
        Y_flat = Y_flat[mask]
        zArr   = zArr[mask]
        dS     = dS[mask]

    # final coords
    rVec = np.column_stack((X_flat, Y_flat, zArr))   # (nPts, 3)
    #nPts = rVec.shape[0]
    #nFreq = len(freqOut)

    # get displacements
    if useSimDisp:
        # expected: (nFreq, nPts, 3)
        # note that zArr negative here, careful
        zUse = -zFaceNeg
        nnMeta = {"rVec": rVec,"dS": dS,"n_hat": face.normVector}
        if(reduceNN):
            I_total, _, _, _ = simDisp.computeFullDispF(zUse,rVec[:, 0], rVec[:, 1], itmAC, xSrc, ySrc, azSrc, srcMeta,
                                                       idxFreq, freqOut, outDispPath,splitAllNew,xMaxGF,
                                                       fInpPath,nCPU=nCPUDisp, nChunk=chunk_size,computeStrategy=computeStrategy,
                                                       saveHV=saveHV,reduceNN=reduceNN,nnMode="hor_surface",nnMeta=nnMeta)
        else:
            dispData, outDirUse, fUse, shmPaths = simDisp.computeFullDispF(zUse,rVec[:, 0], rVec[:, 1], itmAC, xSrc, ySrc, azSrc, srcMeta,
                                                       idxFreq, freqOut, outDispPath,splitAllNew,xMaxGF,
                                                       fInpPath,nCPU=nCPUDisp, nChunk=chunk_size,computeStrategy=computeStrategy,
                                                       saveHV=saveHV,reduceNN=reduceNN,nnMode="hor_surface",nnMeta=nnMeta)
    else:
        if(ifRay):
            # pure Rayleigh wavefield for testing
            # note zFaceNeg is negative, so pass that
            phiSrc = srcMeta["phiV"]; ampSrc = srcMeta["ampV"]
            nSrc = len(azSrc)
            dispData = simDisp.computeRayleighDisp(zFaceNeg, rVec[:,0], rVec[:,1], nSrc, freqOut, azSrc,
                                                phiSrc, ampSrc, vR, vP_for_test, vS_for_test) 
            shmPaths = None
            n_hat = face.normVector
            I_total = compute_surface_parallel(dispData, rVec, itmAC, dS, n_hat, n_jobs=nCPUNN, chunk_size=chunk_size)

        elif(ifBody):
            nSrc = len(azSrc)
            #dispData = simDisp.computeFullBodyDisp(zFaceNeg, rVec[:,0], rVec[:,1], nSrc, freqOut, vP_for_test,
            #                                       vS_for_test, metaBody)
            dispData = simDisp.computeFullBodyDisp_parallel(zFaceNeg, rVec[:,0], rVec[:,1], nSrc, freqOut,
                                                            vP_for_test, vS_for_test, metaBody,n_workers=nCPUDisp)
            shmPaths = None
            n_hat = face.normVector
            I_total = compute_surface_parallel(dispData, rVec, itmAC, dS, n_hat, n_jobs=nCPUNN, chunk_size=chunk_size)
    
    if(not reduceNN):
        n_hat = face.normVector
        #print(n_hat)
        # run parallel surface integration
        #print("I am computing surface parallel")
        I_total = compute_surface_parallel(dispData, rVec, itmAC, dS, n_hat, n_jobs=nCPUNN, chunk_size=chunk_size)

    I_total = I_total*rho

    # delete the displacements stored
    if(not reduceNN):
        del dispData
        if shmPaths is not None:
            simDisp.cleanup_sharedmem(shmPaths)    
    return I_total
    
def compute_surface_parallel(dispData, rVec, itmAC, dS, n_hat, n_jobs=4, chunk_size=20000):
    """
    Parallel surface NN over a horizontal face.

    Parameters
    ----------
    dispData : (nFreq, nPts, 3)
    rVec     : (nPts, 3)
    dS       : (nPts,)
    n_hat    : (3,)
    """
    nFreq, nPts, _ = dispData.shape
    itmAC = np.atleast_2d(itmAC)
    nTM = len(itmAC[:,0])
    I_total = np.zeros((nFreq, 3, nTM), dtype=np.complex128)

    def process_chunk(start, end):
        u_chunk = dispData[:, start:end, :]
        r_chunk = rVec[start:end, :]
        dS_chunk = dS[start:end]
        return computeVertSurfNN_multiTM(u_chunk, r_chunk, n_hat, dS_chunk, itmAC)

    starts = list(range(0, nPts, chunk_size))

    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(process_chunk)(s, min(s + chunk_size, nPts))
        for s in starts
    )

    for i, res in enumerate(results):
        #print(f"[Main] Aggregating result from chunk {i+1}/{len(results)}")
        I_total += res
    
    return I_total
    
def runVertSurfNNCompute(face, gridSize, freqOut, itmAC, xSrc, ySrc, azSrc, srcMeta, idxFreq, fMin, fMax,
                         outDispPath, splitAllNew, xMaxGF, fInpPath, components, vR, nCPUDisp=4, nChunk=20000,
                         useSimDisp=False, ifRay = 0, ifBody = 0, vP_for_test=None, vS_for_test = None,
                         metaBody=None, computeStrategy="threading_shared",saveHV=False, reduceNN=True):
    """
    Surface NN for ONE vertical face, looping over depth.
    face.axis == 'x'  -> fixed x, vary y,z
    face.axis == 'y'  -> fixed y, vary x,z
    """
    nFreq = len(freqOut)
    itmAC = np.atleast_2d(itmAC)
    nTM = len(itmAC[:,0])
    I_total = np.zeros((nFreq, 3, nTM), dtype=np.complex128)

    if(face.axis=='x'):
        totY = face.yLim[1]-face.yLim[0]
        totZ = face.zLim[1]-face.zLim[0]
                    
        # at least three points in depth for integration
        nY = max(3,int(np.round(totY/gridSize)))
        nZ = max(3,int(np.round(totZ/gridSize)))
            
        faceCoords, faceWeightlgwt = lgwtPoints(nY, face.yLim[0], face.yLim[1])
        zlgwt, zWeightlgwt = lgwtPoints(nZ, face.zLim[0], face.zLim[1])
                
    if(face.axis=='y'):
        totX = face.xLim[1]-face.xLim[0]
        totZ = face.zLim[1]-face.zLim[0]
                    
        # at least three points in depth for integration
        nX = max(3,int(np.round(totX/gridSize)))
        nZ = max(3,int(np.round(totZ/gridSize)))
            
        faceCoords, faceWeightlgwt = lgwtPoints(nX, face.xLim[0], face.xLim[1])
        zlgwt, zWeightlgwt = lgwtPoints(nZ, face.zLim[0], face.zLim[1])

    # precompute dS along the face direction (x or y)
    # this is 1D: area element along-face
    dS_line = faceWeightlgwt  # shape (nPts,)

    # switch sign here, because NN geometry returns depth as positive
    zlgwt = -np.abs(zlgwt)
    
    for iz, (zNow, wz) in enumerate(zip(zlgwt, zWeightlgwt)):
        # build coords for THIS depth only
        if face.axis == "x":
            # x fixed, y varies
            xArr = np.full_like(faceCoords, face.position, dtype=float)
            yArr = faceCoords
            zArr = np.full_like(faceCoords, zNow, dtype=float)
        elif face.axis == "y":
            # y fixed, x varies
            xArr = faceCoords
            yArr = np.full_like(faceCoords, face.position, dtype=float)
            zArr = np.full_like(faceCoords, zNow, dtype=float)
        else:
            raise ValueError("face.axis must be 'x' or 'y' for vertical faces")

        rVec = np.column_stack((xArr, yArr, zArr))  # (nPts, 3)

        # full surface element for this depth slice
        # shape (nPts,)
        dS = dS_line * wz

        # get displacement at THIS depth
        if useSimDisp:
            # be carefull here zNow is negative
            zUse = -zNow
            nnMeta = {"rVec": rVec,"dS": dS,"n_hat": face.normVector}
            if(reduceNN):
                I_slice, _, _, _ = simDisp.computeFullDispF(zUse,rVec[:, 0], rVec[:, 1], itmAC, xSrc, ySrc,
                                                            azSrc, srcMeta, idxFreq, freqOut, outDispPath,
                                                            splitAllNew, xMaxGF, fInpPath, nCPU=nCPUDisp, nChunk=nChunk,
                                                            computeStrategy=computeStrategy, saveHV=False,reduceNN=True,
                                                            nnMode="vert_surface",nnMeta=nnMeta)
            else:
                disp_f, outDirUse, fUse, shmPaths = simDisp.computeFullDispF(zUse,rVec[:, 0], rVec[:, 1], itmAC, xSrc, ySrc,
                                                            azSrc, srcMeta, idxFreq, freqOut, outDispPath,
                                                            splitAllNew, xMaxGF, fInpPath, nCPU=nCPUDisp, nChunk=nChunk,
                                                            computeStrategy=computeStrategy, saveHV=saveHV,reduceNN=True,
                                                            nnMode="vert_surface",nnMeta=nnMeta)
        else:
            # plane wave test
            if(ifRay):
                # pure Rayleigh wavefield for testing
                # note zNow is negative, so pass that
                phiSrc = srcMeta["phiV"]; ampSrc = srcMeta["ampV"]
                nSrc = len(azSrc)
                disp_f = simDisp.computeRayleighDisp(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut, azSrc,
                                                phiSrc, ampSrc, vR, vP_for_test, vS_for_test)
                shmPaths = None
                I_slice = computeVertSurfNN_multiTM(disp_fxn=disp_f, rVec=rVec, n_hat=face.normVector, dS=dS, itmAC=itmAC)
            elif(ifBody):
                nSrc = len(azSrc)
                #disp_f = simDisp.computeFullBodyDisp(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut, vP_for_test,
                #                                     vS_for_test, metaBody)
                disp_f = simDisp.computeFullBodyDisp_parallel(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut, vP_for_test,
                                                              vS_for_test, metaBody,n_workers=nCPUDisp)
                shmPaths = None
                I_slice = computeVertSurfNN_multiTM(disp_fxn=disp_f, rVec=rVec, n_hat=face.normVector, dS=dS, itmAC = itmAC)
        # now do the surface kernel for ALL freqs
        if(not reduceNN):
            I_slice = computeVertSurfNN_multiTM(disp_fxn=disp_f, rVec=rVec, n_hat=face.normVector, dS=dS, itmAC=itmAC)
        # add with density
        I_total += face.rhoOut * I_slice

        if(not reduceNN):
            del disp_f
            if shmPaths is not None:
                simDisp.cleanup_sharedmem(shmPaths)

    return I_total

def computeVertSurfNN(disp_fxn, rVec, zCav, n_hat, dS, itmAC):
    """
    disp_fxn : (nFreq, nPts, 3)  complex
        displacement on the surface
    rVec : (nPts, 3)  float
        coordinates of surface points (in *global* coords)
    zCav : float
        cavity depth, BUT we subtract it here: r -> (r - r0)
        so pass +cubeC and we do r_z - cubeC
    n_hat : (3,) float
        outward normal of the surface
    dS : (nPts,) float
        surface area weights (per point)

    returns
    -------
    I_surf : (nFreq, 3) complex
    """
    nFreq, nPts, _ = disp_fxn.shape

    # shift to cavity
    rRel = rVec.copy()
    # old version
    #rRel[:, 2] = rRel[:, 2] - zCav  # z - z_cav

    rRel[:, 0] = rRel[:, 0] -itmAC[0]
    rRel[:, 1] = rRel[:, 1] -itmAC[1]
    rRel[:, 2] = rRel[:, 2] -itmAC[2]
    
    # |r| and r/|r|^3
    rnorm = np.linalg.norm(rRel, axis=1)          # (nPts,)
    rnorm3 = rnorm**3                             # (nPts,)
    #r_over_r3 = (rRel / rnorm3[:, None])          # (nPts, 3)

    # (u · n) for each freq, point
    # disp_fxn: (nFreq, nPts, 3)
    u_dot_n = np.einsum("fpk,k->fp", disp_fxn, n_hat)      # (nFreq, nPts)

    # scalar part: (u·n)/r^3
    scalar = u_dot_n / rnorm3[None, :]                     # (nFreq, nPts)

    # now multiply by r-vector → (nFreq, nPts, 3)
    I_pt = scalar[:, :, None] * rRel[None, :, :]           # (nFreq, nPts, 3)

    # apply surface weight dS
    I_pt *= dS[None, :, None]                              # (nFreq, nPts, 3)

    # integrate over surface points
    I_surf = I_pt.sum(axis=1)                              # (nFreq, 3)

    return I_surf

def computeVertSurfNN_multiTM(disp_fxn, rVec, n_hat, dS, itmAC):
    """
    Compute vertical/side surface NN contribution for one or more test masses.

    Parameters
    ----------
    disp_fxn : ndarray, shape (nFreq, nPts, 3)
        Displacement on the surface.

    rVec : ndarray, shape (nPts, 3)
        Coordinates of surface points in global/code coordinates.

    zCav : float
        Kept for backward compatibility. Not used here if itmAC gives full coordinates.

    n_hat : ndarray, shape (3,)
        Outward normal of the surface.

    dS : ndarray, shape (nPts,)
        Surface area weights per point.

    itmAC : ndarray, shape (3,) or (nTM, 3)
        Test-mass coordinates in the same coordinate system as rVec.

    eps : float
        Small cutoff to avoid division by zero if a surface point lies exactly on a test mass.

    Returns
    -------
    I_surf : ndarray, shape (nFreq, 3, nTM)
        Surface NN contribution for each test mass.
    """

    disp_fxn = np.asarray(disp_fxn)
    rVec = np.asarray(rVec, dtype=float)
    n_hat = np.asarray(n_hat, dtype=float)
    dS = np.asarray(dS, dtype=float)

    itmAC = np.asarray(itmAC, dtype=float)

    # Allow old single-test-mass input shape (3,)
    if itmAC.ndim == 1:
        itmAC = itmAC[None, :]   # (1, 3)

    #nFreq, nPts, _ = disp_fxn.shape
    #nTM = itmAC.shape[0]

    # Relative vectors:
    # rRel[m, p, :] = rVec[p, :] - itmAC[m, :]
    rRel = rVec[None, :, :] - itmAC[:, None, :]      # (nTM, nPts, 3)

    rnorm = np.linalg.norm(rRel, axis=2)             # (nTM, nPts)

    # Avoid divide-by-zero. Surface points should not coincide with test masses.

    rnorm3 = rnorm**3                                # (nTM, nPts)

    # u dot n is independent of test mass:
    # disp_fxn: (nFreq, nPts, 3)
    # n_hat:    (3,)
    u_dot_n = np.einsum("fpk,k->fp", disp_fxn, n_hat)   # (nFreq, nPts)

    # Broadcast to (nFreq, nTM, nPts, 1)
    scalar = (
        u_dot_n[:, None, :, None] /
        rnorm3[None, :, :, None]
    )                                                  # (nFreq, nTM, nPts, 1)

    I_pt = scalar * rRel[None, :, :, :]                # (nFreq, nTM, nPts, 3)

    # Apply surface weights
    I_pt *= dS[None, None, :, None]                    # (nFreq, nTM, nPts, 3)

    # Sum over surface points
    I_sum = I_pt.sum(axis=2)                           # (nFreq, nTM, 3)

    # Return as (nFreq, 3, nTM), same as volume multiTM
    I_surf = np.transpose(I_sum, (0, 2, 1))

    return I_surf
  
def runVolNNComputeBlock(block, gridSize, freqOut, allCavities, itmAC, xSrc, ySrc, azSrc, srcMeta,
                         idxFreq, fMin, fMax, outDispPath, splitAllNew, xMaxGF, fInpPath, components, vR, 
                         useSimDisp=False, ifRay = 0, ifBody = 0, nCPUDisp=4, nCPUNN=4, nChunk=20000,
                         metaBody=None, computeStrategy = 'threading_shared', saveHV = False, reduceNN=True):
    """
    Perform NN computation for one block.
    
    Parameters
    ----------
    block : NNGeometryN.Block
        Current block object (contains x/y/z bounds, rho, vP, etc.)
    gridX, gridY : 1D arrays
        Quadrature or uniform grid coordinates in X and Y.
    gridXW, gridYW : 1D arrays
        Corresponding quadrature weights for X and Y.
    zlgwt, zWeightlgwt : 1D arrays
        Depth coordinates and weights (positive down).
    freqOut : ndarray
        Frequencies at which displacement is evaluated.
    cubeC : float
        Depth of cavity center (positive, in meters).
    rCavity : float
        Cavity radius (in meters).
    useSimDisp : bool
        If True, read displacement from file. Otherwise, generate synthetic plane wave.
    simDispFile : str
        Path to displacement file (if useSimDisp=True).
    """

    nFreq = len(freqOut)
    # new version
    itmAC = np.atleast_2d(itmAC)
    nTM = len(itmAC[:,0])
    I_total = np.zeros((nFreq, 3, nTM), dtype=np.complex128)
    # old version
    #I_total = np.zeros((nFreq, 3), dtype=np.complex128)

    if(block.spaceType == 'lgwt'):
        #print('I am here!')
        totX = (block.xMax-block.xMin)
        totY = (block.yMax-block.yMin)
        totZ = (block.zMax-block.zMin)
            
        nX = max(3,int(np.round(totX/gridSize)))
        nY = max(3,int(np.round(totY/gridSize)))
        nZ = max(3,int(np.round(totZ/gridSize))) # at least three points in depth for integration
            
        gridX, gridXW = lgwtPoints(nX, block.xMin, block.xMax)
        gridY, gridYW = lgwtPoints(nY, block.yMin, block.yMax)
        zlgwt, zWeightlgwt = lgwtPoints(nZ, block.zMin, block.zMax)
        # new update
        gridXMat, gridYMat = np.meshgrid(gridX, gridY, indexing='ij')
        gridXWMat, gridYWMat = np.meshgrid(gridXW, gridYW, indexing='ij')
        gridX_flat = gridXMat.ravel()
        gridY_flat = gridYMat.ravel()
        gridW_flat = (gridXWMat * gridYWMat).ravel()
        #zlgwt = -np.abs(zlgwt)
    else:
        print('Uniform spacing')
        # the case of the block enclosing the cavity, no lgwt sampliing here
        # adpative uniform sampling to be done here
        # new version, the az calculation still needs to happen
        cubeTopAll = np.zeros((len(allCavities),))
        cubeBotAll = np.zeros((len(allCavities),))
        for cavNo, cav in enumerate(allCavities):
            if(cav["shape"]=="cuboid"):
                hBuf = cav["height"]/2  #aangepast
                cubeTopAll[cavNo] = cav["zC"] - hBuf; cubeBotAll[cavNo] = cav["zC"] + hBuf
            elif(cav["shape"] == 'sphere'):
                hBuf = cav["radius"]
                cubeTopAll[cavNo] = cav["zC"] -hBuf ; cubeBotAll[cavNo] = cav["zC"] + hBuf
            else:
                ValueError('Unsupported cavity type')

        # compute max and min to get global cubeTop and cubeBot
        cav_zmin = min(cubeTopAll); cav_zmax = max(cubeBotAll)
        
        xy_result = generate_adaptive_xy_pointcloud(cavities=allCavities,x_min=block.xMin,x_max=block.xMax,
                                                    y_min=block.yMin, y_max=block.yMax, dx_inside=1.0,
                                                    dy_inside=1.0, dx_near=1.0, dy_near=1.0, dx_mid=2.0,
                                                    dy_mid=2.0, dx_far=5.0, dy_far=5.0, near_frac=1.0/3.0, mid_frac=2.0/3.0,
                                                    near_cap=20.0, mid_cap=20.0, gap_tol=10.0)
        
        zlgwt, zWeightlgwt = make_adaptive_z_axis(block.zMin,block.zMax, cav_zmin, cav_zmax, dz_inside=1.0, dz_near=1.0, dz_mid=2.0,
                         dz_far=5.0, near_width=20.0, mid_width=20.0)
        print(zlgwt)
        gridX_flat = xy_result["gridX_flat"]
        gridY_flat = xy_result["gridY_flat"]
        gridW_flat = xy_result["gridW_flat"]
        
        print("Total points =", len(gridX_flat))
        #A_grid = np.sum(gridW_flat)
        #A_true = ((block.xMax - block.xMin)*(block.yMax - block.yMin))
        #print('check discretization')
        #print(A_grid,A_true,A_grid - A_true)

    # --- Flip z to negative (depths below surface) ---
    zlgwt = -np.abs(zlgwt)
    
    # --- Loop over depth slices ---
    for d in range(len(zlgwt)):
        zNow = zlgwt[d]
        #print("Depth = " + str(zNow));
        zWeight = zWeightlgwt[d]
        zWNow = zWeight * gridW_flat

        # full coordinate vectors
        rVec = np.column_stack((gridX_flat, gridY_flat, np.full_like(gridX_flat, zNow)))
        dV = zWNow  # base differential volume weights

        # ============================================================
        # CAVITY REMOVAL
        # ============================================================
        # Apply only to the block containing the cavity (uniform grid)

        #Jasper commented out rDist, mask
        #replaced by: if cavitytype == sphere, cuboid

        if block.spaceType == 'uniform':
            # new version
            mask = mask_points_outside_all_cavities(gridX_flat, gridY_flat, zNow, allCavities)
            # apply mask to geometry and weights
            rInit = len(rVec)
            rVec = rVec[mask, :]
            rFin = len(rVec)
            if(rInit!=rFin):
                print('Cavity removal happening!')
            dV = dV[mask]
            if len(rVec) == 0:
                print(f"[WARNING] All points removed at z={zNow:.2f} in cavity block — skipping this slice.")
                continue

        # --- Generate or read displacements ---
        if useSimDisp:
            # TODO: memory-mapped read from file (later)
            # be careful here zNow is negative
            zUse = -zNow
            nnMeta = {"rVec": rVec, "dV": dV, "rho": block.rho}
            #print('Started disp computation at depth ' + str(zUse))
            if(reduceNN):
                IDepthSlice,_,_,_ = simDisp.computeFullDispF(zUse, rVec[:,0], rVec[:,1], itmAC, xSrc, ySrc, azSrc, srcMeta,
                    idxFreq, freqOut, outDispPath, splitAllNew, xMaxGF, fInpPath, nCPU=nCPUDisp, nChunk=nChunk,
                    computeStrategy=computeStrategy,saveHV=saveHV, reduceNN=reduceNN, nnMode="volume", nnMeta=nnMeta)    
                I_total += IDepthSlice
            else:
                dispData, outDirUse, fUse, shmPaths = simDisp.computeFullDispF(zUse, rVec[:,0], rVec[:,1], itmAC, xSrc, ySrc, azSrc, srcMeta,
                    idxFreq, freqOut, outDispPath, splitAllNew, xMaxGF, fInpPath, nCPU=nCPUDisp, nChunk=nChunk,
                    computeStrategy=computeStrategy,saveHV=saveHV, reduceNN=reduceNN, nnMode="volume", nnMeta=nnMeta)
            #print('Ended disp computation at depth ' + str(zUse))
        
        else:
            if(ifRay):
                # pure Rayleigh wavefield for testing
                # note zNow is negative, so pass that
                phiSrc = srcMeta["phiV"]; ampSrc = srcMeta["ampV"]
                nSrc = len(azSrc)
                dispData = simDisp.computeRayleighDisp(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut, azSrc,
                                                phiSrc, ampSrc, vR, block.vP, block.vS)
                I_total += compute_volume_parallel(dispData, rVec, itmAC, block.rho, dV, n_jobs=nCPUNN)
                shmPaths = None
            elif(ifBody):
                nSrc = len(azSrc)
                
                dispData = simDisp.computeFullBodyDisp_parallel(zNow, rVec[:,0], rVec[:,1], nSrc, freqOut,
                                                                block.vP, block.vS, metaBody,n_workers=nCPUDisp)
                I_total += compute_volume_parallel(dispData, rVec, itmAC, block.rho, dV, n_jobs=nCPUNN)
                shmPaths = None
            else:
                raise ValueError("ifRay and ifBody can't be set to 0 when useQseis=False")

        # --- Volume integral for this depth slice ---
        if(not reduceNN):
            I_total += compute_volume_parallel(dispData, rVec, itmAC, block.rho, dV, n_jobs=nCPUNN)
            #print("grid points = " + str(len(rVec)));
            del dispData
            if shmPaths is not None:
                simDisp.cleanup_sharedmem(shmPaths)

    return I_total
 
def compute_volume_parallel(dispData, rVec, itmAC, rho, dV, n_jobs=4, chunk_size=20000):
    """
    Parallel computation of volume NN contribution using joblib.
    dispData: (nFreq, nGrid, nComp)
    rVec: (nGrid, 3)
    dV:   (nGrid,)
    """

    nFreq, nGrid, _ = dispData.shape
    
    # new version
    itmAC = np.atleast_2d(itmAC)
    nTM = len(itmAC[:,0]) # number of test-masses

    # new version
    I_total = np.zeros((nFreq, 3, nTM), dtype=np.complex128)

    # old version
    #I_total = np.zeros((nFreq, 3), dtype=np.complex128)

    def process_chunk(start, end):
        """Worker task for a subset of grid points."""
        pid = os.getpid()
        npts = end - start
        #print(f"[Worker {pid}] Processing chunk {start}:{end} ({npts} points)")

        u_chunk = dispData[:, start:end, :]
        r_chunk = rVec[start:end, :]
        dV_chunk = dV[start:end]

        # old version
        #result = getVolNN_chunk(u_chunk, r_chunk, zCav, rho, dV_chunk, itmAC)
        # updated
        result = getVolNN_chunk_multiTM(u_chunk, r_chunk, rho, dV_chunk, itmAC)
        #print(f"[Worker {pid}] Finished chunk {start}:{end}")
        return result

    # Build all chunk boundaries
    chunk_starts = list(range(0, nGrid, chunk_size))
    #print(f"Total {len(chunk_starts)} chunks for {nGrid} grid points, "
    #      f"running with {n_jobs} workers...")

    # Dispatch chunks in parallel
    results = Parallel(n_jobs=n_jobs, backend='loky', verbose=0)(
        delayed(process_chunk)(start, min(start + chunk_size, nGrid))
        for start in chunk_starts
    )

    # Aggregate results
    for i, res in enumerate(results):
        #print(f"[Main] Aggregating result from chunk {i+1}/{len(results)}")
        I_total += res

    #print("[Main] All chunks complete. Total integrated volume field ready.")
    return I_total


def getVolNN_chunk(u_chunk, r_chunk, zCav, rho, dV, itmAC):
    """
    Compute the volume NN contribution for a spatial chunk, across all frequencies.
    u_chunk : (nFreq, nChunk, 3)
    r_chunk : (nChunk, 3)
    zCav    : cavity depth (m)
    rho     : density (kg/m³)
    dV      : (nChunk,) volume weights for integration
    """
    rRel = r_chunk.copy()
    rRel[:,0] -= itmAC[0]
    rRel[:,1] -= itmAC[1]
    rRel[:,2] -= itmAC[2]

    #print('zCav inside getVolNNChunk = '+ str(zCav))
    #print('itmAC[2] inside getVolNNChunk = ' + str(itmAC[2]))

    #old version
    #rRel[:, 2] -= zCav
    rDist = np.linalg.norm(rRel, axis=1)
    rCap = rRel / rDist[:, np.newaxis]
    rDist3 = rDist**3

    # broadcast along frequency
    rCap = rCap[np.newaxis, :, :]         # (1, nChunk, 3)
    rDist3 = rDist3[np.newaxis, :, np.newaxis]
    dV = dV[np.newaxis, :, np.newaxis]

    term1 = u_chunk / rDist3
    dot_ru = np.sum(rCap * u_chunk, axis=2, keepdims=True)
    term2 = 3.0 * rCap * dot_ru / rDist3

    I_chunk = term1-term2
    I_local = rho * (term1 - term2) * dV   # (nFreq, nChunk, 3)
    I_chunk = np.sum(I_local, axis=1)      # (nFreq, 3)
    return I_chunk

def getVolNN_chunk_multiTM(u_chunk, r_chunk, rho, dV, itmAC):
    """
    Compute the volume NN contribution for a spatial chunk,
    across all frequencies and one or more test masses.

    Parameters
    ----------
    u_chunk : ndarray, shape (nFreq, nChunk, 3)
        Displacement field at integration points.

    r_chunk : ndarray, shape (nChunk, 3)
        Integration point coordinates.

    zCav : float
        Kept for backward compatibility. Not used here if itmAC gives full coordinates.

    rho : float
        Density [kg/m^3].

    dV : ndarray, shape (nChunk,)
        Volume weights.

    itmAC : ndarray, shape (3,) or (nTM, 3)
        Test-mass coordinates.

    eps : float
        Small cutoff to avoid division by zero if a point lies exactly on a test mass.

    Returns
    -------
    I_chunk : ndarray, shape (nFreq, 3, nTM)
        Volume NN contribution for each test mass.
    """

    u_chunk = np.asarray(u_chunk)
    r_chunk = np.asarray(r_chunk, dtype=float)
    dV = np.asarray(dV, dtype=float)

    itmAC = np.asarray(itmAC, dtype=float)

    # Allow old single-test-mass input shape (3,)
    if itmAC.ndim == 1:
        itmAC = itmAC[None, :]   # (1, 3)

    nFreq = u_chunk.shape[0]
    nTM = itmAC.shape[0]

    # Relative vectors from each test mass to each integration point:
    # rRel[m, q, :] = r_chunk[q, :] - itmAC[m, :]
    rRel = r_chunk[None, :, :] - itmAC[:, None, :]       # (nTM, nChunk, 3)

    rDist = np.linalg.norm(rRel, axis=2)                 # (nTM, nChunk)

    # Avoid division by zero. Points exactly at test masses should normally
    # already be removed by the cavity mask.
    #bad = rDist < eps
    #if np.any(bad):
    #    rDist = rDist.copy()
    #    rDist[bad] = np.inf

    rCap = rRel / rDist[:, :, None]                      # (nTM, nChunk, 3)
    rDist3 = rDist**3                                    # (nTM, nChunk)

    # Broadcast:
    # u_chunk: (nFreq, nChunk, 3)
    # rCap:    (nTM,   nChunk, 3)
    #
    # Work arrays become:
    #          (nFreq, nTM, nChunk, 3)
    u = u_chunk[:, None, :, :]                           # (nFreq, 1, nChunk, 3)
    rc = rCap[None, :, :, :]                             # (1, nTM, nChunk, 3)
    rd3 = rDist3[None, :, :, None]                       # (1, nTM, nChunk, 1)
    weights = dV[None, None, :, None]                    # (1, 1, nChunk, 1)

    term1 = u / rd3

    dot_ru = np.sum(rc * u, axis=3, keepdims=True)       # (nFreq, nTM, nChunk, 1)
    term2 = 3.0 * rc * dot_ru / rd3

    I_local = rho * (term1 - term2) * weights            # (nFreq, nTM, nChunk, 3)

    # Sum over spatial points
    I_sum = np.sum(I_local, axis=2)                      # (nFreq, nTM, 3)

    # Return as (nFreq, 3, nTM)
    I_chunk = np.transpose(I_sum, (0, 2, 1))

    return I_chunk

def getAnalyticRayNN(config,cavityDepth,gamma=0.8):
    # analytic Rayleigh wave calculation
    freqs = np.arange(config.fMin,config.fMax,config.df)
    nF = len(freqs)
    thickness = config.thickness
    vP = config.vP
    vS = config.vS
    rho = config.rho

    periods = 1/freqs
    h = cavityDepth
    #gamma = 0.8

    vR = surf96(thickness,vP,vS,rho,periods,wave="rayleigh",mode=1,velocity="phase",flat_earth=True)
    qP = (2*np.pi*freqs)/(vR*vP[0])*np.sqrt(vP[0]**2 - vR**2)
    qS = (2*np.pi*freqs)/(vR*vS[0])*np.sqrt(vS[0]**2 - vR**2)
    ki = np.sqrt(qP/qS)
    kR = (2*np.pi*freqs)/vR
    RNum = -kR*(1+ki)*np.exp(-kR*h) + 2/3*(2*kR*np.exp(-qP*h) + ki*qS*np.exp(-qS*h))
    RDen = kR*(1-ki)
    R = np.abs(RNum/RDen)
    NNR = np.sqrt(2)*np.pi*const.G*gamma*rho[0]*R
    return NNR,freqs

def getAnalyticBodyNN(config,p=1/3):
    # analytic body wave NN calculation
    freqs = np.arange(config.fMin,config.fMax,0.1)
    nF = len(freqs)
    minNN =(4*np.pi/3*const.G*config.rho[0])*np.ones((nF,))
    bodyNN = (4*np.pi/3*const.G*config.rho[0]*np.sqrt(3*p+1))*np.ones((nF,)) 
    return bodyNN, minNN, freqs

def make_centered_axis(xmin, xmax, target_dx=1.0, min_pts=3):
    """
    Uniform, cell-centered grid inside [xmin, xmax].
    Returns centers and equal weights (~target_dx).
    """
    L = xmax - xmin
    n_int = max(min_pts, int(np.round(L / target_dx)))
    dx = L / n_int
    # center points shifted half a cell inside each end
    x = xmin + dx / 2 + dx * np.arange(n_int)
    w = np.full_like(x, dx)
    return x, w

def make_adaptive_centered_axis(xmin, xmax, cavity_min, cavity_max, dx_inside=1.0, dx_near=1.0, 
                                dx_mid=2.0, dx_far=5.0, near_width=20.0, mid_width=80.0, min_pts_per_segment=1):
    """
    Cell-centered nonuniform axis inside [xmin, xmax].

    The cavity interval [cavity_min, cavity_max] is included in the axis
    because points are later removed by the cavity mask.

    Spacing is finest inside/near the cavity and grows outward.

    Returns
    -------
    x : ndarray
        Cell centers.
    w : ndarray
        Cell widths.
    """

    def add_segment(edges, a, b, dx):
        if b <= a:
            return edges

        L = b - a
        n = max(min_pts_per_segment, int(np.ceil(L / dx)))
        local_edges = np.linspace(a, b, n + 1)

        if len(edges) == 0:
            return list(local_edges)
        else:
            return edges + list(local_edges[1:])

    # Important breakpoints
    c0 = cavity_min
    c1 = cavity_max

    # near shell around cavity
    n0 = max(xmin, c0 - near_width)
    n1 = min(xmax, c1 + near_width)

    # mid shell around cavity
    m0 = max(xmin, c0 - mid_width)
    m1 = min(xmax, c1 + mid_width)

    # Build sorted breakpoints clipped to [xmin, xmax]
    breaks = [xmin, m0,n0,c0,c1,n1,m1,xmax]

    breaks = np.array(sorted(set([float(np.clip(v, xmin, xmax)) for v in breaks])))

    edges = []

    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue

        # Segment classification by location relative to cavity
        seg_mid = 0.5 * (a + b)

        if c0 <= seg_mid <= c1:
            dx = dx_inside
        elif n0 <= seg_mid <= n1:
            dx = dx_near
        elif m0 <= seg_mid <= m1:
            dx = dx_mid
        else:
            dx = dx_far

        edges = add_segment(edges, a, b, dx)

    edges = np.asarray(edges)
    x = 0.5 * (edges[:-1] + edges[1:])
    w = edges[1:] - edges[:-1]

    return x, w

def make_adaptive_z_axis(zmin,zmax, cav_zmin, cav_zmax, dz_inside=1.0, dz_near=1.0, dz_mid=2.0,
                         dz_far=5.0, near_width=15.0, mid_width=30.0, min_pts_per_segment=1):
    """
    Cell-centered nonuniform z axis in [zmin, zmax], classified by distance
    to the cavity interval [cav_zmin, cav_zmax].

    Interpretation:
      inside cavity:      [cav_zmin, cav_zmax]
      near cavity shell:  distance <= near_width
      mid cavity shell:   near_width < distance <= mid_width
      far:                distance > mid_width
    """

    def add_segment(edges, a, b, dz):
        if b <= a:
            return edges

        L = b - a
        n = max(min_pts_per_segment, int(np.ceil(L / dz)))
        local_edges = np.linspace(a, b, n + 1)

        if len(edges) == 0:
            return list(local_edges)
        else:
            return edges + list(local_edges[1:])

    # breakpoints induced by cavity interval and shell widths
    breaks = [
        zmin,
        cav_zmin - (near_width + mid_width),
        cav_zmin - near_width,
        cav_zmin,
        cav_zmax,
        cav_zmax + near_width,
        cav_zmax + (near_width + mid_width),
        zmax,
    ]

    breaks = np.array(sorted(set(float(np.clip(v, zmin, zmax)) for v in breaks)))

    edges = []

    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue

        zm = 0.5 * (a + b)

        # distance of zm to cavity interval
        if cav_zmin <= zm <= cav_zmax:
            dz = dz_inside
        elif zm < cav_zmin:
            dist = cav_zmin - zm
            if dist <= near_width:
                dz = dz_near
            elif dist <= (near_width+mid_width):
                dz = dz_mid
            else:
                dz = dz_far
        else:  # zm > cav_zmax
            dist = zm - cav_zmax
            if dist <= near_width:
                dz = dz_near
            elif dist <= (near_width+ mid_width):
                dz = dz_mid
            else:
                dz = dz_far

        edges = add_segment(edges, a, b, dz)

    edges = np.asarray(edges)
    z = 0.5 * (edges[:-1] + edges[1:])
    w = edges[1:] - edges[:-1]

    return z, w

def choose_adaptive_widths(block_min, block_max, cavity_min, cavity_max,
                           near_cap=20.0, mid_cap=80.0,
                           near_frac=1.0/3.0, mid_frac=2.0/3.0):
    """
    Choose near_width and mid_width measured outward from the cavity wall,
    clipped to the available refined shell size inside this block.

    Returns
    -------
    near_width, mid_width
    """

    # Available space outside the cavity on both sides within this block
    left_shell  = max(0.0, cavity_min - block_min)
    right_shell = max(0.0, block_max - cavity_max)

    # Use the smaller side so the refinement zones fit symmetrically
    shell_half = min(left_shell, right_shell)

    near_width = min(near_cap, near_frac * shell_half)
    mid_width  = min(mid_cap,  mid_frac  * shell_half)

    # Keep ordering sensible
    mid_width = max(mid_width, near_width)

    return near_width, mid_width

def cavity_xy_intervals(cavity: dict) -> tuple[tuple[float, float], tuple[float, float]]:
    shape = cavity["shape"].lower()

    if shape == "sphere":
        r = cavity["radius"]
        x0 = cavity["xC"] - r
        x1 = cavity["xC"] + r
        y0 = cavity["yC"] - r
        y1 = cavity["yC"] + r
        return (x0, x1), (y0, y1)

    if shape == "cuboid":
        xC = cavity["xC"]
        yC = cavity["yC"]
        length = cavity["length"]
        breadth = cavity["breadth"]
        angle_deg = cavity["angleDeg"]

        theta = math.radians(angle_deg)
        u = np.array([math.cos(theta), math.sin(theta)])
        v = np.array([-math.sin(theta), math.cos(theta)])

        hl = 0.5 * length
        hb = 0.5 * breadth
        center = np.array([xC, yC])

        corners = np.array([
            center - hl * u - hb * v,
            center - hl * u + hb * v,
            center + hl * u + hb * v,
            center + hl * u - hb * v,
        ])

        x0 = float(np.min(corners[:, 0]))
        x1 = float(np.max(corners[:, 0]))
        y0 = float(np.min(corners[:, 1]))
        y1 = float(np.max(corners[:, 1]))
        return (x0, x1), (y0, y1)

    raise ValueError(f"Unsupported cavity shape: {cavity['shape']}")


def merge_intervals(intervals: list[tuple[float, float]], gap_tol: float = 0.0) -> list[tuple[float, float]]:
    if not intervals:
        return []

    intervals = sorted(intervals, key=lambda t: t[0])
    merged = [intervals[0]]

    for a, b in intervals[1:]:
        last_a, last_b = merged[-1]
        if a <= last_b + gap_tol:
            merged[-1] = (last_a, max(last_b, b))
        else:
            merged.append((a, b))

    return merged


def build_merged_xy_intervals(
    cavities: list[dict],
    gap_tol: float = 0.0
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    x_intervals = []
    y_intervals = []

    for cav in cavities:
        ix, iy = cavity_xy_intervals(cav)
        x_intervals.append(ix)
        y_intervals.append(iy)

    return merge_intervals(x_intervals, gap_tol), merge_intervals(y_intervals, gap_tol)


def choose_adaptive_widths_multi(
    block_min: float,
    block_max: float,
    intervals: list[tuple[float, float]],
    near_cap: float = 20.0,
    mid_cap: float = 80.0,
    near_frac: float = 1.0 / 3.0,
    mid_frac: float = 2.0 / 3.0,
) -> tuple[float, float]:
    if not intervals:
        return near_cap, mid_cap

    shell_half_all = []
    for cavity_min, cavity_max in intervals:
        left_shell = max(0.0, cavity_min - block_min)
        right_shell = max(0.0, block_max - cavity_max)
        shell_half_all.append(min(left_shell, right_shell))

    shell_half = min(shell_half_all)

    near_width = min(near_cap, near_frac * shell_half)
    mid_width = min(mid_cap, mid_frac * shell_half)
    mid_width = max(mid_width, near_width)

    return near_width, mid_width


def _point_in_any_interval(x: float, intervals: list[tuple[float, float]]) -> bool:
    return any(a <= x <= b for a, b in intervals)


def _distance_to_intervals(x: float, intervals: list[tuple[float, float]]) -> float:
    if not intervals:
        return float("inf")

    vals = []
    for a, b in intervals:
        if x < a:
            vals.append(a - x)
        elif x > b:
            vals.append(x - b)
        else:
            vals.append(0.0)
    return min(vals)


def _add_segment(edges: list[float], a: float, b: float, dx: float, min_pts_per_segment: int = 1) -> list[float]:
    if b <= a:
        return edges

    L = b - a
    n = max(min_pts_per_segment, int(np.ceil(L / dx)))
    local_edges = np.linspace(a, b, n + 1)

    if len(edges) == 0:
        return list(local_edges)
    return edges + list(local_edges[1:])


def make_adaptive_axis_multi(
    xmin: float,
    xmax: float,
    intervals: list[tuple[float, float]],
    dx_inside: float = 1.0,
    dx_near: float = 1.0,
    dx_mid: float = 2.0,
    dx_far: float = 5.0,
    near_width: float = 20.0,
    mid_width: float = 80.0,
    min_pts_per_segment: int = 1,
) -> tuple[np.ndarray, np.ndarray]:
    breaks = [xmin, xmax]

    for c0, c1 in intervals:
        breaks.extend([
            c0, c1,
            c0 - near_width, c1 + near_width,
            c0 - mid_width, c1 + mid_width,
        ])

    breaks = sorted(set(float(np.clip(v, xmin, xmax)) for v in breaks))

    edges = []
    for a, b in zip(breaks[:-1], breaks[1:]):
        if b <= a:
            continue

        xm = 0.5 * (a + b)
        d = _distance_to_intervals(xm, intervals)

        if _point_in_any_interval(xm, intervals):
            dx = dx_inside
        elif d <= near_width:
            dx = dx_near
        elif d <= mid_width:
            dx = dx_mid
        else:
            dx = dx_far

        edges = _add_segment(edges, a, b, dx, min_pts_per_segment=min_pts_per_segment)

    edges = np.asarray(sorted(set(edges)), dtype=float)
    x = 0.5 * (edges[:-1] + edges[1:])
    w = edges[1:] - edges[:-1]
    return x, w


def generate_adaptive_xy_axes(
    cavities: list[dict],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx_inside: float = 1.0,
    dx_near: float = 1.0,
    dx_mid: float = 2.0,
    dx_far: float = 5.0,
    dy_inside: float = 1.0,
    dy_near: float = 1.0,
    dy_mid: float = 2.0,
    dy_far: float = 5.0,
    near_cap: float = 20.0,
    mid_cap: float = 80.0,
    near_frac: float = 1.0 / 3.0,
    mid_frac: float = 2.0 / 3.0,
    gap_tol: float = 0.0,
):
    x_intervals_merged, y_intervals_merged = build_merged_xy_intervals(cavities, gap_tol=gap_tol)

    near_x, mid_x = choose_adaptive_widths_multi(
        x_min, x_max, x_intervals_merged,
        near_cap=near_cap, mid_cap=mid_cap,
        near_frac=near_frac, mid_frac=mid_frac,
    )
    near_y, mid_y = choose_adaptive_widths_multi(
        y_min, y_max, y_intervals_merged,
        near_cap=near_cap, mid_cap=mid_cap,
        near_frac=near_frac, mid_frac=mid_frac,
    )

    gridX, gridXW = make_adaptive_axis_multi(
        x_min, x_max, x_intervals_merged,
        dx_inside=dx_inside, dx_near=dx_near, dx_mid=dx_mid, dx_far=dx_far,
        near_width=near_x, mid_width=mid_x,
    )
    gridY, gridYW = make_adaptive_axis_multi(
        y_min, y_max, y_intervals_merged,
        dx_inside=dy_inside, dx_near=dy_near, dx_mid=dy_mid, dx_far=dy_far,
        near_width=near_y, mid_width=mid_y,
    )

    return {
        "gridX": gridX,
        "gridXW": gridXW,
        "gridY": gridY,
        "gridYW": gridYW,
        "xIntervalsMerged": x_intervals_merged,
        "yIntervalsMerged": y_intervals_merged,
        "near_x": near_x,
        "mid_x": mid_x,
        "near_y": near_y,
        "mid_y": mid_y,
    }

def points_inside_sphere(x, y, z, cavity):
    """
    True where points are inside the spherical cavity.
    """
    xC = cavity["xC"]
    yC = cavity["yC"]
    zC = cavity["zC"]
    r = cavity["radius"]

    r2 = (x - xC)**2 + (y - yC)**2 + (z + zC)**2
    return r2 <= r**2


def points_inside_cuboid(x, y, z, cavity):
    """
    True where points are inside the cuboid cavity.
    Cuboid may be rotated in the x-y plane.
    """
    xC = cavity["xC"]
    yC = cavity["yC"]
    zC = cavity["zC"]

    ax = cavity["length"] / 2.0
    ay = cavity["breadth"] / 2.0
    az = cavity["height"] / 2.0
    angleDeg = cavity.get("angleDeg", 0.0)

    theta = np.deg2rad(angleDeg)

    dx = x - xC
    dy = y - yC
    dz = z + zC

    # transform to cavity-local coordinates
    x_local =  np.cos(theta) * dx + np.sin(theta) * dy
    y_local = -np.sin(theta) * dx + np.cos(theta) * dy

    return (
        (np.abs(x_local) <= ax) &
        (np.abs(y_local) <= ay) &
        (np.abs(dz) <= az)
    )


def mask_points_outside_all_cavities(x, y, z, allCavities):
    """
    Returns boolean mask:
        True  -> point is outside all cavities
        False -> point is inside at least one cavity
    """
    inside_any = np.zeros_like(x, dtype=bool)

    for cav in allCavities:
        shape = cav["shape"].lower()

        if shape == "sphere":
            inside_any |= points_inside_sphere(x, y, z, cav)

        elif shape == "cuboid":
            inside_any |= points_inside_cuboid(x, y, z, cav)

        else:
            raise ValueError(f"Unsupported cavity shape: {cav['shape']}")

    return ~inside_any

def choose_z_shell_widths_from_global_band(allCavities, cubeTop, cubeBot):
    half_heights = []
    for cav in allCavities:
        if cav["shape"] == "cuboid":
            hz = 0.5 * cav["height"]
        elif cav["shape"] == "sphere":
            hz = cav["radius"]
        else:
            raise ValueError("Unsupported cavity type")
        half_heights.append(hz)

    iref = int(np.argmax(half_heights))
    cav_ref = allCavities[iref]

    center_z = cav_ref["zC"]
    h = half_heights[iref]

    phys_top = center_z - h
    phys_bot = center_z + h

    top_buffer = phys_top - cubeTop
    bottom_buffer = cubeBot - phys_bot

    z_buffer = min(top_buffer, bottom_buffer)

    near_z = (1.0/3.0) * z_buffer
    mid_z  = (2.0/3.0) * z_buffer

    return near_z, mid_z


def cavity_xy_bbox(cavity: dict) -> tuple[float, float, float, float]:
    """
    Returns axis-aligned XY bounding box:
        (x_min, x_max, y_min, y_max)
    """
    shape = cavity["shape"].lower()

    if shape == "sphere":
        r = cavity["radius"]
        return (
            cavity["xC"] - r,
            cavity["xC"] + r,
            cavity["yC"] - r,
            cavity["yC"] + r,
        )

    if shape == "cuboid":
        xC = cavity["xC"]
        yC = cavity["yC"]
        length = cavity["length"]
        breadth = cavity["breadth"]
        angle_deg = cavity.get("angleDeg", 0.0)

        theta = math.radians(angle_deg)
        u = np.array([math.cos(theta), math.sin(theta)])
        v = np.array([-math.sin(theta), math.cos(theta)])

        hl = 0.5 * length
        hb = 0.5 * breadth
        ctr = np.array([xC, yC])

        corners = np.array([
            ctr - hl * u - hb * v,
            ctr - hl * u + hb * v,
            ctr + hl * u + hb * v,
            ctr + hl * u - hb * v,
        ])

        return (
            float(np.min(corners[:, 0])),
            float(np.max(corners[:, 0])),
            float(np.min(corners[:, 1])),
            float(np.max(corners[:, 1])),
        )

    raise ValueError(f"Unsupported cavity shape: {cavity['shape']}")


def bboxes_overlap_or_close(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
    gap_tol: float,
) -> bool:
    xa0, xa1, ya0, ya1 = bbox_a
    xb0, xb1, yb0, yb1 = bbox_b

    x_sep = max(xa0, xb0) - min(xa1, xb1)
    y_sep = max(ya0, yb0) - min(ya1, yb1)

    return (x_sep <= gap_tol) and (y_sep <= gap_tol)


def merge_two_bboxes(
    bbox_a: tuple[float, float, float, float],
    bbox_b: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    return (
        min(bbox_a[0], bbox_b[0]),
        max(bbox_a[1], bbox_b[1]),
        min(bbox_a[2], bbox_b[2]),
        max(bbox_a[3], bbox_b[3]),
    )


def merge_bboxes_xy(
    bboxes: list[tuple[float, float, float, float]],
    gap_tol: float,
) -> list[tuple[float, float, float, float]]:
    if not bboxes:
        return []

    working = bboxes[:]
    changed = True

    while changed:
        changed = False
        used = [False] * len(working)
        out = []

        for i in range(len(working)):
            if used[i]:
                continue

            cur = working[i]
            used[i] = True

            grew = True
            while grew:
                grew = False
                for j in range(len(working)):
                    if used[j]:
                        continue
                    if bboxes_overlap_or_close(cur, working[j], gap_tol):
                        cur = merge_two_bboxes(cur, working[j])
                        used[j] = True
                        grew = True
                        changed = True

            out.append(cur)

        working = out

    return working


def choose_patch_widths_for_cluster(
    cluster_bbox: tuple[float, float, float, float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    near_cap: float,
    mid_cap: float,
    near_frac: float,
    mid_frac: float,
) -> tuple[float, float, float, float]:
    """
    Returns:
        near_x, mid_x, near_y, mid_y

    IMPORTANT:
    near_x / near_y are near-shell widths
    mid_x  / mid_y  are mid-shell widths
    not cumulative outer paddings.
    """
    bx0, bx1, by0, by1 = cluster_bbox

    left_gap = max(0.0, bx0 - x_min)
    right_gap = max(0.0, x_max - bx1)
    shell_x = min(left_gap, right_gap)

    bottom_gap = max(0.0, by0 - y_min)
    top_gap = max(0.0, y_max - by1)
    shell_y = min(bottom_gap, top_gap)

    near_x = min(near_cap, near_frac * shell_x)
    mid_x = min(mid_cap, mid_frac * shell_x)
    mid_x = max(mid_x, near_x)

    near_y = min(near_cap, near_frac * shell_y)
    mid_y = min(mid_cap, mid_frac * shell_y)
    mid_y = max(mid_y, near_y)

    return near_x, mid_x, near_y, mid_y


def expand_and_clip_bbox_xy(
    bbox: tuple[float, float, float, float],
    pad_x: float,
    pad_y: float,
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
) -> tuple[float, float, float, float]:
    return (
        max(x_min, bbox[0] - pad_x),
        min(x_max, bbox[1] + pad_x),
        max(y_min, bbox[2] - pad_y),
        min(y_max, bbox[3] + pad_y),
    )

def subtract_rect_from_cells(cells: np.ndarray,rect_bbox: tuple[float, float, float, float],
                             tol: float = 1e-12,) -> np.ndarray:
    """
    Subtract a rectangular region from an existing rectangular-cell grid.

    Partially intersected cells are split so no area is lost
    and no area is double counted.
    """

    if cells.size == 0:
        return cells

    rx0, rx1, ry0, ry1 = rect_bbox

    out = []

    for x0, x1, y0, y1 in cells:

        # intersection between this cell and replacement rectangle
        ix0 = max(x0, rx0)
        ix1 = min(x1, rx1)
        iy0 = max(y0, ry0)
        iy1 = min(y1, ry1)

        # no intersection
        if ix1 <= ix0 + tol or iy1 <= iy0 + tol:
            out.append([x0, x1, y0, y1])
            continue

        # portion left of overlap
        if ix0 > x0 + tol:
            out.append([x0, ix0, y0, y1])

        # portion right of overlap
        if ix1 < x1 - tol:
            out.append([ix1, x1, y0, y1])

        # portion below overlap
        if iy0 > y0 + tol:
            out.append([ix0, ix1, y0, iy0])

        # portion above overlap
        if iy1 < y1 - tol:
            out.append([ix0, ix1, iy1, y1])

    if len(out) == 0:
        return np.zeros((0, 4), dtype=float)

    return np.asarray(out, dtype=float)

def cells_to_pointcloud(cells: np.ndarray):
    """
    Convert rectangular quadrature cells to midpoint coordinates
    and exact cell-area weights.
    """

    if cells.size == 0:
        return (np.zeros((0, 2), dtype=float),np.zeros((0,), dtype=float))

    x0 = cells[:, 0]
    x1 = cells[:, 1]
    y0 = cells[:, 2]
    y1 = cells[:, 3]

    x_mid = 0.5 * (x0 + x1)
    y_mid = 0.5 * (y0 + y1)

    XY = np.column_stack((x_mid, y_mid))

    dA = (x1 - x0) * (y1 - y0)

    return XY, dA

def make_rect_patch_cells(x_min: float,x_max: float,y_min: float,y_max: float,dx: float,
                          dy: float,) -> np.ndarray:
    """
    Construct rectangular midpoint-quadrature cells.

    Returns
    -------
    cells : ndarray, shape (N, 4)
        Each row is [x0, x1, y0, y1].
    """

    if x_max <= x_min or y_max <= y_min:
        return np.zeros((0, 4), dtype=float)

    nx = max(1, int(np.ceil((x_max - x_min) / dx)))
    ny = max(1, int(np.ceil((y_max - y_min) / dy)))

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    cells = []

    for iy in range(ny):
        for ix in range(nx):
            cells.append([x_edges[ix],x_edges[ix + 1],y_edges[iy],y_edges[iy + 1]])

    return np.asarray(cells, dtype=float)

def make_rect_patch_points(
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx: float,
    dy: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns:
        XY: shape (N, 2)
        dA: shape (N,)
    """
    if x_max <= x_min or y_max <= y_min:
        return np.zeros((0, 2), dtype=float), np.zeros((0,), dtype=float)

    nx = max(1, int(np.ceil((x_max - x_min) / dx)))
    ny = max(1, int(np.ceil((y_max - y_min) / dy)))

    x_edges = np.linspace(x_min, x_max, nx + 1)
    y_edges = np.linspace(y_min, y_max, ny + 1)

    x_centers = 0.5 * (x_edges[:-1] + x_edges[1:])
    y_centers = 0.5 * (y_edges[:-1] + y_edges[1:])

    XX, YY = np.meshgrid(x_centers, y_centers, indexing="xy")
    XY = np.column_stack((XX.ravel(), YY.ravel()))

    cell_dx = x_edges[1:] - x_edges[:-1]
    cell_dy = y_edges[1:] - y_edges[:-1]
    DX, DY = np.meshgrid(cell_dx, cell_dy, indexing="xy")
    dA = (DX * DY).ravel()

    return XY, dA


def replace_points_in_rect(
    XY: np.ndarray,
    dA: np.ndarray,
    rect_bbox: tuple[float, float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Remove existing points inside rect_bbox.
    """
    if XY.size == 0:
        return XY, dA

    x0, x1, y0, y1 = rect_bbox
    inside = (
        (XY[:, 0] >= x0) & (XY[:, 0] <= x1) &
        (XY[:, 1] >= y0) & (XY[:, 1] <= y1)
    )

    return XY[~inside, :], dA[~inside]


def generate_adaptive_xy_pointcloud(
    cavities: list[dict],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    dx_inside: float = 1.0,
    dy_inside: float = 1.0,
    dx_near: float = 1.0,
    dy_near: float = 1.0,
    dx_mid: float = 2.0,
    dy_mid: float = 2.0,
    dx_far: float = 5.0,
    dy_far: float = 5.0,
    near_frac: float = 1.0 / 3.0,
    mid_frac: float = 2.0 / 3.0,
    near_cap: float = 20.0,
    mid_cap: float = 80.0,
    gap_tol: float = 10.0,
) -> dict:
    """
    Build adaptive XY point cloud from local patches.

    Returns dict with:
        gridX_flat
        gridY_flat
        gridW_flat
        XY
        dA
        cavity_bboxes
        cluster_bboxes
    """
    cavity_bboxes = [cavity_xy_bbox(cav) for cav in cavities]
    cluster_bboxes = merge_bboxes_xy(cavity_bboxes, gap_tol=gap_tol)

    # Start with far patch over full block
    #XY, dA = make_rect_patch_points(x_min, x_max, y_min, y_max, dx_far, dy_far)
    # adding this on Aug 22, 2026
    cells = make_rect_patch_cells(x_min, x_max,y_min, y_max,dx_far, dy_far)

    # Mid patches
    for cluster_bbox in cluster_bboxes:
        near_x, mid_x, near_y, mid_y = choose_patch_widths_for_cluster(
            cluster_bbox, x_min, x_max, y_min, y_max,
            near_cap, mid_cap, near_frac, mid_frac,
        )

        # IMPORTANT: mid shell width is mid_x itself, so cumulative pad is near + mid
        mid_bbox = expand_and_clip_bbox_xy(
            cluster_bbox,
            near_x + mid_x,
            near_y + mid_y,
            x_min, x_max, y_min, y_max,
        )

        #XY, dA = replace_points_in_rect(XY, dA, mid_bbox)
        #XY_new, dA_new = make_rect_patch_points(
        #    mid_bbox[0], mid_bbox[1], mid_bbox[2], mid_bbox[3], dx_mid, dy_mid
        #)
        #XY = np.vstack((XY, XY_new))
        #dA = np.concatenate((dA, dA_new))
        # updated Aug 22, 2026
        cells = subtract_rect_from_cells(cells, mid_bbox)

        cells_new = make_rect_patch_cells(mid_bbox[0], mid_bbox[1],mid_bbox[2], mid_bbox[3],
                                          dx_mid, dy_mid)

        cells = np.vstack((cells, cells_new))

    # Near patches
    for cluster_bbox in cluster_bboxes:
        near_x, mid_x, near_y, mid_y = choose_patch_widths_for_cluster(
            cluster_bbox, x_min, x_max, y_min, y_max,
            near_cap, mid_cap, near_frac, mid_frac,
        )

        near_bbox = expand_and_clip_bbox_xy(
            cluster_bbox,
            near_x,
            near_y,
            x_min, x_max, y_min, y_max,
        )

        #XY, dA = replace_points_in_rect(XY, dA, near_bbox)
        #XY_new, dA_new = make_rect_patch_points(
        #    near_bbox[0], near_bbox[1], near_bbox[2], near_bbox[3], dx_near, dy_near
        #)
        #XY = np.vstack((XY, XY_new))
        #dA = np.concatenate((dA, dA_new))
        # updated Aug 22, 2026
        cells = subtract_rect_from_cells(cells, near_bbox)
        cells_new = make_rect_patch_cells(near_bbox[0], near_bbox[1],near_bbox[2], near_bbox[3],
                                                  dx_near, dy_near)
        
        cells = np.vstack((cells, cells_new))

    # Inside patches
    for cluster_bbox in cluster_bboxes:
        #XY, dA = replace_points_in_rect(XY, dA, cluster_bbox)
        #XY_new, dA_new = make_rect_patch_points(
        #    cluster_bbox[0], cluster_bbox[1], cluster_bbox[2], cluster_bbox[3],
        #    dx_inside, dy_inside
        #)
        #XY = np.vstack((XY, XY_new))
        #dA = np.concatenate((dA, dA_new))
        # update Aug 22, 2026
        cells = subtract_rect_from_cells(cells, cluster_bbox)
        cells_new = make_rect_patch_cells(cluster_bbox[0], cluster_bbox[1],cluster_bbox[2], cluster_bbox[3],
                                                  dx_inside, dy_inside)
        cells = np.vstack((cells, cells_new))

    XY, dA = cells_to_pointcloud(cells)
    gridX_flat = XY[:, 0].copy()
    gridY_flat = XY[:, 1].copy()
    gridW_flat = dA.copy()

    A_grid = np.sum(dA)
    A_true = (x_max - x_min) * (y_max - y_min)

    if not np.isclose(A_grid, A_true, rtol=1e-12, atol=1e-8):
        raise RuntimeError(f"Adaptive XY quadrature does not conserve area: "
                           f"A_grid={A_grid}, A_true={A_true}, "
                           f"difference={A_grid - A_true}")
    return {
        "gridX_flat": gridX_flat,
        "gridY_flat": gridY_flat,
        "gridW_flat": gridW_flat,
        "XY": XY,
        "dA": dA,
        "cavity_bboxes": cavity_bboxes,
        "cluster_bboxes": cluster_bboxes,
    }

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

def getGridSize(thickness,vP, vS, rho, fMin, fMax, lambdaFrac, lambdaRes):
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
    [freqMin, vRMin] = getMinVelFreq(thickness,vP, vS, rho, fMin, fMax, lambdaFrac)
    lambdaRMin = vRMin/freqMin*1000; # in meters
    
    lambdaSMin = vS/fMax; # in meters
    
    lambdaMin = np.minimum(lambdaRMin,lambdaSMin)

    gridSize = lambdaMin/lambdaRes
    
    return gridSize

def getMinVelFreq(thickness,vP, vS, rho, fMin, fMax, lambdaFrac):
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
    thick = thickness/1000; # converted to kilometers
    vP = vP/1000; vS = vS/1000; # converted to km/s
    rho = rho/1000; # converted to gm/cc
    
    # Periods we are interested in
    freqs = np.arange(fMin,fMax,0.2);
    
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
    lambdaValBy3 = lambdaVal*lambdaFrac;
    depths = np.cumsum(thick);
    depths = np.insert(depths,0,0.0,axis=0);
    vMin = np.zeros((len(depths)-1),);
    freqMin = np.zeros((len(depths)-1),);
    
    for depthNo in range(0,(len(depths)-1)):
        fInd = np.where((lambdaValBy3<=depths[depthNo+1]) & (lambdaValBy3>=depths[depthNo]))[0]
        if(len(fInd)>0):
            vMin[depthNo] = vMinRayLove[fInd[-1]]
            freqMin[depthNo] = freqs[fInd[-1]];
            if(vMin[depthNo]>vS[depthNo]):
                vMin[depthNo] = vS[depthNo];
        else:
            vMin[depthNo] = vS[depthNo];
            freqMin[depthNo] = fMax;
    return freqMin, vMin;    
if __name__ == "__main__":
    main()