#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
from scipy.io import loadmat
import matplotlib.pyplot as plt
from modules import simDisp

def applyScaleDispAllComp(itfname, tmName, configVertA, configVertB, configVertC, sF):
    
    # finds the scale factor such that the displacement at the desired itm at depth or surface
    # is 1, if scaleToTrue is set to false, and to observed ASD at Terziet scaleToTrue = True

    # first map the test-mass
    itmStr, itmInd, vertStr, vertInd = mapTM(itfname, tmName)

    # determine the path to read all realizations
    if(vertStr == 'A'):
        configUse = configVertA
    elif(vertStr == 'B'):
        configUse = configVertB
    elif(vertStr == 'C'):
        configUse = configVertC
    else:
        raise ValueError('Vertex string must be either A , B or C!')

    # now loop over realizations
    # dispPointFull is saved as nFreq x nRec x nComp x nLoc
    freqOut, _, _ = simDisp.getFreqGrid(configUse.tMax, configUse.nSamp, 
                                                      configUse.fMin, configUse.fMax, configUse.df)
    nFreq = len(freqOut)
    dispPointAllRea = np.zeros((nFreq,3,2))
    for i in range(0,configUse.nRea):
        fName = 'surfDeepDispRea' + str(i) + '.npz'
        fPathFull = os.path.join(configUse.outDispPathRea,fName)
        data = np.load(fPathFull)
        # dispPitnFull is aleready magnitude, so no abs here
        dispPointAllRea += data['dispPointFull'][:,itmInd,:,:]**2
    
    dispPointAllRea = np.sqrt(dispPointAllRea/configUse.nRea)

    dispOut = dispPointAllRea*sF[:,None,None]

    #dispOut is of the shape (nFreq x nCom x nLoc)
    return dispOut, freqOut

def getDispScaleFact(itfname, tmName, compName, location, configVertA, configVertB, configVertC,
                     scaleToTrue = False):
    
    # finds the scale factor such that the displacement at the desired itm at depth or surface
    # is 1, if scaleToTrue is set to false, and to observed ASD at Terziet scaleToTrue = True

    # first map the test-mass
    itmStr, itmInd, vertStr, vertInd = mapTM(itfname, tmName)

    # determine the path to read all realizations
    if(vertStr == 'A'):
        configUse = configVertA
    elif(vertStr == 'B'):
        configUse = configVertB
    elif(vertStr == 'C'):
        configUse = configVertC
    else:
        raise ValueError('Vertex string must be either A , B or C!')
    
    # determine component index to read
    if(compName =='E'):
        compInd = 0
    elif(compName == 'N'):
        compInd = 1
    elif(compName == 'Z'):
        compInd = 2
    else:
        raise ValueError('Component must be E, N, or Z')

    # determine surface or depth index
    if(location == 'depth'):
        depthInd = 1
    elif(location == 'surface'):
        depthInd = 0
    else:
        raise ValueError('location must be depth or surface')

    # now loop over realizations
    # dispPointFull is saved as nFreq x nRec x nComp x nLoc
    freqOut, _, _ = simDisp.getFreqGrid(configUse.tMax, configUse.nSamp, 
                                                      configUse.fMin, configUse.fMax, configUse.df)
    nFreq = len(freqOut)
    dispPointAllRea = np.zeros((nFreq,))
    for i in range(0,configUse.nRea):
        fName = 'surfDeepDispRea' + str(i) + '.npz'
        fPathFull = os.path.join(configUse.outDispPathRea,fName)
        data = np.load(fPathFull)
        dispPointAllRea += data['dispPointFull'][:,itmInd,compInd,depthInd]**2
    
    dispPointAllRea = np.sqrt(dispPointAllRea/configUse.nRea)
    if(scaleToTrue):
        # load the relevant site ASDs
        siteASDAllPath = configUse.siteASDPath
        if(location=='depth'):
            siteASDFName = 'asdBH.mat'
            siteASDFullPath = os.path.join(siteASDAllPath,siteASDFName)
            asdLoad = loadmat(siteASDFullPath)
            if(compName == 'E'):
                asdData = asdLoad['asdBHX']
            elif(compName=='N'):
                asdData = asdLoad['asdBHY']
            elif(compName=='Z'):
                asdData = asdLoad['asdBHZ']
            else:
                raise ValueError('Component must be E, N, Z')

        elif(location=='surface'):
            siteASDFName = 'asdSurf.mat'
            siteASDFullPath = os.path.join(siteASDAllPath,siteASDFName)
            asdLoad = loadmat(siteASDFullPath)
            if(compName == 'E'):
                asdData = asdLoad['asdSurfX']
            elif(compName=='N'):
                asdData = asdLoad['asdSurfY']
            elif(compName=='Z'):
                asdData = asdLoad['asdSurfZ']
            else:
               raise ValueError('Component must be E, N, Z')

        else:
            raise ValueError('location must be depth or surface')

        # asdData has four columns, freq, 10th prct, 50th prct, 90th prct
        # we use the 50th prct to scale
        # but before an interpolation on freqOut is necessary
        asd10PrctIntp = np.interp(freqOut, asdData[:,0],asdData[:,2])
        sF = asd10PrctIntp/dispPointAllRea        
    else:
        sF = 1/dispPointAllRea

    return sF

def getSiteASDs(itfname, tmName, configVertA, configVertB, configVertC):
    # loads the site ASDs using the path in configFile and then returns
    # asdBHX, asdBHY, asdBHZ, asdSurfX, asdSurfY, asdSurfZ
    # first map the test-mass
    itmStr, itmInd, vertStr, vertInd = mapTM(itfname, tmName)

    # determine the path to read all realizations
    if(vertStr == 'A'):
        configUse = configVertA
    elif(vertStr == 'B'):
        configUse = configVertB
    elif(vertStr == 'C'):
        configUse = configVertC
    else:
        raise ValueError('Vertex string must be either A , B or C!')
    freqOut, _, _ = simDisp.getFreqGrid(configUse.tMax, configUse.nSamp, 
                                                      configUse.fMin, configUse.fMax, configUse.df)
    nF = len(freqOut)
    siteASDAllPath = configUse.siteASDPath
    # --- load site ASDs ---
    siteASDSurf = loadmat(os.path.join(siteASDAllPath, "asdSurf.mat"))
    siteASDBH   = loadmat(os.path.join(siteASDAllPath, "asdBH.mat"))

    # helper: interpolate [freq, col1, col2, col3] -> (nF,3)
    def interp3(mat, key):
        src = mat[key]
        out = np.zeros((nF, 3))
        for i in range(3):
            out[:, i] = np.interp(freqOut, src[:, 0], src[:, i+1])
        return out

    asdSurfX = interp3(siteASDSurf, "asdSurfX")
    asdSurfY = interp3(siteASDSurf, "asdSurfY")
    asdSurfZ = interp3(siteASDSurf, "asdSurfZ")
    asdBHX   = interp3(siteASDBH,   "asdBHX")
    asdBHY   = interp3(siteASDBH,   "asdBHY")
    asdBHZ   = interp3(siteASDBH,   "asdBHZ")
    
    return asdSurfX, asdSurfY, asdSurfZ, asdBHX, asdBHY, asdBHZ

def getTMPerITF(itfName, configVertA,configVertB,configVertC):
    # function to extract the test-mass locations corresponding to
    # a particular interferometer
    allTMLoc = np.zeros((4,3))
    if(itfName == 'A'):
        allTMName = ['itmAC','itmAB','etmAC','etmAB']
        for tmNo, tmName in enumerate(allTMName):
            allTMLoc[tmNo,:] = getSingleTM(itfName, tmName, configVertA, configVertB, configVertC)
    
    if(itfName == 'B'):
        allTMName = ['itmBA','itmBC','etmBA','etmBC']
        for tmNo, tmName in enumerate(allTMName):
            allTMLoc[tmNo,:] = getSingleTM(itfName, tmName, configVertA, configVertB, configVertC)
    
    if(itfName == 'C'):
        allTMName = ['itmCB','itmCA','etmCB','etmCA']
        for tmNo, tmName in enumerate(allTMName):
            allTMLoc[tmNo,:] = getSingleTM(itfName, tmName, configVertA, configVertB, configVertC)
    return allTMLoc

    
def getSingleTM(itfName, itmName, configVertA, configVertB, configVertC):
    
    allTMLoc = collectAllTMLoc(configVertA, configVertB, configVertC)
    itmStr, itmInd, vertStr, vertInd = mapTM(itfName, itmName)
    tmLoc = allTMLoc[itmInd,:,vertInd]
    return tmLoc

def collectAllTMLoc(configVertA, configVertB, configVertC):
    allTMLoc = np.zeros((4,3,3))
    
    allTMLoc[0,:,0] = configVertA.itmAC
    allTMLoc[1,:,0] = configVertA.itmAB
    allTMLoc[2,:,0] = configVertA.etmCA
    allTMLoc[3,:,0] = configVertA.etmBA

    allTMLoc[0,:,1] = configVertB.itmAC
    allTMLoc[1,:,1] = configVertB.itmAB
    allTMLoc[2,:,1] = configVertB.etmCA
    allTMLoc[3,:,1] = configVertB.etmBA

    allTMLoc[0,:,2] = configVertC.itmAC
    allTMLoc[1,:,2] = configVertC.itmAB
    allTMLoc[2,:,2] = configVertC.etmCA
    allTMLoc[3,:,2] = configVertC.etmBA

    return allTMLoc

def mapTM(itfName, itmName):
    # returns the itm index given the vertex which is a str 'A', 'B', or 'C'
    # and itmName which is string like 'itmAC', 'etmAC' and so on
    
    if(itfName=='A'):
        if(itmName=='itmAC'):
            itmStr = 'itmAC'
            vertStr = 'A'
            itmInd = 0
            vertInd = 0
        elif(itmName == 'itmAB'):
            itmStr = 'itmAB'
            vertStr = 'A'
            itmInd = 1
            vertInd = 0
        elif(itmName == 'etmAC'):
            itmStr = 'etmCA' # the one which the config file thinks
            vertStr = 'C'
            itmInd = 2
            vertInd = 2

        elif(itmName == 'etmAB'):
            itmStr = 'etmCA'
            vertStr = 'B'
            itmInd = 2
            vertInd = 1
        else:
            raise ValueError('Incorrect testmass name for itf A!')
    
    if(itfName == 'B'):
        if(itmName == 'itmBC'):
            itmStr = 'itmAC'
            itmInd = 0
            vertStr = 'B'
            vertInd = 1

        elif(itmName == 'itmBA'):
            itmStr = 'itmAB'
            itmInd = 1
            vertStr = 'B'
            vertInd = 1

        elif(itmName == 'etmBC'):
            itmStr = 'etmBA'
            itmInd = 3
            vertStr = 'C'
            vertInd = 2

        elif(itmName == 'etmBA'):
            itmStr = 'etmBA'
            itmInd = 3
            vertStr = 'A'
            vertInd = 0

        else:
            raise ValueError('Incorrect test-mass name for the itf B')

    if(itfName == 'C'):
        if(itmName == 'itmCA'):
            itmStr = 'itmAC'
            itmInd = 0
            vertStr = 'C'
            vertInd = 2
        elif(itmName == 'itmCB'):
            itmStr = 'itmAB'
            itmInd = 1
            vertStr = 'C'
            vertInd = 2

        elif(itmName == 'etmCB'):
            itmStr = 'etmBA'
            itmInd = 3
            vertStr = 'B'
            vertInd = 1

        elif(itmName == 'etmCA'):
            itmStr = 'etmCA'
            itmInd = 2
            vertStr = 'A'
            vertInd = 0

        else:
            raise ValueError('Incorrect test-mass name for the itf C')

    return itmStr, int(itmInd), vertStr, int(vertInd)

def unitVec(Xi, Xj):
    return (Xj - Xi)/np.linalg.norm(Xj - Xi)

def projectNNAlongArm(nnInp, nnEnd, uVec):
    inpAlongArm = nnInp[:,0]*uVec[0] + nnInp[:,1]*uVec[1]
    endAlongArm = nnEnd[:,0]*uVec[0] + nnEnd[:,1]*uVec[1]
    totNNAlongArm = endAlongArm - inpAlongArm
    return totNNAlongArm

def getNullStream(configVertA, configVertB, configVertC):
    # computes the null stream from all the three interferometers
    # additionally also computes the single interferometer output and its correlation
    # with the null stream
    configAll = []
    configAll.append(configVertA)
    configAll.append(configVertB)
    configAll.append(configVertC)

    freqOut, _, _ = simDisp.getFreqGrid(configAll[0].tMax, configAll[0].nSamp, 
                                                      configAll[0].fMin, configAll[0].fMax, configAll[0].df)
    nITF = 3
    itfName = ['A','B','C']
    allTMList = []
    tmList = ['itmAB','etmAB','itmAC','etmAC']
    allTMList.append(tmList)
    tmList = ['itmBC', 'etmBC', 'itmBA', 'etmBA']
    allTMList.append(tmList)
    tmList = ['itmCA','etmCA','itmCB','etmCB']
    allTMList.append(tmList)
   
    nF = len(freqOut)
    # get the 4x2 tmLoc array
    nTM = len(allTMList[0])
    tmLoc = np.zeros((nTM,3,nITF))
    for j in range(0,nITF):
        for i in range(0,nTM):
            tmLoc[i,:,j] = getSingleTM(itfName[j], allTMList[j][i], configVertA, configVertB, configVertC)

    # extract the unit vectors along the arms
    uVec = np.zeros((2,2,nITF))
    for j in range(0,nITF):
        uVec[0,:,j] = unitVec(tmLoc[0,0:2,j], tmLoc[1,0:2,j])
        uVec[1,:,j] = unitVec(tmLoc[2,0:2,j], tmLoc[3,0:2,j])
    print(uVec[:,:,0])
    print(uVec[:,:,1])
    print(uVec[:,:,2])
    
    # extract the file path for reading realizations
    tmIndAll = np.zeros((nTM,nITF),dtype=int); vertIndAll = np.zeros((nTM,nITF),dtype=int)
    reaPathAll = []
    nReaAll = np.zeros((nTM,nITF))
    for j in range(0,nITF):
        reaPathPerITF = []
        for i in range(0,nTM):
            _, tmIndAll[i,j], _, vertIndAll[i,j] = mapTM(itfName[j], allTMList[j][i])
            reaPathPerITF.append(configAll[int(vertIndAll[i,j])].outDispPathRea) 
            nReaAll[i,j] = configAll[int(vertIndAll[i,j])].nRea
        reaPathAll.append(reaPathPerITF)
    
    #print(reaPathAll)
    nRea = int(np.min(nReaAll))
    #print('nRea = ' + str(nRea))
    # loop over realizations
    nnFullITFA = np.zeros((nF,))
    nnFullITFB = np.zeros((nF,))
    nnFullITFC = np.zeros((nF,))
    nullFullITF = np.zeros((nF,))

    autoITFA = np.zeros((nF,))
    autoITFB = np.zeros((nF,))
    autoITFC = np.zeros((nF,))
    autoNull = np.zeros((nF,))
    crossITFANull = np.zeros((nF,),dtype = np.complex128)
    crossITFBNull = np.zeros((nF,),dtype = np.complex128)
    crossITFCNull = np.zeros((nF,),dtype = np.complex128)
    crossITFAB = np.zeros((nF,),dtype = np.complex128)
    crossITFAC = np.zeros((nF,),dtype = np.complex128)
    
    for i in range(0,nRea):
        fName = 'NNFullRea' + str(i) + '.npz'
        nnAllTM = np.zeros((nF,3,nTM,nITF),dtype = np.complex128)
        for j in range(0,nITF):
            for tmNo in range(0,nTM):
                data = np.load(os.path.join(reaPathAll[j][tmNo],fName))
                #print(np.shape(data['ITot'][:,:,tmIndAll[tmNo,j]]))
                nnAllTM[:,:,tmNo,j] = data['ITot'][:,:,tmIndAll[tmNo,j]]
        
        # now use the X,Y components of nnAllTM and tm locations to project
        nnArmAB = projectNNAlongArm(nnAllTM[:,:,0,0], nnAllTM[:,:,1,0], uVec[0,:,0])
        #print(np.shape(nnArm1))
        nnArmAC = projectNNAlongArm(nnAllTM[:,:,2,0], nnAllTM[:,:,3,0], uVec[1,:,0])
        #print(np.shape(nnArm2))
        nnITFA = nnArmAB - nnArmAC

        # do the same for interferomter B
        # now use the X,Y components of nnAllTM and tm locations to project
        nnArmBC = projectNNAlongArm(nnAllTM[:,:,0,1], nnAllTM[:,:,1,1], uVec[0,:,1])
        #print(np.shape(nnArm1))
        nnArmBA = projectNNAlongArm(nnAllTM[:,:,2,1], nnAllTM[:,:,3,1], uVec[1,:,1])
        #print(np.shape(nnArm2))
        nnITFB = nnArmBC - nnArmBA

        # do the same for interferomter C
        # now use the X,Y components of nnAllTM and tm locations to project
        nnArmCA = projectNNAlongArm(nnAllTM[:,:,0,2], nnAllTM[:,:,1,2], uVec[0,:,2])
        #print(np.shape(nnArm1))
        nnArmCB = projectNNAlongArm(nnAllTM[:,:,2,2], nnAllTM[:,:,3,2], uVec[1,:,2])
        #print(np.shape(nnArm2))
        nnITFC = nnArmCA - nnArmCB

        nullITF = nnITFA + nnITFB + nnITFC

        # get the normalized cross-correlation
        autoITFA = autoITFA + (np.abs(nnITFA))**2
        autoITFB = autoITFB + (np.abs(nnITFB))**2
        autoITFC = autoITFC + (np.abs(nnITFC))**2
        autoNull = autoNull + (np.abs(nullITF))**2

        crossITFANull = crossITFANull + nnITFA*np.conj(nullITF)
        crossITFBNull = crossITFBNull + nnITFB*np.conj(nullITF)
        crossITFCNull = crossITFCNull + nnITFC*np.conj(nullITF)
        crossITFAB = crossITFAB + nnITFA*np.conj(nnITFB)
        crossITFAC = crossITFAC + nnITFA*np.conj(nnITFC)
        
        nnFullITFA  = nnFullITFA + (np.abs(nnITFA))**2
        nnFullITFB  = nnFullITFB + (np.abs(nnITFB))**2
        nnFullITFC  = nnFullITFC + (np.abs(nnITFC))**2
        nullFullITF = nullFullITF + (np.abs(nullITF)**2)

    nnOutA = np.sqrt(nnFullITFA/nRea)
    nnOutB = np.sqrt(nnFullITFB/nRea)
    nnOutC = np.sqrt(nnFullITFC/nRea)
    nullOut = np.sqrt(nullFullITF/nRea)

    ccITFANull = crossITFANull/np.sqrt(autoITFA*autoNull)
    ccITFBNull = crossITFBNull/np.sqrt(autoITFB*autoNull)
    ccITFCNull = crossITFCNull/np.sqrt(autoITFC*autoNull)
    ccITFAB = crossITFAB/np.sqrt(autoITFA*autoITFB)
    ccITFAC = crossITFAC/np.sqrt(autoITFA*autoITFC)
    
    
    return nnOutA, nnOutB, nnOutC, nullOut, ccITFANull, ccITFBNull, ccITFCNull, ccITFAB, ccITFAC

def getSingleITFStrain(itfName, configVertA, configVertB, configVertC):
    # returns the strain corresponding a itf given by itfName
    # define an empty lits of the test-mass names
    
    configAll = []
    configAll.append(configVertA)
    configAll.append(configVertB)
    configAll.append(configVertC)

    freqOut, _, _ = simDisp.getFreqGrid(configAll[0].tMax, configAll[0].nSamp, 
                                                      configAll[0].fMin, configAll[0].fMax, configAll[0].df)
    
    if(itfName == 'A'):
        tmList = ['itmAB','etmAB','itmAC','etmAC']
    elif(itfName == 'B'):
        tmList = ['itmBC', 'etmBC', 'itmBA', 'etmBA']
    elif(itfName == 'C'):
        tmList = ['itmCA','etmCA','itmCB','etmCB']

    nF = len(freqOut)
    # get the 4x2 tmLoc array
    nTM = len(tmList)
    tmLoc = np.zeros((nTM,3))
    for i in range(0,nTM):
        tmLoc[i,:] = getSingleTM(itfName, tmList[i], configVertA, configVertB, configVertC)

    # extract the unit vectors along the arms
    uVec = np.zeros((2,2))
    uVec[0,:] = unitVec(tmLoc[0,0:2], tmLoc[1,0:2])
    uVec[1,:] = unitVec(tmLoc[2,0:2], tmLoc[3,0:2])
    
    # read the NN acceleration per test-mass over all realizations and do the projection
    tmIndAll = np.zeros((nTM,),dtype=int); vertIndAll = np.zeros((nTM,),dtype=int)
    reaPathAll = []
    nReaAll = np.zeros((nTM,))
    for i in range(0,nTM):
        _, tmIndAll[i], _, vertIndAll[i] = mapTM(itfName, tmList[i])
        reaPathAll.append(configAll[int(vertIndAll[i])].outDispPathRea) 
        nReaAll[i] = configAll[int(vertIndAll[i])].nRea
    # extract the file path for reading realizations
    nRea = int(min(nReaAll))

    # loop over realizations
    nnFullITF = np.zeros((nF,))
    for i in range(0,nRea):
        fName = 'NNFullRea' + str(i) + '.npz'
        nnAllTM = np.zeros((nF,3,nTM),dtype = np.complex128)
        for tmNo in range(0,nTM):
            data = np.load(os.path.join(reaPathAll[tmNo],fName))
            nnAllTM[:,:,tmNo] = data['ITot'][:,:,tmIndAll[tmNo]]
        
        # now use the X,Y components of nnAllTM and tm locations to project
        nnArm1 = projectNNAlongArm(nnAllTM[:,:,0], nnAllTM[:,:,1], uVec[0,:])
        #print(np.shape(nnArm1))
        nnArm2 = projectNNAlongArm(nnAllTM[:,:,2], nnAllTM[:,:,3], uVec[1,:])
        #print(np.shape(nnArm2))
        nnArm12 = nnArm1 - nnArm2

        nnFullITF  = nnFullITF + (np.abs(nnArm12))**2

    nnOut = np.sqrt(nnFullITF/nRea)

    return nnOut

def getSingleTMITFStrain(itfName, tmName, configVertA, configVertB, configVertC):
    # very similar to teh function getSingleITFStrain
    # but it does not calculate the differential displacements
    # it just uses one test-mass given by tmName and assuming all four test-masses
    # are uncorrelated, multiplies single testmass acceleration by 2 (added in quadrature)
    configAll = []
    configAll.append(configVertA)
    configAll.append(configVertB)
    configAll.append(configVertC)

    freqOut, _, _ = simDisp.getFreqGrid(configAll[0].tMax, configAll[0].nSamp, 
                                                      configAll[0].fMin, configAll[0].fMax, configAll[0].df)
    
    nF = len(freqOut)
    
    tmLoc = getSingleTM(itfName, tmName, configVertA, configVertB, configVertC)
    
    # read the NN acceleration per test-mass over all realizations
    _, tmIndAll, _, vertIndAll = mapTM(itfName, tmName)
    reaPathAll = configAll[int(vertIndAll)].outDispPathRea 
    nRea = configAll[int(vertIndAll)].nRea

    # loop over realizations
    nnFullITF = np.zeros((nF,2))
    for i in range(0,nRea):
        fName = 'NNFullRea' + str(i) + '.npz'
        data = np.load(os.path.join(reaPathAll,fName))
        nnAllTM = data['ITot'][:,:,tmIndAll]
        
        # multiply by 4 for uncorrelated behavior

        nnFullITF  = nnFullITF + 4*(np.abs(nnAllTM[:,0:2]))**2

    nnOut = np.sqrt(nnFullITF/nRea)
    return nnOut

def NNProjectToET(nnITFA, nnITFB, nnITFC, nullITF, freqOut, config, L = 10000):
    """
    function converts the NN acceleration ASD to strain ASD
    return the ET design sensitivity as well for comparing the projection
    scaledNN -> nFx3 array where the Newtonian acceleration have been scaled with site observed
    displacements
    freqOut -> array((nF,)) like
    config is used to get the path to the ET design curve
    L = 10000 is set to default as the interferometer arm length
    corrFactor = 2 as default for the contribution of 4 test-masses which is assumed to be
    independent and added in quadrature, so ASD the factor is sqrt(4) 
    """
    # first load the design sensitivity
    ETD = loadmat(os.path.join(config.etdPath,'ETD.mat'))
    ETDOut = ETD['ETD']
    nFreq = len(freqOut)

    nnProj = np.zeros((nFreq,3))
    nnProj[:,0] = nnITFA/(L*4*(np.pi**2)*(freqOut**2))
    nnProj[:,1] = nnITFB/(L*4*(np.pi**2)*(freqOut**2))
    nnProj[:,2] = nnITFC/(L*4*(np.pi**2)*(freqOut**2))
    nullProj = nullITF/(L*4*(np.pi**2)*(freqOut**2))

    return nnProj, nullProj, ETDOut