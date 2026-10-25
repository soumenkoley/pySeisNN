#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define the velocity model in km and km/s
    thickness = np.array([20.0, 25.0, 75.0, 180.0, 700.0, 4000.0, 0.0]) # units in meters
    vs = np.array([300.0, 500.0, 1500.0, 2500.0, 3000.0, 3360.0, 3750.0]) # units in m/s
    vp = np.array([1000, 2000.0, 3000.0, 4500.0, 5000.0, 5800.0, 6000.0]) # units in m/s
    rho = np.array([1743.0, 2073.1, 2294.3, 2539.0, 2606.8, 2720.0, 2800.0]) # units in kg/m^3
    fMin = 2.0; fMax = 8.0; # units in Hz
    lambdaFrac = 1/3; # fraction
    lambdaRes = 6; #must be greater than 4
    xMin = -3000.0; xMax = 3000.0; # minimum and maximum of the simulation domain in X-direction (EW)
    yMin = -3000.0; yMax = 3000.0; # maximum and minimum of the simulation domain in Y-direction (NS)
    zMin = 0.0; zMax = 4000.0; # maximum and minimum of the simulation domain in Z-direction (depth)
    domXYBounds = (xMin,xMax,yMin,yMax);
    cubeC = 290.0; rCavity = 20.0; 
    # just make sure zMax never coincides with an actual horizontal interface
    # bug to be fixed later if needed
    cubeS = 2*rCavity;
    cubeTop = cubeC-cubeS; cubeBot = cubeC+cubeS;
    
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
        layers.append(Layer(xMin=xMin, xMax=xMax, yMin= yMin, yMax = yMax, zTop=newDepths[i], zBot=newDepths[i+1], vP=vp[i], vS=vs[i], rho=rho[i]));

    #print(len(layers));

    layer = layers[0]

    layer.updateCubeInteraction(cubeTop, cubeBot)
    layer.generateBlocks(xMin, xMax, yMin, yMax, cubeC, cubeS, domXYBounds)

    print(layer)
    for blk in layer.blocks:
        print(blk)
        for f in blk.horizontalFaces:
            print("   ", f)
            
    """
    for i, layer in enumerate(layers):
        #print(f"Layer {i}:");
        layer.updateCubeInteraction(cubeTop, cubeBot);
        layer.generateBlocks(xMin, xMax, yMin, yMax, cubeC, cubeS, domXYBounds);
        #layer.describe();
        print(layer);
        for blk in layer.blocks:
            print(blk);
            for f in blk.horizontalFaces:
                print("   ", f);
    """
    

class VerticalFace:
    def __init__(self, axis, position, xLim, yLim, zLim, rhoOut, domainBounds = None, isBoundary=False):
        """
        Represents a vertical face (normal to x or y).

        Parameters
        ----------
        axis : str
            Axis normal to the face ('x' or 'y').
        position : float
            Coordinate value along the normal axis where the face lies (e.g. xMin, xMax, etc.).
        xLim : tuple
            (xMin, xMax) range covered by the face in x-direction.
        yLim : tuple
            (yMin, yMax) range covered by the face in y-direction.
        zLim : tuple
            (zMin, zMax) range of the face vertically.
        isBoundary : bool
            True if this face lies on the *outer boundary* of the simulation domain.
        """
        assert axis in ['x', 'y'], "VerticalFace axis must be 'x' or 'y'"
        self.axis = axis
        self.position = position
        self.xLim = xLim
        self.yLim = yLim
        self.zLim = zLim
        self.rhoOut = rhoOut
        self.domainBounds = domainBounds
        self.isBoundary = isBoundary
        self.normVector = self.computeNormalVector()

    def computeNormalVector(self):

        if self.domainBounds is None:
            # fall back to default behavior, warning
            if self.axis == 'x':
                # normal along ±x
                if self.position <0:  # xMin face → -x normal
                    return np.array([-1.0, 0.0, 0.0])
                else:  # xMax face → +x normal
                    return np.array([1.0, 0.0, 0.0])
            elif self.axis == 'y':
                # normal along ±y
                if self.position <0:  # yMin face → -y normal
                    return np.array([0.0, -1.0, 0.0])
                else:  # yMax face → +y normal
                    return np.array([0.0, 1.0, 0.0])
        
        dom_xmin, dom_xmax, dom_ymin, dom_ymax = self.domainBounds

        if self.axis == 'x':
            if np.isclose(self.position, dom_xmin):
                return np.array([-1.0, 0.0, 0.0])   # left boundary
            elif np.isclose(self.position, dom_xmax):
                return np.array([1.0, 0.0, 0.0])    # right boundary
            else:
                # internal x-face → pick convention, say +x
                return np.array([1.0, 0.0, 0.0])

        elif self.axis == 'y':
            if np.isclose(self.position, dom_ymin):
                return np.array([0.0, -1.0, 0.0])   # bottom (south)
            elif np.isclose(self.position, dom_ymax):
                return np.array([0.0, 1.0, 0.0])    # top (north)
            else:
                return np.array([0.0, 1.0, 0.0])

    def __repr__(self):
        tag = 'BOUNDARY' if self.isBoundary else 'internal'
        return (f"<VerticalFace axis={self.axis}, pos={self.position}, "
                f"zRange={self.zLim}, norm={self.normVector}, {tag}>")

class HorizontalFace:
    def __init__(self, position, xLim, yLim, zLim, normVector=None, isBoundary=False):
        """
        Represents a horizontal face (normal to z).

        Parameters
        ----------
        position : float
            z-coordinate of the face (e.g., zMin or zMax)
        xLim : tuple
            (xMin, xMax) range of the face in x-direction
        yLim : tuple
            (yMin, yMax) range of the face in y-direction
        zLim : tuple
            (zMin, zMax) vertical extent of the block
        normVector : np.ndarray
            Normal vector (e.g., [0,0,1] or [0,0,-1])
        isBoundary : bool
            True if the face coincides with the top or bottom of the simulation domain
        """
        self.position = position
        self.xLim = xLim
        self.yLim = yLim
        self.zLim = zLim
        # assign default if not provided
        self.normVector = normVector if normVector is not None else np.array([0.0, 0.0, 1.0])
        self.isBoundary = isBoundary

    def __repr__(self):
        tag = 'BOUNDARY' if self.isBoundary else 'internal'
        return (f"<HorizontalFace z={self.position}, norm={self.normVector}, {tag}>")
        
class Block:
    def __init__(self, xMin, xMax, yMin, yMax, zMin, zMax, vP, vS, rho,spaceType,interpType,domainBounds=None,domainZBounds=None):
        self.xMin = xMin;
        self.xMax = xMax;
        self.yMin = yMin;
        self.yMax = yMax;
        self.zMin = zMin;
        self.zMax = zMax;
        self.vP = vP;
        self.vS = vS;
        self.rho = rho;
        self.spaceType = spaceType;
        self.interpType = interpType;
        self.domainBounds = domainBounds;
        self.domainZBounds = domainZBounds
        self.verticalFaces = self.buildVerticalFaces();
        self.horizontalFaces = self.buildHorizontalFaces();

    def buildVerticalFaces(self):
        """Generate vertical faces for the block and mark boundary ones."""
        faces = []
        if self.domainBounds is None:
            dom_xMin = dom_xMax = dom_yMin = dom_yMax = None
        else:
            dom_xMin, dom_xMax, dom_yMin, dom_yMax = self.domainBounds

        # x-normal faces
        for xVal in [self.xMin, self.xMax]:
            isBnd = (xVal == dom_xMin or xVal == dom_xMax)
            face = VerticalFace(
                axis='x',
                position=xVal,
                xLim=(xVal, xVal),
                yLim=(self.yMin, self.yMax),
                zLim=(self.zMin, self.zMax),
                rhoOut = self.rho,
                domainBounds=self.domainBounds,
                isBoundary=isBnd
            )
            faces.append(face)

        # y-normal faces
        for yVal in [self.yMin, self.yMax]:
            isBnd = (yVal == dom_yMin or yVal == dom_yMax)
            face = VerticalFace(
                axis='y',
                position=yVal,
                xLim=(self.xMin, self.xMax),
                yLim=(yVal, yVal),
                zLim=(self.zMin, self.zMax),
                rhoOut = self.rho,
                domainBounds=self.domainBounds,
                isBoundary=isBnd
            )
            faces.append(face)

        return faces;

    
    def buildHorizontalFaces(self):
        """Generate the two horizontal faces (+z, -z) and tag boundary faces."""
        faces = []

        # Use the full domain Z-limits if available
        zDom = self.domainZBounds

        for zVal, norm in zip([self.zMin, self.zMax],
                          [np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, -1.0])]):
            isBnd = False
            if zDom is not None:
                isBnd = np.isclose(zVal, zDom[0]) or np.isclose(zVal, zDom[1])

            faces.append(
                HorizontalFace(
                    position=zVal,
                    xLim=(self.xMin, self.xMax),
                    yLim=(self.yMin, self.yMax),
                    zLim=(self.zMin, self.zMax),
                    normVector=norm,
                    isBoundary=isBnd
                )
            )

        return faces

    
    def containsPoint(self, x, y, z):
        return (self.xMin <= x <= self.xMax and
                self.yMin <= y <= self.yMax and
                self.zMin <= z <= self.zMax);

    def __repr__(self):
        return (f"<Block x:[{self.xMin},{self.xMax}], y:[{self.yMin},{self.yMax}], "
                f"z:[{self.zMin},{self.zMax}] spaceT:[{self.spaceType}] intpType:[{self.interpType}] >")


class Layer:
    def __init__(self, xMin, xMax, yMin, yMax, zTop, zBot, vP, vS, rho):
        self.zTop = zTop;
        self.zBot = zBot;
        self.xMax = xMax;
        self.xMin = xMin;
        self.yMin = yMin;
        self.yMax = yMax;
        self.vP = vP;
        self.vS = vS;
        self.rho = rho;
        self.nBlocks = 1;
        self.intersectionType = None;
        self.blocks = [];

    def addBlock(self, xMin, xMax, yMin, yMax, zMin, zMax):
        block = Block(xMin, xMax, yMin, yMax, zMin, zMax, self.vP, self.vS, self.rho);
        self.blocks.append(block);
    
    def classifyIntersection(self, cubeTop, cubeBot):
        #Classify how the cube intersects with this layer (z-down positive)."""
        if cubeBot <= self.zTop or cubeTop >= self.zBot:
            return 'no_intersection';
        if cubeTop >= self.zTop and cubeBot <= self.zBot:
            return 'contains';
        if cubeTop < self.zTop and cubeBot > self.zTop and cubeBot <= self.zBot:
            return 'cut_bottom';
        if cubeTop >= self.zTop and cubeTop < self.zBot and cubeBot > self.zBot:
            return 'cut_top';
        if cubeTop < self.zTop and cubeBot > self.zBot:
            return 'cut_middle';
        return 'other'; # should never be reached, tight logic
    
    def getNBlocks(self, intersectionType):
        #Determine number of blocks based on intersection type."""
        if intersectionType == 'no_intersection':
            return 1
        elif intersectionType == 'contains':
            return 7
        elif intersectionType in ['cut_top', 'cut_bottom']:
            return 6
        elif intersectionType == 'cut_middle':
            return 5
        else:
            # additional logic if needed, check later
            return 1
     
    def updateCubeInteraction(self, cubeTop, cubeBot):
        #Set cube interaction info: intersection type + block count."""
        self.intersectionType = self.classifyIntersection(cubeTop, cubeBot)
        self.nBlocks = self.getNBlocks(self.intersectionType)

    def generateBlocks(self, xMin, xMax, yMin, yMax, zC, zS,domXYBounds):
        # xMin, xMax are the minimum and the maximum extent of the simulation domain in X-direction
        # yMin, yMax are the minimum and the maximum extent of the simulation domain in Y-direction
        # zC is the depth of the center of the cavity
        # zS is half the side length of the inner cube
        
        self.blocks = []  # Clear previous

        # Example: add the cube part (always present if intersects)
        if self.intersectionType == 'no_intersection':
            block1 = Block(
                xMin=xMin, xMax=xMax,yMin=yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType='lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block1.buildVerticalFaces();
            block1.buildHorizontalFaces();
            self.blocks.append(block1);

        if self.intersectionType == 'contains':
            # block -x
            block1 = Block(
                xMin = xMin, xMax = -zS, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block1.buildVerticalFaces();
            block1.buildHorizontalFaces();
            self.blocks.append(block1);
            
            # block +x
            block2 = Block(
                xMin = zS, xMax = xMax, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block2.buildVerticalFaces();
            block2.buildHorizontalFaces();
            self.blocks.append(block2);

            # block -z
            block3 = Block(
                xMin = -zS, xMax = zS, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=(zC-zS),
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block3.buildVerticalFaces();
            block3.buildHorizontalFaces();
            self.blocks.append(block3);

            # block +z
            block4 = Block(
                xMin = -zS, xMax = zS, yMin = yMin, yMax=yMax,
                zMin=(zC+zS), zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block4.buildVerticalFaces();
            block4.buildHorizontalFaces();
            self.blocks.append(block4);

            # block -y
            block5 = Block(
                xMin = -zS, xMax = zS, yMin = yMin, yMax=-zS,
                zMin=(zC-zS), zMax=(zC+zS),
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block5.buildVerticalFaces();
            block5.buildHorizontalFaces();
            self.blocks.append(block5);

            # block +y
            block6 = Block(
                xMin = -zS, xMax = zS, yMin = zS, yMax=yMax,
                zMin=(zC-zS), zMax=(zC+zS),
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block6.buildVerticalFaces();
            block6.buildHorizontalFaces();
            self.blocks.append(block6);

            # block center
            block7 = Block(
                xMin = -zS, xMax = zS, yMin = -zS, yMax=zS,
                zMin=(zC-zS), zMax=(zC+zS),
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'uniform', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block7.buildVerticalFaces();
            block7.buildHorizontalFaces();
            self.blocks.append(block7);

        if self.intersectionType == 'cut_top':
            # block -x
            block1 = Block(
                xMin = xMin, xMax = -zS, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block1.buildVerticalFaces();
            block1.buildHorizontalFaces();
            self.blocks.append(block1);

            # block +x
            block2 = Block(
                xMin = zS, xMax = xMax, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block2.buildVerticalFaces();
            block2.buildHorizontalFaces();
            self.blocks.append(block2);

            # block -z
            block3 = Block(
                xMin = -zS, xMax = zS, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=(zC-zS),
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block3.buildVerticalFaces();
            block3.buildHorizontalFaces();
            self.blocks.append(block3);

            # note +z is absent
            # block -y
            block5 = Block(
                xMin = -zS, xMax = zS, yMin = yMin, yMax=-zS,
                zMin=(zC-zS), zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block5.buildVerticalFaces();
            block5.buildHorizontalFaces();
            self.blocks.append(block5);

            # block +y
            block6 = Block(
                xMin = -zS, xMax = zS, yMin = zS, yMax=yMax,
                zMin=(zC-zS), zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block6.buildVerticalFaces();
            block6.buildHorizontalFaces();
            self.blocks.append(block6);

            # block center
            block7 = Block(
                xMin = -zS, xMax = zS, yMin = -zS, yMax=zS,
                zMin=(zC-zS), zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'uniform', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block7.buildVerticalFaces();
            block7.buildHorizontalFaces();
            self.blocks.append(block7);

        
        if self.intersectionType == 'cut_bottom':
            # block -x
            block1 = Block(
                xMin = xMin, xMax = -zS, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1,domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block1.buildVerticalFaces();
            block1.buildHorizontalFaces();
            self.blocks.append(block1);

            # block +x
            block2 = Block(
                xMin = zS, xMax = xMax, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block2.buildVerticalFaces();
            block2.buildHorizontalFaces();
            self.blocks.append(block2);

            # block +z
            block3 = Block(
                xMin = -zS, xMax = zS, yMin = yMin, yMax=yMax,
                zMin=(zC+zS), zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block3.buildVerticalFaces();
            block3.buildHorizontalFaces();
            self.blocks.append(block3);

            # note -z is absent
            # block -y
            block5 = Block(
                xMin = -zS, xMax = zS, yMin = yMin, yMax=-zS,
                zMin=self.zTop, zMax=(zC+zS),
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block5.buildVerticalFaces();
            block5.buildHorizontalFaces();
            self.blocks.append(block5);

            # block +y
            block6 = Block(
                xMin = -zS, xMax = zS, yMin = zS, yMax=yMax,
                zMin=self.zTop, zMax=(zC+zS),
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block6.buildVerticalFaces();
            block6.buildHorizontalFaces();
            self.blocks.append(block6);

            # block center
            block7 = Block(
                xMin = -zS, xMax = zS, yMin = -zS, yMax=zS,
                zMin=self.zTop, zMax=(zC+zS),
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'uniform', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block7.buildVerticalFaces();
            block7.buildHorizontalFaces();
            self.blocks.append(block7);

        if self.intersectionType == 'cut_middle':
            # block -x
            block1 = Block(
                xMin = xMin, xMax = -zS, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block1.buildVerticalFaces();
            block1.buildHorizontalFaces();
            self.blocks.append(block1);

            # block +x
            block2 = Block(
                xMin = zS, xMax = xMax, yMin = yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block2.buildVerticalFaces();
            block2.buildHorizontalFaces();
            self.blocks.append(block2);

            # note -z and +z is absent
            
            # block -y
            block5 = Block(
                xMin = -zS, xMax = zS, yMin = yMin, yMax=-zS,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block5.buildVerticalFaces();
            block5.buildHorizontalFaces();
            self.blocks.append(block5);

            # block +y
            block6 = Block(
                xMin = -zS, xMax = zS, yMin = zS, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'lgwt', interpType = 1,domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block6.buildVerticalFaces();
            block6.buildHorizontalFaces();
            self.blocks.append(block6);

            # block center
            block7 = Block(
                xMin = -zS, xMax = zS, yMin = -zS, yMax=zS,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho, spaceType = 'uniform', interpType = 2, domainBounds=domXYBounds,
                domainZBounds=(self.zTop, self.zBot)
            )
            block7.buildVerticalFaces();
            block7.buildHorizontalFaces();
            self.blocks.append(block7);

        # TODO: Add other surrounding blocks as needed based on intersection_type
    def getBlock(self, i):
        return self.blocks[i];

    def __repr__(self):
        return f"<Layer z: {self.zTop}-{self.zBot}, type: {self.intersectionType}, nblocks: {self.nBlocks}>";
        
    def describe(self):
        print(f"Layer {self.zTop} to {self.zBot} with {self.getNBlocks()} blocks");


if __name__ == "__main__":
    main()
