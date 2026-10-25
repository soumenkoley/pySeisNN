#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt

def main():
    # Define the velocity model in km and km/s
    thickness = np.array([10000.0, 0.0]) # units in meters
    vs = np.array([2000.0, 2500.0]) # units in m/s
    vp = np.array([4000.0, 5000.0]) # units in m/s
    rho = np.array([2465.0, 2606.8]) # units in kg/m^3
    fMin = 2.0; fMax = 8.0; # units in Hz
    lambdaFrac = 1/3; # fraction
    lambdaRes = 6; #must be greater than 4
    xMin = -2000.0; xMax = 2000.0; # minimum and maximum of the simulation domain in X-direction (EW)
    yMin = -2000.0; yMax = 2000.0; # maximum and minimum of the simulation domain in Y-direction (NS)
    zMin = 0.0; zMax = 4000.0; # maximum and minimum of the simulation domain in Z-direction (depth)
    domXYBounds = (xMin,xMax,yMin,yMax);
    cubeC = 250.0; rCavity = 20.0; 
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
            return 11
        elif intersectionType in ['cut_top', 'cut_bottom']:
            return 10
        elif intersectionType == 'cut_middle':
            return 9
        else:
            # additional logic if needed, check later
            return 1
     
    def updateCubeInteraction(self, cubeTop, cubeBot):
        #Set cube interaction info: intersection type + block count."""
        self.intersectionType = self.classifyIntersection(cubeTop, cubeBot)
        self.nBlocks = self.getNBlocks(self.intersectionType)

    def generateBlocks(self, xMin, xMax, yMin, yMax, zC, ax, ay, az, domXYBounds):
        # xMin, xMax are the minimum and the maximum extent of the simulation domain in X-direction
        # yMin, yMax are the minimum and the maximum extent of the simulation domain in Y-direction
        # zC is the depth of the center of the cavity
        # zS is half the side length of the inner cube
        xC = 0.0; yC = 0.0
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
            xL, xR = xC - ax, xC + ax
            yB, yT = yC - ay, yC + ay
            zL, zU = zC - az, zC + az

            xIntervals = [(xMin, xL), (xL, xR), (xR, xMax)]
            yIntervals = [(yMin, yB), (yB, yT), (yT, yMax)]
            zIntervals_center = [(self.zTop, zL), (zL, zU), (zU, self.zBot)]

            for ix, (x1, x2) in enumerate(xIntervals):
                for iy, (y1, y2) in enumerate(yIntervals):

                    # Central tile
                    if ix == 1 and iy == 1:
                        for z1, z2 in zIntervals_center:
                            spaceType = 'uniform' if (z1 == zL and z2 == zU) else 'lgwt'
                            interpType = 1 if(z1 ==zL and z2 == zU) else 2

                            block = Block(xMin=x1, xMax=x2, yMin=y1, yMax=y2,zMin=z1, zMax=z2,
                                          vP=self.vP, vS=self.vS, rho=self.rho,
                                          spaceType=spaceType, interpType=interpType,domainBounds=domXYBounds,
                                          domainZBounds=(self.zTop, self.zBot)
                                          )
                            block.buildVerticalFaces()
                            block.buildHorizontalFaces()
                            self.blocks.append(block)

                    # Outer tiles
                    else:
                        block = Block(xMin=x1, xMax=x2, yMin=y1, yMax=y2, zMin=self.zTop, zMax=self.zBot,
                                      vP=self.vP, vS=self.vS, rho=self.rho, spaceType='lgwt', interpType=2,
                                      domainBounds=domXYBounds,domainZBounds=(self.zTop, self.zBot)
                                      )
                        block.buildVerticalFaces()
                        block.buildHorizontalFaces()
                        self.blocks.append(block)

        if self.intersectionType == 'cut_top':
            # cavity bounds
            cubeTop = zC - az
            cubeBot = zC + az   # not directly used here

            # symmetric x–y tiling
            xL, xR = xC-ax, xC+ax
            yB, yT = yC-ay, yC+ay

            xIntervals = [(xMin, xL), (xL, xR), (xR, xMax)]
            yIntervals = [(yMin, yB), (yB, yT), (yT, yMax)]

            for ix, (x1, x2) in enumerate(xIntervals):
                for iy, (y1, y2) in enumerate(yIntervals):

                    is_center = (ix == 1 and iy == 1)

                    if is_center:
                        # central tile split into TWO z blocks
                        zIntervals = [(self.zTop, cubeTop), (cubeTop, self.zBot)]

                        for z1, z2 in zIntervals:
                            if z2 <= z1:
                                continue  # skip degenerate blocks

                            spaceType = "uniform" if (z1 == cubeTop) else "lgwt"
                            interpType = 1 if(z1==cubeTop) else 2

                            block = Block(xMin=x1, xMax=x2, yMin=y1, yMax=y2, zMin=z1, zMax=z2,
                                          vP=self.vP, vS=self.vS, rho=self.rho, spaceType=spaceType,
                                          interpType=interpType, domainBounds=domXYBounds,
                                          domainZBounds=(self.zTop, self.zBot)
                                          )
                            block.buildVerticalFaces()
                            block.buildHorizontalFaces()
                            self.blocks.append(block)

                    else:
                        # outer tiles: full z, always lgwt
                        block = Block(xMin=x1, xMax=x2, yMin=y1, yMax=y2, zMin=self.zTop, zMax=self.zBot,
                                      vP=self.vP, vS=self.vS, rho=self.rho, spaceType="lgwt", interpType=2,
                                      domainBounds=domXYBounds, domainZBounds=(self.zTop, self.zBot)
                                      )
                        block.buildVerticalFaces()
                        block.buildHorizontalFaces()
                        self.blocks.append(block)

        if self.intersectionType == "cut_bottom":
            # cavity bounds
            cubeTop = zC - az
            cubeBot = zC + az

            # symmetric x–y tiling
            xL, xR = -ax, ax
            yB, yT = -ay, ay

            xIntervals = [(xMin, xL), (xL, xR), (xR, xMax)]
            yIntervals = [(yMin, yB), (yB, yT), (yT, yMax)]

            for ix, (x1, x2) in enumerate(xIntervals):
                for iy, (y1, y2) in enumerate(yIntervals):

                    is_center = (ix == 1 and iy == 1)

                    if is_center:
                        # central tile split into TWO z blocks
                        zIntervals = [(self.zTop, cubeBot),(cubeBot, self.zBot)]

                        for z1, z2 in zIntervals:
                            if z2 <= z1:
                                continue  # skip degenerate blocks

                            spaceType = "uniform" if (z2 == cubeBot) else "lgwt"
                            interpType = 1 if(z2 == cubeBot) else 2

                            block = Block(xMin=x1, xMax=x2, yMin=y1, yMax=y2, zMin=z1, zMax=z2,
                                          vP=self.vP, vS=self.vS, rho=self.rho, spaceType=spaceType, interpType = 2,
                                          domainBounds=domXYBounds, domainZBounds=(self.zTop, self.zBot)
                                          )
                            block.buildVerticalFaces()
                            block.buildHorizontalFaces()
                            self.blocks.append(block)

                    else:
                        # outer tiles: full z, always lgwt
                        block = Block(xMin=x1, xMax=x2, yMin=y1, yMax=y2, zMin=self.zTop, zMax=self.zBot,
                                      vP=self.vP, vS=self.vS, rho=self.rho, spaceType="lgwt", interpType = 2,
                                      domainBounds=domXYBounds, domainZBounds=(self.zTop, self.zBot)
                                      )
                        block.buildVerticalFaces()
                        block.buildHorizontalFaces()
                        self.blocks.append(block)

        if self.intersectionType == 'cut_middle':
            cubeTop = zC - az
            cubeBot = zC + az

            # symmetric x–y tiling (centered cavity at xC=yC=0; generalize if needed)
            xL, xR = xC-ax, xC+ax #change this to x and y values
            yB, yT = yC-ay, yC + ay

            xIntervals = [(xMin, xL), (xL, xR), (xR, xMax)]
            yIntervals = [(yMin, yB), (yB, yT), (yT, yMax)]

            for ix, (x1, x2) in enumerate(xIntervals):
                for iy, (y1, y2) in enumerate(yIntervals):

                    is_center = (ix == 1 and iy == 1)

                    spaceType = "uniform" if is_center else "lgwt"
                    interpType = 1 if(is_center) else 2

                    block = Block(xMin=x1, xMax=x2, yMin=y1, yMax=y2, zMin=self.zTop, zMax=self.zBot,
                                  vP=self.vP, vS=self.vS, rho=self.rho, spaceType=spaceType, interpType=interpType,
                                  domainBounds=domXYBounds, domainZBounds=(self.zTop, self.zBot)
                                  )
                    block.buildVerticalFaces()
                    block.buildHorizontalFaces()
                    self.blocks.append(block)

    def getBlock(self, i):
        return self.blocks[i];

    def __repr__(self):
        return f"<Layer z: {self.zTop}-{self.zBot}, type: {self.intersectionType}, nblocks: {self.nBlocks}>";
        
    def describe(self):
        print(f"Layer {self.zTop} to {self.zBot} with {self.getNBlocks()} blocks");

def partitionLayers(z_interfaces, vp, vs, rho, cubeTop, cubeBot, max_host_thickness=500.0, tol=1e-9):
    """
    Only splits layers that overlap the cavity band [cubeTop, cubeBot].
    For each overlapping layer thicker than max_host_thickness:
      - create ONE sublayer window of thickness <= max_host_thickness that fully contains [cubeTop,cubeBot]
      - keep the rest of that layer untouched (no further subdivision)
    Non-overlapping layers remain unchanged.

    z is depth-positive.

    Returns refined (z_new, vp_new, vs_new, rho_new).
    """

    z = np.asarray(z_interfaces, float)
    vp = np.asarray(vp, float)
    vs = np.asarray(vs, float)
    rho = np.asarray(rho, float)

    if len(z) < 2:
        raise ValueError("z_interfaces must have length >= 2")
    if len(vp) != len(z) - 1 or len(vs) != len(z) - 1 or len(rho) != len(z) - 1:
        raise ValueError("vp/vs/rho must have length len(z_interfaces)-1")

    cav_span = cubeBot - cubeTop
    if max_host_thickness + tol < cav_span:
        raise ValueError(
            f"max_host_thickness ({max_host_thickness}) must be >= cavity span ({cav_span})"
        )

    def overlaps_cavity(z0, z1):
        # overlap with [cubeTop, cubeBot]
        return (z1 > cubeTop + tol) and (z0 < cubeBot - tol)

    z_new = [z[0]]
    vp_new, vs_new, rho_new = [], [], []

    for i in range(len(z) - 1):
        z0, z1 = z[i], z[i+1]
        dz = z1 - z0

        # If it doesn't overlap cavity OR already short enough, keep as is
        if (not overlaps_cavity(z0, z1)) or (dz <= max_host_thickness + tol):
            z_new.append(z1)
            vp_new.append(vp[i]); vs_new.append(vs[i]); rho_new.append(rho[i])
            continue

        # We need to split this hosting layer.
        # Choose a window [a,b] of length max_host_thickness that contains [cubeTop,cubeBot]
        # and lies inside [z0,z1].

        # Start with a centered window around cavity band mid
        z_mid = 0.5 * (cubeTop + cubeBot)
        a = z_mid - 0.5 * max_host_thickness
        b = a + max_host_thickness

        # Clamp window into [z0,z1]
        if a < z0:
            a = z0
            b = a + max_host_thickness
        if b > z1:
            b = z1
            a = b - max_host_thickness

        # Now ensure cavity band is inside [a,b]; if not, shift minimally
        if cubeTop < a:
            shift = a - cubeTop
            a -= shift
            b -= shift
        if cubeBot > b:
            shift = cubeBot - b
            a += shift
            b += shift

        # Re-clamp in case of numerical issues
        a = max(z0, a)
        b = min(z1, b)

        # At this point [cubeTop,cubeBot] should lie inside [a,b] (within tol)
        if cubeTop < a - 10*tol or cubeBot > b + 10*tol:
            # Fallback: just take the smallest window that works inside the layer
            a = max(z0, min(cubeTop, z1 - max_host_thickness))
            b = a + max_host_thickness
            b = min(b, z1)
            a = b - max_host_thickness

        # Build up to 3 sublayers: [z0,a], [a,b], [b,z1], skipping degenerate
        cuts = [z0, a, b, z1]
        cuts = [cuts[0]] + [c for c in cuts[1:] if c > cuts[0] + tol]  # basic cleanup
        # Ensure strictly increasing
        cuts_clean = [cuts[0]]
        for c in cuts[1:]:
            if c > cuts_clean[-1] + tol:
                cuts_clean.append(c)

        for j in range(len(cuts_clean) - 1):
            zz0, zz1 = cuts_clean[j], cuts_clean[j+1]
            if zz1 <= zz0 + tol:
                continue
            z_new.append(zz1)
            vp_new.append(vp[i]); vs_new.append(vs[i]); rho_new.append(rho[i])

    # Final cleanup: remove nearly-duplicate interfaces (and remap props by midpoint)
    z_new = np.array(z_new)
    z_clean = [z_new[0]]
    for k in range(1, len(z_new)):
        if not np.isclose(z_new[k], z_clean[-1], atol=tol, rtol=0):
            z_clean.append(z_new[k])
    z_clean = np.array(z_clean)

    vp_out, vs_out, rho_out = [], [], []
    for i in range(len(z_clean) - 1):
        mid = 0.5 * (z_clean[i] + z_clean[i+1])
        j = np.searchsorted(z, mid, side="right") - 1
        j = np.clip(j, 0, len(vp) - 1)
        vp_out.append(vp[j]); vs_out.append(vs[j]); rho_out.append(rho[j])

    return z_clean, np.array(vp_out), np.array(vs_out), np.array(rho_out)

def mapGridSizeToPartition(z_orig_interfaces, gridSize_orig, z_new_interfaces, tol=1e-9):
    """
    z_orig_interfaces: (nOld+1,) original depth interfaces (0, z1, z2, ...)
    gridSize_orig: (nOld,) grid size per original layer
    z_new_interfaces: (nNew+1,) partitioned depth interfaces

    Returns gridSize_new: (nNew,)
    """
    z0 = np.asarray(z_orig_interfaces, float)
    gs0 = np.asarray(gridSize_orig, float)
    z1 = np.asarray(z_new_interfaces, float)

    if len(gs0) != len(z0) - 1:
        raise ValueError("gridSize_orig must have length len(z_orig_interfaces)-1")

    gs_new = np.zeros(len(z1) - 1, float)

    for i in range(len(z1) - 1):
        a, b = z1[i], z1[i+1]
        mid = 0.5 * (a + b)

        j = np.searchsorted(z0, mid, side="right") - 1
        j = np.clip(j, 0, len(gs0) - 1)
        gs_new[i] = gs0[j]

    return gs_new

def createAllLayers(xMin, xMax, yMin, yMax, thickness, zMax, vp, vs, rho, gridSize, cubeTop, cubeBot):
    
    # Create and add multiple layers
    lenThick = len(thickness)
    depths = np.cumsum(thickness[0:(lenThick-1)])
    depths = np.insert(depths,0,0.0,axis=0)
    
    # find the index to insert zMax
    zMaxInd = np.where(depths<zMax)[0]
    
    newDepths = np.append(depths[0:(zMaxInd[-1]+1)],zMax)
    
    print('Layer depths given by user before partitioning = ' + str(newDepths) + ' m')

    newDepths2, vp2, vs2, rho2 = partitionLayers(z_interfaces=newDepths,vp=vp[:len(newDepths)-1],
                                                             vs=vs[:len(newDepths)-1],rho=rho[:len(newDepths)-1],
                                                             cubeTop=cubeTop, cubeBot=cubeBot,max_host_thickness=400.0  # tune
                                                             )
    lenNewDepth = len(newDepths2)
    print('New layer depths after partitioning for computationional speed = ' + str(newDepths2) + ' m')
    layers = []
    for i in range(len(newDepths2)-1):
        layers.append(Layer(xMin=xMin, xMax=xMax, yMin=yMin, yMax=yMax, zTop=newDepths2[i], zBot=newDepths2[i+1],
                            vP=vp2[i], vS=vs2[i], rho=rho2[i]))

    gridSize2 = mapGridSizeToPartition(z_orig_interfaces=newDepths, gridSize_orig=gridSize[:len(newDepths)-1],
                                               z_new_interfaces=newDepths2)

    return layers, vp2, vs2, rho2, gridSize2 
    
if __name__ == "__main__":
    main()