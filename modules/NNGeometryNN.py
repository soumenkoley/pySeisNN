#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Circle, Polygon

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

class RefineBox:
    """
    Axis-aligned refinement box used for block generation.

    This is not the true cavity geometry.
    It is the rectangular refinement region around a cavity or merged cavities.
    """

    def __init__(self, xC, yC, zC, ax, ay, az, sourceNames=None):
        self.xC = xC
        self.yC = yC
        self.zC = zC
        self.ax = ax
        self.ay = ay
        self.az = az
        self.sourceNames = sourceNames[:] if sourceNames is not None else []

    @property
    def xMin(self):
        return self.xC - self.ax

    @property
    def xMax(self):
        return self.xC + self.ax

    @property
    def yMin(self):
        return self.yC - self.ay

    @property
    def yMax(self):
        return self.yC + self.ay

    @property
    def zMin(self):
        return self.zC - self.az

    @property
    def zMax(self):
        return self.zC + self.az

    def intersectsLayer(self, zTop, zBot, tol=1e-9):
        return not (self.zMax <= zTop + tol or self.zMin >= zBot - tol)

    def __repr__(self):
        src = ",".join(self.sourceNames) if self.sourceNames else "None"
        return (f"<RefineBox x:[{self.xMin},{self.xMax}], "
                f"y:[{self.yMin},{self.yMax}], "
                f"z:[{self.zMin},{self.zMax}], "
                f"sources:[{src}]>")

def makeSphereRefineBox(xC, yC, zC,rCavity,wallRefine,domainZBounds,name="sphere"):

    # ------------------------------------------------------
    # Requested refinement extent
    # ------------------------------------------------------
    ax = rCavity + wallRefine
    ay = rCavity + wallRefine

    zMin_req = zC - rCavity - wallRefine
    zMax_req = zC + rCavity + wallRefine

    # ------------------------------------------------------
    # Clip to computational z-domain
    # ------------------------------------------------------
    zDomMin, zDomMax = domainZBounds

    zMin = max(zMin_req, zDomMin)
    zMax = min(zMax_req, zDomMax)

    if zMax <= zMin:
        raise ValueError(
            f"Invalid spherical refinement box after z clipping: "
            f"zMin={zMin}, zMax={zMax}"
        )

    # ------------------------------------------------------
    # Refinement box may no longer be centered at cavity zC
    # ------------------------------------------------------
    zC_ref = 0.5 * (zMin + zMax)
    az = 0.5 * (zMax - zMin)

    return RefineBox(xC=xC,yC=yC,zC=zC_ref,ax=ax,ay=ay,az=az,sourceNames=[name])


def makeCuboidRefineBox(xC, yC, zC,length, breadth, height,angleDeg=0.0,wallRefineWidth=None,
                        domainZBounds=None,name="cuboid"):
    """
    Build an axis-aligned global RefineBox around a rotated cuboid cavity.

    wallRefineWidth must contain the six face-specific refinement
    extensions in the LOCAL cuboid coordinate system:

        {
            'xMin': ...,
            'xMax': ...,
            'yMin': ...,
            'yMax': ...,
            'zMin': ...,
            'zMax': ...
        }
    """

    if wallRefineWidth is None:
        wallRefineWidth = {'xMin': 20.0,'xMax': 20.0,'yMin': 20.0,'yMax': 20.0,
                           'zMin': 20.0,'zMax': 20.0}

    # ------------------------------------------------------
    # Physical half-sizes of cavity in LOCAL coordinates
    # ------------------------------------------------------
    hl = 0.5 * length
    hb = 0.5 * breadth
    hh = 0.5 * height

    # ------------------------------------------------------
    # Expanded bounds in LOCAL cuboid coordinates
    # ------------------------------------------------------
    xLocMin = -hl - wallRefineWidth['xMin']
    xLocMax =  hl + wallRefineWidth['xMax']

    yLocMin = -hb - wallRefineWidth['yMin']
    yLocMax =  hb + wallRefineWidth['yMax']

    # ------------------------------------------------------
    # Local XY corners of expanded refinement rectangle
    # ------------------------------------------------------
    localCorners = np.array([[xLocMin, yLocMin],
                             [xLocMin, yLocMax],
                             [xLocMax, yLocMax],
                             [xLocMax, yLocMin]], 
                             dtype=float)

    # ------------------------------------------------------
    # Rotate LOCAL -> GLOBAL
    # ------------------------------------------------------
    theta = np.deg2rad(angleDeg)

    c = np.cos(theta)
    s = np.sin(theta)

    RlocalToGlobal = np.array([[c, -s],
                               [s,  c]])

    globalCorners = localCorners @ RlocalToGlobal.T

    # Translate to actual cavity center
    globalCorners[:, 0] += xC
    globalCorners[:, 1] += yC

    # ------------------------------------------------------
    # Smallest global axis-aligned box enclosing
    # the expanded rotated rectangle
    # ------------------------------------------------------
    xMin = np.min(globalCorners[:, 0])
    xMax = np.max(globalCorners[:, 0])

    yMin = np.min(globalCorners[:, 1])
    yMax = np.max(globalCorners[:, 1])

    # ------------------------------------------------------
    # z does not rotate
    # ------------------------------------------------------
    zMin = zC - hh - wallRefineWidth['zMin']
    zMax = zC + hh + wallRefineWidth['zMax']
    zDomMin, zDomMax = domainZBounds
    zMin = max(zMin, zDomMin)
    zMax = min(zMax, zDomMax)
    # ------------------------------------------------------
    # Convert bounds to RefineBox center + half-widths
    # ------------------------------------------------------
    xC_ref = 0.5 * (xMin + xMax)
    yC_ref = 0.5 * (yMin + yMax)
    zC_ref = 0.5 * (zMin + zMax)

    ax_ref = 0.5 * (xMax - xMin)
    ay_ref = 0.5 * (yMax - yMin)
    az_ref = 0.5 * (zMax - zMin)

    return RefineBox(xC=xC_ref,yC=yC_ref,zC=zC_ref,ax=ax_ref,ay=ay_ref,az=az_ref,
                     sourceNames=[name])


def getHBG(thickness,allCav,gridSize):
    nCav = len(allCav)
    hBG = np.zeros((nCav,))
    lThick = len(thickness)
    layerDepths = np.cumsum(thickness[0:(lThick-1)])

    for i in range(0,nCav):
        dInd = np.where(layerDepths>=allCav[i]['zC'])[0][0]
        hBG[i] = gridSize[dInd]

    return(hBG)

def findTMInCavity(allCav, itmAll):

    nTM = itmAll.shape[0]
    nCav = len(allCav)
    cavOut = []
    for i in range(nCav):
        centerNow = [allCav[i]['xC'],allCav[i]['yC'],allCav[i]['zC']]
        tmOut = []
        for j in range(nTM):
            if(allCav[i]['shape']=='cuboid'):
                pIn, pDist = pointInRotatedCuboid(itmAll[j, :],centerNow, allCav[i]['length'],
                                                      allCav[i]['breadth'],allCav[i]['height'],allCav[i]['angleDeg'])
            elif(allCav[i]['shape']=='sphere'):
                pIn,pDist = pointInSphere(itmAll[j,:], centerNow, allCav[i]['radius'])

            tmOut.append({'tmNo': j,'pIn': pIn,'pDist': pDist})
        cavOut.append({'cavNo': i,'tm': tmOut})
    return cavOut


def getRefineWidths(tmInCavity,allCavities,hBGAll,eta=0.3,minRefine=20.0):
    """
    Compute the required refinement extension outward from the cavity walls.

    The required TM-centered refinement radius is

        rReq = 3 * hBG / eta

    For a cuboidal cavity, the refinement extension is computed
    independently for each of the six cavity faces, using distances
    measured in the local cavity coordinate system.

    For a spherical cavity, a single isotropic wall-refinement
    extension is returned.

    If multiple test masses lie inside the same cavity, the most
    demanding refinement requirement is retained independently for
    each face.

    If no test mass lies inside a cavity, minRefine is used.

    Parameters
    ----------
    tmInCavity : list
        Output from findTMInCavity().

    allCavities : list of dict
        Cavity definitions.

    hBGAll : array-like
        Background grid spacing associated with each cavity.

    eta : float
        Allowed fractional variation of the gravitational kernel.

    minRefine : float
        Minimum refinement extension outward from a cavity wall.

    Returns
    -------
    wallRefineWidth : list

        For a cuboid:

            {
                'xMin': float,
                'xMax': float,
                'yMin': float,
                'yMax': float,
                'zMin': float,
                'zMax': float
            }

        where x/y refer to the LOCAL cuboid axes.

        For a sphere:

            float
    """

    nCav = len(allCavities)

    if len(tmInCavity) != nCav:
        raise ValueError(
            "tmInCavity and allCavities must have the same length."
        )

    if len(hBGAll) != nCav:
        raise ValueError(
            "hBGAll and allCavities must have the same length."
        )

    wallRefineWidth = []

    faces = ('xMin', 'xMax','yMin', 'yMax','zMin', 'zMax')

    for i in range(nCav):

        shape = allCavities[i]['shape'].lower()

        # Required radius measured from the TM
        rReq = 3.0 * hBGAll[i] / eta

        # Test masses lying inside this cavity
        tmInside = [
            tm for tm in tmInCavity[i]['tm']
            if tm['pIn']
        ]

        # ======================================================
        # Cuboidal cavity
        # ======================================================
        if shape == 'cuboid':

            # No TM inside -> minimum geometry refinement
            if len(tmInside) == 0:

                refineNow = {
                    face: float(minRefine)
                    for face in faces
                }

            else:

                refineNow = {}

                for face in faces:

                    requiredWidths = []

                    for tm in tmInside:

                        # Distance from this TM to this cavity face
                        dFace = float(tm['pDist'][face])

                        # Required extension outward from that face
                        widthTM = max(
                            minRefine,
                            rReq - dFace
                        )

                        requiredWidths.append(widthTM)

                    # Must satisfy every TM in the cavity
                    refineNow[face] = max(requiredWidths)

            wallRefineWidth.append(refineNow)

        # ======================================================
        # Spherical cavity
        # ======================================================
        elif shape == 'sphere':

            if len(tmInside) == 0:

                widthNow = float(minRefine)

            else:

                requiredWidths = []

                for tm in tmInside:

                    dWall = float(tm['pDist'])

                    widthTM = max(
                        minRefine,
                        rReq - dWall
                    )

                    requiredWidths.append(widthTM)

                widthNow = max(requiredWidths)

            wallRefineWidth.append(widthNow)

        else:

            raise ValueError(
                f"Unsupported cavity shape: "
                f"{allCavities[i]['shape']}"
            )

    return wallRefineWidth


def pointInRotatedCuboid(point,center,length,breadth,height,angleDeg):
    """
    Check whether a point lies inside a rotated cuboid and, if so,
    return its distances from all six cavity faces.

    Parameters
    ----------
    point : array-like, shape (3,)
        Global coordinates [x, y, z] of the query point.

    center : array-like, shape (3,)
        Global coordinates [xc, yc, zc] of the cuboid center.

    length : float
        Cuboid size along its local x-axis.

    breadth : float
        Cuboid size along its local y-axis.

    height : float
        Cuboid size along its local z-axis.

    angleDeg : float
        Counterclockwise rotation of the cuboid local x-axis
        relative to the global x-axis, about global z.

    Returns
    -------
    inside : bool
        True if point lies inside or on the cavity boundary.

    distances : dict or None
        Distances from the point to the six faces, measured
        perpendicular to each face. None if point is outside.

    localPoint : ndarray
        Point coordinates in the cavity's local coordinate system.
    """

    pointUse = np.array([point[0],point[1],-point[2]])

    print(pointUse)
    center = np.asarray(center, dtype=float)

    # Translate point so cavity center is at origin
    p = pointUse - center

    # Rotate GLOBAL coordinates into LOCAL cavity coordinates
    theta = np.deg2rad(angleDeg)

    c = np.cos(theta)
    s = np.sin(theta)

    RglobalToLocal = np.array([
        [ c,  s, 0.0],
        [-s,  c, 0.0],
        [0.0, 0.0, 1.0]
    ])

    pLocal = RglobalToLocal @ p

    x, y, z = pLocal

    hx = length  / 2.0
    hy = breadth / 2.0
    hz = height  / 2.0

    # Check whether point lies inside cuboid
    inside = ((-hx <= x <= hx) and(-hy <= y <= hy) and(-hz <= z <= hz))

    if not inside:
        return False, None

    # Distances to the six faces
    distances = {"xMin": x + hx,"xMax": hx - x,"yMin": y + hy,"yMax": hy - y,
                 "zMin": z + hz,"zMax": hz - z}

    return True, distances

def pointInSphere(point, center, radius):
    """
    Check whether a point lies inside a spherical cavity and,
    if so, return its shortest distance to the cavity wall.

    Parameters
    ----------
    point : array-like, shape (3,)
        Global coordinates [x, y, z] of the point.

    center : array-like, shape (3,)
        Global coordinates [xc, yc, zc] of the sphere center.

    radius : float
        Radius of the spherical cavity.

    Returns
    -------
    pIn : bool
        True if the point lies inside or on the sphere.

    pDist : float or None
        Shortest distance from the point to the spherical wall.
        None if the point is outside.
    """

    pointUse= np.array([point[0],point[1],-point[2]])
    center = np.asarray(center, dtype=float)

    # Distance from sphere center to point
    rPoint = np.linalg.norm(pointUse - center)

    # Check whether point lies inside
    pIn = rPoint <= radius

    if not pIn:
        return False, None

    # Shortest distance to spherical cavity wall
    pDist = radius - rPoint

    return True, pDist

def boxesOverlapOrCloseXY(box1, box2, tol=0.0):
    """
    Check overlap / near-overlap in x-y only.
    """
    xSep = max(box1.xMin, box2.xMin) - min(box1.xMax, box2.xMax)
    ySep = max(box1.yMin, box2.yMin) - min(box1.yMax, box2.yMax)
    return (xSep <= tol) and (ySep <= tol)


def mergeTwoRefineBoxes(box1, box2):
    xMin = min(box1.xMin, box2.xMin)
    xMax = max(box1.xMax, box2.xMax)
    yMin = min(box1.yMin, box2.yMin)
    yMax = max(box1.yMax, box2.yMax)
    zMin = min(box1.zMin, box2.zMin)
    zMax = max(box1.zMax, box2.zMax)

    xC = 0.5 * (xMin + xMax)
    yC = 0.5 * (yMin + yMax)
    zC = 0.5 * (zMin + zMax)

    ax = 0.5 * (xMax - xMin)
    ay = 0.5 * (yMax - yMin)
    az = 0.5 * (zMax - zMin)

    names = sorted(set(box1.sourceNames + box2.sourceNames))

    return RefineBox(xC=xC, yC=yC, zC=zC, ax=ax, ay=ay, az=az, sourceNames=names)


def mergeRefineBoxes(boxes, xyTol=0.0):
    """
    Merge boxes that overlap or are closer than xyTol in x-y.
    """
    if len(boxes) == 0:
        return []

    working = boxes[:]
    changed = True

    while changed:
        changed = False
        used = [False] * len(working)
        merged = []

        for i in range(len(working)):
            if used[i]:
                continue

            cur = working[i]
            used[i] = True

            again = True
            while again:
                again = False
                for j in range(len(working)):
                    if used[j]:
                        continue
                    if boxesOverlapOrCloseXY(cur, working[j], tol=xyTol):
                        cur = mergeTwoRefineBoxes(cur, working[j])
                        used[j] = True
                        again = True
                        changed = True

            merged.append(cur)

        working = merged

    return working
def getEnclosingRefineBox(refineBoxes, layer, name="mergedBox"):
    """
    Build one enclosing RefineBox from all boxes that intersect this layer.

    The returned box is clipped in z only through its center/half-width definition;
    the actual layer/block clipping is still handled in generateBlocks.
    """
    active = [b for b in refineBoxes if b.intersectsLayer(layer.zTop, layer.zBot)]

    if len(active) == 0:
        return None

    xMin = min(b.xMin for b in active)
    xMax = max(b.xMax for b in active)
    yMin = min(b.yMin for b in active)
    yMax = max(b.yMax for b in active)
    zMin = min(b.zMin for b in active)
    zMax = max(b.zMax for b in active)

    xC = 0.5 * (xMin + xMax)
    yC = 0.5 * (yMin + yMax)
    zC = 0.5 * (zMin + zMax)

    ax = 0.5 * (xMax - xMin)
    ay = 0.5 * (yMax - yMin)
    az = 0.5 * (zMax - zMin)

    return RefineBox(xC=xC, yC=yC, zC=zC, ax=ax, ay=ay, az=az, sourceNames=[name])

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

    def generateBlocksNew(self, refineBoxes, domXYBounds, domainZBounds=None):
        """
        Generate blocks for this layer from a list of cavity-derived RefineBox objects.

        Logic
        -----
        1. Find refine boxes active in this layer.
        2. Replace them by one enclosing central refine box.
        3. Apply the old NNGeometryN 3x3 x-y logic around that one box.
        4. Keep old z intersection logic:
             - no_intersection -> 1 block
            - contains       -> 11 blocks
            - cut_top        -> 10 blocks
            - cut_bottom     -> 10 blocks
            - cut_middle     -> 9 blocks

        Notes
        -----
        - The layer intersection type (self.intersectionType) is still assumed to have
        been set by updateCubeInteraction(cubeTop, cubeBot) beforehand.
        - The central hosting region is treated as 'uniform' for now.
        """
        self.blocks = []

        if domainZBounds is None:
            domainZBounds = (self.zTop, self.zBot)

        # full layer/domain extents in x-y
        xMin = self.xMin
        xMax = self.xMax
        yMin = self.yMin
        yMax = self.yMax

        # no global cavity-hosting interaction in this layer
        if self.intersectionType == 'no_intersection':
            block = Block(
                xMin=xMin, xMax=xMax,
                yMin=yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho,
                spaceType='lgwt', interpType=1,
                domainBounds=domXYBounds,
                domainZBounds=domainZBounds
            )
            self.blocks.append(block)
            self.nBlocks = len(self.blocks)
            return

        # build one enclosing active refine box for this layer
        refineBox = getEnclosingRefineBox(refineBoxes, self)

        # if global cubeInteraction says intersecting but no active refine box is found,
        # fall back to one full-domain lgwt block
        if refineBox is None:
            block = Block(
                xMin=xMin, xMax=xMax,
                yMin=yMin, yMax=yMax,
                zMin=self.zTop, zMax=self.zBot,
                vP=self.vP, vS=self.vS, rho=self.rho,
                spaceType='lgwt', interpType=1,
                domainBounds=domXYBounds,
                domainZBounds=domainZBounds
            )
            self.blocks.append(block)
            self.nBlocks = len(self.blocks)
            return

        # enclosing box bounds
        xL, xR = refineBox.xMin, refineBox.xMax
        yB, yT = refineBox.yMin, refineBox.yMax
        zL, zU = refineBox.zMin, refineBox.zMax

        # clip x-y box to simulation/layer bounds just in case
        xL = max(xMin, xL)
        xR = min(xMax, xR)
        yB = max(yMin, yB)
        yT = min(yMax, yT)

        # x-y tiling
        xIntervals = [(xMin, xL), (xL, xR), (xR, xMax)]
        yIntervals = [(yMin, yB), (yB, yT), (yT, yMax)]

        # -------------------------
        # contains
        # -------------------------
        if self.intersectionType == 'contains':
            zIntervals_center = [(self.zTop, zL), (zL, zU), (zU, self.zBot)]

            for ix, (xa, xb) in enumerate(xIntervals):
                for iy, (ya, yb) in enumerate(yIntervals):

                    if xb <= xa or yb <= ya:
                        continue

                    is_center = (ix == 1 and iy == 1)

                    if is_center:
                        for za, zb in zIntervals_center:
                            if zb <= za:
                                continue

                            spaceType = 'uniform' if (np.isclose(za, zL) and np.isclose(zb, zU)) else 'lgwt'
                            interpType = 1 if (np.isclose(za, zL) and np.isclose(zb, zU)) else 2

                            block = Block(
                                xMin=xa, xMax=xb,
                                yMin=ya, yMax=yb,
                                zMin=za, zMax=zb,
                                vP=self.vP, vS=self.vS, rho=self.rho,
                                spaceType=spaceType, interpType=interpType,
                                domainBounds=domXYBounds,
                                domainZBounds=domainZBounds
                            )
                            self.blocks.append(block)
                    else:
                        block = Block(
                            xMin=xa, xMax=xb,
                            yMin=ya, yMax=yb,
                            zMin=self.zTop, zMax=self.zBot,
                            vP=self.vP, vS=self.vS, rho=self.rho,
                            spaceType='lgwt', interpType=2,
                            domainBounds=domXYBounds,
                            domainZBounds=domainZBounds
                        )
                        self.blocks.append(block)

            self.nBlocks = len(self.blocks)
            return

        # -------------------------
        # cut_top
        # -------------------------
        if self.intersectionType == 'cut_top':
            for ix, (xa, xb) in enumerate(xIntervals):
                for iy, (ya, yb) in enumerate(yIntervals):

                    if xb <= xa or yb <= ya:
                        continue

                    is_center = (ix == 1 and iy == 1)

                    if is_center:
                        zIntervals = [(self.zTop, zL), (zL, self.zBot)]

                        for za, zb in zIntervals:
                            if zb <= za:
                                continue

                            spaceType = 'uniform' if np.isclose(za, zL) else 'lgwt'
                            interpType = 1 if np.isclose(za, zL) else 2

                            block = Block(
                                xMin=xa, xMax=xb,
                                yMin=ya, yMax=yb,
                                zMin=za, zMax=zb,
                                vP=self.vP, vS=self.vS, rho=self.rho,
                                spaceType=spaceType, interpType=interpType,
                                domainBounds=domXYBounds,
                                domainZBounds=domainZBounds
                            )
                            self.blocks.append(block)
                    else:
                        block = Block(
                            xMin=xa, xMax=xb,
                            yMin=ya, yMax=yb,
                            zMin=self.zTop, zMax=self.zBot,
                            vP=self.vP, vS=self.vS, rho=self.rho,
                            spaceType='lgwt', interpType=2,
                            domainBounds=domXYBounds,
                            domainZBounds=domainZBounds
                        )
                        self.blocks.append(block)

            self.nBlocks = len(self.blocks)
            return

        # -------------------------
        # cut_bottom
        # -------------------------
        if self.intersectionType == 'cut_bottom':
            for ix, (xa, xb) in enumerate(xIntervals):
                for iy, (ya, yb) in enumerate(yIntervals):

                    if xb <= xa or yb <= ya:
                        continue

                    is_center = (ix == 1 and iy == 1)

                    if is_center:
                        zIntervals = [(self.zTop, zU), (zU, self.zBot)]

                        for za, zb in zIntervals:
                            if zb <= za:
                                continue

                            spaceType = 'uniform' if np.isclose(zb, zU) else 'lgwt'
                            interpType = 1 if np.isclose(zb, zU) else 2

                            block = Block(
                                xMin=xa, xMax=xb,
                                yMin=ya, yMax=yb,
                                zMin=za, zMax=zb,
                                vP=self.vP, vS=self.vS, rho=self.rho,
                                spaceType=spaceType, interpType=interpType,
                                domainBounds=domXYBounds,
                                domainZBounds=domainZBounds
                            )
                            self.blocks.append(block)
                    else:
                        block = Block(
                            xMin=xa, xMax=xb,
                            yMin=ya, yMax=yb,
                            zMin=self.zTop, zMax=self.zBot,
                            vP=self.vP, vS=self.vS, rho=self.rho,
                            spaceType='lgwt', interpType=2,
                            domainBounds=domXYBounds,
                            domainZBounds=domainZBounds
                        )
                        self.blocks.append(block)

            self.nBlocks = len(self.blocks)
            return

        # -------------------------
        # cut_middle
        # -------------------------
        if self.intersectionType == 'cut_middle':
            for ix, (xa, xb) in enumerate(xIntervals):
                for iy, (ya, yb) in enumerate(yIntervals):

                    if xb <= xa or yb <= ya:
                        continue

                    is_center = (ix == 1 and iy == 1)
                    spaceType = 'uniform' if is_center else 'lgwt'
                    interpType = 1 if is_center else 2

                    block = Block(
                        xMin=xa, xMax=xb,
                        yMin=ya, yMax=yb,
                        zMin=self.zTop, zMax=self.zBot,
                        vP=self.vP, vS=self.vS, rho=self.rho,
                        spaceType=spaceType, interpType=interpType,
                        domainBounds=domXYBounds,
                        domainZBounds=domainZBounds
                    )
                    self.blocks.append(block)

            self.nBlocks = len(self.blocks)
            return

        # fallback
        block = Block(
            xMin=xMin, xMax=xMax,
            yMin=yMin, yMax=yMax,
            zMin=self.zTop, zMax=self.zBot,
            vP=self.vP, vS=self.vS, rho=self.rho,
            spaceType='lgwt', interpType=1,
            domainBounds=domXYBounds,
            domainZBounds=domainZBounds
        )
        self.blocks.append(block)
        self.nBlocks = len(self.blocks)
    
    def getBlock(self, i):
        return self.blocks[i]

    def __repr__(self):
        return f"<Layer z: {self.zTop}-{self.zBot}, type: {self.intersectionType}, nblocks: {self.nBlocks}>";
        
    def describe(self):
        print(f"Layer {self.zTop} to {self.zBot} with {self.getNBlocks()} blocks");

def makeRefineBoxesFromCavities(allCavities,wallRefineWidth,domainZBounds): 
    """
    Build a list of RefineBox objects from a list of cavity dictionaries.
    Parameters 
    ---------- 
    allCavities : list of dict 
    Each cavity dict must contain: Sphere: shape, xC, yC, zC, 
    radius Cuboid: shape, xC, yC, zC, length, breadth, height, angleDeg 
    Optionally each cavity may also contain: zRefineFactor xyRefineFactor 
    name zRefineFactor : float Default z refinement factor if not specified per cavity. 
    xyRefineFactor : float Default xy refinement factor if not specified per cavity.
    
    Returns ------- refineBoxes : list of RefineBox """ 
    refineBoxes = [] 
    for i, cav in enumerate(allCavities): 
        shape = cav["shape"].lower() 
        cav_name = cav.get("name", f"Cavity_{i+1}") 
        cav_zref = cav.get("zRefineFactor") 
        cav_xref = cav.get("xRefineFactor") 
        cav_yref = cav.get("yRefineFactor") 
        if shape == "sphere": 
            box = makeSphereRefineBox( xC=cav["xC"], yC=cav["yC"], zC=cav["zC"], rCavity=cav["radius"],
                                      wallRefine=wallRefineWidth[i],domainZBounds =domainZBounds,name=cav_name )

        elif shape == "cuboid": 
            box = makeCuboidRefineBox( xC=cav["xC"], yC=cav["yC"], zC=cav["zC"], length=cav["length"],
                                           breadth=cav["breadth"], height=cav["height"], 
                                           angleDeg=cav.get("angleDeg", 0.0), 
                                           wallRefineWidth = wallRefineWidth[i],domainZBounds=domainZBounds,name=cav_name ) 
        else: 
            raise ValueError(f"Unsupported cavity shape: {cav['shape']}") 

        refineBoxes.append(box) 
    return refineBoxes

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

def createAllLayers(xMin, xMax, yMin, yMax, thickness, zMax, vp, vs, rho, gridSize, cubeTop, cubeBot,maxHostThickness=550.0):
    
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
                                                             cubeTop=cubeTop, cubeBot=cubeBot,max_host_thickness=maxHostThickness  # tune
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