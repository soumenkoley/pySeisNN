#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker
from matplotlib.patches import Circle, Polygon
from matplotlib.patches import Rectangle

def main():
    print('Activating plot modiule')
    xP = 0; yP = 0;
    outDir = '/data/gravwav/koley/OutDisp/Depth0p00/'
    #plotDispSpect(outDir,xP,yP)


def getLayerAtZ(layers, zSlice, tol=1e-9):
    """
    Return the layer containing zSlice.
    """
    for layer in layers:
        if (layer.zTop - tol) <= zSlice <= (layer.zBot + tol):
            return layer
    return None


def getBlocksAtZ(layer, zSlice, tol=1e-9):
    """
    Return blocks in the given layer whose z-range contains zSlice.
    """
    out = []
    for blk in layer.blocks:
        if (blk.zMin - tol) <= zSlice <= (blk.zMax + tol):
            out.append(blk)
    return out


def getCavityFootprintAtZ(cavity, zSlice, tol=1e-9):
    """
    Return a matplotlib patch for the cavity footprint at zSlice,
    or None if the cavity does not intersect the slice.

    Supported cavity dict formats
    -----------------------------
    Sphere:
        {
            "shape": "sphere",
            "xC": ...,
            "yC": ...,
            "zC": ...,
            "radius": ...
        }

    Cuboid:
        {
            "shape": "cuboid",
            "xC": ...,
            "yC": ...,
            "zC": ...,
            "length": ...,
            "breadth": ...,
            "height": ...,
            "angleDeg": ...
        }
    """

    shape = cavity["shape"].lower()

    if shape == "sphere":
        xC = cavity["xC"]
        yC = cavity["yC"]
        zC = cavity["zC"]
        r = cavity["radius"]

        dz = zSlice - zC
        if abs(dz) > r + tol:
            return None

        rSlice = np.sqrt(max(0.0, r**2 - dz**2))
        return Circle((xC, yC), rSlice, fill=False, edgecolor="black", linewidth=2.0)

    elif shape == "cuboid":
        xC = cavity["xC"]
        yC = cavity["yC"]
        zC = cavity["zC"]
        length = cavity["length"]
        breadth = cavity["breadth"]
        height = cavity["height"]
        angleDeg = cavity["angleDeg"]

        zTop = zC - 0.5 * height
        zBot = zC + 0.5 * height

        if zSlice < zTop - tol or zSlice > zBot + tol:
            return None

        theta = np.deg2rad(angleDeg)

        u = np.array([np.cos(theta), np.sin(theta)])
        v = np.array([-np.sin(theta), np.cos(theta)])

        hl = 0.5 * length
        hb = 0.5 * breadth
        center = np.array([xC, yC])

        corners = np.array([
            center - hl*u - hb*v,
            center - hl*u + hb*v,
            center + hl*u + hb*v,
            center + hl*u - hb*v
        ])

        return Polygon(corners, closed=True, fill=False, edgecolor="black", linewidth=2.0)

    else:
        raise ValueError(f"Unsupported cavity shape: {cavity['shape']}")
    
def plotTopViewAtZ(
    layers,
    zSlice,
    cavities=None,
    xlim=None,
    ylim=None,
    annotateBlocks=False,
    figsize=(8, 8),
    tol=1e-9
):
    """
    Plot the top view of generated blocks at a particular z slice.

    Parameters
    ----------
    layers : list
        List of Layer objects. Their .blocks must already be generated.
    zSlice : float
        Depth at which to plot the x-y slice.
    cavities : list or None
        Optional list of cavity dictionaries. See getCavityFootprintAtZ().
    xlim, ylim : tuple or None
        Optional axis limits, e.g. xlim=(-200, 200), ylim=(-100, 100)
    annotateBlocks : bool
        If True, annotate visible block centers.
    figsize : tuple
        Figure size.
    tol : float
        Numerical tolerance.
    """
    layer = getLayerAtZ(layers, zSlice, tol=tol)
    if layer is None:
        raise ValueError(f"No layer found containing zSlice = {zSlice}")

    blocks = getBlocksAtZ(layer, zSlice, tol=tol)

    # Default limits: if not given, use layer bounds
    if xlim is None:
        xlim = (layer.xMin, layer.xMax)
    if ylim is None:
        ylim = (layer.yMin, layer.yMax)

    fig, ax = plt.subplots(figsize=figsize)

    # Plot blocks
    for i, blk in enumerate(blocks):
        if blk.spaceType.lower() == "uniform":
            color = "blue"
            lw = 2.0
        else:
            color = "red"
            lw = 1.5

        rect = Rectangle(
            (blk.xMin, blk.yMin),
            blk.xMax - blk.xMin,
            blk.yMax - blk.yMin,
            fill=False,
            edgecolor=color,
            linewidth=lw
        )
        ax.add_patch(rect)

        if annotateBlocks:
            xc = 0.5 * (blk.xMin + blk.xMax)
            yc = 0.5 * (blk.yMin + blk.yMax)

            # Only annotate if center lies inside current visible window
            if (xlim[0] <= xc <= xlim[1]) and (ylim[0] <= yc <= ylim[1]):
                ax.text(
                    xc,
                    yc,
                    str(i),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=color,
                    clip_on=True
                )

    # Plot cavity footprints
    if cavities is not None:
        for cav in cavities:
            patch = getCavityFootprintAtZ(cav, zSlice, tol=tol)
            if patch is not None:
                ax.add_patch(patch)

    # Apply limits explicitly
    ax.set_xlim(*xlim)
    ax.set_ylim(*ylim)

    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("X (meters)")
    ax.set_ylabel("Y (meters)")
    ax.set_title(f"Top view at z = {zSlice:.3f} m  |  Layer z=[{layer.zTop}, {layer.zBot}]")

    # Legend
    ax.plot([], [], color="blue", linewidth=2.0, label="Uniform blocks")
    ax.plot([], [], color="red", linewidth=1.5, label="LGWT blocks")
    ax.plot([], [], color="black", linewidth=2.0, label="Cavity footprints")
    ax.legend()

    plt.tight_layout()
    plt.show()

    return fig, ax

if __name__ == "__main__":
    main()
