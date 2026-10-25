#!/usr/bin/env python
# coding: utf-8

import os
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import matplotlib.ticker as mticker

def main():
    print('Activating plot modiule')
    xP = 0; yP = 0;
    outDir = '/data/gravwav/koley/OutDisp/Depth0p00/'
    plotDispSpect(outDir,xP,yP)

def set_plot_style(style="default"):
    styles = {
        "default": dict(labelsize=18, ticksize=16, legendsize=16, titlesize=18),
        "small":   dict(labelsize=14, ticksize=12, legendsize=12, titlesize=14),
        "medium":  dict(labelsize=16, ticksize=16, legendsize=16, titlesize=16),
        "large":   dict(labelsize=18, ticksize=16, legendsize=16, titlesize=18),
    }

    s = styles[style]

    mpl.rcParams.update({
        "axes.labelsize": s["labelsize"],
        "xtick.labelsize": s["ticksize"],
        "ytick.labelsize": s["ticksize"],
        "legend.fontsize": s["legendsize"],
        "axes.titlesize": s["titlesize"],
        "figure.titlesize": s["titlesize"],
    })

def plotPSDDeepSurfMulti(freq, data, depth_label, color, fig=None, axs=None, quantity="ASD",subFigs=3):

    if(subFigs==2):
        a = 12; b = 6
    else:
        a = 16; b = 6

    if fig is None or axs is None:
        fig, axs = plt.subplots(
            1, subFigs, figsize=(a, b),
            sharey=True,              # <<< THIS FIXES THE OVERLAP
            constrained_layout=True   # better than tight_layout
        )

        component_titles = ["(a) X Component", "(b) Y Component", "(c) Z Component"]
        for i, ax in enumerate(axs):
            ax.set_title(component_titles[i])
            ax.set_xlabel("Frequency (Hz)")
            if i == 0:                # <<< Only left subplot gets the y-label
                ax.set_ylabel(quantity)
            ax.grid(True)

    for i in range(subFigs):
        axs[i].semilogy(freq, np.abs(data[:, i]), color=color, label=depth_label)
        #axs[i].legend(loc="upper right")

    return fig, axs

def plotCCDeepSurfMulti(freq, ccZ, ccX, ccY, depth_label, color, fig=None, axs=None, quantity="real(CC)"):
    """
    Plot 3-component (Z, X, Y) spectra vs frequency on shared subplots.

    Returns
    -------
    fig, axs : updated figure and axes
    """

    # Create figure if not passed
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharey=True,constrained_layout=True)
        component_titles = ["(a) Z Component", "(b) X Component", "(c) Y Component"]
        for i, ax in enumerate(axs):
            ax.set_title(component_titles[i])
            ax.set_xlabel("Frequency (Hz)")
            if(i==0):
                ax.set_ylabel(quantity)
            ax.grid(True)

    # Plot data on the same axes
    #labels = ["X", "Y", "Z"]
    axs[0].plot(freq, ccZ, color=color, label=depth_label)
    axs[0].legend(loc="upper left")
    axs[1].plot(freq, ccX, color=color, label=depth_label)
    axs[1].legend(loc="upper left")
    axs[2].plot(freq, ccY, color=color, label=depth_label)
    axs[1].legend(loc="upper left")
    #plt.tight_layout();
    return fig, axs

    
def plotCC(ccRealZ, ccRealX, ccRealY, freqOut, j0R=None, j0L=None, fJ=None):

    # Create figure with shared y-axis and better spacing
    fig, axs = plt.subplots(
        1, 3,
        figsize=(12, 3),
        sharey=True,
        constrained_layout=True
    )

    # --- Z Component ---
    axs[0].plot(freqOut, ccRealZ, 'b', label="Simulated")
    if j0R is not None and fJ is not None:
        axs[0].plot(fJ, j0R, 'r', label=r"Theoretical $J_0(2\pi f\,d/v_R)$")
    
    axs[0].set_xlim([0, 8])
    axs[0].set_xlabel("Frequency (Hz)")
    axs[0].set_ylabel(r"$\langle XY^*\rangle / \sqrt{\langle XX^*\rangle \langle YY^*\rangle}$")
    axs[0].set_title("(a) Z component")
    axs[0].grid(True)
    axs[0].legend(loc="upper right")

    # --- X Component ---
    axs[1].plot(freqOut, ccRealX, 'b', label="Simulated")
    if j0R is not None and fJ is not None:
        axs[1].plot(fJ, j0R, 'r', label=r"Theoretical $J_0(2\pi f\,d/v_R)$")
    
    axs[1].set_xlim([0, 8])
    axs[1].set_xlabel("Frequency (Hz)")
    axs[1].set_title("(b) X component")
    axs[1].grid(True)
    axs[1].legend(loc="upper right")

    # --- Y Component ---
    axs[2].plot(freqOut, ccRealY, 'b', label="Simulated")
    if j0R is not None and fJ is not None:
        axs[2].plot(fJ, j0R, 'r', label=r"Theoretical $J_0(2\pi f\,d/v_R)$")
    axs[2].set_xlim([0, 8])
    axs[2].set_xlabel("Frequency (Hz)")
    axs[2].set_title("(c) Y component")
    axs[2].grid(True)
    axs[2].legend(loc="upper right")

    return fig, axs

def plotAttn(freqOut, attnSim, attnX, attnY, attnZ,
                     title="Surface / Depth attenuation: site bands vs simulation",
                     yscale="log",
                     plot_median=True):
    """
    Parameters
    ----------
    freqOut : (nF,) array
    attnSim : (nF,3) array (complex or real)
        Simulated attenuation ratios for [X,Y,Z] (e.g. surf/depth).
    attnX, attnY, attnZ : (nF,3) arrays
        Observed attenuation percentiles: columns [p10, p50, p90].
    yscale : 'log' or None
    plot_median : bool
        If True, plots the 50th percentile lines.
    """

    freqOut = np.asarray(freqOut)
    attnSim = np.asarray(attnSim)

    # take magnitude if complex
    sim = np.abs(attnSim)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True)
    fig.suptitle(title)

    comps = [
        ("X", attnX, 0),
        ("Y", attnY, 1),
        ("Z", attnZ, 2),
    ]

    for ax, (name, obs, ci) in zip(axes, comps):
        obs = np.asarray(obs)

        # Shaded 10–90% band
        ax.fill_between(freqOut, obs[:, 0], obs[:, 2], alpha=0.25,
                        label="Observed (10–90%)")

        # Optional median line
        if plot_median:
            ax.plot(freqOut, obs[:, 1], linewidth=1.2, alpha=0.9,
                    label="Observed (50%)")

        # Simulated attenuation
        ax.plot(freqOut, sim[:, ci], linewidth=2.2, label="Simulated")

        ax.set_title(name)
        ax.set_xlabel("Frequency (Hz)")
        if ci == 0:
            ax.set_ylabel("Attenuation (surface / depth)")

        if yscale is not None:
            ax.set_yscale(yscale)

        ax.grid(True, which="both", alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    #fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    return fig, axes

def plotNullStreamET(fNN, nnITFSum, nullITF, ETD, fig = None, axs = None):
    # plots the null stream and sum of individual itf output on a same plot with
    # ETD
    fNN = np.asarray(fNN).ravel()
    ETD = np.asarray(ETD)
    
    if fig is None or axs is None:
        fig, axs = plt.subplots(figsize=(8, 4), sharex=True, sharey=True)
    
    axs.plot(ETD[:, 0], ETD[:, 1], linewidth=1.8, label='ETD')
    axs.plot(ETD[:,0], 2*ETD[:,1], 'k--', linewidth=1.8, label='2*ETD')
    axs.plot(ETD[:,0], 5*ETD[:,1], 'k--', linewidth=1.8, label='5*ETD')
    axs.plot(fNN, nnITFSum,'r', label = r'$\sqrt{|h_A|^2 + |h_B|^2 + |h_C|^2}$')
    axs.plot(fNN, nullITF, 'g', label = r'$h_{null}$')

    axs.set_xscale("log")
    axs.set_yscale("log")
    # only show these major ticks
    axs.set_xticks([1, 2, 4, 8])
    axs.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))

    # keep minor ticks (optional) but hide their labels
    axs.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1))
    axs.xaxis.set_minor_formatter(mticker.NullFormatter())
        
    axs.set_xlabel("Frequency (Hz)")
    axs.set_ylabel("Strain / √Hz")

    axs.grid(True, which="both", alpha=0.3)

    axs.legend(frameon=True)
    axs.set_xlim(1,8)
    #fig.tight_layout(rect=[0, 0, 0.98, 0.92])
    return fig, axs

def plotNNStrainET(fNN, nnProj, ETD, title="Projected NN strain ASD", xlim=(1, 8), xscale="log",
                   yscale="log", comp_labels=("X", "Y", "Z"), sim_label="Simulated", etd_label="ET-D",
                   legend_once=True, fig=None,axs=None):
    """
    fNN: (nF,) frequency vector
    nnProj: (nF,3) projected NN strain ASD (complex or real)
    ETD: (nM,2) array [f, ASD] reference curve to overlay
    """

    fNN = np.asarray(fNN).ravel()
    nnProj = np.asarray(nnProj)
    ETD = np.asarray(ETD)

    if nnProj.shape[0] != fNN.size or nnProj.shape[1] != 3:
        raise ValueError(f"nnProj must be (nF,3); got {nnProj.shape} with nF={fNN.size}")
    if ETD.ndim != 2 or ETD.shape[1] < 2:
        raise ValueError("ETD must be (n,2+) with columns [f, curve]")

    # magnitude in case complex
    nn_asd = np.abs(nnProj)

    # Create figure/axes if not provided
    addET = False
    if fig is None or axs is None:
        fig, axs = plt.subplots(1, 3, figsize=(16, 6), sharex=True, sharey=True)
        
        addET = True

    fig.suptitle(title)

    for i, ax in enumerate(axs):
        ax.plot(fNN, nn_asd[:, i], linewidth=2.0, label=sim_label)
        if(addET):
            ax.plot(ETD[:, 0], ETD[:, 1], linewidth=1.8, label=etd_label)
            ax.plot(ETD[:,0], 2*ETD[:,1], 'k--', linewidth=1.8, label='2*ETD')
            ax.plot(ETD[:,0], 5*ETD[:,1], 'k--', linewidth=1.8, label='5*ETD')
        ax.set_title(comp_labels[i])
        ax.set_xlim(*xlim)

        if xscale == "log":
            ax.set_xscale("log")

            # only show these major ticks
            ax.set_xticks([1, 2, 4, 8])
            ax.xaxis.set_major_formatter(mticker.FormatStrFormatter('%g'))

            # keep minor ticks (optional) but hide their labels
            ax.xaxis.set_minor_locator(mticker.LogLocator(base=10, subs=np.arange(2, 10)*0.1))
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        else:
            ax.set_xscale(xscale)

        if yscale is not None:
            ax.set_yscale(yscale)

        ax.set_xlabel("Frequency (Hz)")
        if i == 0:
            ax.set_ylabel("Strain / √Hz")

        ax.grid(True, which="both", alpha=0.3)

    # Add legend (once or per subplot)
    if legend_once:
        handles, labels = axs[0].get_legend_handles_labels()
        fig.legend(handles, labels, loc="upper right", frameon=True)
    else:
        for ax in axs:
            ax.legend(frameon=True)

    fig.tight_layout(rect=[0, 0, 0.98, 0.92])
    return fig, axs
  
def plotScaledVsSiteASD(freqOut, scaled,
                            asdSurfX, asdSurfY, asdSurfZ,
                            asdBHX,   asdBHY,   asdBHZ,
                            title="Displacement ASD: site bands vs scaled simulation",
                            yscale="log"):
    """
    Plots site ASD percentile bands (10–90%) for surface and borehole, plus scaled simulation lines.
    scaled: (nFreq, 3, 2) complex or real, components [X,Y,Z], depths [surface(0), depth(1)]
    asdSurf*, asdBH*: (nFreq, 3) columns [p10, p50, p90]
    """

    freqOut = np.asarray(freqOut)
    scaled = np.asarray(scaled)

    if scaled.shape[1:] != (3, 2):
        raise ValueError(f"scaled must be (nFreq,3,2), got {scaled.shape}")

    # take magnitude in case scaled is complex
    sim = np.abs(scaled)

    fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharex=True)
    fig.suptitle(title)

    comps = [
        ("X", asdSurfX, asdBHX, 0),
        ("Y", asdSurfY, asdBHY, 1),
        ("Z", asdSurfZ, asdBHZ, 2),
    ]

    for ax, (name, surf, bh, ci) in zip(axes, comps):
        surf = np.asarray(surf)
        bh   = np.asarray(bh)

        # Shaded bands: 10th–90th percentiles
        ax.fill_between(freqOut, surf[:, 0], surf[:, 2], alpha=0.25, label="Site surface (10-90%)")
        ax.fill_between(freqOut, bh[:, 0],   bh[:, 2],   alpha=0.25, label="Site 250 m (10-90%)")

        # Optional: median lines (comment out if you don't want them)
        ax.plot(freqOut, surf[:, 1], linewidth=1.0, alpha=0.8, label="Site surface (50%)")
        ax.plot(freqOut, bh[:, 1],   linewidth=1.0, alpha=0.8, label="Site 250 m (50%)")

        # Scaled simulation: surface and depth
        ax.plot(freqOut, sim[:, ci, 0], linewidth=2.0, label="Sim scaled (surface)")
        ax.plot(freqOut, sim[:, ci, 1], linewidth=2.0, label="Sim scaled (250 m)")

        ax.set_title(name)
        ax.set_xlabel("Frequency (Hz)")
        if ci == 0:
            ax.set_ylabel("ASD")

        if yscale is not None:
            ax.set_yscale(yscale)

        ax.grid(True, which="both", alpha=0.3)

    # One legend for the whole figure
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", frameon=True)

    fig.tight_layout(rect=[0, 0, 0.98, 0.95])
    return fig, axes
    
def plotDispSpect(outDir, xP, yP):
    """
    Plot absolute value of displacement spectra for a point (xP, yP)
    on a depth slice (3 components, H-force & V-force).
    """

    # ---- Load receiver grid ----
    data = np.load(os.path.join(outDir, "receiverGrid.npz"))
    xGrid = data["xGrid"]
    yGrid = data["yGrid"]
    freqOut = data["freqOut"]
    nFreq = data["nFreq"]
    nRec  = len(xGrid)

    # ---- Load displacement fields ----
    dispHForce = np.memmap(
        os.path.join(outDir, "dispHForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    dispVForce = np.memmap(
        os.path.join(outDir, "dispVForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )

    # ---- Get receiver index ----
    rec_index, distRec = findRecIndex(xP, yP, xGrid, yGrid)

    dispH = np.abs(dispHForce[:, rec_index, :])
    dispV = np.abs(dispVForce[:, rec_index, :])

    # ---- Plotting ----
    fig, axs = plt.subplots(
        1, 3,
        figsize=(12, 3),
        sharey=True,
        constrained_layout=True
    )

    component_titles = ["Z component", "X component", "Y component"]

    # Loop through components
    for i in range(3):

        axs[i].semilogy(freqOut, dispH[:, i], "b", label="H-force")
        axs[i].semilogy(freqOut, dispV[:, i], "r", label="V-force")

        axs[i].set_xlim([0, 8])
        axs[i].set_xlabel("Frequency (Hz)")
        axs[i].set_title(component_titles[i])
        axs[i].grid(True)
        axs[i].legend(loc="upper right")

        if i == 0:
            axs[i].set_ylabel(r"$|\,\mathrm{disp}(f)\,|$")

    return fig, axs

def plotDispSpectAllRea(dispPointFull, freqOut):
    """
    Plot absolute value of displacement spectra for a point (after many realizations),
    showing Z, X, Y components on shared subplots.
    """

    # Create improved figure layout
    fig, axs = plt.subplots(
        1, 3,
        figsize=(12, 3),
        sharey=True,
        constrained_layout=True
    )

    component_titles = ["Z component", "X component", "Y component"]

    for i in range(3):
        axs[i].semilogy(freqOut, np.abs(dispPointFull[:, i]), 'b', label="V+H force")

        axs[i].set_xlim([0, 8])
        axs[i].set_xlabel("Frequency (Hz)")
        axs[i].set_title(component_titles[i])
        axs[i].grid(True)
        axs[i].legend(loc="upper right")

        if i == 0:
            axs[i].set_ylabel(r"$|\,\mathrm{disp}(f)\,|$")

    return fig, axs
    
def plotDispSurf(outDir, fTarget=None):
    """
    Plot a surface map of |displacement| at a chosen frequency.
    """

    # --- Load receiver grid ---
    data = np.load(os.path.join(outDir, "receiverGrid.npz"))
    xGrid = data["xGrid"]
    yGrid = data["yGrid"]
    freqOut = data["freqOut"]
    nFreq = int(data["nFreq"])
    nRec = len(xGrid)

    # --- Open memmap files ---
    dispHForce = np.memmap(
        os.path.join(outDir, "dispHForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )
    dispVForce = np.memmap(
        os.path.join(outDir, "dispVForce.dat"),
        dtype="complex128",
        mode="r",
        shape=(nFreq, nRec, 3)
    )

    # --- Optional surface maps ---
    if fTarget is not None:
        iFreq = np.argmin(np.abs(freqOut - fTarget))
        fSel = freqOut[iFreq]
        print(f"[plotDispSpectMemap] Plotting field magnitude at {fSel:.2f} Hz")

        nx = ny = int(np.sqrt(nRec))
        X = xGrid.reshape(nx, ny)
        Y = yGrid.reshape(nx, ny)

        magH = np.abs(dispHForce[iFreq, :, :]).reshape(nx, ny, 3)
        magV = np.abs(dispVForce[iFreq, :, :]).reshape(nx, ny, 3)
        comps = ["Z", "X", "Y"]

        # --- Horizontal Force field ---
        figH, axsH = plt.subplots(1, 3, figsize=(9,3))
        for i in range(3):
            im = axsH[i].pcolormesh(X, Y, magH[:, :, i], shading="auto", cmap="viridis")
            axsH[i].set_title(f"H-Force {comps[i]}")
            axsH[i].set_xlabel("x (m)")
            axsH[i].set_ylabel("y (m)")
            figH.colorbar(im, ax=axsH[i])
        figH.suptitle(f"|Displacement| (Horizontal Force) at {fSel:.2f} Hz")
        figH.tight_layout()

        # --- Vertical Force field ---
        figV, axsV = plt.subplots(1, 3, figsize=(9,3))
        for i in range(3):
            im = axsV[i].pcolormesh(X, Y, magV[:, :, i], shading="auto", cmap="viridis")
            axsV[i].set_title(f"V-Force {comps[i]}")
            axsV[i].set_xlabel("x (m)")
            axsV[i].set_ylabel("y (m)")
            figV.colorbar(im, ax=axsV[i])
        figV.suptitle(f"|Displacement| (Vertical Force) at {fSel:.2f} Hz")
        figV.tight_layout()

    plt.show()


def plotDispDepthSlice(U,freqOut, gridXMat, gridYMat, nX, nY, fPlot, zdepth=0, cUnit = 'm'):
    """
    function plots the absolute the value of the complex displacement
    obtained from superposition of many sources
    U - array of the type (nFreq,nGrid,nComp)
    freqOut - len(freqOut) = first dimension of U
    gridXMat, gridYMat = np.meshgrid(gridX, gridY, indexing='ij')
    where gridX, gridY are the grid points along X,Y -> note the indexing
    nX*nY = second dimension of U
    the third dimension are just the components
    fPlot is the frequency in Hz that you want to plot

    """    
    iFreq = np.where(freqOut>=fPlot)[0][0]

    labels = ["|Ux|", "|Uy|", "|Uz|"]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for k in range(3):
        comp = (U[iFreq, :, k]).reshape((nX, nY), order='C')
        im = axes[k].pcolormesh(gridXMat, gridYMat, comp, shading="auto")
        axes[k].set_title(labels[k])
        axes[k].set_aspect("equal")
        axes[k].set_xlabel('X (m)')
        if(k==0):
            axes[k].set_ylabel('Y (m)')
        cbar = fig.colorbar(im, ax=axes[k])
        cbar.ax.text(0.5, -0.08, f"[{cUnit}]",transform=cbar.ax.transAxes,ha="center",va="top")

    plt.suptitle(f"Displacement amplitude at f = {freqOut[iFreq]:.2f} Hz, z = {zdepth} m")
    plt.show()
    return fig, axes

def plotSrcRec(xSrc, ySrc, xGrid, yGrid, pA):
    
    fig1, axs1 = plt.subplots(1, 1, figsize=(4,4))
    axs1.plot(xSrc,ySrc,'b*',label="Source locations");

    minX = np.min(xGrid); maxX = np.max(xGrid);
    minY = np.min(yGrid); maxY = np.max(yGrid);
    simBoundX = np.array((minX,maxX,maxX,minX,minX));
    simBoundY = np.array((minY,minY,maxY,maxY,minY));

    axs1.plot(simBoundX, simBoundY, 'r', label= "Receiver boundary");
    axs1.plot(pA[0],pA[1], 'k*', label = "Point A");
    axs1.legend(loc='best');
    axs1.set_xlabel('X coordinate (m)');
    axs1.set_ylabel('Y coordinate (m)');
    axs1.set_title('Source-receiver locations');
    fig1.tight_layout();
    plt.show();
    
def findRecIndex(x1, y1, xGrid, yGrid, xVec=None, yVec=None):
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
    
def plotITFNullCC(ccITFANull, ccITFBNull, ccITFCNull, freqOut, figCC=None, axCC=None):
    if figCC is None or axCC is None:
        figCC, axCC = plt.subplots(1, 3, figsize=(16, 6), sharey=True,constrained_layout=True)
        component_titles = ["(a) ITF A - Null stream", "(b) ITF B - Null stream", "(c) ITF C - Null Stream"]
        for i, ax in enumerate(axCC):
            ax.set_title(component_titles[i])
            ax.set_xlabel("Frequency (Hz)")
            if(i==0):
                ax.set_ylabel('cross-correlation')
            ax.grid(True)
    
    axCC[0].plot(freqOut, np.real(ccITFANull), color='b', label='real')
    axCC[0].plot(freqOut, np.imag(ccITFANull), color='r', label='imaginary')
    axCC[0].plot(freqOut, 1/np.sqrt(3)*np.ones((len(freqOut),)),'k', label = r'$\frac{1}{\sqrt{3}}$')
    axCC[0].legend(loc="lower right")
    axCC[0].set_ylim(-1,1)

    axCC[1].plot(freqOut, np.real(ccITFBNull), color='b', label='real')
    axCC[1].plot(freqOut, np.imag(ccITFBNull), color='r', label='imaginary')
    axCC[1].plot(freqOut, 1/np.sqrt(3)*np.ones((len(freqOut),)),'k', label = r'$\frac{1}{\sqrt{3}}$')
    axCC[1].legend(loc="lower right")
    axCC[1].set_ylim(-1,1)

    axCC[2].plot(freqOut, np.real(ccITFCNull), color='b', label='real')
    axCC[2].plot(freqOut, np.imag(ccITFCNull), color='r', label='imaginary')
    axCC[2].plot(freqOut, 1/np.sqrt(3)*np.ones((len(freqOut),)),'k', label = r'$\frac{1}{\sqrt{3}}$')
    axCC[2].legend(loc="lower right")
    axCC[2].set_ylim(-1,1)

    return figCC, axCC

def plotITF_A_B_C_CC(ccITFAB, ccITFAC, freqOut, figCC=None, axCC=None):
    if figCC is None or axCC is None:
        figCC, axCC = plt.subplots(1, 2, figsize=(10, 4), sharey=True, constrained_layout=True)
        component_titles = ["(a) ITF A - ITF B", "(b) ITF A - ITF C"]
        for i, ax in enumerate(axCC):
            ax.set_title(component_titles[i])
            ax.set_xlabel("Frequency (Hz)")
            if(i==0):
                ax.set_ylabel('cross-correlation')
            ax.grid(True)
    axCC[0].plot(freqOut, np.real(ccITFAB), color='b', label='real')
    axCC[0].plot(freqOut, np.imag(ccITFAC), color='r', label='imaginary')  
    axCC[0].plot(freqOut, 1/np.sqrt(100)*np.ones((len(freqOut),)), 'k--', label = 'crosscorr significance')
    axCC[0].plot(freqOut, -1/np.sqrt(100)*np.ones((len(freqOut),)), 'k--')
    
    axCC[0].legend(loc="lower right")
    axCC[0].set_ylim(-1,1)

    axCC[1].plot(freqOut, np.real(ccITFAC), color='b', label='real')
    axCC[1].plot(freqOut, np.imag(ccITFAC), color='r', label='imaginary')
    axCC[1].plot(freqOut, 1/np.sqrt(100)*np.ones((len(freqOut),)), 'k--', label = 'crosscorr significance')
    axCC[1].plot(freqOut, -1/np.sqrt(100)*np.ones((len(freqOut),)), 'k--')
    
    axCC[1].legend(loc="lower right")
    axCC[1].set_ylim(-1,1)

    return figCC, axCC

def plotSingleITF(allTMLoc, fig, ax, colVal):
    for i in range(0,4):
        ax.plot(allTMLoc[i,0],allTMLoc[i,1], color = colVal, marker='o')
    # plot lines joining the input and end test-masses
    plotLine(allTMLoc[0,0:2], allTMLoc[2,0:2], ax, colVal)
    plotLine(allTMLoc[1,0:2], allTMLoc[3,0:2], ax, colVal)
    return ax

def plotLine(pointA, pointB, ax, colVal):
    x = np.zeros((2,))
    y = np.zeros((2,))
    x[0] = pointA[0]
    x[1] = pointB[0]
    y[0] = pointA[1]
    y[1] = pointB[1]

    ax.plot(x,y,color=colVal)
if __name__ == "__main__":
    main()
