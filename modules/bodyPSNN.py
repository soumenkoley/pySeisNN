#!/usr/bin/env python
# coding: utf-8

import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.constants as const
from scipy.integrate import quad, dblquad
from numpy.polynomial.legendre import leggauss

# -----------------------------
# Correct SH y-propagating, x-polarized analytic pieces
# -----------------------------


AXIS = {"x": 0, "y": 1, "z": 2}


def sphere_cavity_H(
    k,
    wave_axis,
    pol_axis,
    radius,
    cav_center,
    tm,
    n_mu=64,
    n_phi=128,
    n_r=64,
):
    """
    Ordinary dipole-volume integral over a spherical cavity for

        xi = xi0 * exp(i k q) * e_pol

    where q is wave_axis and e_pol is pol_axis.

    Returns H_cav, shape (3,), such that

        H_bulk = H_outer_box - H_cav.

    Important:
    This includes the -4pi/3 distribution correction if the test mass
    lies inside the spherical cavity.
    """

    qind = AXIS[wave_axis]
    jind = AXIS[pol_axis]

    cav_center = np.asarray(cav_center, dtype=float)
    tm = np.asarray(tm, dtype=float)

    R0 = float(radius)

    # ------------------------------------------------------------
    # Surface term:
    #
    # ∫_{∂C} exp(i k q) n_j R_i/R^3 dS
    # ------------------------------------------------------------
    mu, wmu = leggauss(n_mu)
    phi = 2.0 * np.pi * (np.arange(n_phi) + 0.5) / n_phi
    wphi = 2.0 * np.pi / n_phi

    MU, PHI = np.meshgrid(mu, phi, indexing="ij")
    WMU = wmu[:, None]

    sinth = np.sqrt(1.0 - MU**2)

    nvec = np.empty(MU.shape + (3,), dtype=float)
    nvec[..., 0] = sinth * np.cos(PHI)
    nvec[..., 1] = sinth * np.sin(PHI)
    nvec[..., 2] = MU

    r_surf = cav_center[None, None, :] + R0 * nvec
    Rvec = r_surf - tm[None, None, :]

    Rnorm = np.linalg.norm(Rvec, axis=2)
    Kvec = Rvec / Rnorm[..., None]**3

    phase = np.exp(1j * k * r_surf[..., qind])

    dS_weight = R0**2 * WMU * wphi

    H_surf = np.sum(
        phase[..., None]
        * nvec[..., jind, None]
        * Kvec
        * dS_weight[..., None],
        axis=(0, 1),
    )

    # ------------------------------------------------------------
    # Phase-gradient volume correction:
    #
    # - i k q_j ∫_C exp(i k q) K_i dV
    #
    # This is nonzero only when propagation axis == polarization axis,
    # i.e. for the P-wave cases.
    # ------------------------------------------------------------
    qj = 1.0 if qind == jind else 0.0

    H_volcorr = np.zeros(3, dtype=np.complex128)

    if qj != 0.0:
        xr, wr = leggauss(n_r)
        rr = 0.5 * R0 * (xr + 1.0)
        wrr = 0.5 * R0 * wr

        H_K = np.zeros(3, dtype=np.complex128)

        for r_now, wr_now in zip(rr, wrr):
            r_pts = cav_center[None, None, :] + r_now * nvec
            Rv = r_pts - tm[None, None, :]

            Rn = np.linalg.norm(Rv, axis=2)
            K = Rv / Rn[..., None]**3

            ph = np.exp(1j * k * r_pts[..., qind])

            dV_weight = (r_now**2) * wr_now * WMU * wphi

            H_K += np.sum(
                ph[..., None] * K * dV_weight[..., None],
                axis=(0, 1),
            )

        H_volcorr = -1j * k * qj * H_K

    # ------------------------------------------------------------
    # Distribution correction:
    #
    # D_ij = ∂_j K_i - (4pi/3) δ_ij δ(R)
    #
    # If the test mass is inside the removed sphere, subtract this
    # from the ordinary cavity-volume integral.
    # ------------------------------------------------------------
    H_delta = np.zeros(3, dtype=np.complex128)

    if np.linalg.norm(tm - cav_center) < R0:
        phase_tm = np.exp(1j * k * tm[qind])
        H_delta[jind] = -(4.0 * np.pi / 3.0) * phase_tm

    H_cav = H_surf + H_volcorr + H_delta

    return H_cav

def sphere_P_cavity_centered_exact(k, wave_axis, radius, sphere_center):
    """
    Exact removed-cavity contribution for a centered spherical cavity,
    with the test mass at the sphere center, for a P wave:

        xi = xi0 exp(i k q) e_q

    Used as:
        H_bulk = H_out - H_cav

    Returns
    -------
    H_cav : complex ndarray, shape (3,)
    """

    qind = AXIS[wave_axis]
    R = float(radius)
    sphere_center = np.asarray(sphere_center, dtype=float)

    alpha = k * R
    H = np.zeros(3, dtype=np.complex128)

    if abs(alpha) < 1e-4:
        # Small-alpha expansion:
        # H_cav = 4*pi*(alpha^2/15 - alpha^4/420 + ...)
        Hscalar = 4.0*np.pi * (alpha**2 / 15.0 - alpha**4 / 420.0)
    else:
        Hscalar = 4.0*np.pi * (
            2.0*np.cos(alpha)/alpha**2
            -
            2.0*np.sin(alpha)/alpha**3
            +
            2.0/3.0
        )

    H[qind] = np.exp(1j * k * sphere_center[qind]) * Hscalar

    return H

def A_xface_zint(x, y, z1, z2):
    """
    z-integral of x/(x^2 + y^2 + z^2)^(3/2)
    from z1 to z2.
    """
    q = x*x + y*y
    if q == 0.0:
        return 0.0

    return (
        x*z2 / (q*np.sqrt(q + z2*z2))
        -
        x*z1 / (q*np.sqrt(q + z1*z1))
    )


def H_box_Dxx_SH_y(k, x1, x2, y1, y2, z1, z2):
    """
    Integral over a rectangular box of

        cos(k y) D_xx dV

    where D_xx = d/dx (x/r^3).

    This is the dipole bulk contribution over that box.
    """

    def integrand(y):
        return np.cos(k*y) * (
            A_xface_zint(x2, y, z1, z2)
            -
            A_xface_zint(x1, y, z1, z2)
        )

    val, err = quad(
        integrand,
        y1, y2,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500
    )
    return val


def H_xsurface_SH_y(k, L, z1, z2):
    """
    Outer vertical x-face surface term:

        ∫_{x=+L,-L} cos(k y) n_x x/r^3 dS

    with z in [z1,z2], y in [-L,L].

    For both x faces, n_x*x = L.
    """

    def integrand(y):
        # integrate over z analytically:
        # ∫ dz 1/(L^2+y^2+z^2)^(3/2)
        q = L*L + y*y
        zterm = (
            z2/(q*np.sqrt(q + z2*z2))
            -
            z1/(q*np.sqrt(q + z1*z1))
        )
        return np.cos(k*y) * L * zterm

    # two x-faces
    val, err = quad(
        integrand,
        -L, L,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500
    )
    return 2.0 * val


def SH_y_x_pieces(freq, vS, L=4000.0, a=50.0, b=20.0, c=20.0):
    """
    Returns dimensionless H_bulk, H_vert, H_total for

        xi = xi0 exp(i k y) xhat

    in the same convention as your saved code pieces:

        ITot = IBlockTot - IVertFaceTot - IHorFaceTot.

    Assumes the test mass is at the origin and the full domain is
    [-L,L] x [-L,L] x [-L,L], with cavity
    [-a,a] x [-b,b] x [-c,c].
    """

    k = 2*np.pi*freq/vS

    # Dipole bulk over outer cube
    H_out = H_box_Dxx_SH_y(k, -L, L, -L, L, -L, L)

    # Dipole bulk over removed cuboidal cavity
    H_cav = H_box_Dxx_SH_y(k, -a, a, -b, b, -c, c)

    # Code bulk = rock volume = outer - cavity
    H_bulk = H_out - H_cav

    # Code vertical surface = outer x faces.
    # y-faces do not contribute because u dot n = 0 for x-polarized SH.
    H_vert = H_xsurface_SH_y(k, L, -L, L)

    # Horizontal surface should be zero for this SH test
    H_hor = 0.0

    # Code total convention
    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_total

# for SH wave propagating along +x, but displacement field along +y

def A_yface_zint(y, x, z1, z2):
    """
    z-integral of y/(x^2 + y^2 + z^2)^(3/2)
    from z1 to z2.

    This is the y-face analogue of A_xface_zint.
    """
    q = x*x + y*y
    if q == 0.0:
        return 0.0

    return (
        y*z2 / (q*np.sqrt(q + z2*z2))
        -
        y*z1 / (q*np.sqrt(q + z1*z1))
    )


def H_box_Dyy_SH_x(k, x1, x2, y1, y2, z1, z2):
    """
    Integral over a rectangular box of

        cos(k x) D_yy dV

    where

        D_yy = 1/r^3 - 3 y^2/r^5
             = d/dy (y/r^3).

    This is the dipole bulk contribution over that box
    for an S/SH wave:

        xi = xi0 exp(i k x) yhat.
    """

    def integrand(x):
        return np.cos(k*x) * (
            A_yface_zint(y2, x, z1, z2)
            -
            A_yface_zint(y1, x, z1, z2)
        )

    val, err = quad(
        integrand,
        x1, x2,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500
    )
    return val


def H_ysurface_SH_x(k, L, z1, z2):
    """
    Outer vertical y-face surface term:

        ∫_{y=+L,-L} cos(k x) n_y y/r^3 dS

    with z in [z1,z2], x in [-L,L].

    For both y faces, n_y*y = L.
    """

    def integrand(x):
        # integrate over z analytically:
        # ∫ dz 1/(x^2+L^2+z^2)^(3/2)
        q = L*L + x*x
        zterm = (
            z2/(q*np.sqrt(q + z2*z2))
            -
            z1/(q*np.sqrt(q + z1*z1))
        )
        return np.cos(k*x) * L * zterm

    # two y-faces
    val, err = quad(
        integrand,
        -L, L,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500
    )
    return 2.0 * val


def SH_x_y_pieces(freq, vS, L=4000.0, a=50.0, b=20.0, c=20.0):
    """
    Returns dimensionless H_bulk, H_vert, H_total for

        xi = xi0 exp(i k x) yhat

    in the same convention as your saved code pieces:

        ITot = IBlockTot - IVertFaceTot - IHorFaceTot.

    Assumes the test mass is at the origin and the full domain is
    [-L,L] x [-L,L] x [-L,L], with cavity

        [-a,a] x [-b,b] x [-c,c].

    Here:
        a = cavity half-length along x
        b = cavity half-width  along y
        c = cavity half-height along z
    """

    k = 2*np.pi*freq/vS

    # Dipole bulk over outer cube
    H_out = H_box_Dyy_SH_x(k, -L, L, -L, L, -L, L)

    # Dipole bulk over removed cuboidal cavity
    H_cav = H_box_Dyy_SH_x(k, -a, a, -b, b, -c, c)

    # Code bulk = rock volume = outer - cavity
    H_bulk = H_out - H_cav

    # Code vertical surface = outer y faces.
    # x-faces do not contribute because u dot n = 0 for y-polarized SH.
    H_vert = H_ysurface_SH_x(k, L, -L, L)

    # Horizontal surface should be zero for this SH test
    H_hor = 0.0

    # Code total convention
    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_total

def A_zface_xyint(z, x, y):
    """
    Integrand after using

        D_zz = d/dz (z/r^3)

    This is z/(x^2 + y^2 + z^2)^(3/2)
    evaluated at a fixed z.
    """
    r2 = x*x + y*y + z*z
    if r2 == 0.0:
        return 0.0
    return z / (r2**1.5)


def H_box_Dzz_SV_x(k, x1, x2, y1, y2, z1, z2):
    """
    Integral over a rectangular box of

        cos(k x) D_zz dV

    where

        D_zz = d/dz (z/r^3).

    This is the dipole bulk contribution over that box
    for an SV wave:

        xi = xi0 exp(i k x) zhat.

    The z integral is analytic, leaving a 2D integral over x,y.
    """

    def integrand(y, x):
        return np.cos(k*x) * (
            A_zface_xyint(z2, x, y)
            -
            A_zface_xyint(z1, x, y)
        )

    val, err = dblquad(
        integrand,
        x1, x2,
        lambda x: y1,
        lambda x: y2,
        epsabs=1e-10,
        epsrel=1e-10
    )
    return val


def H_zsurface_SV_x(k, L, x1, x2, y1, y2, zface):
    """
    Horizontal z-face surface contribution for

        xi = xi0 exp(i k x) zhat.

    On a horizontal face, n_z z contributes.

    This returns the surface integral over one z-face:

        ∫ cos(k x) n_z z/r^3 dS

    The caller supplies zface and the correct sign through n_z.
    """

    # n_z is sign(zface) for a centered outer cube:
    # z = +L -> n_z = +1
    # z = -L -> n_z = -1
    n_z = np.sign(zface)
    pref = n_z * zface

    def integrand(y, x):
        r2 = x*x + y*y + zface*zface
        return np.cos(k*x) * pref / (r2**1.5)

    val, err = dblquad(
        integrand,
        x1, x2,
        lambda x: y1,
        lambda x: y2,
        epsabs=1e-10,
        epsrel=1e-10
    )
    return val


def SV_x_z_pieces(freq, vS, L=4000.0, a=50.0, b=20.0, c=20.0):
    """
    Returns dimensionless H_bulk, H_hor, H_total for

        xi = xi0 exp(i k x) zhat

    in the same convention as your saved code pieces:

        ITot = IBlockTot - IVertFaceTot - IHorFaceTot.

    Assumes full domain:
        [-L,L] x [-L,L] x [-L,L]

    and cavity:
        [-a,a] x [-b,b] x [-c,c].
    """

    k = 2*np.pi*freq/vS

    # Dipole bulk over outer cube
    H_out = H_box_Dzz_SV_x(k, -L, L, -L, L, -L, L)

    # Removed cavity
    H_cav = H_box_Dzz_SV_x(k, -a, a, -b, b, -c, c)

    H_bulk = H_out - H_cav

    # For z-polarized displacement, vertical x/y side faces do not contribute
    H_vert = 0.0

    # Horizontal outer surfaces z=+/-L.
    H_top = H_zsurface_SV_x(k, L, -L, L, -L, L, +L)
    H_bot = H_zsurface_SV_x(k, L, -L, L, -L, L, -L)

    H_hor = H_top + H_bot

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_hor, H_total

def H_box_Dzz_SV_y(k, x1, x2, y1, y2, z1, z2):
    """
    Integral over a rectangular box of

        cos(k y) D_zz dV

    where

        D_zz = d/dz (z/r^3).

    This is the dipole bulk contribution over that box
    for an SV wave:

        xi = xi0 exp(i k y) zhat.

    The z integral is analytic, leaving a 2D integral over x,y.
    """

    def integrand(y, x):
        return np.cos(k*y) * (
            A_zface_xyint(z2, x, y)
            -
            A_zface_xyint(z1, x, y)
        )

    val, err = dblquad(
        integrand,
        x1, x2,
        lambda x: y1,
        lambda x: y2,
        epsabs=1e-10,
        epsrel=1e-10
    )
    return val


def H_zsurface_SV_y(k, L, x1, x2, y1, y2, zface):
    """
    Horizontal z-face surface contribution for

        xi = xi0 exp(i k y) zhat.

    Over one z-face:

        ∫ cos(k y) n_z z/r^3 dS.
    """

    n_z = np.sign(zface)
    pref = n_z * zface

    def integrand(y, x):
        r2 = x*x + y*y + zface*zface
        return np.cos(k*y) * pref / (r2**1.5)

    val, err = dblquad(
        integrand,
        x1, x2,
        lambda x: y1,
        lambda x: y2,
        epsabs=1e-10,
        epsrel=1e-10
    )
    return val


def SV_y_z_pieces(freq, vS, L=4000.0, a=50.0, b=20.0, c=20.0):
    """
    Returns dimensionless H_bulk, H_hor, H_total for

        xi = xi0 exp(i k y) zhat

    in the same convention as your saved code pieces:

        ITot = IBlockTot - IVertFaceTot - IHorFaceTot.

    Assumes full domain:
        [-L,L] x [-L,L] x [-L,L]

    and cavity:
        [-a,a] x [-b,b] x [-c,c].
    """

    k = 2*np.pi*freq/vS

    # Dipole bulk over outer cube
    H_out = H_box_Dzz_SV_y(k, -L, L, -L, L, -L, L)

    # Removed cavity
    H_cav = H_box_Dzz_SV_y(k, -a, a, -b, b, -c, c)

    H_bulk = H_out - H_cav

    # For z-polarized displacement, vertical side surfaces do not contribute
    H_vert = 0.0

    # Horizontal outer surfaces
    H_top = H_zsurface_SV_y(k, L, -L, L, -L, L, +L)
    H_bot = H_zsurface_SV_y(k, L, -L, L, -L, L, -L)

    H_hor = H_top + H_bot

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_hor, H_total
def Q_rect_xface(x, Y, Z):
    """
    Integral over a rectangular face at coordinate x:

        Q(x;Y,Z) = ∫_{-Y}^{Y} ∫_{-Z}^{Z} x/(x^2+y^2+z^2)^(3/2) dz dy

    for x > 0.

    This equals the solid angle of the rectangular face.
    """
    if x == 0.0:
        return 2.0*np.pi

    return 4.0*np.arctan(
        (Y*Z) / (x*np.sqrt(x*x + Y*Y + Z*Z))
    )


def H_box_Dxx_P_x(k, A, B, C):
    """
    Integral over a centered rectangular box

        [-A,A] x [-B,B] x [-C,C]

    of

        cos(k x) D_xx dV

    where

        D_xx = 1/r^3 - 3x^2/r^5
             = d/dx (x/r^3).

    This is the dipole bulk contribution over that box for

        xi = xi0 exp(i k x) xhat.
    """

    QA = Q_rect_xface(A, B, C)

    def integrand(x):
        return np.sin(k*x) * Q_rect_xface(x, B, C)

    I, err = quad(
        integrand,
        0.0, A,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500
    )

    # 2 * [ cos(kA) Q(A) - Q(0) + k ∫_0^A sin(kx) Q(x) dx ]
    # with Q(0)=2pi
    H_box = 2.0*np.cos(k*A)*QA - 4.0*np.pi + 2.0*k*I

    return H_box


def H_xsurface_P_x(k, L):
    """
    Outer x-face surface term for

        xi = xi0 exp(i k x) xhat

    over the outer cube [-L,L]^3.

    This computes

        ∫_{x=+L,-L} exp(i k x) n_x x/r^3 dS

    taking the real symmetric part.
    """
    QL = Q_rect_xface(L, L, L)

    # two x faces, phases exp(+ikL) and exp(-ikL)
    return 2.0*np.cos(k*L)*QL


def P_x_x_pieces(freq, vP, L=4000.0, a=50.0, b=20.0, c=20.0):
    """
    Returns dimensionless H_bulk, H_vert, H_total for

        xi = xi0 exp(i k x) xhat

    in the same convention as your code pieces:

        ITot = IBlockTot - IVertFaceTot - IHorFaceTot.

    Domain:
        outer cube [-L,L]^3

    Cavity:
        [-a,a] x [-b,b] x [-c,c]
    """

    k = 2.0*np.pi*freq/vP

    # Dipole bulk over outer cube
    H_out = H_box_Dxx_P_x(k, L, L, L)

    # Dipole bulk over removed cuboidal cavity
    H_cav = H_box_Dxx_P_x(k, a, b, c)

    # Code bulk = rock volume = outer - cavity
    H_bulk = H_out - H_cav

    # Outer vertical surface: only x-faces contribute
    H_vert = H_xsurface_P_x(k, L)

    # Horizontal surfaces do not contribute because displacement is x-polarized
    H_hor = 0.0

    # Code convention
    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_total

def Q_rect_zface(z, X, Y):
    """
    Integral over a rectangular face at coordinate z:

        Q(z;X,Y) = ∫_{-X}^{X} ∫_{-Y}^{Y}
                   z/(x^2+y^2+z^2)^(3/2) dx dy

    for z > 0.

    This equals the solid angle of the rectangular face.
    """
    if z == 0.0:
        return 2.0*np.pi

    return 4.0*np.arctan(
        (X*Y) / (z*np.sqrt(z*z + X*X + Y*Y))
    )


def H_box_Dzz_P_z(k, A, B, C):
    """
    Integral over a centered rectangular box

        [-A,A] x [-B,B] x [-C,C]

    of

        cos(k z) D_zz dV

    where

        D_zz = 1/r^3 - 3z^2/r^5
             = d/dz (z/r^3).

    This is the dipole bulk contribution over that box for

        xi = xi0 exp(i k z) zhat.

    A, B, C are half-lengths along x, y, z.
    """

    QC = Q_rect_zface(C, A, B)

    def integrand(z):
        return np.sin(k*z) * Q_rect_zface(z, A, B)

    I, err = quad(
        integrand,
        0.0, C,
        epsabs=1e-10,
        epsrel=1e-10,
        limit=500
    )

    # 2 * [ cos(kC) Q(C) - Q(0) + k ∫_0^C sin(kz) Q(z) dz ]
    # with Q(0)=2pi
    H_box = 2.0*np.cos(k*C)*QC - 4.0*np.pi + 2.0*k*I

    return H_box

def H_zsurface_P_z(k, L):
    """
    Outer z-face surface term for

        xi = xi0 exp(i k z) zhat

    over the outer cube [-L,L]^3.

    This computes

        ∫_{z=+L,-L} exp(i k z) n_z z/r^3 dS

    taking the real symmetric part.
    """
    QL = Q_rect_zface(L, L, L)

    # two z faces, phases exp(+ikL) and exp(-ikL)
    return 2.0*np.cos(k*L)*QL


def P_z_z_pieces(freq, vP, L=4000.0, a=50.0, b=20.0, c=20.0):
    """
    Returns dimensionless H_bulk, H_hor, H_total for

        xi = xi0 exp(i k z) zhat

    in the same convention as your code pieces:

        ITot = IBlockTot - IVertFaceTot - IHorFaceTot.

    Domain:
        outer cube [-L,L]^3

    Cavity:
        [-a,a] x [-b,b] x [-c,c]

    Here:
        a = half-length along x
        b = half-length along y
        c = half-length along z
    """

    k = 2.0*np.pi*freq/vP

    # Dipole bulk over outer cube
    H_out = H_box_Dzz_P_z(k, L, L, L)

    # Dipole bulk over removed cuboidal cavity
    H_cav = H_box_Dzz_P_z(k, a, b, c)

    # Code bulk = rock volume = outer - cavity
    H_bulk = H_out - H_cav

    # For z-polarized P-wave, vertical x/y side surfaces do not contribute
    H_vert = 0.0

    # Outer horizontal surface: only z-faces contribute
    H_hor = H_zsurface_P_z(k, L)

    # Code convention
    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_hor, H_total

# helpers for whe test-mass can move inside the cavity

def _contains_zero(a, b):
    return (a < 0.0) and (0.0 < b)


def _asinh_safe(z, A):
    """
    asinh(z/A), with protection for A=0.
    """
    if A == 0.0:
        # This should only occur at a measure-zero singular line.
        # Avoid NaNs in quadrature.
        A = 1e-300
    return np.arcsinh(z / A)

def complex_quad_vec(func, a, b, epsabs=1e-10, epsrel=1e-10, limit=500):
    """
    Integrate a vector-valued complex function over [a,b].
    func(x) must return shape (3,).
    """

    out = np.zeros(3, dtype=np.complex128)

    for i in range(3):
        real_val, _ = quad(
            lambda xx: np.real(func(xx)[i]),
            a, b,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )

        imag_val, _ = quad(
            lambda xx: np.imag(func(xx)[i]),
            a, b,
            epsabs=epsabs,
            epsrel=epsrel,
            limit=limit,
        )

        out[i] = real_val + 1j * imag_val

    return out


def Qvec_xface_shifted(X, Y1, Y2, Z1, Z2):
    """
    Computes the vector face integral over a rectangle at fixed X:

        Q_i(X) = ∫_{Y1}^{Y2} ∫_{Z1}^{Z2}
                 R_i / R^3 dZ dY

    where R = (X,Y,Z), relative to the test mass.

    Returns:
        [Qx, Qy, Qz]

    This is used for x-normal faces and for reducing the D_ix
    volume integrals.
    """

    if np.isclose(X, 0.0):
        X = 1e-300

    # ---- Qx = ∫∫ X/R^3 dZ dY ----
    def F_x(Y, Z):
        R = np.sqrt(X*X + Y*Y + Z*Z)
        return np.arctan((Y * Z) / (X * R))

    Qx = (
        F_x(Y2, Z2)
        - F_x(Y1, Z2)
        - F_x(Y2, Z1)
        + F_x(Y1, Z1)
    )

    # ---- Qy = ∫∫ Y/R^3 dZ dY ----
    # ∫ Y/R^3 dY = -1/R
    # so Qy = ∫ [1/R(Y1) - 1/R(Y2)] dZ
    A_y1 = np.sqrt(X*X + Y1*Y1)
    A_y2 = np.sqrt(X*X + Y2*Y2)

    G_y1 = _asinh_safe(Z2, A_y1) - _asinh_safe(Z1, A_y1)
    G_y2 = _asinh_safe(Z2, A_y2) - _asinh_safe(Z1, A_y2)

    Qy = G_y1 - G_y2

    # ---- Qz = ∫∫ Z/R^3 dZ dY ----
    # ∫ Z/R^3 dZ = -1/R
    # so Qz = ∫ [1/R(Z1) - 1/R(Z2)] dY
    A_z1 = np.sqrt(X*X + Z1*Z1)
    A_z2 = np.sqrt(X*X + Z2*Z2)

    G_z1 = _asinh_safe(Y2, A_z1) - _asinh_safe(Y1, A_z1)
    G_z2 = _asinh_safe(Y2, A_z2) - _asinh_safe(Y1, A_z2)

    Qz = G_z1 - G_z2

    return np.array([Qx, Qy, Qz], dtype=np.complex128)

def Qvec_zface_shifted(Z, X1, X2, Y1, Y2):
    """
    Computes the vector face integral over a rectangle at fixed Z:

        Q_i(Z) = ∫_{X1}^{X2} ∫_{Y1}^{Y2}
                 R_i / R^3 dY dX

    where R = (X,Y,Z), relative to the test mass.

    Returns:
        [Qx, Qy, Qz]

    This is used for z-normal faces and for reducing the D_iz
    volume integrals.
    """

    if np.isclose(Z, 0.0):
        Z = 1e-300

    # ---- Qz = ∫∫ Z/R^3 dY dX ----
    def F_z(X, Y):
        R = np.sqrt(X*X + Y*Y + Z*Z)
        return np.arctan((X * Y) / (Z * R))

    Qz = (
        F_z(X2, Y2)
        - F_z(X1, Y2)
        - F_z(X2, Y1)
        + F_z(X1, Y1)
    )

    # ---- Qx = ∫∫ X/R^3 dY dX ----
    # ∫ X/R^3 dX = -1/R
    # Qx = ∫ [1/R(X1) - 1/R(X2)] dY
    A_x1 = np.sqrt(X1*X1 + Z*Z)
    A_x2 = np.sqrt(X2*X2 + Z*Z)

    G_x1 = _asinh_safe(Y2, A_x1) - _asinh_safe(Y1, A_x1)
    G_x2 = _asinh_safe(Y2, A_x2) - _asinh_safe(Y1, A_x2)

    Qx = G_x1 - G_x2

    # ---- Qy = ∫∫ Y/R^3 dY dX ----
    # ∫ Y/R^3 dY = -1/R
    # Qy = ∫ [1/R(Y1) - 1/R(Y2)] dX
    A_y1 = np.sqrt(Y1*Y1 + Z*Z)
    A_y2 = np.sqrt(Y2*Y2 + Z*Z)

    G_y1 = _asinh_safe(X2, A_y1) - _asinh_safe(X1, A_y1)
    G_y2 = _asinh_safe(X2, A_y2) - _asinh_safe(X1, A_y2)

    Qy = G_y1 - G_y2

    return np.array([Qx, Qy, Qz], dtype=np.complex128)

# wave specific box integrals
def H_box_Dix_P_x_shifted(k, x1, x2, y1, y2, z1, z2, tm):
    """
    Vector dipole-bulk integral over a rectangular box for a P-wave along +x:

        xi = xi0 exp(i k x) xhat

    Computes the three acceleration components:

        H_i = ∫_box exp(i k x) D_{i x}(r - r_TM) dV

    where

        D_{i x} = delta_{i x}/R^3 - 3 R_i R_x/R^5.

    The x-integral is reduced using

        D_{i x} = d/dX (R_i/R^3) - (4pi/3) delta_{i x} delta(R).

    Parameters
    ----------
    x1,x2,y1,y2,z1,z2 : float
        Physical/global box bounds.

    tm : array_like
        Test-mass coordinate [x_TM, y_TM, z_TM].
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    # Shift bounds relative to the test mass
    X1, X2 = x1 - x0, x2 - x0
    Y1, Y2 = y1 - y0, y2 - y0
    Z1, Z2 = z1 - z0, z2 - z0

    def Qvec(X):
        return Qvec_xface_shifted(X, Y1, Y2, Z1, Z2)

    # Boundary term:
    # [ exp(i k x_phys) Qvec(X) ]_{X1}^{X2}
    boundary = (
        np.exp(1j * k * (X2 + x0)) * Qvec(X2)
        -
        np.exp(1j * k * (X1 + x0)) * Qvec(X1)
    )

    # Integral term from integration by parts:
    # - i k ∫ exp(i k x_phys) Qvec(X) dX
    def integrand(X):
        return np.exp(1j * k * (X + x0)) * Qvec(X)

    # Split around X=0 to avoid evaluating face integral at singular plane
    scale = max(abs(X1), abs(X2), abs(Y1), abs(Y2), abs(Z1), abs(Z2), 1.0)
    eps = 1e-12 * scale

    integral = np.zeros(3, dtype=np.complex128)

    if X1 < -eps:
        upper = min(X2, -eps)
        if upper > X1:
            integral += complex_quad_vec(integrand, X1, upper)

    if X2 > eps:
        lower = max(X1, eps)
        if X2 > lower:
            integral += complex_quad_vec(integrand, lower, X2)

    H = boundary - 1j * k * integral

    # Distribution correction only for D_xx if test mass is inside the box
    if (
        _contains_zero(X1, X2)
        and _contains_zero(Y1, Y2)
        and _contains_zero(Z1, Z2)
    ):
        H[0] -= (4.0 * np.pi / 3.0) * np.exp(1j * k * x0)

    return H

def H_box_Diz_P_z_shifted(k, x1, x2, y1, y2, z1, z2, tm):
    """
    Vector dipole-bulk integral over a rectangular box for a P-wave along +z:

        xi = xi0 exp(i k z) zhat

    Computes the three acceleration components:

        H_i = ∫_box exp(i k z) D_{i z}(r - r_TM) dV

    where

        D_{i z} = delta_{i z}/R^3 - 3 R_i R_z/R^5.

    The z-integral is reduced using

        D_{i z} = d/dZ (R_i/R^3) - (4pi/3) delta_{i z} delta(R).

    Parameters
    ----------
    x1,x2,y1,y2,z1,z2 : float
        Physical/global box bounds.

    tm : array_like
        Test-mass coordinate [x_TM, y_TM, z_TM].
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    # Shift bounds relative to the test mass
    X1, X2 = x1 - x0, x2 - x0
    Y1, Y2 = y1 - y0, y2 - y0
    Z1, Z2 = z1 - z0, z2 - z0

    def Qvec(Z):
        return Qvec_zface_shifted(Z, X1, X2, Y1, Y2)

    # Boundary term:
    # [ exp(i k z_phys) Qvec(Z) ]_{Z1}^{Z2}
    boundary = (
        np.exp(1j * k * (Z2 + z0)) * Qvec(Z2)
        -
        np.exp(1j * k * (Z1 + z0)) * Qvec(Z1)
    )

    # Integral term from integration by parts:
    # - i k ∫ exp(i k z_phys) Qvec(Z) dZ
    def integrand(Z):
        return np.exp(1j * k * (Z + z0)) * Qvec(Z)

    # Split around Z=0 to avoid evaluating the face integral at singular plane
    scale = max(abs(X1), abs(X2), abs(Y1), abs(Y2), abs(Z1), abs(Z2), 1.0)
    eps = 1e-12 * scale

    integral = np.zeros(3, dtype=np.complex128)

    if Z1 < -eps:
        upper = min(Z2, -eps)
        if upper > Z1:
            integral += complex_quad_vec(integrand, Z1, upper)

    if Z2 > eps:
        lower = max(Z1, eps)
        if Z2 > lower:
            integral += complex_quad_vec(integrand, lower, Z2)

    H = boundary - 1j * k * integral

    # Distribution correction only for D_zz if test mass is inside this box
    if (
        _contains_zero(X1, X2)
        and _contains_zero(Y1, Y2)
        and _contains_zero(Z1, Z2)
    ):
        H[2] -= (4.0 * np.pi / 3.0) * np.exp(1j * k * z0)

    return H

def H_box_Dix_SH_y_shifted(k, x1, x2, y1, y2, z1, z2, tm):
    """
    Vector dipole-bulk integral over a rectangular box for

        xi = xi0 exp(i k y) xhat.

    Computes

        H_i = ∫_box exp(i k y) D_{i x}(r-r_TM) dV.

    Since D_{i x} = d/dX(R_i/R^3) - (4pi/3) delta_ix delta(R),
    the X integral is done analytically. The remaining phase is in Y,
    so this becomes a 1D integral over Y.
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    X1, X2 = x1 - x0, x2 - x0
    Y1, Y2 = y1 - y0, y2 - y0
    Z1, Z2 = z1 - z0, z2 - z0

    def face_diff(Y):
        return (
            A_y_reduced_for_xface(X2, Y, Z1, Z2)
            -
            A_y_reduced_for_xface(X1, Y, Z1, Z2)
        )

    def integrand(Y):
        # physical y = Y + y0
        return np.exp(1j * k * (Y + y0)) * face_diff(Y)

    H = complex_quad_vec(integrand, Y1, Y2)

    # Distribution correction only for D_xx if test mass is inside box
    if (
        _contains_zero(X1, X2)
        and _contains_zero(Y1, Y2)
        and _contains_zero(Z1, Z2)
    ):
        H[0] -= (4.0 * np.pi / 3.0) * np.exp(1j * k * y0)

    return H

def H_box_Diy_SH_x_shifted(k, x1, x2, y1, y2, z1, z2, tm):
    """
    Vector dipole-bulk integral over a rectangular box for

        xi = xi0 exp(i k x) yhat.

    Computes

        H_i = ∫_box exp(i k x) D_{i y}(r-r_TM) dV.

    Since D_iy = d/dY(R_i/R^3) - (4pi/3) delta_iy delta(R),
    the Y integral is done analytically. The remaining phase is in X,
    so this becomes a 1D integral over X.
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    X1, X2 = x1 - x0, x2 - x0
    Y1, Y2 = y1 - y0, y2 - y0
    Z1, Z2 = z1 - z0, z2 - z0

    def face_diff(X):
        return (
            A_x_reduced_for_yface(Y2, X, Z1, Z2)
            -
            A_x_reduced_for_yface(Y1, X, Z1, Z2)
        )

    def integrand(X):
        # physical x = X + x0
        return np.exp(1j * k * (X + x0)) * face_diff(X)

    H = complex_quad_vec(integrand, X1, X2)

    # Distribution correction only for D_yy if test mass is inside box
    if (
        _contains_zero(X1, X2)
        and _contains_zero(Y1, Y2)
        and _contains_zero(Z1, Z2)
    ):
        H[1] -= (4.0 * np.pi / 3.0) * np.exp(1j * k * x0)

    return H

def H_box_Diz_SV_x_shifted(k, x1, x2, y1, y2, z1, z2, tm):
    """
    Vector dipole-bulk integral over a rectangular box for

        xi = xi0 exp(i k x) zhat.

    Computes

        H_i = ∫_box exp(i k x) D_{i z}(r-r_TM) dV.

    Since

        D_iz = d/dZ(R_i/R^3) - (4pi/3) delta_iz delta(R),

    the Z integral is done analytically. The remaining phase is in X,
    so this becomes a 1D integral over X.
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    X1, X2 = x1 - x0, x2 - x0
    Y1, Y2 = y1 - y0, y2 - y0
    Z1, Z2 = z1 - z0, z2 - z0

    def face_diff(X):
        return (
            A_x_reduced_for_zface(Z2, X, Y1, Y2)
            -
            A_x_reduced_for_zface(Z1, X, Y1, Y2)
        )

    def integrand(X):
        # physical x = X + x0
        return np.exp(1j * k * (X + x0)) * face_diff(X)

    H = complex_quad_vec(integrand, X1, X2)

    # Distribution correction only for D_zz if test mass is inside box
    if (
        _contains_zero(X1, X2)
        and _contains_zero(Y1, Y2)
        and _contains_zero(Z1, Z2)
    ):
        H[2] -= (4.0 * np.pi / 3.0) * np.exp(1j * k * x0)

    return H

# reduced face wrappers

def A_y_reduced_for_xface(X, Y, Z1, Z2):
    """
    z-integral of R_i/R^3 over z at fixed X,Y,
    for i = x,y,z, from Z1 to Z2.

    Returns vector:
        ∫ [X,Y,Z]/R^3 dZ.
    """

    q = X*X + Y*Y

    if q == 0.0:
        q = 1e-300

    # ∫ X/R^3 dz
    Ix = X * (
        Z2 / (q * np.sqrt(q + Z2*Z2))
        -
        Z1 / (q * np.sqrt(q + Z1*Z1))
    )

    # ∫ Y/R^3 dz
    Iy = Y * (
        Z2 / (q * np.sqrt(q + Z2*Z2))
        -
        Z1 / (q * np.sqrt(q + Z1*Z1))
    )

    # ∫ Z/R^3 dz = -1/R evaluated
    Iz = (
        1.0 / np.sqrt(q + Z1*Z1)
        -
        1.0 / np.sqrt(q + Z2*Z2)
    )

    return np.array([Ix, Iy, Iz], dtype=np.complex128)

def A_x_reduced_for_yface(Y, X, Z1, Z2):
    """
    z-integral of R_i/R^3 over z at fixed X,Y,
    for i = x,y,z, from Z1 to Z2.

    Used after reducing D_iy through the y-face identity.
    Returns:
        ∫ [X,Y,Z]/R^3 dZ.
    """

    q = X*X + Y*Y

    if q == 0.0:
        q = 1e-300

    # ∫ X/R^3 dz
    Ix = X * (
        Z2 / (q * np.sqrt(q + Z2*Z2))
        -
        Z1 / (q * np.sqrt(q + Z1*Z1))
    )

    # ∫ Y/R^3 dz
    Iy = Y * (
        Z2 / (q * np.sqrt(q + Z2*Z2))
        -
        Z1 / (q * np.sqrt(q + Z1*Z1))
    )

    # ∫ Z/R^3 dz = -1/R evaluated from Z1 to Z2
    Iz = (
        1.0 / np.sqrt(q + Z1*Z1)
        -
        1.0 / np.sqrt(q + Z2*Z2)
    )

    return np.array([Ix, Iy, Iz], dtype=np.complex128)

def A_x_reduced_for_zface(Z, X, Y1, Y2):
    """
    y-integral of R_i/R^3 over y at fixed X,Z,
    for i = x,y,z, from Y1 to Y2.

    Used after reducing D_iz through the z-face identity.

    Returns:
        ∫ [X,Y,Z]/R^3 dY.
    """

    q = X*X + Z*Z

    if q == 0.0:
        q = 1e-300

    # ∫ X/R^3 dy
    Ix = X * (
        Y2 / (q * np.sqrt(q + Y2*Y2))
        -
        Y1 / (q * np.sqrt(q + Y1*Y1))
    )

    # ∫ Y/R^3 dy = -1/R evaluated from Y1 to Y2
    Iy = (
        1.0 / np.sqrt(q + Y1*Y1)
        -
        1.0 / np.sqrt(q + Y2*Y2)
    )

    # ∫ Z/R^3 dy
    Iz = Z * (
        Y2 / (q * np.sqrt(q + Y2*Y2))
        -
        Y1 / (q * np.sqrt(q + Y1*Y1))
    )

    return np.array([Ix, Iy, Iz], dtype=np.complex128)

# wave specific surface integrals
def H_xsurface_P_x_shifted_vec(k, x_min, x_max, y_min, y_max, z_min, z_max, tm):
    """
    Outer x-face surface vector for P-wave along +x:

        H_surf,i = ∫_{x faces} exp(i k x) n_x R_i/R^3 dS

    where R = r - r_TM.

    Returns vector [Hx_surf, Hy_surf, Hz_surf].
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    Y1, Y2 = y_min - y0, y_max - y0
    Z1, Z2 = z_min - z0, z_max - z0

    # Right face x = x_max, n_x = +1
    Xr = x_max - x0
    Qr = Qvec_xface_shifted(Xr, Y1, Y2, Z1, Z2)
    Sr = np.exp(1j * k * x_max) * (+1.0) * Qr

    # Left face x = x_min, n_x = -1
    Xl = x_min - x0
    Ql = Qvec_xface_shifted(Xl, Y1, Y2, Z1, Z2)
    Sl = np.exp(1j * k * x_min) * (-1.0) * Ql

    return Sr + Sl

def H_zsurface_P_z_shifted_vec(k, x_min, x_max, y_min, y_max, z_min, z_max, tm, fullSpace=True):
    """
    Outer z-face surface vector for P-wave along +z:

        H_surf,i = ∫_{z faces} exp(i k z) n_z R_i/R^3 dS

    where R = r - r_TM.

    Returns vector [Hx_surf, Hy_surf, Hz_surf].
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    X1, X2 = x_min - x0, x_max - x0
    Y1, Y2 = y_min - y0, y_max - y0

    # Top/right z face z = z_max, n_z = +1
    Zr = z_max - z0
    Qr = Qvec_zface_shifted(Zr, X1, X2, Y1, Y2)
    Sr = np.exp(1j * k * z_max) * (+1.0) * Qr

    # Bottom/left z face z = z_min, n_z = -1
    Zl = z_min - z0
    Ql = Qvec_zface_shifted(Zl, X1, X2, Y1, Y2)
    Sl = np.exp(1j * k * z_min) * (-1.0) * Ql

    if(fullSpace):
        return Sr + Sl
    else:
        return Sl

def H_xsurface_SH_y_shifted_vec(k, x_min, x_max, y_min, y_max, z_min, z_max, tm):
    """
    Outer x-face surface vector for

        xi = xi0 exp(i k y) xhat.

    Computes

        H_surf,i = ∫_{x faces} exp(i k y) n_x R_i/R^3 dS.

    Returns [Hx, Hy, Hz].
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    Xr = x_max - x0
    Xl = x_min - x0
    Y1, Y2 = y_min - y0, y_max - y0
    Z1, Z2 = z_min - z0, z_max - z0

    def integrand(Y):
        phase = np.exp(1j * k * (Y + y0))

        # right face, n_x = +1
        right = A_y_reduced_for_xface(Xr, Y, Z1, Z2)

        # left face, n_x = -1
        left = A_y_reduced_for_xface(Xl, Y, Z1, Z2)

        return phase * (right - left)

    return complex_quad_vec(integrand, Y1, Y2)

def H_ysurface_SH_x_shifted_vec(k, x_min, x_max, y_min, y_max, z_min, z_max, tm):
    """
    Outer y-face surface vector for

        xi = xi0 exp(i k x) yhat.

    Computes

        H_surf,i = ∫_{y faces} exp(i k x) n_y R_i/R^3 dS.

    Returns [Hx, Hy, Hz].
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    Yt = y_max - y0
    Yb = y_min - y0
    X1, X2 = x_min - x0, x_max - x0
    Z1, Z2 = z_min - z0, z_max - z0

    def integrand(X):
        phase = np.exp(1j * k * (X + x0))

        # y = y_max face, n_y = +1
        top = A_x_reduced_for_yface(Yt, X, Z1, Z2)

        # y = y_min face, n_y = -1
        bottom = A_x_reduced_for_yface(Yb, X, Z1, Z2)

        return phase * (top - bottom)

    return complex_quad_vec(integrand, X1, X2)

def H_zsurface_SV_x_shifted_vec(k, x_min, x_max, y_min, y_max, z_min, z_max, tm,fullSpace=True):
    """
    Outer z-face surface vector for

        xi = xi0 exp(i k x) zhat.

    Computes

        H_surf,i = ∫_{z faces} exp(i k x) n_z R_i/R^3 dS.

    Returns [Hx, Hy, Hz].
    """

    x0, y0, z0 = np.asarray(tm, dtype=float)

    Zt = z_max - z0
    Zb = z_min - z0
    X1, X2 = x_min - x0, x_max - x0
    Y1, Y2 = y_min - y0, y_max - y0

    def integrand(X):
        phase = np.exp(1j * k * (X + x0))

        # z = z_max face, n_z = +1
        top = A_x_reduced_for_zface(Zt, X, Y1, Y2)

        # z = z_min face, n_z = -1
        bottom = A_x_reduced_for_zface(Zb, X, Y1, Y2)

        if(fullSpace):
            return phase * (top - bottom)
        else:
            return phase * (-bottom)

    return complex_quad_vec(integrand, X1, X2)

# final wrappers for specific wave type
def P_x_all_components_shifted(freq, vP, outer_bounds, cavity_bounds, tm):
    """
    Finite-box analytical pieces for a P-wave propagating along +x:

        xi = xi0 exp(i k x) xhat

    Returns
    -------
    H_bulk : complex ndarray, shape (3,)
        Bulk vector coefficient [Hx, Hy, Hz].

    H_vert : complex ndarray, shape (3,)
        Outer vertical x-face surface coefficient [Hx, Hy, Hz].

    H_hor : complex ndarray, shape (3,)
        Horizontal surface coefficient. Zero for x-polarized P wave.

    H_total : complex ndarray, shape (3,)
        Total coefficient in your code convention:

            H_total = H_bulk - H_vert - H_hor

    Units:
        delta_a = G * rho * xi0 * H_total.
    """

    k = 2.0 * np.pi * freq / vP

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds
    xc1, xc2, yc1, yc2, zc1, zc2 = cavity_bounds

    # Bulk over outer box
    H_out = H_box_Dix_P_x_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    # Bulk over removed cavity
    H_cav = H_box_Dix_P_x_shifted(
        k, xc1, xc2, yc1, yc2, zc1, zc2, tm
    )

    # Rock bulk = outer - cavity
    H_bulk = H_out - H_cav

    # Outer surface: only x-faces contribute because displacement is xhat
    H_vert = H_xsurface_P_x_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    # Horizontal surface: zero for x-polarized displacement
    H_hor = np.zeros(3, dtype=np.complex128)

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def P_x_sphere_all_components_shifted(
    freq,
    vP,
    outer_bounds,
    sphere_center,
    sphere_radius,
    tm,
    **sphere_quad_kwargs,
):
    """
    P wave:
        xi = xi0 exp(i k x) xhat
    """

    k = 2.0 * np.pi * freq / vP

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds

    H_out = H_box_Dix_P_x_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    H_cav = sphere_cavity_H(
        k,
        wave_axis="x",
        pol_axis="x",
        radius=sphere_radius,
        cav_center=sphere_center,
        tm=tm,
        **sphere_quad_kwargs,
    )

    H_bulk = H_out - H_cav

    H_vert = H_xsurface_P_x_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    H_hor = np.zeros(3, dtype=np.complex128)

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def P_z_all_components_shifted(
    freq,
    vP,
    outer_bounds,
    cavity_bounds,
    tm,
    fullSpace=True
):
    """
    Finite-box analytical pieces for a P-wave propagating along +z:

        xi = xi0 exp(i k z) zhat

    Returns
    -------
    H_bulk : complex ndarray, shape (3,)
        Bulk vector coefficient [Hx, Hy, Hz].

    H_vert : complex ndarray, shape (3,)
        Vertical surface coefficient. Zero for z-polarized P wave.

    H_hor : complex ndarray, shape (3,)
        Outer horizontal z-face surface coefficient [Hx, Hy, Hz].

    H_total : complex ndarray, shape (3,)
        Total coefficient in your code convention:

            H_total = H_bulk - H_vert - H_hor

    Units:
        delta_a = G * rho * xi0 * H_total.
    """

    k = 2.0 * np.pi * freq / vP

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds
    xc1, xc2, yc1, yc2, zc1, zc2 = cavity_bounds

    # Bulk over outer box
    H_out = H_box_Diz_P_z_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    # Bulk over removed cavity
    H_cav = H_box_Diz_P_z_shifted(
        k, xc1, xc2, yc1, yc2, zc1, zc2, tm
    )

    # Rock bulk = outer - cavity
    H_bulk = H_out - H_cav

    # Vertical surfaces: zero for z-polarized displacement
    H_vert = np.zeros(3, dtype=np.complex128)

    # Horizontal z-face surface term
    H_hor = H_zsurface_P_z_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm, fullSpace=fullSpace
    )

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def P_z_sphere_all_components_shifted(
    freq,
    vP,
    outer_bounds,
    sphere_center,
    sphere_radius,
    tm,
    fullSpace=True,
    **sphere_quad_kwargs,
):
    """
    P wave:
        xi = xi0 exp(i k z) zhat
    """

    k = 2.0 * np.pi * freq / vP

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds

    H_out = H_box_Diz_P_z_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    H_cav = sphere_cavity_H(
        k,
        wave_axis="z",
        pol_axis="z",
        radius=sphere_radius,
        cav_center=sphere_center,
        tm=tm,
        **sphere_quad_kwargs,
    )

    H_bulk = H_out - H_cav

    H_vert = np.zeros(3, dtype=np.complex128)

    H_hor = H_zsurface_P_z_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm, fullSpace=fullSpace,
    )

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def SH_y_x_all_components_shifted(
    freq,
    vS,
    outer_bounds,
    cavity_bounds,
    tm,
):
    """
    Finite-box analytical pieces for an SH wave propagating along +y:

        xi = xi0 exp(i k y) xhat.

    Returns
    -------
    H_bulk : complex ndarray, shape (3,)
        Bulk vector coefficient [Hx, Hy, Hz].

    H_vert : complex ndarray, shape (3,)
        Outer x-face surface coefficient [Hx, Hy, Hz].

    H_hor : complex ndarray, shape (3,)
        Horizontal surface coefficient. Zero for x-polarized displacement.

    H_total : complex ndarray, shape (3,)
        Total coefficient in your code convention:

            H_total = H_bulk - H_vert - H_hor

    Units:
        delta_a = G * rho * xi0 * H_total.
    """

    k = 2.0 * np.pi * freq / vS

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds
    xc1, xc2, yc1, yc2, zc1, zc2 = cavity_bounds

    # Bulk over outer box
    H_out = H_box_Dix_SH_y_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    # Bulk over removed cavity
    H_cav = H_box_Dix_SH_y_shifted(
        k, xc1, xc2, yc1, yc2, zc1, zc2, tm
    )

    # Rock bulk = outer - cavity
    H_bulk = H_out - H_cav

    # Vertical surface: only x-faces contribute for x-polarized displacement
    H_vert = H_xsurface_SH_y_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    # Horizontal surface: zero for x-polarized displacement
    H_hor = np.zeros(3, dtype=np.complex128)

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def SH_y_x_sphere_all_components_shifted(freq, vS, outer_bounds, sphere_center, sphere_radius, tm, **sphere_quad_kwargs):
    """
    SH wave:
        xi = xi0 exp(i k y) xhat
    """

    k = 2.0 * np.pi * freq / vS

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds

    H_out = H_box_Dix_SH_y_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    H_cav = sphere_cavity_H(
        k,
        wave_axis="y",
        pol_axis="x",
        radius=sphere_radius,
        cav_center=sphere_center,
        tm=tm,
        **sphere_quad_kwargs,
    )

    H_bulk = H_out - H_cav

    H_vert = H_xsurface_SH_y_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    H_hor = np.zeros(3, dtype=np.complex128)

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def SH_x_y_all_components_shifted(freq, vS, outer_bounds, cavity_bounds, tm):
    """
    Finite-box analytical pieces for an SH wave propagating along +x:

        xi = xi0 exp(i k x) yhat.

    Returns
    -------
    H_bulk : complex ndarray, shape (3,)
        Bulk vector coefficient [Hx, Hy, Hz].

    H_vert : complex ndarray, shape (3,)
        Outer y-face surface coefficient [Hx, Hy, Hz].

    H_hor : complex ndarray, shape (3,)
        Horizontal surface coefficient. Zero for y-polarized displacement.

    H_total : complex ndarray, shape (3,)
        Total coefficient in your code convention:

            H_total = H_bulk - H_vert - H_hor

    Units:
        delta_a = G * rho * xi0 * H_total.
    """

    k = 2.0 * np.pi * freq / vS

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds
    xc1, xc2, yc1, yc2, zc1, zc2 = cavity_bounds

    # Bulk over outer box
    H_out = H_box_Diy_SH_x_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    # Bulk over removed cavity
    H_cav = H_box_Diy_SH_x_shifted(
        k, xc1, xc2, yc1, yc2, zc1, zc2, tm
    )

    # Rock bulk = outer - cavity
    H_bulk = H_out - H_cav

    # Vertical surface: only y-faces contribute for y-polarized displacement
    H_vert = H_ysurface_SH_x_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    # Horizontal surface: zero for y-polarized displacement
    H_hor = np.zeros(3, dtype=np.complex128)

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def SH_x_y_sphere_all_components_shifted(freq, vS, outer_bounds, sphere_center, sphere_radius, tm, **sphere_quad_kwargs):
    """
    SH wave:
        xi = xi0 exp(i k x) yhat
    """

    k = 2.0 * np.pi * freq / vS

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds

    H_out = H_box_Diy_SH_x_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    H_cav = sphere_cavity_H(
        k,
        wave_axis="x",
        pol_axis="y",
        radius=sphere_radius,
        cav_center=sphere_center,
        tm=tm,
        **sphere_quad_kwargs,
    )

    H_bulk = H_out - H_cav

    H_vert = H_ysurface_SH_x_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    H_hor = np.zeros(3, dtype=np.complex128)

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def SV_x_z_all_components_shifted(freq, vS, outer_bounds, cavity_bounds,tm,fullSpace=True):
    """
    Finite-box analytical pieces for an SV wave propagating along +x:

        xi = xi0 exp(i k x) zhat.

    Returns
    -------
    H_bulk : complex ndarray, shape (3,)
        Bulk vector coefficient [Hx, Hy, Hz].

    H_vert : complex ndarray, shape (3,)
        Vertical x/y side-surface coefficient. Zero for z-polarized displacement.

    H_hor : complex ndarray, shape (3,)
        Outer z-face surface coefficient [Hx, Hy, Hz].

    H_total : complex ndarray, shape (3,)
        Total coefficient in your code convention:

            H_total = H_bulk - H_vert - H_hor

    Units:
        delta_a = G * rho * xi0 * H_total.
    """

    k = 2.0 * np.pi * freq / vS

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds
    xc1, xc2, yc1, yc2, zc1, zc2 = cavity_bounds

    # Bulk over outer box
    H_out = H_box_Diz_SV_x_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    # Bulk over removed cavity
    H_cav = H_box_Diz_SV_x_shifted(
        k, xc1, xc2, yc1, yc2, zc1, zc2, tm
    )

    # Rock bulk = outer - cavity
    H_bulk = H_out - H_cav

    # Vertical side surfaces: zero for z-polarized displacement
    H_vert = np.zeros(3, dtype=np.complex128)

    # Horizontal z-face surfaces
    H_hor = H_zsurface_SV_x_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm,fullSpace=fullSpace,
    )

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total

def SV_x_z_sphere_all_components_shifted(freq, vS, outer_bounds, sphere_center, sphere_radius, tm, fullSpace=True, 
                                         **sphere_quad_kwargs):
    """
    SV wave:
        xi = xi0 exp(i k x) zhat
    """

    k = 2.0 * np.pi * freq / vS

    xo1, xo2, yo1, yo2, zo1, zo2 = outer_bounds

    H_out = H_box_Diz_SV_x_shifted(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm
    )

    H_cav = sphere_cavity_H(
        k,
        wave_axis="x",
        pol_axis="z",
        radius=sphere_radius,
        cav_center=sphere_center,
        tm=tm,
        **sphere_quad_kwargs,
    )

    H_bulk = H_out - H_cav

    H_vert = np.zeros(3, dtype=np.complex128)

    H_hor = H_zsurface_SV_x_shifted_vec(
        k, xo1, xo2, yo1, yo2, zo1, zo2, tm, fullSpace=fullSpace
    )

    H_total = H_bulk - H_vert - H_hor

    return H_bulk, H_vert, H_hor, H_total