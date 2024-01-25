#!/usr/bin/python
#**********************************************************************#
#    Copyright (C) 2020 Guillaume Moroz <guillaume.moroz@inria.fr>     #
#                                                                      #
# This program is free software: you can redistribute it and/or modify #
# it under the terms of the GNU General Public License as published by #
# the Free Software Foundation, either version 2 of the License, or    #
# (at your option) any later version.                                  #
#                  http://www.gnu.org/licenses/                        #
#**********************************************************************#

import argparse
import sys as _sys
import numpy as np
import numba as _nb
from numba import jit
from scipy import special as _special
from scipy import optimize as _optimize

from random import randrange, choice
import matplotlib.pyplot as plt
from scipy.spatial import Delaunay

import tempfile
import itertools

# Main Sampling functions
def sample_indices(kernel, R, epsilon=2**-53):
    I = []
    Lambda = 1
    i = 0
    R2 = R**2
    while Lambda > epsilon:
        Lambda = kernel.F(i, R2)
        if np.random.binomial(1, Lambda) == 1:
            I.append(i)
        i += 1
    return np.array(I, dtype='int64')

def sample_module(C, R, I, invLambdas, F, epsilon):
    i = np.random.multinomial(1, C).argmax()
    c = np.random.uniform()
    f = lambda r: F(I[i], r**2)*invLambdas[i] - c
    r = _optimize.brentq(f, 0, R, xtol=epsilon)
    return r

def sample_argument(V, r, I, invLambdas, g, epsilon):
    G = g(I,r)*np.sqrt(invLambdas)
    l, u = _argtruncate(G, epsilon)
    p = np.zeros(I[u-1] - I[l] + 1, dtype='complex128')
    _instantiate_polynomial(p, I, V, G, l, u)
    p[1:] /= 0.5*p[0]*1j*np.arange(1, p.size)
    p[0] = -np.sum(p[1:])
    n = np.arange(p.size)
    c = np.random.uniform()
    f = lambda alpha: alpha + np.real(_horner(p, np.exp(1j*alpha))) - c*2*np.pi
    alpha = _optimize.brentq(f, 0, 2*np.pi, xtol=epsilon)
    return alpha

def sample_points(kernel, R, I, epsilon=2**-53, print_point=lambda x,y,i:None):
    global points_list
    F, g = kernel
    n = len(I)
    W = np.zeros(n, dtype='complex128')
    U = np.ones(n, dtype='float64')
    V = np.identity(n, dtype='complex128')
    Lambdas = np.array([F(i, R**2) for i in I])
    invLambdas = 1/Lambdas
    for i in range(n, 0, -1):
        # Draw point Wi
        r = sample_module(U/U.sum(), R, I, invLambdas, F, epsilon)
        alpha = sample_argument(V, r, I, invLambdas, g, epsilon)
        p = r*np.exp(1j*alpha)
        W[n-i] = p
        px = p.real
        py = p.imag
        print_point(px, py, n-i)

        # Compute new vector ei
        phi = g(I,r)*np.exp(1j*alpha*I)*np.sqrt(invLambdas)
        l, u = _argtruncate(phi, epsilon)
        phi = V[:, l:u].dot(phi[l:u])
        e = phi/np.linalg.norm(phi)

        # Update arrays U and V
        U -= e.real**2 + e.imag**2
        U[U<0] = 0
        _V_minus_e_estar(V, e, epsilon)
    return V, W

# Kernels

from collections import namedtuple as _namedtuple
Kernel = _namedtuple('Kernel', ['F','g'])

kernels = {
    # Ginibre point process
    'ginibre': Kernel(lambda i, r: _special.gammainc(i+1,r),
                      lambda i, r: np.where(i!=0, np.exp(i*np.log(r) - 0.5*(_special.gammaln(i+1) + r**2)),
                                                   np.exp(-0.5*r**2))),

    # Zeros of an analytic function with Gaussian coefficients
    'gaussian': Kernel(lambda i, r: np.power(r,i+1),
                       lambda i, r: np.power(r,i)*np.sqrt(i+1)),

    ## Experimental kernels
    # Gaussian kernel times 1 - r**2
    'weighted': Kernel(lambda i, r: 3*(np.power(r,i+1) - 2*np.power(r, i+2)*(i+1)/(i+2) + np.power(r, i+3)*(i+1)/(i+3)),
                       lambda i, r: np.power(r,i)*np.sqrt(i+1)),

    # Uniform modules
    'pseudo-uniform': Kernel(lambda i, r: r,
                             lambda i, r: 1),
}

# Parser  for command line arguments

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-R', type=float, help="radius", default=1.)

    parser.add_argument('-N', metavar='N', type=int,
                        help="preset N points by truncating the kernel to the N first eigenfunctions",
                        default=None)

    parser.add_argument('-k', '--kernel', metavar='kernel', type=str, help='kernel to sample : ginibre or gaussian',
                        default='ginibre')

    parser.add_argument('-p', '--precision', metavar='prec  ', type=float, help="error tolerated for internal computations",
                        default=2**-53)

    parser.add_argument('-s', '--size', metavar='size  ', type=float, help="points size in pixels", default=5)

    parser.add_argument('-t', '--time', metavar='time  ', type=int, help="refresh time in miliseconds", default=100)

    parser.add_argument('-o', '--output', metavar='output', type=str,
                        help='name of file to output the data, implies --nogui', default=None)

    parser.add_argument('-e ', '--error', action='store_true',
                        help="compute the error and the condition number for the result", default=False)

    parser.add_argument('-pg', '--profile', action='store_true', help="output time indicator some functions", default=False)

    parser.add_argument('-q ', '--quiet', action='store_true', help="disable information messages on standard output",
                        default=False)

    parser.add_argument('--nogui', action='store_true', help="output points coordinate on the terminal", default=False)

    args, unknown = parser.parse_known_args()

    if not args.quiet:
        print("Importing libraries ...")
        
        
# Util functions compiled with numba

if __name__ == '__main__':
    if not args.quiet:
        print("Compiling functions ...")
        
        
# Critical loops compiled

@jit(nopython=True)
# @_nb.njit((_nb.complex128[::1], _nb.int64[::1], _nb.complex128[:,::1], _nb.float64[::1], _nb.int64, _nb.int64))

def _instantiate_polynomial(p, I, M, G, l, u):
    for i in range(l, u):
        for j in range(i, u):
            p[I[j]-I[i]] += M[i,j]*G[i]*G[j]

@_nb.njit((_nb.complex128[:,::1], _nb.complex128[::1], _nb.complex128[::1]))
def _fused_minus_outer(V, e, f):
    for i in range(V.shape[0]):
        for j in range(V.shape[1]):
            V[i,j] -= e[i]*f[j]

@_nb.guvectorize([(_nb.complex128[::1], _nb.complex128, _nb.complex128[::1], _nb.complex128[::1])],
                '(n),(),(n)->(n)')
def _fused_minus_outer_vec(v, c, f, res):
        for j in range(res.shape[0]):
            res[j] = v[j] - c*f[j]

def _V_minus_e_estar(V, e, epsilon):
        l, u = _argtruncate(e, epsilon)
        f = e.conjugate()
        if u-l > 0.9*len(e):
            _fused_minus_outer(V, e, f)
        else:
            _fused_minus_outer_vec(V[l:u, l:u], e[l:u], f[l:u], out=V[l:u, l:u])

@_nb.njit((_nb.complex128[::1], _nb.complex128))
def _horner(p, v):
    c = p[-1]
    for i in range(len(p)-2, -1, -1):
        c = c*v + p[i]
    return c

def _argtruncate(v, epsilon):
    vbig = abs(v) > epsilon
    l = np.argmax(vbig)
    u = len(vbig) - np.argmax(vbig[::-1])
    return l, u

# Qt interface functions

def _init_figure(R, size):
    global _scatter, _view, _spot, _app
    # Launch app
    pg.setConfigOptions(background = 'w', foreground = 'k')
    _app = pg.mkQApp()

    # Create the main view
    _view = pg.PlotWidget()
    _view.setRenderHint(pg.Qt.QtGui.QPainter.HighQualityAntialiasing)
    _view.resize(800, 600)
    _view.setRange(xRange=(-R,R), yRange=(-R,R))
    _view.setWindowTitle('Determinantal point process')
    _view.setTitle('Sampling the number of points ...')
    _view.setAspectLocked(True)
    _view.show()
    
    # Create the circle and add it to the view
    circle = pg.Qt.QtWidgets.QGraphicsEllipseItem()
    circle.setRect(-R, -R, 2*R, 2*R)
    circle.setPen(pg.mkPen(width=2, color='k'))
    _view.addItem(circle)
    
    # Create the scatter plot and add it to the view
    _scatter = pg.ScatterPlotItem(symbol='o')
    _scatter.setSize(size)
    _view.addItem(_scatter)

    # Spot
    _spot = np.empty(1, dtype=_scatter.data.dtype)
    _spot['pen'] = pg.mkPen(width=1, color='b')
    _spot['brush'] = pg.mkBrush(None)
    _spot['size'] = size
    _spot['visible'] = True
    if pg.Qt.QT_LIB not in ['PySide2', 'PySide6']:
        _spot['targetQRectValid'] = False
    _scatter.updateSpots(_spot)

    _app.processEvents()

def _update_figure():
    global _scatter, _view, _Npoints
    pad =  len(str(_Npoints))
    _view.setTitle('<pre>Sampling: {0: >{2}}/{1} points</pre>'.format(_scatter.data.size, _Npoints, pad))
    _scatter.prepareGeometryChange()
    _scatter.bounds = [None, None]

def _print_point_qt(px, py, i):
    _scatter.data.resize(i+1, refcheck=False)
    _scatter.data[i] = _spot
    if pg.Qt.QT_LIB not in ['PySide2', 'PySide6']:
        _scatter.data[i]['targetQRect'] = pg.Qt.QtCore.QRectF()
    _scatter.data[i]['x'] = px
    _scatter.data[i]['y'] = py
    _app.processEvents()

def qt_sample(R, N = None, kernel=kernels['ginibre'], precision=2**-53, size=5, refresh=100, error=False, quiet=False):
    global _Npoints, pg
    import pyqtgraph as pg
    if N is not None and kernel.F(N-1, R**2) == 0:
        raise ValueError("N is too big")
    _init_figure(R, size)
    if N is None:
        I = sample_indices(kernel, R, precision)
    else:
        I = np.arange(N)
    _Npoints = len(I)
    timer = pg.Qt.QtCore.QTimer()
    timer.timeout.connect(_update_figure)
    timer.start(refresh)
    V, W = sample_points(kernel, R, I, precision, _print_point_qt)
    timer.stop()
    _update_figure()
    _app.processEvents()
    if error:
        _view.setTitle('Computing the error and the condition number ...')
        _app.processEvents()
        Error = np.linalg.norm(V)
        tI = I.reshape(-1,1)
        M = kernel.g(tI, np.abs(W))*np.exp(1j*np.angle(W)*tI)/np.sqrt(kernel.F(tI, R**2))
        ConditionNumber = np.linalg.cond(M)
        _view.setTitle('<pre>Number of points: {0}        Error: {1:.3e}        Condition number: {2:.3e}</pre>'
                      .format(_Npoints, Error, ConditionNumber))
    else:
        _view.setTitle('<pre>Number of points: {0}</pre>'.format(_Npoints))
    Blue = pg.mkBrush('b')
    _scatter.setBrush([Blue]*len(_scatter.data))
    _app.exec_()
    
# Text interface functions

def _build_print_point(output, quiet, n):
    pad =  len(str(n))
    message = '\r{{0: >{0}}}/{1} '.format(pad, n)
    if output is None and quiet:
        print_point_txt = lambda x, y, i: None
    elif output is None and not quiet:
        print_point_txt = lambda x, y, i: _sys.stdout.write(message.format(i+1))
    elif output is not None and quiet:
        print_point_txt = lambda x, y, i: output.write("{0} {1}\n".format(x, y))
    else:
        print_point_txt = lambda x, y, i: _sys.stdout.write(message.format(i+1)) and output.write("{0} {1}\n".format(x, y))
    return print_point_txt

def sample(R, N = None, kernel=kernels['ginibre'], precision=2**-53, error=False, quiet=False, output=None):
    if N is None:
        if not quiet:
            print('Sampling the number of points ...')
        I = sample_indices(kernel, R, precision)
    else:
        if kernel.F(N-1, R**2) == 0:
            raise ValueError("N is too big")
        I = np.arange(N)
    print_point_txt = _build_print_point(output, quiet, len(I))
    if not quiet:
        print('Sampling the points ...')
    V, W = sample_points(kernel, R, I, precision, print_point_txt)
    if not quiet:
        print()
    if error:
        if not quiet:
            print('Computing the error and the condition number ...')
        Error = np.linalg.norm(V)
        tI = I.reshape(-1,1)
        M = kernel.g(tI, np.abs(W))*np.exp(1j*np.angle(W)*tI)/np.sqrt(kernel.F(tI, R**2))
        ConditionNumber = np.linalg.cond(M)
        if not quiet:
            print('Error: {0:.3e}'.format(Error))
            print('Condition number: {0:.3e}'.format(ConditionNumber))
        if output is not None:
            output.write('# Error: {0:.3e}\n'.format(Error))
            output.write('# Condition number: {0:.3e}\n'.format(ConditionNumber))
        return W, Error, ConditionNumber
    else:
        return W
    
# Main script if called from command line


if __name__ == '__main__':
    if args.profile:
        import line_profiler as lp
        import atexit
        profile = lp.LineProfiler()
        sample_points = profile(sample_points)
        sample_argument = profile(sample_argument)
        atexit.register(profile.print_stats)
    if args.nogui or args.output is not None:
        if args.output is None:
            output = _sys.stdout
        else:
            output = open(args.output, 'w')
        sample(args.R, args.N, kernels[args.kernel], args.precision, args.error, args.quiet, output)
        if args.output is not None:
            output.close()

    # else:
        # qt_sample(args.R, args.N, kernels[args.kernel], args.precision, args.size, args.time, args.error, args.quiet)