#!/usr/bin/env python
"""Library of plotting tools, used by the user or plotwrap.py"""


import math
import warnings
import numpy as np
import itertools
import copy
from scipy import signal
from numpy import pi

with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=PendingDeprecationWarning)
    import matplotlib
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D

GRAPH_OUTPUT_LOCATION = 'graphs/' # don't forget the trailing slash
GRAPHDUMP_OUTPUT_LOCATION = 'graphdump/' # don't forget the trailing slash
GRAPH_OUTPUT_FORMAT = 'eps'


__MARKS = 'xov^<>12348sp*hH+,Dd|_'
__COLORS = 'kgrcmy' + 'k'*len(__MARKS)

FONTSIZE = 15
MARKERSIZE = FONTSIZE*2
ERR_MARKERSIZE = 8
matplotlib.rcParams.update({'font.size': FONTSIZE})
#matplotlib.rc('font',**{'sans-serif':['Helvetica']})
#matplotlib.rc('text', usetex=True)

#----- HELPER FCTS
def remove_zeropad(x,y,repad_ratio):
    """Returns the truncated x and y arrays with the zero padding removed"""
    if len(x) > len(y):
        raise ValueError('Expected x and y to have same length')
    
    first, last = y.nonzero()[0][[0,-1]]

    padding = int(round(repad_ratio*(last-first+1)))
    lo = max(0, first - padding)
    hi = min(len(y), last + padding)

    outx = x[lo:(hi+1)]
    outy = y[lo:(hi+1)]
    return outx,outy

def discrete(*args, label='curve0', axes=None):
    """Plots a discrete graph with vertical bars"""
    if len(args) < 1 and len(args) > 2:
        raise Exception("Too many or too little inputs")
    elif len(args) == 2:
        y = args[1]
        x = args[0]
    else:
        y = args[0]
        x = np.arange(len(y))

    if axes == None:
        ax = plt.axes()
    else:
        ax = axes

    ax.plot((x, x) , (y, np.zeros(len(y))), 'k-')

    # Pad the limits so we always see the leftmost-rightmost point
    x_pad = (x[1]-x[0])
    x_lims = [min(x)-x_pad, max(x)+x_pad]
    ax.plot(x_lims, [0,0], 'k-')
    ax.set_xlim(x_lims)
    return ax

def continuous(*args, label='curve0', axes=None):
    """Plots a continuous graph"""
    if len(args) < 1 and len(args) > 2:
        raise Exception("Too many or too little inputs")
    elif len(args) == 2:
        y = args[1]
        x = args[0]
    else:
        y = args[0]
        x = np.arange(len(y))

    if axes == None:
        ax = plt.axes()
    else:
        ax = axes

    lh = ax.plot(x, y, 'k-', label=label)
    #plt.scatter(x, y, marker='.')
    #plt.plot(indep_ax,curve,'k-')
    x_lims = [min(x), max(x)]
    ax.set_xlim(x_lims)

    return ax

def scatter(x, y, yerr, x_label='', y_label='',axes=None, savename='',show_std=True, **kwargs):
    """Scatter plot, with errorbars if specified"""

    if axes == None:
        ax = plt.axes()
    else:
        ax = axes

    lh = ax.errorbar(x, y, yerr, capsize=None, markersize=ERR_MARKERSIZE, **kwargs )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    save(savename)
    return ax

def scatter_noerr(x, y, x_label='', y_label='',axes=None, s=MARKERSIZE, savename='',show_std=True, **kwargs):
    """Scatter plot with no errorbars"""

    if axes == None:
        ax = plt.axes()
    else:
        ax = axes

    lh = ax.scatter(x, y, s=s, **kwargs )
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)

    save(savename)
    return ax

def surface3d(x,y,z, density=20, **kwargs):
    """3d plot of the x, y vectors and z 2d array"""

    xstride = max(int(round(len(x)/density)),1)
    ystride = max(int(round(len(y)/density)),1)
    
    X, Y = np.meshgrid(x,y)
    fig = plt.figure()
    ax = fig.gca(projection='3d')
    ax.plot_surface(X,Y,z,cstride=xstride, rstride=ystride, cmap=matplotlib.cm.coolwarm, **kwargs)
    return fig, ax

def save(name, **kwargs):
    """Saves the current figure to """
    if name:
        fname = GRAPH_OUTPUT_LOCATION + name + '.' + GRAPH_OUTPUT_FORMAT
        plt.savefig(fname , bbox_inches='tight', format=GRAPH_OUTPUT_FORMAT)

def show():
    plt.show()

def change_fontsize(fsize):
    globals()['FONTSIZE'] = fsize
    matplotlib.rcParams.update({'font.size': fsize})

#----- GRAPHS
def train_and_test(err_list, y_label='Error', legend_labels=('Train', 'Test'), axes=None, savename='', **kwargs):
    """Plots an evolution graph of the test and validation errors"""
    if axes==None:
        ax = plt.axes()
    else:
        ax = axes

    avg = np.zeros(err_list[0].shape)
    for errs in err_list:
        ax.plot(errs[0,:], color='#CC0000')
        ax.plot(errs[1,:], color='#00CC00')
        avg += errs
    avg /= len(err_list)

    ax.plot(avg[0,:], color='#660000', linewidth=4, label=legend_labels[0])
    ax.plot(avg[1,:], color='#006600', linewidth=4, label=legend_labels[1])
    ax.legend()

    #ymin, ymax = (0, 1)
    #xmin, xmax = ax.get_xlim()
    #ax.set_ylim([ymin,ymax])
    #ax.set_xlim([xmin,xmax])

    ax.set_xlabel('Epochs')
    ax.set_ylabel(y_label)

    save(savename)
    return ax


#----- CATTED GRAPHS
def cat_graphs(graphs, rows=2,subplotsize=(8,6), savename=''):
    """Concatenate the figures together together
    graphlist: list of tuples of (fct name, args, kwarg)"""
    
    # Make sure lst is a list, indep of input.
    if type(graphs) is tuple:
        lst = [graphs.copy]
    elif type(graphs) is list:
        lst = graphs.copy()
    else:
        raise ValueError("Expected first argument to be a tuple or list. Currently is: " + str(type(graphs)))

    spcount = len(lst) # Subplotcount
    spargs = (rows, (spcount+1)//2) # Premade argument for rows-cols in add_subplots
    figsize = (spargs[1]*subplotsize[0], spargs[0]*subplotsize[1])
    fig = plt.figure(figsize=figsize)
    
    # Build axes and draw in them
    for k, tpl in enumerate(lst):
        # Break down the tuple if needed
        fct = tpl[0]
        fargs = tpl[1] if len(tpl) > 1 else tuple()
        fkwargs = tpl[2] if len(tpl) > 2 else dict()
        if len(tpl) > 4: raise ValueError('Input list element is a length ' + len(tpl) + 'iterable. Exected length 3 or 4')
        # Build and populate axes
        ax = fig.add_subplot(*spargs, k+1)
        fkwargs['axes'] = ax
        fct(*fargs, **fkwargs)

        #Make axes title
        if len(tpl) == 4:
            ax.set_title(tpl[3])
        else:
            ax.set_title(fct.__name__)

    # Finalize figure
    fig.tight_layout()
    save(savename)

def all_graphs(p,ctrl=None):
    
    glist = []
    glist.append((crosscorr_zneg, [p]))
    glist.append((analog, [p]))
    glist.append((pulse, [p]))

    # CTRL dependent graphs
    if ctrl is not None:
        glist.append((barywidth,
                        [p],
                        dict(axes=None, fit_type='linear', reach=ctrl.bmap_reach , scaling_fct=ctrl.bmap_scaling, residuals=False, force_calculate=False, disp=False)
                        ))

    cat_graphs(glist)

#----- SIMBD GRAPHS


if  __name__ == '__main__':
    x = np.sqrt(np.arange(30)+1)
    y = np.log(np.arange(30)+1)

    data = np.vstack((x,y))

    num_folds = 10
    err_list = []
    for k in range(num_folds):
        rnd = 0.7*np.random.rand(*data.shape)
        err_list.append(data+rnd)

    hair(err_list); show()

