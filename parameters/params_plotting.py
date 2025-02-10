# -*- coding: utf-8 -*-
"""
@author: juschu

Parameters needed to define appearance of figures

 - colors
 - fontsizes
"""

import matplotlib as mpl
from matplotlib.colors import LinearSegmentedColormap


params = {}

## colors
# AU task: attended->unattended
params['AU'] = {'clim': 0.2, 'color': 'red'}
# UA task: unattended->attended
params['UA'] = {'clim': 0.2, 'color': '#5c5794ff'}
# UU task/control: unattended->unattended
params['UU'] = {'color': 'black'}

# colormap with pure white at lower values
cdict_red = {'red':   ((0.0, 1.0, 1.0),
                       (1.0, 1.0, 1.0)),
            'green': ((0.0, 1.0, 1.0),
                      (0.1, 1.0, 1.0),
                      (0.3, 0.5, 0.5),
                      (1.0, 0.0, 0.0)),
            'blue':  ((0.0, 1.0, 1.0),
                      (0.1, 1.0, 1.0),
                      (0.3, 0.5, 0.5),
                      (1.0, 0.0, 0.0))
            }
mpl.colormaps.register(cmap=LinearSegmentedColormap(name='myReds', segmentdata=cdict_red))

## fontsizes
params['fontsizes'] = {'legend': 12, 'axes': 15, 'text': 16, 'title': 18, 'setup_text': 50, 'setup_axes': 30}
