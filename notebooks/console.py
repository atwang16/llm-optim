#%%
from IPython import get_ipython

# Get the current IPython instance
ipython = get_ipython()

if ipython is not None:
    # Run magic commands
    ipython.run_line_magic('matplotlib', 'inline')
    ipython.run_line_magic('config', "InlineBackend.figure_format = 'retina'")
    ipython.run_line_magic('reload_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%%
# generate mesh grid 2D
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
gamma = .9
# z is the conditional pdf p(y|x) = N(y|(1-gamma*2)x; gamma^2)
mu = (1 - gamma**2) * X
sigma = gamma**2
# use numpy pdf
from scipy.stats import norm
Z = norm.pdf(Y, loc=mu, scale=sigma)
#Z = np.exp(-0.5 * ((Y - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
# show imag
plt.imshow(Z, origin='lower', extent=(-2, 2, -2, 2))
# %%


#%%