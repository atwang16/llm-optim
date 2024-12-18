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

import os
# change dir to parent dir of this file
ROOT_DIR = os.path.dirname(os.path.abspath(__file__)) + "/.."
os.chdir(ROOT_DIR)
print( os.getcwd() )
#%%
from llmoptim.data_toy2d import *

# %%
def grid_run(lski, gmin = [-1,-1.], gmax = [1., 1.], steps = [3, 3]):
    grid = np.meshgrid(np.linspace(gmin[0], gmax[0], steps[0]), np.linspace(gmin[1], gmax[1], steps[1]))
    grid = np.array(grid).reshape(2, -1).T
    trajs=  []
    for init_theta in grid:
        print(init_theta)
        infersgd_name=f'grid_{init_theta[0]:.3f}_{init_theta[1]:.3f}'
        lski.infer_sgd(infer_init_thetas = init_theta, infersgd_name = infersgd_name)

        traj = np.load(f'{lski.output_root}/{infersgd_name}/sgd_infer_trajectory.npz', allow_pickle=True)['arr_0'].item()['thetas']
        traj = np.concatenate([[init_theta[:, None]], traj], axis=0)
        trajs.append( traj )
        # plot_progressive_trajectory(torch.from_numpy(traj), lski.model, frame_dirname=f'{infersgd_name}_frames', plot_range=[[-2, -2], [2,5]])
    return trajs
def visgrid(trajs):
    for traj in trajs:
        plt.plot(traj[:, 0], traj[:, 1], '--')
        plt.scatter(traj[:, 0], traj[:, 1], c=np.arange(traj.shape[0]), cmap='viridis')
        plt.xlim(-2, 2)
        plt.ylim(-2, 5)
        plt.show()

# %%
if __name__ == "__main__":

    # lski = LSKI_convex_underparam('multirun', theta_init = [0.6, 0.5])
    lski = LSKI_nonconvex_underparam('multirun_nonconvex2', theta_init = [1.0, -2.])
    #%%
    lski.generate_data(lr=.2, num_steps = 50, plot_range=[[-2, -2], [2,5]])
    #%%
    lski.infer_kernels()
    #%%
    trajs = grid_run(lski, gmin = [-1.8, -1.8], gmax = [1.6, 4.8], steps = [5, 5])
    visgrid(trajs)
    #%%
    lski.infer_sgd(infer_init_thetas=[-1.5, -1.5])
    #%%
    #traj = np.load(f'{lski.output_root}/inferred_sgd/sgd_infer_trajectory.npz', allow_pickle=True)['arr_0'].item()['thetas']
    traj = np.load(f'{lski.output_root}/inferred_sgd/sgd_infer_trajectory.npz', allow_pickle=True)['arr_0'].item()['thetas']
    print(traj.shape)
    plot_progressive_trajectory(torch.from_numpy(traj), lski.model, frame_dirname='sgdrunframes2', plot_range=[[-2, -2], [2,5]])
#%%