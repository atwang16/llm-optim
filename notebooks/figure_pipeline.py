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
def visgrid(trajs, plot_range=None, model=None, margin=0.5, frame_dirname='gridframes'):
    for i, traj in enumerate(trajs):
        # plt.plot(traj[:, 0], traj[:, 1], '--')
        # plt.scatter(traj[:, 0], traj[:, 1], c=np.arange(traj.shape[0]), cmap='viridis')
        # if plot_range is not None:
        #     plt.xlim(plot_range[0][0], plot_range[1][0])
        #     plt.ylim(plot_range[0][1], plot_range[1][1])
        # plt.show()
        save_path = f'{model.visual_dir}/{frame_dirname}/{i:03d}.png'
        plot_trajectory(traj, model, save_path, plot_range=plot_range, margin=margin)
        # if plot_range is None:
        #     minr, maxr = trajectory.min(axis=0)[0].numpy(), trajectory.max(axis=0)[0].numpy()
        #     x_range = [minr[0] - margin, maxr[0] + margin]
        #     y_range = [minr[1] - margin, maxr[1] + margin]
        # else:
        #     minr, maxr = plot_range
        #     x_range = [minr[0], maxr[0] ]
        #     y_range = [minr[1], maxr[1] ]
        # X, Y, Z = compute_grid_values(
        #     model, 
        #     x_range=x_range, 
        #     y_range=y_range
        # )
def plot_trajectory(trajectory, model, save_path=None, plot_range=None, margin=0.4, cmap='viridis', init_theta=None, simple=True):
    '''
        trajectory: torch.tensor of shape (num_steps, 2, 1)
        model: ConvexProblemModel
        plot_range: list of list of 2 elements, [[x_min, y_min], [x_max, y_max]]
        margin: float, margin to add to the plot range
        cmap: str, colormap

    '''
    trajectory = trajectory[..., 0 ].copy()
    if plot_range is None:
        minr, maxr = trajectory.min(axis=0), trajectory.max(axis=0)
        x_range = [minr[0] - margin, maxr[0] + margin]
        y_range = [minr[1] - margin, maxr[1] + margin]
    else:
        minr, maxr = plot_range
        x_range = [minr[0], maxr[0] ]
        y_range = [minr[1], maxr[1] ]
    X, Y, Z = compute_grid_values(
        model, 
        x_range=x_range, 
        y_range=y_range
    )
    init_theta = trajectory[0] if init_theta is None else init_theta    
    # Generate frames
    for i in range(len(trajectory), len(trajectory) + 1):
        fig, ax = plt.subplots()
        c = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
        ax.plot(trajectory[:i, 0], trajectory[:i, 1], 'o-', color='magenta', label='SGD run')
        plt.plot(init_theta[0], init_theta[1], 'gX', label='Initial theta', markersize=10)
        if not simple:
            plt.colorbar(c, ax=ax, label='Function value')
            ax.set_xlabel(r'$\theta_1$')
            ax.set_ylabel(r'$\theta_2$')
            ax.set_title('SGD Trajectory')
            ax.legend()
        else:
            ax.axis('off')
        
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close(fig)
    
    print(f"Frames saved in '{save_path}'.")
# %%
if __name__ == "__main__":

    # lski = LSKI_convex_underparam('multirun', theta_init = [0.6, 0.5])
    lski = LSKI_convex_underparam('figure_pipeline', theta_init = [0.0, 0.5])
    #lski = LSKI_nonconvex_underparam('multirun_nonconvex0', theta_init = [-1.9, 1.9])
    #%%
    lski.generate_data(lr=.1, num_steps = 50, plot_range=[[-1, -1], [5,5]])
    #%%
    lski.infer_kernels()
    #%%
    infer_init_thetas = [0.0, 1.5]
    lski.infer_sgd(infer_init_thetas=infer_init_thetas)
    #%%
    #traj = np.load(f'{lski.output_root}/inferred_sgd/sgd_infer_trajectory.npz', allow_pickle=True)['arr_0'].item()['thetas']
    traj = np.load(f'{lski.output_root}/inferred_sgd/sgd_infer_trajectory.npz', allow_pickle=True)['arr_0'].item()['thetas']
    traj = np.concatenate([[np.array(infer_init_thetas)[:, None]], traj], axis=0)
    print(traj.shape)
    plot_progressive_trajectory(torch.from_numpy(traj), lski.model, frame_dirname='sgdrunframes', plot_range=[[-1, -1], [5,5]])
#%%
    trajs = grid_run(lski, gmin = [-.0, .5], gmax = [4, 4.], steps = [5, 5])
    #%%
    visgrid(trajs, plot_range=[[-.8, -.8], [4.8, 4.8]])
#%%
    import einops
    import glob
    imgs = glob.glob("/localhome/xya120/studio/llm-optim/output/figure_pipeline/visuals/gridframes/*")
    # sort
    imgs = sorted(imgs)
    imgs = [ plt.imread(img) for img in imgs ]
    grid = einops.rearrange(imgs, '(b1 b2) h w c -> (b1 h) (b2 w) c', b1=5, b2=5)
    plt.axis('off')
    plt.imshow(grid)


#%%
