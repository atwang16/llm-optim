import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import imageio

from data_toy2d import (
    LSKI_nonconvex_underparam, 
    plot_progressive_trajectory, 
    compute_grid_values,
    # TODO: Possibly other needed imports 
)

def run_nonconvex_experiment(refinement_depth):
    """
    Creates a nonconvex underparam scenario, generates data (SGD checkpoints),
    then runs kernel inference + LLM-based 'SGD' inference with the specified refinement depth.
    Outputs are stored in separate directories for easy comparison.
    Returns the final inferred trajectory for plotting/comparison.
    """

    exp_name = f"nonconvex_underparam_depth{refinement_depth}"
    lski = LSKI_nonconvex_underparam()  

    # override the default experiment name/output directories
    lski.exp_name = exp_name
    lski.output_root = f"output/{exp_name}"
    lski.model.output_dir = lski.output_root
    lski.model.ckpt_dir   = os.path.join(lski.model.output_dir, 'ckpts')
    lski.model.visual_dir = os.path.join(lski.model.output_dir, 'visuals')
    os.makedirs(lski.model.ckpt_dir, exist_ok=True)
    os.makedirs(lski.model.visual_dir, exist_ok=True)

    print(f"\n=== Running Nonconvex Underparam Experiment with depth={refinement_depth} ===\n")

    # 1) generate the base SGD trajectory as the ground truth
    #    this also produces a baseline set of visuals (the "true" SGD training).
    lski.generate_data()  # calls generate_sgd_traj_and_visuals() with the given LR, etc.

    # 2) run kernel_inference.py with the specified refinement depth
    ckpts_path = f"{lski.output_root}/ckpts/"
    output_dir = f"{lski.output_root}/inferred_kernels/" # TODO: check the output dirs
    kernel_inference_cmd = (
        f"python kernel_inference.py "
        f"--ckpts_path {ckpts_path} "
        f"--llama_v 2 "
        f"--output_dir {output_dir} "
        f"--depth {refinement_depth}"  # <--- KEY PART
    )
    print(kernel_inference_cmd)
    os.system(kernel_inference_cmd)

    # 3) run sgd_inference.py to generate the LLM-based “SGD inference” trajectory
    #    we choose some initialization, e.g. [1.5, 2.5]
    lski.infer_sgd(infer_init_thetas=[1.5, 2.5])

    # 4) load the final LLM-inferred trajectory for comparison
    #    The trajectory is stored in 'output_dir/inferred_sgd/sgd_infer_trajectory.npz'
    infer_sgd_path = f"{lski.output_root}/inferred_sgd/sgd_infer_trajectory.npz"
    data = np.load(infer_sgd_path, allow_pickle=True)['arr_0'].item()
    # 'data' is a dict of parameter_name -> [timesteps, ...]
    # For toy2D, the parameter key should be "thetas" or something similar:
    trajectory = data["thetas"]  # shape: (num_steps, 2,1)

    return trajectory, lski.model


def plot_comparison(traj_dict, model_dict):
    """
    Plots the final LLM-inferred trajectories for each refinement depth side-by-side
    on the same figure (or multiple figures), using the same data distribution and loss landscape.
    
    :param traj_dict: dict[refinement_depth] -> np.ndarray of shape (steps, 2, 1)
    :param model_dict: dict[refinement_depth] -> NonConvexProblemModel instance 
    """
    fig, ax = plt.subplots()
    
    # take just one "representative" model instance to compute the contour,
    # since all runs used the same (theta_star, data). E.g. pick the smallest refinement depth's model:
    any_depth = sorted(model_dict.keys())[0]
    representative_model = model_dict[any_depth]

    # generate the same 2D contour
    # If the existing "plot_range" was [-2, -2], [2, 2], let's reuse that:
    x_range = [-2, 2]
    y_range = [-2, 2]
    X, Y, Z = compute_grid_values(representative_model, x_range=x_range, y_range=y_range, resolution=100)
    c = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(c, ax=ax, label='Function Value')
    
    colors = ["red", "green", "blue", "orange"]  # For variety if needed
    for idx, (depth, traj) in enumerate(traj_dict.items()):
        # traj shape: (steps, 2, 1)
        color = colors[idx % len(colors)]
        label = f"RefDepth={depth}"
        # Flatten for x,y: (steps, 2)
        flat_traj = traj[..., 0]
        ax.plot(flat_traj[:,0], flat_traj[:,1], "o-", color=color, label=label)
    
    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Comparison of LLM-Inferred Trajectories (NonConvex Underparam)")
    ax.legend()
    
    plt.tight_layout()
    out_plot = "output/nonconvex_underparam_comparison.png"
    plt.savefig(out_plot)
    print(f"Comparison plot saved to {out_plot}")
    plt.show()


if __name__ == "__main__":
    # store final trajectories (and references to the model) in dictionaries
    all_trajectories = {}
    all_models = {}

    for depth in [1, 2, 3]:
        traj, model = run_nonconvex_experiment(refinement_depth=depth)
        all_trajectories[depth] = traj
        all_models[depth] = model
    
    # create a single comparison plot
    plot_comparison(all_trajectories, all_models)
