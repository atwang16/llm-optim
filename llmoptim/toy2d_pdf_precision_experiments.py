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
    load_ckpt_to_traj
)


def generate_baseline_data(baseline_exp_name="nonconvex_underparam_baseline", init_thetas=[1.0, -1.5]):
    """
    Generate the 'ground truth' SGD trajectory once. Saves checkpoints and visuals in a baseline folder.
    Returns the model object and the path to the baseline ckpts folder.
    """
    print(f"=== Generating Baseline Data in '{baseline_exp_name}' ===")
    lski = LSKI_nonconvex_underparam()
    lski.exp_name = baseline_exp_name
    lski.output_root = f"output/{baseline_exp_name}"
    lski.model.output_dir = lski.output_root
    lski.model.ckpt_dir   = os.path.join(lski.model.output_dir, 'ckpts')
    lski.model.visual_dir = os.path.join(lski.model.output_dir, 'visuals')
    os.makedirs(lski.model.ckpt_dir, exist_ok=True)
    os.makedirs(lski.model.visual_dir, exist_ok=True)

    # Override the init thetas to match for baseline
    lski.model.thetas = torch.nn.Parameter(torch.tensor(init_thetas).reshape(-1,1), requires_grad=True)

    # Generate the baseline SGD trajectory (and visuals)
    lski.generate_data()  # This calls generate_sgd_traj_and_visuals, storing baseline in ckpts/ & visuals/

    baseline_ckpts_path = lski.model.ckpt_dir
    return lski.model, baseline_ckpts_path


def run_inference_experiment(refinement_depth, baseline_ckpts_path, init_thetas=[1.0, -1.5]):
    """
    Uses the baseline ckpts (ground truth data) to run kernel_inference.py for a given refinement_depth.
    Then runs sgd_inference.py from the same init_thetas to produce the LLM-based 'SGD' trajectory.
    Returns the final trajectory for plotting.
    """
    exp_name = f"nonconvex_underparam_depth{refinement_depth}"
    output_root = f"output/{exp_name}"
    os.makedirs(output_root, exist_ok=True)

    # 1) Run kernel_inference.py (builds transition kernel from the baseline ckpts)
    output_dir = os.path.join(output_root, "inferred_kernels")
    os.makedirs(output_dir, exist_ok=True)
    kernel_inference_cmd = (
        f"python kernel_inference.py "
        f"--ckpts_path {baseline_ckpts_path} "
        f"--llama_v 2 "
        f"--output_dir {output_dir} "
        f"--depth {refinement_depth}"
    )
    print(kernel_inference_cmd)
    os.system(kernel_inference_cmd)

    # 2) Run sgd_inference.py to simulate LLM-based 'SGD' from the same init
    infer_init_ckpt_path = os.path.join(output_root, "infer_init_ckpt.pth")
    # Create a dummy checkpoint with the initial thetas:
    dummy_ckpt = {"model_state_dict": {"thetas": torch.tensor(init_thetas).reshape(-1,1)}}
    torch.save(dummy_ckpt, infer_init_ckpt_path)

    inferred_sgd_path = os.path.join(output_root, "inferred_sgd")
    os.makedirs(inferred_sgd_path, exist_ok=True)
    sgd_inference_cmd = (
        f"python sgd_inference.py "
        f"--init_ckpt_path {infer_init_ckpt_path} "
        f"--output_dir {inferred_sgd_path} "
        f"--kernels_dir {output_dir}/kernel/ "
        f"--sample "
        f"--steps 50"
    )
    print(sgd_inference_cmd)
    os.system(sgd_inference_cmd)

    # 3) Load the final LLM-inferred trajectory
    sgd_infer_trajectory_npz = os.path.join(inferred_sgd_path, "sgd_infer_trajectory.npz")
    data = np.load(sgd_infer_trajectory_npz, allow_pickle=True)['arr_0'].item()
    trajectory = data["thetas"]  # shape: (num_steps, 2, 1)
    return trajectory


def plot_comparison(baseline_model, baseline_ckpts_path, all_trajectories, depth_list):
    """
    Plots the ground-truth baseline trajectory plus the LLM-inferred trajectories for each refinement depth
    on the same 2D loss surface.
    """
    fig, ax = plt.subplots()

    # 1) Load the baseline trajectory from ckpts
    baseline_traj = load_ckpt_to_traj(baseline_ckpts_path)  # shape: (num_steps+1, 2, 1) if 50 steps
    baseline_traj = baseline_traj[..., 0]  # flatten the last dim

    # 2) Plot the 2D contour
    x_range = [-2, 2]
    y_range = [-2, 5]  # as mentioned in your final report needs y up to 5
    X, Y, Z = compute_grid_values(baseline_model, x_range=x_range, y_range=y_range, resolution=100)
    c = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(c, ax=ax, label='Function Value')

    # 3) Plot the baseline ground-truth trajectory
    ax.plot(baseline_traj[:,0], baseline_traj[:,1], "o-", color="black", label="Baseline SGD")

    # 4) Plot each LLM-inferred trajectory
    colors = ["red", "green", "blue", "orange"]
    for i, depth in enumerate(depth_list):
        traj = all_trajectories[depth][..., 0]  # shape (steps, 2)
        color = colors[i % len(colors)]
        label = f"LLM-Inferred (depth={depth})"
        ax.plot(traj[:,0], traj[:,1], "o-", color=color, label=label)

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title("Comparison: Baseline vs LLM-Inferred (NonConvex Underparam)")
    ax.legend()

    out_plot = "output/nonconvex_underparam_compare_depths.png"
    plt.savefig(out_plot, dpi=150)
    print(f"Comparison plot saved to {out_plot}")
    plt.show()


if __name__ == "__main__":
    """
    1) Generate baseline data once (SGD).
    2) For each refinement depth, run kernel_inference using the same baseline ckpts + the same init.
    3) Plot all on the same figure.
    """
    # 1) Generate baseline data
    baseline_model, baseline_ckpts_path = generate_baseline_data(
        baseline_exp_name="nonconvex_underparam_baseline",
        init_thetas=[-1.9, 1.9]  # choose your consistent init for baseline
    )

    # 2) Run LLM-based experiments with different refinements (only 1,2 here)
    depth_list = [1, 2]
    all_trajectories = {}
    for depth in depth_list:
        traj = run_inference_experiment(
            refinement_depth=depth,
            baseline_ckpts_path=baseline_ckpts_path,
            init_thetas=[-1.9, 1.9]  # same init as baseline
        )
        all_trajectories[depth] = traj

    # 3) Plot comparison
    plot_comparison(baseline_model, baseline_ckpts_path, all_trajectories, depth_list)
