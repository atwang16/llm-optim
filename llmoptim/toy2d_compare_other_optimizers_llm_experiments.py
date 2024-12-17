#!/usr/bin/env python
"""
toy2d_compare_course_optimizers_llm.py

Compares multiple optimization methods from your course 
(GD, Momentum, Nesterov, AdaGrad, Adam) on the toy2D NonConvexProblemModel 
with an LLM-based inference approach.

Steps for each optimizer method (baseline):
  1. Train NonConvexProblemModel -> produce checkpoints (ckpts_<method>).
  2. kernel_inference.py -> build transition kernels from those ckpts.
  3. sgd_inference.py -> produce LLM-based param trajectory (starting from baseline's *initial* param, step_000).
  4. Plot baseline vs. LLM overlay in param space.

Usage:
  python toy2d_compare_course_optimizers_llm.py
"""

import os
import subprocess
import torch
import numpy as np
import matplotlib.pyplot as plt
import random

from data_toy2d import (
    NonConvexProblemModel,
    compute_grid_values,
)

def generate_baseline_ckpts(model, method="GD", lr=0.1, num_steps=50):
    """
    Trains 'model' for 'num_steps' using the specified method (GD, Momentum, Nesterov, AdaGrad, Adam).
    Saves checkpoints in model.output_dir + f"/ckpts_{method}".
    Returns the ckpt_dir path.
    """
    from data_toy2d import NonConvexProblemModel
    
    # Clone the model to avoid overwriting the original's parameters
    model_clone = NonConvexProblemModel(
        init_params=model.thetas.detach().cpu().numpy().flatten(),
        random_seed=model.random_seed,
        theta_star=model.theta_star.detach().cpu().numpy().flatten(),
        batch_size=model.batch_size,
        dataset_size_N=model.dataset_size_N,
        name=model.name + f"_{method}",
        output_root=model.output_dir
    )
    model_clone.thetas.data = model.thetas.data.clone().detach()

    ckpt_dir = os.path.join(model_clone.output_dir, f"ckpts_{method}")
    os.makedirs(ckpt_dir, exist_ok=True)

    # Pick PyTorch optimizer
    if method.lower() in ["gd","gradientdescent"]:
        optimizer = torch.optim.SGD(model_clone.parameters(), lr=lr, momentum=0.0)
    elif method.lower() == "momentum":
        optimizer = torch.optim.SGD(model_clone.parameters(), lr=lr, momentum=0.9)
    elif method.lower() == "nesterov":
        optimizer = torch.optim.SGD(model_clone.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif method.lower() == "adagrad":
        optimizer = torch.optim.Adagrad(model_clone.parameters(), lr=lr)
    elif method.lower() == "adam":
        optimizer = torch.optim.Adam(model_clone.parameters(), lr=lr)
    else:
        raise ValueError(f"Unknown method: {method}")

    for step in range(num_steps+1):
        x_batch, y_batch = model_clone.get_random_batch()
        optimizer.zero_grad()
        loss = model_clone(x_batch, y_batch)
        if step > 0:
            loss.backward()
            optimizer.step()

        ckpt_path = os.path.join(ckpt_dir, f"step_{step:03d}.pth")
        torch.save(model_clone.state_dict(), ckpt_path)
        print(f"[{method}] Step {step:03d}: thetas={model_clone.thetas.detach().cpu().numpy().flatten()}, loss={loss.item():.4f}")

    print(f"Baseline '{method}' finished. Checkpoints in: {ckpt_dir}")
    return ckpt_dir


def run_llm_inference(ckpt_dir, output_dir, llama_v=2, depth=1, steps=50):
    """
    1) kernel_inference.py on ckpt_dir -> builds transition kernels in output_dir
    2) sgd_inference.py -> produce LLM-based param trajectory starting from *step_000* (initial param)
    Returns the final param trajectory (numpy array shape (steps,2,1)).
    """
    # 1) kernel_inference
    kernel_inference_cmd = [
        "python", "kernel_inference.py",
        "--ckpts_path", ckpt_dir,
        "--llama_v", str(llama_v),
        "--output_dir", output_dir,
        "--depth", str(depth)
    ]
    print(" ".join(kernel_inference_cmd))
    subprocess.run(kernel_inference_cmd, check=True)

    # 2) sgd_inference - start from *initial param* => step_000
    init_ckpt_path = os.path.join(output_dir, "infer_init_ckpt.pth")
    initial_ckpt = os.path.join(ckpt_dir, f"step_{0:03d}.pth")  # step_000 => baseline's initial param
    torch.save(torch.load(initial_ckpt), init_ckpt_path)

    inferred_sgd_dir = os.path.join(output_dir, "inferred_sgd")
    os.makedirs(inferred_sgd_dir, exist_ok=True)
    sgd_inference_cmd = [
        "python", "sgd_inference.py",
        "--init_ckpt_path", init_ckpt_path,
        "--output_dir", inferred_sgd_dir,
        "--kernels_dir", os.path.join(output_dir,"kernel"),
        "--steps", str(steps),
        "--sample"
    ]
    print(" ".join(sgd_inference_cmd))
    subprocess.run(sgd_inference_cmd, check=True)

    # Load the final LLM-based trajectory
    traj_npz = os.path.join(inferred_sgd_dir, "sgd_infer_trajectory.npz")
    data = np.load(traj_npz, allow_pickle=True)['arr_0'].item()
    llm_traj = data["thetas"]  # shape (steps,2,1)
    return llm_traj


def load_baseline_traj(ckpt_dir):
    """
    Loads the entire baseline param trajectory from checkpoint folder (step_000, step_001, ...).
    Returns shape (num_steps+1,2,1).
    """
    import glob
    ckpt_files = sorted(glob.glob(os.path.join(ckpt_dir, "*.pth")),
                        key=lambda x: int(x.split('_')[-1].split('.')[0]))
    baseline_traj = []
    for f in ckpt_files:
        ckpt = torch.load(f)
        baseline_traj.append(ckpt['thetas']) # shape (2,1)
    baseline_traj = torch.stack(baseline_traj, dim=0) # shape (num_steps+1,2,1)
    return baseline_traj


def plot_comparison(baseline_traj, llm_traj, model, out_fig, title="Comparison: Baseline vs LLM"):
    """
    Overlays baseline_traj vs. llm_traj in param space (theta1,theta2) on the same 2D contour.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6,5))
    X, Y, Z = compute_grid_values(model, x_range=[-2,2], y_range=[-2,5], resolution=100)
    c = ax.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(c, ax=ax, label='Loss')

    # baseline
    # base_2d = baseline_traj[...,0].numpy() # shape (num_steps+1,2)
    # ax.plot(base_2d[:,0], base_2d[:,1], "o-", color="black", label="Baseline (from step_000)")

    # LLM
    llm_2d = llm_traj[...,0] # shape (steps,2)
    ax.plot(llm_2d[:,0], llm_2d[:,1], "o-", color="red", label="LLM inference (same init)")

    ax.set_xlabel(r"$\theta_1$")
    ax.set_ylabel(r"$\theta_2$")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.savefig(out_fig, dpi=150)
    print(f"Saved comparison plot to {out_fig}")
    plt.show()


if __name__ == "__main__":
    """
    We'll run multiple methods from your course: 
     - GD (no momentum)
     - Momentum (heavy-ball / polyak)
     - Nesterov
     - AdaGrad
     - Adam
    Compare each baseline vs. LLM inference starting from the same initial param (step_000).
    """
    # seeds for reproducibility
    np.random.seed(256)
    random.seed(256)
    torch.manual_seed(256)

    # 1) Create NonConvexProblemModel
    from data_toy2d import NonConvexProblemModel
    model = NonConvexProblemModel(
        init_params=[-1.9, 1.9],
        random_seed=315,
        theta_star=[-1, -1],
        batch_size=10,
        dataset_size_N=100,
        name='nonconvex_course_compare',
        output_root='output/nonconvex_course_compare'
    )

    # 2) Define the methods, learning rates, etc.
    methods = [
        ("GD",       0.1),
        ("Momentum", 0.1),
        ("Nesterov", 0.1),
        ("AdaGrad",  0.1),
        ("Adam",     0.01),
    ]
    num_steps = 50
    llama_v = 2
    depth = 1  # refinement depth for kernel_inference

    for (method, lr) in methods:
        # A) generate baseline
        ckpt_dir = generate_baseline_ckpts(model, method=method, lr=lr, num_steps=num_steps)

        # B) LLM-based inference *from baseline's initial param (step_000)
        output_dir = os.path.join(model.output_dir, f"inferred_kernels_{method}")
        os.makedirs(output_dir, exist_ok=True)
        llm_traj = run_llm_inference(
            ckpt_dir=ckpt_dir,
            output_dir=output_dir,
            llama_v=llama_v,
            depth=depth,
            steps=num_steps
        )

        # C) load baseline trajectory
        baseline_traj = load_baseline_traj(ckpt_dir)

        # D) plot side-by-side
        out_fig = os.path.join(model.output_dir, f"compare_{method}.png")
        plot_comparison(
            baseline_traj, llm_traj, model,
            out_fig,
            title=f"{method} Baseline vs. LLM (Same Init: step_000)"
        )

    print("Done! Check output in output/nonconvex_course_compare/")
