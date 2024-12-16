import argparse
import os
import shutil
from glob import glob
from tqdm import tqdm

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d


def load_ckpts_into_seq(ckpts_path):
    # Loads checkpoints and constructs sequence for each parameter
    # Returns a dict {"{param_name}": np.ndarray({n_ckpts}), ...},
    # where n_ckpts practically is time series length we provide as an input
    # remember to account for param_name being actually {layer_name}_{param_flattened_index}

    model_state_dicts = []
    for ckpt_path in sorted(glob(f"{ckpts_path}/*.pth")):
        try:
            checkpoint = torch.load(ckpt_path)
            if "model_state_dict" in checkpoint:
                model_state_dicts.append(checkpoint["model_state_dict"])
            else:
                model_state_dicts.append(checkpoint)
        except Exception as e:
            print(f"Error loading {ckpt_path}: {e}")

    param_specs = [(param_name, param_mat.size()) for param_name, param_mat in model_state_dicts[0].items()]
    sequences = {}
    for param_spec in param_specs:
        param_name, param_size = param_spec
        total_flattened = np.prod(param_size)
        for i in range(total_flattened):
            param_seq = np.zeros(len(model_state_dicts))
            for j, model_state_dict in enumerate(model_state_dicts):
                param_seq[j] = model_state_dict[param_name].flatten()[i]
            sequences[f"{param_name}_{i}"] = param_seq
    return sequences


def generate_random_kernel(kernel_dim: int = 1000, sigma: float = 10):
    init = np.eye(kernel_dim)
    kernel = gaussian_filter1d(init, sigma=sigma)
    return kernel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, help="path to directory with checkpoints")
    parser.add_argument("--kernel-dim", "-k", dest="kernel_dim", type=int, default=1000)
    parser.add_argument("--sigma", type=float, default=10)
    parser.add_argument("--output", type=str, help="path to output directory")
    args = parser.parse_args()

    sequences = load_ckpts_into_seq(args.ckpt)

    # compute kernel per parameter
    if os.path.exists(args.output):
        shutil.rmtree(args.output)
    os.makedirs(args.output, exist_ok=True)
    for idx, (param_name, param_seq) in enumerate(tqdm(sequences.items())):
        kernel = generate_random_kernel(args.kernel_dim, args.sigma)
        init_min = list(sequences.values())[idx].min()
        init_max = list(sequences.values())[idx].max()

        output_file = os.path.join(args.output, f"{param_name}.npz")
        np.savez(output_file, kernel=kernel, init_min=init_min, init_max=init_max)
