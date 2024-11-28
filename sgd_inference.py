import argparse
import copy
import os
from glob import glob

import numpy as np
import torch
from llmoptim.tokenizer import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer(None)


def param_to_state(param_val, init_min, init_max):
    data = np.array([param_val, init_min, init_max])
    data = tokenizer.rescale(data)
    data = np.round(data, tokenizer.n_digits - 1)
    state = np.zeros(1000, dtype=float)
    state[data[0]] = 1
    return state


def state_to_param(state, init_min, init_max):
    float_state = float(state) / 100

    original_value = (float_state - 1.5) * (init_max - init_min) / (8.5 - 1.5) + init_min
    return original_value


def load_kernels(kernels_dir):
    kernels_dict = {}
    for kernel_path in glob(f"{kernels_dir}/*.npz"):
        kernel = np.load(kernel_path, allow_pickle=True)

        curr_dict = {}
        for key in kernel.keys():
            curr_dict[key] = kernel[key]

        curr_dict["seq"] = []
        kernels_dict[os.path.basename(kernel_path)[:-4]] = curr_dict
    return kernels_dict


def get_new_state(new_state_probs, cur_state, sample=False):
    # Sort in decreasing order
    if not sample:
        sorted_probs = np.argsort(new_state_probs)[::-1]
        cond1, cond2 = False, False
        counter = 0
        while not (cond1 and cond2):
            cond1 = sorted_probs[counter] != cur_state  # Skip the current state if
            cond2 = len(str(sorted_probs[counter])) == 3  # Skip if the state is not 3 digits
            counter += 1
        return sorted_probs[counter - 1]
    else:
        # Construct a distribution and sample from it
        new_state_probs[cur_state] = 0
        new_state_probs = new_state_probs / np.sum(new_state_probs)
        return np.random.choice(np.arange(1000), p=new_state_probs)


def apply_kernel(kernels_dict, state_mat, model_state_dict, sample_flag=False):
    # TODO: debug
    counter = 0
    new_state_mat = np.zeros_like(state_mat)
    for key in model_state_dict.keys():
        for i, _ in enumerate(model_state_dict[key].flatten()):
            kernel = kernels_dict[f"{key}_{i}"]["kernel"]
            param_state_vec = state_mat[:, counter]
            new_state_probs = np.matmul(kernel, param_state_vec)
            # Argmax over probabilities
            new_state = get_new_state(new_state_probs, np.argmax(param_state_vec), sample_flag)
            new_state_mat[new_state, counter] = 1
            counter += 1
    return new_state_mat


def state_mat_to_param(state_mat, kernels_dict, model_state_dict):
    # TODO: debug
    params_mat = copy.deepcopy(model_state_dict)
    counter = 0
    for key in model_state_dict.keys():
        for i, _ in enumerate(model_state_dict[key].flatten()):
            # i is flattened index
            # j1, j2 are indices in the original shape
            j1 = i // model_state_dict[key].shape[1]
            j2 = i % model_state_dict[key].shape[1]
            params_mat[key][j1][j2] = state_to_param(
                np.argmax(state_mat[:, counter]), kernels_dict[f"{key}_{i}"]["init_min"], kernels_dict[f"{key}_{i}"]["init_max"]
            )
            counter += 1
    return params_mat


def load_params_to_state_mat(model_state_dict, kernels_dict):
    # TODO: debug
    params_size = 0
    for key in model_state_dict.keys():
        params_size += model_state_dict[key].flatten().shape[0]
    state_mat = np.zeros((1000, params_size), dtype=float)
    counter = 0
    for key in model_state_dict.keys():
        for i, param in enumerate(model_state_dict[key].flatten()):
            state = param_to_state(
                param, kernels_dict[f"{key}_{i}"]["init_min"], kernels_dict[f"{key}_{i}"]["init_max"]
            )
            state_mat[:, counter] = state
            counter += 1
    return state_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt_path", type=str, required=True, help="Path to initial checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory to save inferred ckpts")
    parser.add_argument("--kernels_dir", type=str, required=True, help="Path to previously inferred kernels")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to infer")
    parser.add_argument("--sample", action="store_true", help="Sample from the distribution instead of argmax")
    args = parser.parse_args()

    # Load initial checkpoint
    ckpt = torch.load(args.init_ckpt_path)

    # TODO import MLP and uncomment this
    # model = MLP()
    # model.load_state_dict(ckpt["model_state_dict"])

    # Load kernels
    kernels_dict = load_kernels(args.kernels_dir)

    init_state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
    state_mat = load_params_to_state_mat(init_state_dict, kernels_dict)
    print(np.where(state_mat == 1))
    trajectory = dict()
    for key in init_state_dict.keys():
        trajectory[key] = []

    for i in tqdm(range(args.steps)):
        state_mat = apply_kernel(kernels_dict, state_mat, init_state_dict, args.sample)
        # print(np.where(state_mat == 1))
        param_dict = state_mat_to_param(state_mat, kernels_dict, init_state_dict)
        for key in param_dict.keys():
            trajectory[key].append(param_dict[key])
        # TODO
        # Save checkpoint
        # model.load_state_dict(param_dict)
        # torch.save({"model_state_dict": model.state_dict()}, f"{args.output_dir}/ckpt_{i}.pt")
        # metrics = model.get_metrics()
        # np.savez(f"{args.output_dir}/metrics_{i}.npz", metrics)
        # print(param_dict)
    # save trajectory
    for key in trajectory.keys():
        trajectory[key] = torch.stack(trajectory[key], axis=0).numpy()
    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(f"{args.output_dir}/sgd_infer_trajectory.npz", trajectory)
