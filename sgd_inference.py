import argparse
import os
from glob import glob

import numpy as np
import torch
from llmoptim.tokenizer import Tokenizer
from tqdm import tqdm

tokenizer = Tokenizer(None)


def param_to_state(param_val, init_min, init_max):
    data = np.array([param_val, init_min, init_max])
    data = tokenizer._rescale(data)
    data = np.round(data, tokenizer.n_digits - 1)
    state = np.zeros(1000, dtype=float)
    state[data[0]] = 1
    return state


def state_to_param(state, init_min, init_max):
    float_state = float(state) / 100
    return float_state * (init_max - init_min) + init_min


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


def apply_kernel(kernels_dict, state_mat, model_state_dict):
    # TODO: debug
    counter = 0
    new_state_mat = np.zeros_like(state_mat)
    for key in model_state_dict.keys():
        for i, _ in enumerate(model_state_dict[key].flatten()):
            kernel = kernels_dict[f"{key}_{i}"]["kernel"]
            param_state_vec = state_mat[:, counter]
            new_state_probs = np.matmul(kernel, param_state_vec)
            # Argmax over probabilities
            new_state_mat[torch.argmax(new_state_probs, dim=0), counter] = 1
            counter += 1
    return new_state_mat


def state_mat_to_param(state_mat, kernels_dict, model_state_dict):
    # TODO: debug
    params_mat = model_state_dict.clone()
    counter = 0
    for key in model_state_dict.keys():
        for i, _ in enumerate(model_state_dict[key].flatten()):
            # i is flattened index
            # j1, j2 are indices in the original shape
            j1 = i // model_state_dict[key].shape[1]
            j2 = i % model_state_dict[key].shape[1]
            params_mat[key][j1][j2] = state_to_param(state_mat[:, counter], kernels_dict[f"{key}_{i}"]["init_min"], kernels_dict[f"{key}_{i}"]["init_max"])
            counter += 1
    return params_mat


def load_params(model_state_dict, kernels_dict):
    # TODO: debug
    params_size = 0
    for key in model_state_dict.keys():
        params_size += model_state_dict[key].flatten().shape[0]
    state_mat = np.zeros((1000, params_size), dtype=float)
    counter = 0
    for key in model_state_dict.keys():
        for i, param in enumerate(model_state_dict[key].flatten()):
            state = param_to_state(param, kernels_dict[f"{key}_{i}"]["init_min"], kernels_dict[f"{key}_{i}"]["init_max"])
            state_mat[:, counter] = state
            counter += 1
    return state_mat


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--init_ckpt_path", type=str, required=True, help="Path to initial checkpoint")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory to save inferred ckpts")
    parser.add_argument("--kernels_dir", type=str, required=True, help="Path to previously inferred kernels")
    parser.add_argument("--steps", type=int, required=True, help="Number of steps to infer")
    args = parser.parse_args()

    # Load initial checkpoint
    ckpt = torch.load(args.init_ckpt_path)

    # TODO import MLP and uncomment this 
    # model = MLP()
    # model.load_state_dict(ckpt["model_state_dict"])

    # Load kernels
    kernels_dict = load_kernels(args.kernels_dir)

    state_mat = load_params(ckpt["model_state_dict"], kernels_dict)

    for i in tqdm(range(args.steps)):
        state_mat = apply_kernel(kernels_dict, state_mat)
        param_dict = state_mat_to_param(state_mat, kernels_dict)

        # TODO
        # Save checkpoint
        # model.load_state_dict(param_dict)
        # torch.save({"model_state_dict": model.state_dict()}, f"{args.output_dir}/ckpt_{i}.pt")
        # metrics = model.get_metrics()
        # np.savez(f"{args.output_dir}/metrics_{i}.npz", metrics)
