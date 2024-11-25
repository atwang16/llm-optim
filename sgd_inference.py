import argparse

import numpy as np
import torch
from llmoptim.tokenizer import Tokenizer
from tqdm import tqdm
from glob import glob
import os


def param_to_state(param_val, init_min, init_max):
    tokenizer = Tokenizer(None)
    data = np.array([param_val, init_min, init_max])
    data = tokenizer._rescale(data)
    data = np.round(data, tokenizer.n_digits - 1)
    state = np.zeros(len(data), dtype=float)
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


def apply_kernel(kernels_dict, state_mat)
    pass


def state_mat_to_param(state_mat, kernels_dict):
    pass


def load_params(model_state_dict, kernels_dict):
    pass


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
        # metrics = model.eval()
        # np.save(f"{args.output_dir}/metrics_{i}.npy", metrics)