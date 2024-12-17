import argparse
import os
import re

import numpy as np
import torch
from sklearn.decomposition import PCA

from llmoptim.utils import plot_progressive_trajectory, create_mp4_from_frames
from models.toy_mnist_mlp import MLP as ToyMNISTMLP

model_mapping = {"ToyMNISTMLP": ToyMNISTMLP}


def apply_pca(weights_traj):
    traj_flattened = np.reshape(weights_traj, (weights_traj.shape[0], -1))
    pca = PCA(n_components=2)
    traj = pca.fit_transform(traj_flattened)
    return torch.from_numpy(traj)


def visualize_gt(ckpt_dir, param, param_indexes, output_dir, use_pca: bool = False):
    # build trajectory
    traj = {}
    for ckpt_file in sorted(os.listdir(ckpt_dir)):
        m = re.match(r"ckpt_(\d+).pt", ckpt_file)
        if m is not None:
            ckpt_file = os.path.join(ckpt_dir, ckpt_file)
            state_dict = torch.load(ckpt_file)["model_state_dict"]
            for key in state_dict:
                if key not in traj:
                    traj[key] = []
                traj[key].append(state_dict[key])
    for key in traj:
        traj[key] = torch.stack(traj[key], dim=0)

    if not use_pca and param_indexes:
        traj_subset = []
        for indexes in param_indexes:
            index_out, index_in = indexes.split("/")
            index_out = int(index_out)
            index_in = int(index_in)
            traj_subset.append(traj[param][:, index_out, index_in])
        traj_subset = np.stack(traj_subset, axis=-1)
        traj = torch.from_numpy(traj_subset)  # (num_steps, num_params)
        params = [f"{param}/{idx}" for idx in param_indexes]
    elif use_pca:
        traj = apply_pca(traj[param])
        params = [r"\theta_1", r"\theta_2"]
    else:
        traj = torch.from_numpy(traj)  # (num_steps, num_params)

    print(f"Number of steps: {traj.shape[0]}")
    plot_progressive_trajectory(traj, None, params, frame_dirname=output_dir)
    create_mp4_from_frames(output_dir, name="sgd_trajectory_gt.mp4")


def visualize_pred(traj_file, model_name, param, param_indexes, ckpt_file, output_dir, use_pca: bool = False):
    traj = np.load(traj_file, allow_pickle=True)["arr_0"].item()
    if model_name is not None:
        model = model_mapping[model_name]()
        weights = torch.load(ckpt_file)["state_dict"]
        model.load_state_dict(weights)
    else:
        model = None

    if not use_pca and param_indexes:
        print("Using selected model parameters...")
        traj_subset = []
        for indexes in param_indexes:
            index_out, index_in = indexes.split("/")
            index_out = int(index_out)
            index_in = int(index_in)
            traj_subset.append(traj[param][:, index_out, index_in])
        traj_subset = np.stack(traj_subset, axis=-1)
        traj = torch.from_numpy(traj_subset)  # (num_steps, num_params)
        params = [f"{param}/{idx}" for idx in param_indexes]
    elif use_pca:
        print("Using PCA...")
        traj = apply_pca(traj[param])
        params = [r"\theta_1", r"\theta_2"]
    else:
        traj = torch.from_numpy(traj)  # (num_steps, num_params)

    print(traj.shape)
    plot_progressive_trajectory(traj, model, params, frame_dirname=output_dir)
    create_mp4_from_frames(output_dir, name="sgd_trajectory_pred.mp4")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", dest="traj_file", type=str, required=False, help="Path to trajectory file")
    parser.add_argument("--gt", action="store_true", help="Plot ground truth trajectory")
    parser.add_argument("--param", type=str, required=True, help="params to plot")
    parser.add_argument("--indexes", type=str, nargs="+", required=False, default=None, help="indexes to plot")
    parser.add_argument("--pca", action="store_true", help="Apply PCA to trajectory")
    parser.add_argument("--model", type=str, required=False, default=None, help="Name of model")
    parser.add_argument("--ckpt", type=str, required=False, default=None, help="Path to model checkpoint (true SGD)")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    if args.gt:
        visualize_gt(args.ckpt, args.param, args.indexes, output_dir=args.output, use_pca=args.pca)
    else:
        visualize_pred(
            args.traj_file,
            args.model,
            args.param,
            args.indexes,
            ckpt_file=args.ckpt,
            output_dir=args.output,
            use_pca=args.pca,
        )
