import argparse

import numpy as np
import torch
from sklearn.decomposition import PCA

from llmoptim.utils import plot_progressive_trajectory, create_mp4_from_frames
from models.toy_mnist_mlp import MLP as ToyMNISTMLP

model_mapping = {"ToyMNISTMLP": ToyMNISTMLP}


def apply_pca():
    pass


def visualize(traj_file, model_name, params, ckpt_file, output_dir, use_pca: bool = False):
    traj = np.load(traj_file, allow_pickle=True)["arr_0"].item()
    if model_name is not None:
        model = model_mapping[model_name]()
        weights = torch.load(ckpt_file)["state_dict"]
        model.load_state_dict(weights)
    else:
        model = None

    if not use_pca and params is not None:
        traj_subset = []
        for param in params:
            param_name, index_out, index_in = param.split("/")
            index_out = int(index_out)
            index_in = int(index_in)
            traj_subset.append(traj[param_name][:, index_out, index_in])
        traj_subset = np.stack(traj_subset, axis=-1)
        traj = torch.from_numpy(traj_subset)  # (num_steps, num_params)
    elif use_pca:
        traj = traj[params[0]]
        traj_flattened = np.reshape(traj, (traj.shape[0], -1))
        pca = PCA(n_components=2)
        traj = pca.fit_transform(traj_flattened)
        traj = torch.from_numpy(traj)
        params = [r"\theta_1", r"\theta_2"]
    else:
        traj = torch.from_numpy(traj)  # (num_steps, num_params)

    print(traj.shape)
    plot_progressive_trajectory(traj, model, params, frame_dirname=output_dir)
    create_mp4_from_frames(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj", dest="traj_file", type=str, required=True, help="Path to trajectory file")
    parser.add_argument("--params", type=str, nargs="+", required=False, default=None, help="params to plot")
    parser.add_argument("--pca", action="store_true", help="Apply PCA to trajectory")
    parser.add_argument("--model", type=str, required=False, default=None, help="Name of model")
    parser.add_argument("--ckpt", type=str, required=False, default=None, help="Path to model checkpoint (true SGD)")
    parser.add_argument("--output", type=str, required=True, help="Path to output directory")
    args = parser.parse_args()

    visualize(args.traj_file, args.model, args.params, ckpt_file=args.ckpt, output_dir=args.output, use_pca=args.pca)
