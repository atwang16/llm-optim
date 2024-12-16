import os

import imageio
import matplotlib.pyplot as plt
import torch
from PIL import Image


def str_seq_to_int(s_traj: str) -> list[list[int]]:
    out = []
    states = s_traj.split(",")
    for state in states:
        state = state.replace(" ", "")  # remove whitespace
        state = [int(s) for s in state]
        out.append(state)
    return out


def int_seq_to_str(states: list[list[int]]) -> str:
    return ",".join(["".join([str(s) for s in state]) for state in states]) + ","


def int_to_list_int(num: int) -> list[int]:
    return [int(s) for s in str(num)]


# Function to compute grid values of the target function
def compute_grid_values(model, x_range=[0, 2], y_range=[0, 3], resolution=100):
    x_vals = torch.linspace(x_range[0], x_range[1], resolution)
    y_vals = torch.linspace(y_range[0], y_range[1], resolution)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing="ij")
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)  # Flatten and stack coordinates

    with torch.no_grad():
        Z = model.get_loss_i(thetas=points.T)  # Batch process and reshape
        Z = Z.mean(axis=0)  # Average over the batch dimension
        Z = Z.view(resolution, resolution)

    return X.numpy(), Y.numpy(), Z.numpy()
    # Example usage
    # X, Y, Z = compute_grid_values(model)
    # print(X.shape, Y.shape, Z.shape)
    # # show as image
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(Z, origin='lower', extent=(0, 2, 0, 3))
    # plt.show()


def load_ckpt_to_traj(ckpt_dir):
    import glob

    ckpt_files = glob.glob(ckpt_dir + "/*.pth")
    # sort by step
    ckpt_files.sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))
    # load the last checkpoint
    xs = []
    for ckpt_file in ckpt_files:
        ckpt = torch.load(ckpt_file)
        xs.append(ckpt["thetas"])
    xs = torch.stack(xs)
    return xs


# Function to plot and save frames progressively
def plot_progressive_trajectory(
    trajectory,
    model=None,
    params: list[str] = None,
    frame_dirname="frames",
    plot_range=None,
    margin=0.1,
    cmap="viridis",
):
    """
    trajectory: torch.tensor of shape (num_steps, 2, 1)
    model: ConvexProblemModel
    plot_range: list of list of 2 elements, [[x_min, y_min], [x_max, y_max]]
    margin: float, margin to add to the plot range
    cmap: str, colormap

    """
    if params is None:
        params = [r"\theta_1", r"\theta_2"]
    else:
        assert len(params) == 2, "Only 2D plots are supported."

    save_dir = frame_dirname
    os.makedirs(os.path.join(save_dir, "frames"), exist_ok=True)
    if trajectory.ndim == 3:
        trajectory = trajectory[..., 0].clone()
    if plot_range is None:
        minr, maxr = trajectory.min(axis=0)[0].numpy(), trajectory.max(axis=0)[0].numpy()
        margin_abs = (maxr - minr) * margin
        x_range = [minr[0] - margin_abs[0], maxr[0] + margin_abs[0]]
        y_range = [minr[1] - margin_abs[1], maxr[1] + margin_abs[1]]
    else:
        minr, maxr = plot_range
        x_range = [minr[0], maxr[0]]
        y_range = [minr[1], maxr[1]]

    if model is not None:
        X, Y, Z = compute_grid_values(model, x_range=x_range, y_range=y_range)
    else:
        X, Y, Z = None, None, None

    # Generate frames
    for i in range(1, len(trajectory) + 1):
        fig, ax = plt.subplots()
        if X is not None and Y is not None and Z is not None:
            c = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
            plt.colorbar(c, ax=ax, label="Function value")
        ax.plot(trajectory[:i, 0], trajectory[:i, 1], "o-", color="magenta", label="SGD run")
        ax.set_xlabel(f"${params[0]}$")
        ax.set_ylabel(f"${params[1]}$")
        ax.set_xbound(x_range)
        ax.set_ybound(y_range)
        ax.set_title("SGD Trajectory")
        ax.legend()
        plt.savefig(os.path.join(save_dir, "frames", f"frame_{i:03d}.png"))
        plt.close(fig)

    print(f"Frames saved in '{save_dir}'.")


# Convert frames to a GIF
def create_gif(visual_dir, duration=100):
    frames_dir = os.path.join(visual_dir, "frames")
    output_file = os.path.join(visual_dir, "sgd_trajectory.gif")
    frames = [Image.open(os.path.join(frames_dir, f)) for f in sorted(os.listdir(frames_dir)) if f.endswith(".png")]
    frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"GIF saved as '{output_file}'.")


# %%
def create_mp4_from_frames(visual_dir, fps=10):
    """
    Create an MP4 video from a folder of image frames.
    Args:
        folder (str): Path to the folder containing the image frames.
        output_file (str): Path to save the MP4 video.
        fps (int): Frames per second for the video.
    """
    frames_dir = os.path.join(visual_dir, "frames")
    output_file = os.path.join(visual_dir, "sgd_trajectory.mp4")
    # Collect all frame files in sorted order
    frame_files = sorted([os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith(".png")])

    # Use imageio to write frames to an MP4 file
    with imageio.get_writer(output_file, fps=fps, codec="libx264", quality=8) as writer:
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
            writer.append_data(frame)

    print(f"MP4 video saved as '{output_file}'.")
