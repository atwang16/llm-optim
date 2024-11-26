#%%
import torch
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import imageio

# chdir to the parent dir of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
print(os.getcwd())
#%%
# Define the convex function
class ConvexProblemModel(torch.nn.Module):
    def __init__(self, init_params):
        super(ConvexProblemModel, self).__init__()
        self.thetas = torch.nn.Parameter(torch.tensor(init_params), requires_grad=True)

    def forward(self, x=None):
        if x is None:
            x = self.thetas
        return (x[0] - 2)**2 + (x[1] - 3)**2 + x[0] - x[1]

# Create a directory for saving checkpoints
ckpt_dir = "checkpoints_2dtoy"
os.makedirs(ckpt_dir, exist_ok=True)

def generate_sgd_trajectory(model, num_steps=50, lr=0.1):
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Optimization loop
    for step in range(0, num_steps + 1):
        optimizer.zero_grad()
        loss = model()
        if step > 0:
            loss.backward()
            optimizer.step()
        # Save checkpoint
        ckpt_path = os.path.join(ckpt_dir, f"step_{step:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)

        print(f"Step {step:03d}: params = {list(model.parameters())}, loss = {loss.item():.4f}")
    print(f"Optimization finished. Checkpoints saved in '{ckpt_dir}'")
    return [list(model.parameters())]
    # Example usage
    # Initialize the problem
    # x = torch.tensor([0.0, 0.5], requires_grad=True)  # Initial values 
    # model = ConvexProblemModel(x)
    # generate_sgd_trajectory(model )

#%%
# Function to compute grid values of the target function
def compute_grid_values(model, x_range=[0,2], y_range=[0,3], resolution=100):
    x_vals = torch.linspace(x_range[0], x_range[1], resolution)
    y_vals = torch.linspace(y_range[0], y_range[1], resolution)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)  # Flatten and stack coordinates
    
    with torch.no_grad():
        Z = model(points.T).view(resolution, resolution)  # Batch process and reshape

    return X.numpy(), Y.numpy(), Z.numpy()
    # Example usage
    # X, Y, Z = compute_grid_values(model)
    # print(X.shape, Y.shape, Z.shape)
    # # show as image
    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # ax.imshow(Z, origin='lower', extent=(0, 2, 0, 3))
    # plt.show()

def load_ckpt_to_traj(ckpt_dir = ckpt_dir):
    import glob 
    ckpt_files = glob.glob(ckpt_dir + '/*.pth')
    # sort by step
    ckpt_files.sort(key=lambda x: int(x.split('_')[-1].split('.')[0]))
    # load the last checkpoint
    xs = []
    for ckpt_file in ckpt_files:
        ckpt = torch.load(ckpt_file)
        xs.append(ckpt['thetas'])
    xs = torch.stack(xs)
    return xs 
# trajectory = load_ckpt_to_traj()

# %%

# Function to plot and save frames progressively
def plot_progressive_trajectory(trajectory, model, folder='assets/frames', margin=0.4, cmap='viridis'):
    os.makedirs(folder, exist_ok=True)
    minr, maxr = trajectory.min(axis=0)[0].numpy(), trajectory.max(axis=0)[0].numpy()
    X, Y, Z = compute_grid_values(
        model, 
        x_range=[minr[0] - margin, maxr[0] + margin], 
        y_range=[minr[1] - margin, maxr[1] + margin]
    )
    
    # Generate frames
    for i in range(1, len(trajectory) + 1):
        fig, ax = plt.subplots()
        c = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
        plt.colorbar(c, ax=ax, label='Function value')
        ax.plot(trajectory[:i, 0], trajectory[:i, 1], 'o-', color='magenta', label='Optimization trajectory')
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        ax.set_title('Optimization trajectory with function values')
        ax.legend()
        plt.savefig(f'{folder}/frame_{i:03d}.png')
        plt.close(fig)
    
    print(f"Frames saved in '{folder}'.")

# Convert frames to a GIF
def create_gif(folder='assets/frames', output_file='assets/toy2d_gt_output.gif', duration=100):
    frames = [Image.open(os.path.join(folder, f)) for f in sorted(os.listdir(folder)) if f.endswith('.png')]
    frames[0].save(output_file, save_all=True, append_images=frames[1:], duration=duration, loop=0)
    print(f"GIF saved as '{output_file}'.")

# %%
def create_mp4_from_frames(folder='assets/frames', output_file='assets/toy2d_gt_output.mp4', fps=10):
    """
    Create an MP4 video from a folder of image frames.
    Args:
        folder (str): Path to the folder containing the image frames.
        output_file (str): Path to save the MP4 video.
        fps (int): Frames per second for the video.
    """
    # Collect all frame files in sorted order
    frame_files = sorted(
        [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.png')]
    )
    
    # Use imageio to write frames to an MP4 file
    with imageio.get_writer(output_file, fps=fps, codec='libx264', quality=8) as writer:
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
            writer.append_data(frame)
    
    print(f"MP4 video saved as '{output_file}'.")
# %%
if __name__ == "__main__":
    # Initialize the problem
    x = torch.tensor([0.0, 0.5], requires_grad=True)  # Initial values 
    model = ConvexProblemModel(x)
    generate_sgd_trajectory(model )
    trajectory = load_ckpt_to_traj()
    
    print("Now plotting animations")
    plot_progressive_trajectory(trajectory, model)
    print("Now creating gif")
    create_gif()
    print("Now creating mp4")
    create_mp4_from_frames()
# %%
