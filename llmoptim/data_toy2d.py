#%%

#%%
import torch
import os

import matplotlib.pyplot as plt
import matplotlib.animation as animation
from PIL import Image
import numpy as np
import imageio

# chdir to the parent dir of this file
os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/..")
print(os.getcwd())

# Create a directory for saving checkpoints
CKPT_DIR = "checkpoints_2dtoy"
os.makedirs(CKPT_DIR, exist_ok=True)
OUTPUT_DIR = "dataprep_output/2dtoy/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

#%%
# Define the convex function
class ConvexProblemModel(torch.nn.Module):
    def __init__(self, init_params, output_root=OUTPUT_DIR, random_seed=314, theta_star = [4, 3], batch_size=10, dataset_size_N=100, name=''):
        super(ConvexProblemModel, self).__init__()
        self.__dict__.update(locals()) # save all arguments to self
        self.thetas = torch.nn.Parameter(torch.tensor(init_params).reshape(-1,1), requires_grad=True)

        # Generate data
        self.generate_data(theta_star, dataset_size_N)

        self.output_dir = output_root #os.path.join(output_root, name)
        self.ckpt_dir   = os.path.join(self.output_dir, 'ckpts/')
        self.visual_dir = os.path.join(self.output_dir, 'visuals/')
        os.makedirs(self.ckpt_dir, exist_ok=True)
        os.makedirs(self.visual_dir, exist_ok=True)

    def get_random_batch(self, batch_size=None):
        if batch_size is None:
            batch_size = self.batch_size
        indices = np.random.choice(len(self.x), batch_size)
        return torch.tensor(self.x[indices], dtype=torch.float32), torch.tensor(self.y[indices], dtype=torch.float32)
    def generate_data(self, theta_star, dataset_size_N):
        self.x = torch.randn(dataset_size_N, 2)  # 100 points in R^2, normal distribution (mean=0, std=1)
        self.theta_star = torch.tensor(theta_star, dtype=torch.float32).reshape(-1,1)  # Ground truth
        self.y = self.x @ self.theta_star + torch.randn(dataset_size_N, 1) * 0.5  # Add small noise
    def get_loss_i(self, x=None, y=None, thetas=None):
        if thetas is None:
            thetas = self.thetas
        if x is None or y is None:
            x = self.x # use all data
            y = self.y # use all data
        return .5*((x@thetas - y)**2) # (m, thetas.shape[1])
    def forward(self, x=None, y=None, thetas=None):
        '''
            x: torch.tensor of shape (m, 2) where m is the batch size
            y: torch.tensor of shape (m,)
            thetas: torch.tensor of shape (2,)
        '''
        #return (x[0] - 2)**2 + (x[1] - 3)**2 + x[0] - x[1]
        # 1/N Σ (f(xi, θ))
        # f(xi, θ) = 1/2 ( ||xi-θ|| − y)^2
        loss_i = self.get_loss_i(x, y, thetas) # shape (m, 1)
        loss =  loss_i.mean() 
        # f(xi, θ) = 1/2 ( ||xi-θ|| − y)^2
        return loss

    
class NonConvexProblemModel(ConvexProblemModel):
    def generate_data(self, theta_star, dataset_size_N):
        self.x = torch.randn(dataset_size_N, 1)  # 100 points in R^2, normal distribution (mean=0, std=1)
        self.theta_star = torch.tensor(theta_star, dtype=torch.float32).reshape(-1,1)  # Ground truth
        self.y = self.theta_star[0,0] * torch.sin(self.theta_star[1,0] * self.x) + 0.2 * torch.randn_like(self.x)  # Add noise
    def get_loss_i(self, x=None, y=None, thetas=None):
        if thetas is None:
            thetas = self.thetas
        if x is None or y is None:
            x = self.x # use all data
            y = self.y # use all data
        pred = torch.sin(x@thetas[1:2,:]) * thetas[0:1,:] # (m, thetas.shape[1])
        f_theta = .5* ( ( pred - y )**2 )
        return f_theta
    

def generate_sgd_trajectory(model, num_steps=50, lr=0.1):
    # Define optimizer
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    # Optimization loop
    for step in range(0, num_steps + 1):
        x, y = model.get_random_batch()
        optimizer.zero_grad()
        loss = model(x, y)
        if step > 0:
            loss.backward()
            optimizer.step()
        # Save checkpoint
        ckpt_path = os.path.join(f"{model.ckpt_dir}/step_{step:03d}.pth")
        torch.save(model.state_dict(), ckpt_path)

        print(f"Step {step:03d}: params = {list(model.parameters())}, loss = {loss.item():.4f}")
    print(f"Optimization finished. Checkpoints saved in '{model.ckpt_dir}'")
    return [list(model.parameters())]
    # Example usage
    # Initialize the problem
    # x = torch.tensor([0.0, 0.5], requires_grad=True)  # Initial values 
    # model = ConvexProblemModel(x)
    # generate_sgd_trajectory(model )

# Function to compute grid values of the target function
def compute_grid_values(model, x_range=[0,2], y_range=[0,3], resolution=100):
    x_vals = torch.linspace(x_range[0], x_range[1], resolution)
    y_vals = torch.linspace(y_range[0], y_range[1], resolution)
    X, Y = torch.meshgrid(x_vals, y_vals, indexing='ij')
    points = torch.stack([X.flatten(), Y.flatten()], dim=1)  # Flatten and stack coordinates
    
    with torch.no_grad():
        Z = model.get_loss_i(thetas=points.T)  # Batch process and reshape
        Z = Z.mean(axis=0) # Average over the batch dimension
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
def plot_progressive_trajectory(trajectory, model, frame_dirname='frames', plot_range=None, margin=0.4, cmap='viridis'):
    '''
        trajectory: torch.tensor of shape (num_steps, 2, 1)
        model: ConvexProblemModel
        plot_range: list of list of 2 elements, [[x_min, y_min], [x_max, y_max]]
        margin: float, margin to add to the plot range
        cmap: str, colormap

    '''
    save_dir = model.visual_dir + f'/{frame_dirname}/'
    os.makedirs(save_dir, exist_ok=True)
    trajectory = trajectory[..., 0 ].clone()
    if plot_range is None:
        minr, maxr = trajectory.min(axis=0)[0].numpy(), trajectory.max(axis=0)[0].numpy()
        x_range = [minr[0] - margin, maxr[0] + margin]
        y_range = [minr[1] - margin, maxr[1] + margin]
    else:
        minr, maxr = plot_range
        x_range = [minr[0], maxr[0] ]
        y_range = [minr[1], maxr[1] ]
    X, Y, Z = compute_grid_values(
        model, 
        x_range=x_range, 
        y_range=y_range
    )
    
    # Generate frames
    for i in range(1, len(trajectory) + 1):
        fig, ax = plt.subplots()
        c = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
        plt.colorbar(c, ax=ax, label='Function value')
        ax.plot(trajectory[:i, 0], trajectory[:i, 1], 'o-', color='magenta', label='SGD run')
        ax.set_xlabel(r'$\theta_1$')
        ax.set_ylabel(r'$\theta_2$')
        ax.set_title('SGD Trajectory')
        ax.legend()
        plt.savefig(f'{save_dir}/frame_{i:03d}.png')
        plt.close(fig)
    
    print(f"Frames saved in '{save_dir}'.")

# Convert frames to a GIF
def create_gif(visual_dir, duration=100):
    frames_dir = visual_dir + '/frames/'
    output_file = visual_dir + '/sgd_trajectory.gif'
    frames = [Image.open(os.path.join(frames_dir, f)) for f in sorted(os.listdir(frames_dir)) if f.endswith('.png')]
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
    frames_dir = visual_dir + '/frames/'
    output_file = visual_dir + '/sgd_trajectory.mp4'
    # Collect all frame files in sorted order
    frame_files = sorted(
        [os.path.join(frames_dir, f) for f in os.listdir(frames_dir) if f.endswith('.png')]
    )
    
    # Use imageio to write frames to an MP4 file
    with imageio.get_writer(output_file, fps=fps, codec='libx264', quality=8) as writer:
        for frame_file in frame_files:
            frame = imageio.imread(frame_file)
            writer.append_data(frame)
    
    print(f"MP4 video saved as '{output_file}'.")
#%%
def generate_sgd_traj_and_visuals(model, num_steps=50, lr=.1, plot_range=None):
    generate_sgd_trajectory( model, num_steps=num_steps, lr=lr)
    trajectory = load_ckpt_to_traj(ckpt_dir = model.ckpt_dir)
    
    print("Now plotting animations")
    plot_progressive_trajectory(trajectory, model, plot_range=plot_range)
    print("Now creating gif")
    create_gif(model.visual_dir)
    print("Now creating mp4")
    create_mp4_from_frames(model.visual_dir)

# %%
class LLMSGDKernelInfer():
    def __init__(self, exp_name):
        self.__dict__.update(locals())
        self.model = None 
        self.exp_name = exp_name
        self.output_root = 'output/' + exp_name
        self.infer_init_ckpt_path = f'{self.output_root}/infer_init_ckpt.pth'
    def generate_data(self, theta_star, dataset_size_N):
        raise NotImplementedError()

    def infer_kernels(self, llama_v=2, ckpts_path=None, output_dir=None):
        ckpts_path = f'{self.output_root}/ckpts/'
        output_dir = f'{self.output_root}/inferred_kernels/'

        kernel_inference_cmd = \
        f"python kernel_inference.py --ckpts_path {ckpts_path} --llama_v {llama_v} --output_dir {output_dir}"
        print(kernel_inference_cmd)
        os.system(kernel_inference_cmd)

    def infer_sgd(self, infer_init_thetas=[1.,2.]):
        self.model.thetas = torch.nn.Parameter(torch.tensor(infer_init_thetas).reshape(-1,1), requires_grad=True)
        torch.save({"model_state_dict": self.model.state_dict()}, self.infer_init_ckpt_path)

        init_ckpt_path = self.infer_init_ckpt_path
        output_dir  = f'{self.output_root}/inferred_sgd'
        kernels_dir = f'{self.output_root}/inferred_kernels/kernel/'
        steps = 50
        sgd_inference_cmd = \
        f"python sgd_inference.py --init_ckpt_path {init_ckpt_path} --output_dir {output_dir} --kernels_dir {kernels_dir} --steps {steps}"
        print(sgd_inference_cmd)
        os.system(sgd_inference_cmd)

    def visualize_sgd_inference(self,):
        pass

class LSKI_convex_underparam(LLMSGDKernelInfer):
    def __init__(self):
        super().__init__(exp_name = 'convex_underparam')
        theta_init = torch.tensor([0.0, 0.5], requires_grad=True)  # Initial values 
        self.model = ConvexProblemModel(theta_init, random_seed=315, theta_star = [4, 3], batch_size=10, dataset_size_N=100, name='convex_underparam', output_root=self.output_root) 
    def generate_data(self):
        generate_sgd_traj_and_visuals(self.model, lr=.1)
class LSKI_convex_overparam(LLMSGDKernelInfer):
    def __init__(self):
        super().__init__(exp_name = 'convex_overparam')
        theta_init = torch.tensor([0.0, 0.5], requires_grad=True)  # Initial values 
        self.model = ConvexProblemModel(theta_init, random_seed=315, theta_star = [4, 3], batch_size=1, dataset_size_N=2, name='convex_overparam', output_root=self.output_root) 
    def generate_data(self):
        generate_sgd_traj_and_visuals(self.model, lr=.1)
class LSKI_nonconvex_underparam(LLMSGDKernelInfer):
    def __init__(self):
        super().__init__(exp_name = 'nonconvex_underparam')
        theta_init = torch.tensor([1.0, -1.5], requires_grad=True)  # Initial values 
        self.model = NonConvexProblemModel(theta_init, random_seed=315, theta_star = [-1, -1], batch_size=10, dataset_size_N=100, name='nonconvex_underparam', output_root=self.output_root) 
    def generate_data(self):
        generate_sgd_traj_and_visuals(self.model, lr=.4, plot_range=[[-2, -2], [2,2]])
class LSKI_nonconvex_underparam_run2(LLMSGDKernelInfer):
    def __init__(self):
        super().__init__(exp_name = 'nonconvex_underparam_run2')
        theta_init = torch.tensor([1.5, -1.], requires_grad=True)  # Initial values 
        self.model = NonConvexProblemModel(theta_init, random_seed=315, theta_star = [-1, -1], batch_size=10, dataset_size_N=100, name='nonconvex_underparam_run2') 
    def generate_data(self):
        generate_sgd_traj_and_visuals(self.model, lr=.4, plot_range=[[-2, -2], [2,2]])
class LSKI_nonconvex_underparam_run3(LLMSGDKernelInfer):
    def __init__(self):
        super().__init__(exp_name = 'nonconvex_underparam_run3')
        theta_init = torch.tensor([1.9, -.5], requires_grad=True)  # Initial values 
        self.model = NonConvexProblemModel(theta_init, random_seed=315, theta_star = [-1, -1], batch_size=10, dataset_size_N=100, name='nonconvex_underparam_run3', output_root=self.output_root) 
    def generate_data(self):
        generate_sgd_traj_and_visuals(self.model, lr=.4, plot_range=[[-2, -2], [2,2]])
class LSKI_nonconvex_underparam_run4(LLMSGDKernelInfer):
    def __init__(self):
        super().__init__(exp_name = 'nonconvex_underparam_run4')
        theta_init = torch.tensor([-1.9, 1.9], requires_grad=True)  # Initial values 
        self.model = NonConvexProblemModel(theta_init, random_seed=315, theta_star = [-1, -1], batch_size=10, dataset_size_N=100, name='nonconvex_underparam_run4', output_root=self.output_root) 
    def generate_data(self):
        generate_sgd_traj_and_visuals(self.model, lr=.4, plot_range=[[-2, -2], [2,2]])


# %%
if __name__ == "__main__":

    #%%
    lski = LSKI_convex_underparam()
    lski.generate_data()
    #%%
    lski.infer_kernels()
    #%%
    lski.infer_sgd(infer_init_thetas=[1.5, 2.5])
    
    traj = np.load('llm-optim/output/convex_underparam/inferred_sgd/sgd_infer_trajectory.npz', allow_pickle=True)['arr_0'].item()['thetas']
    print(traj.shape)
    plot_progressive_trajectory(torch.from_numpy(traj), model, frame_dirname='sgdrunframes3')

    #%%
    #%%

    # Initialize the problem
    theta_init = torch.tensor([0.0, 0.5], requires_grad=True)  # Initial values 

    # model = ConvexProblemModel(theta_init, random_seed=315, theta_star = [4, 3], batch_size=10, dataset_size_N=100, name='convex_underparam') 
    # model.generate_sgd_traj_and_visuals()
    # model = ConvexProblemModel(theta_init, random_seed=315, theta_star = [4, 3], batch_size=1, dataset_size_N=2, name='convex_overparam') 
    # model.generate_sgd_traj_and_visuals()

    # theta_init = torch.tensor([1.0, -1.5], requires_grad=True)  # Initial values 
    # model = NonConvexProblemModel(theta_init, random_seed=315, theta_star = [-1, -1], batch_size=10, dataset_size_N=100, name='nonconvex_underparam') 
    # generate_sgd_traj_and_visuals(model, lr=.4, plot_range=[[-2, -2], [2,2]])

    # theta_init = torch.tensor([1.5, -1.], requires_grad=True)  # Initial values 
    # model = NonConvexProblemModel(theta_init, random_seed=315, theta_star = [-1, -1], batch_size=10, dataset_size_N=100, name='nonconvex_underparam_run2') 
    # generate_sgd_traj_and_visuals(model, lr=.4, plot_range=[[-2, -2], [2,2]])

    # theta_init = torch.tensor([1.9, -.5], requires_grad=True)  # Initial values 
    # model = NonConvexProblemModel(theta_init, random_seed=315, theta_star = [-1, -1], batch_size=10, dataset_size_N=100, name='nonconvex_underparam_run3') 
    # generate_sgd_traj_and_visuals(model, lr=.4, plot_range=[[-2, -2], [2,2]])

    # theta_init = torch.tensor([-1.9, 1.9], requires_grad=True)  # Initial values 
    # model = NonConvexProblemModel(theta_init, random_seed=315, theta_star = [-1, -1], batch_size=10, dataset_size_N=100, name='nonconvex_underparam_run4') 
    # generate_sgd_traj_and_visuals(model, lr=.4, plot_range=[[-2, -2], [2,2]])
