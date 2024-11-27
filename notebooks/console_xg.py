#%%
from IPython import get_ipython

# Get the current IPython instance
ipython = get_ipython()

if ipython is not None:
    # Run magic commands
    ipython.run_line_magic('matplotlib', 'inline')
    ipython.run_line_magic('config', "InlineBackend.figure_format = 'retina'")
    ipython.run_line_magic('reload_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import os
# change dir to parent dir of this file
os.chdir(os.path.dirname(os.getcwd()))
print( os.getcwd() )

#%%
import kernel_inference
#%%
import tqdm
import torch
import argparse
import kernel_inference
import sgd_inference
import pickle
class SGDInferToy2D():
    def __init__(self, datamodel, output_dir):
        self.__dict__.update(locals())

        self.data_ckpt_dir = os.path.join(self.output_dir, 'ckpts/')
        self.output_ckpt_dir = os.path.join(self.output_dir, 'output_ckpts/')
        self.visual_dir = os.path.join(self.output_dir, 'visuals/')

        self.kernels_dir = os.path.join(self.output_dir, 'kernels/')

        self.infer_init_ckpt_path = os.path.join(self.output_dir, 'infer_init_ckpt.pt')
    def kernel_infer(self, ckpts_path=None, ):
        pass
        parser = argparse.ArgumentParser()
        parser.add_argument("--ckpts_path", type=str, required=False, help="Path to directory containing the checkpoints")
        parser.add_argument("--llama_v", choices=[2, 3], type=int, required=True, help="Version of Llama model")
        parser.add_argument(
            "--output_dir", type=str, required=True, help="Path to directory to save inferred PDFs and kernels"
        )
        parser.add_argument("--load", action="store_true", help="if true, load PDFs from output_dir if possible")
        parser.add_argument("--use-dummy", dest="use_dummy", action="store_true", help="Use dummy data for testing")
        args = parser.parse_args()

        llama = kernel_inference.Llama(llama_v=args.llama_v)

        if args.use_dummy:
            sequences = kernel_inference.load_dummy("tmp/brownian_motion_0.pkl")
            rescaled_sequences = sequences
        else:
            sequences = kernel_inference.load_ckpts_into_seq(self.data_ckpt_dir)
            rescaled_sequences = {
                param_name: llama.tokenizer.rescale(param_seq) for param_name, param_seq in sequences.items()
            }

        good_tokens_str = list("0123456789")
        good_tokens = [llama.llama_tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

        # Calculate PDFs
        pdf_dict = {}
        # TODO: parallelize??
        os.makedirs(os.path.join(args.output_dir, "pdfs"), exist_ok=True)
        for idx, (param_name, param_seq) in enumerate(rescaled_sequences.items()):
            # Abstraction for easy parallelization
            pdf_path = os.path.join(args.output_dir, "pdfs", f"{param_name}.pkl")
            if args.load and os.path.exists(pdf_path):
                with open(pdf_path, "rb") as pdf_file:
                    pdfs = pickle.load(pdf_file)["pdf"]
            else:
                pdfs = get_pdf(param_seq, llama, good_tokens, output_file=pdf_path)
            pdf_dict[param_name] = {
                "pdf": pdfs,
                "states": param_seq,
                "init_min": list(sequences.values())[idx].min(),
                "init_max": list(sequences.values())[idx].max(),
            }

        # TODO: Add .get() loop if parallelized to populate pdf_dict

        # Calculate kernels
        kernels_dict = {}
        # TODO: parallelize??
        os.makedirs(os.path.join(args.output_dir, "kernel"), exist_ok=True)
        for param_name, param_dict in pdf_dict.items():
            # Abstraction for easy parallelization
            output_file = os.path.join(args.output_dir, "kernel", f"{param_name}.npz")
            kernel = get_kernel(
                param_dict["pdf"],
                param_dict["states"],
                param_dict["init_min"],
                param_dict["init_max"],
                output_file=output_file,
            )
            kernels_dict[param_name] = kernel

        # TODO: Add .get() loop if parallelized to populate kernels_dict

    def sgd_infer(self, init_model, steps=50, ):
        model = init_model.clone()
        torch.save({"model_state_dict": init_model.state_dict()}, self.infer_init_ckpt_path)

        # Load initial checkpoint
        ckpt = torch.load(self.infer_init_ckpt_path)

        # TODO import MLP and uncomment this
        # model = MLP()
        # model.load_state_dict(ckpt["model_state_dict"])

        # Load kernels
        kernels_dict = sgd_inference.load_kernels(self.kernels_dir)

        init_state_dict = ckpt["model_state_dict"] if "model_state_dict" in ckpt else ckpt
        state_mat = sgd_inference.load_params_to_state_mat(init_state_dict, kernels_dict)

        traj = dict()
        for key in init_state_dict.keys():
            traj[key] = []

        for i in tqdm(range(steps)):
            state_mat = sgd_inference.apply_kernel(kernels_dict, state_mat, init_state_dict)
            param_dict = sgd_inference.state_mat_to_param(state_mat, kernels_dict, init_state_dict)
            for key in param_dict.keys():
                traj[key].append(param_dict[key])
            # Save checkpoint
            # model.load_state_dict(param_dict)
            # torch.save({"model_state_dict": model.state_dict()}, f"{self.output_dir}/ckpt_{i}.pt")
            # metrics = model.get_metrics()
            # np.savez(f"{args.output_dir}/metrics_{i}.npz", metrics)

            # ckpt_path = os.path.join(f"{model.ckpt_dir}/step_{step:03d}.pth")
            # torch.save(model.state_dict(), ckpt_path)
        return traj
#%%
# generate mesh grid 2D
x = np.linspace(-2, 2, 100)
y = np.linspace(-2, 2, 100)
X, Y = np.meshgrid(x, y)
gamma = .9
# z is the conditional pdf p(y|x) = N(y|(1-gamma*2)x; gamma^2)
mu = (1 - gamma**2) * X
sigma = gamma**2
# use numpy pdf
from scipy.stats import norm
Z = norm.pdf(Y, loc=mu, scale=sigma)
#Z = np.exp(-0.5 * ((Y - mu) / sigma)**2) / (np.sqrt(2 * np.pi) * sigma)
# show imag
plt.imshow(Z, origin='lower', extent=(-2, 2, -2, 2))
# %%


#%%