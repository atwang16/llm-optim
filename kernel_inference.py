import argparse

import numpy as np
import torch
from llmoptim.hierarchy_pdf import HierarchyCache, HierarchyPDF
from llmoptim.kernel import fill_rows
from llmoptim.tokenizer import Tokenizer
from models.llama import Llama
from tqdm import tqdm


def load_ckpts_into_seq(ckpts_path):
    # Loads checkpoints and constructs sequence for each parameter
    # Returns a dict {"{param_name}": np.ndarray({n_ckpts}), ...},
    # where n_ckpts practically is time series length we provide as an input
    # remember to account for param_name being actually {layer_name}_{param_flattened_index}
    # TODO
    pass


def get_pdf(sequence, llama, good_tokens, output_dir=None):
    # Get HierarchyPDF
    # If output_dir is not None, save probs into npy (this way we can split jobs for even more distributed computing)
    # TODO
    pass


def get_kernel(pdf, output_dir=None):
    # Get kernel
    # If output_dir is not None, save kernel into npz
    pdf = param_dict["pdf"]
    P = pdf.probs  # TODO: Probably wrong, fix once HierarchyPDF is finished
    kernel = fill_rows(P)
    if output_dir is not None:
        np.savez(output_dir, kernel=kernel, init_min=param_dict["init_min"], init_max=param_dict["init_max"])
    return kernel


def load_dummy(pkl_path):
    pkl = np.load(pkl_path, allow_pickle=True)
    sequence = pkl["full_series"]
    # Back to floats
    sequence = np.array([float(num) / 100 for num in sequence.split(",")[:-1]])
    return {"full_series": sequence}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts_path", type=str, required=True, help="Path to directory containing the checkpoints")
    parser.add_argument("--llama_v", choices=[2, 3], type=int, required=True, help="Version of Llama model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory to save inferred PDFs and kernels")
    args = parser.parse_args()

    # Load dummy for now
    # sequences = load_ckpts_into_seq(args.ckpts_path)
    sequences = load_dummy("tmp/brownian_motion_0.pkl")

    llama = Llama(llama_v=args.llama_v)
    good_tokens_str = list("0123456789")
    good_tokens = [llama.llama_tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

    # Calculate PDFs
    pdf_dict = {}
    # TODO: parallelize??
    for param_name, param_seq in sequences.items():
        # Abtraction for easy parallelization
        # output_dir = f"{args.output_dir}/pdf/{param_name}.npy"
        pdf = get_pdf(param_seq, llama, good_tokens, output_dir=None)
        pdf_dict[param_name] = {"pdf": pdf, "init_min:": param_seq.min(), "init_max": param_seq.max()}

    # TODO: Add .get() loop if parallelized to populate pdf_dict

    # Calculate kernels
    kernels_dict = {}
    # TODO: parallelize??
    for param_name, param_dict in pdf_dict.items():
        # Abstraction for easy parallelization
        # output_dir = f"{args.output_dir}/kernel/{param_name}.npy"
        kernel = get_kernel(param_dict, output_dir=None)
        kernels_dict[param_name] = kernel

    # TODO: Add .get() loop if parallelized to populate kernels_dict