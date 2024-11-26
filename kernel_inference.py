import argparse
import pickle
from glob import glob

import numpy as np
import torch
from llmoptim.hierarchy_pdf import HierarchyCache, HierarchyPDF
from llmoptim.kernel import fill_rows
from llmoptim.tokenizer import Tokenizer
from llmoptim.utils import int_to_list_int, str_seq_to_int
from models.llama import Llama
from tqdm import tqdm


def load_ckpts_into_seq(ckpts_path):
    # Loads checkpoints and constructs sequence for each parameter
    # Returns a dict {"{param_name}": np.ndarray({n_ckpts}), ...},
    # where n_ckpts practically is time series length we provide as an input
    # remember to account for param_name being actually {layer_name}_{param_flattened_index}
    print("DEBUG< LOADING 5 CHECKPOINT ONLY !!!!!!!!\n\n\n\n")
    model_state_dicts = [torch.load(ckpt_path) for ckpt_path in sorted(glob(f"{ckpts_path}/*")[:5])]
    param_specs = [(param_name, param_mat.size()) for param_name, param_mat in model_state_dicts[0].items()]
    sequences = {}
    for param_spec in param_specs:
        param_name, param_size = param_spec
        total_flattened = np.prod(param_size)
        for i in range(total_flattened):
            param_seq = np.zeros(len(model_state_dicts))
            for j, model_state_dict in enumerate(model_state_dicts):
                param_seq[j] = model_state_dict[param_name].flatten()[i]
            sequences[f"{param_name}_{i}"] = param_seq
    return sequences


def get_pdf(sequence: np.ndarray, llama: Llama, good_tokens: str, output_file: str = None):
    """_summary_

    :param sequence: _description_
    :param llama: _description_
    :param good_tokens: _description_
    :param output_dir: _description_, defaults to None
    """
    # Get HierarchyPDF
    # If output_dir is not None, save probs into npy (this way we can split jobs for even more distributed computing)
    # build sequence string
    sequence = llama.tokenizer._rescale(sequence)
    sequence_str = llama.tokenizer._to_string(sequence)
    delimiters = [i for i, char in enumerate(sequence_str) if char == ","]

    probs, kv_cache, _ = llama.forward_probs(sequence_str, good_tokens, use_cache=True)
    start_idx = 0
    pdf_list = []
    for num_list, delim_idx in tqdm(zip(str_seq_to_int(sequence_str), delimiters), total=len(sequence)):
        pdf = HierarchyPDF.from_sample(num_list, probs[0, start_idx:delim_idx])
        rel_idx_from_end = delim_idx - len(sequence)
        kv_cache_trimmed = kv_cache.trim(rel_idx_from_end)
        pdf.refine(
            llama,
            s_traj=sequence_str[:delim_idx],
            refinement_depth=3,
            kv_cache=kv_cache_trimmed,
            good_tokens=good_tokens,
            mode="neighbor",
        )
        pdf_list.append(pdf)

        start_idx = delim_idx + 1

    if output_file is not None:
        with open(output_file, "wb") as output_file:
            pickle.dump(output_file, {"pdf": pdf_list})

    return pdf_list


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
    parser.add_argument("--ckpts_path", type=str, required=False, help="Path to directory containing the checkpoints")
    parser.add_argument("--llama_v", choices=[2, 3], type=int, required=True, help="Version of Llama model")
    parser.add_argument(
        "--output_dir", type=str, required=False, help="Path to directory to save inferred PDFs and kernels"
    )
    args = parser.parse_args()

    sequences = load_ckpts_into_seq(args.ckpts_path)

    llama = Llama(llama_v=args.llama_v)
    good_tokens_str = list("0123456789")
    good_tokens = [llama.llama_tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

    # Calculate PDFs
    pdf_dict = {}
    # TODO: parallelize??
    for param_name, param_seq in sequences.items():
        # Abstraction for easy parallelization
        # output_dir = f"{args.output_dir}/pdf/{param_name}.npy"

        import time
        start = time.time()
        pdf = get_pdf(param_seq, llama, good_tokens, output_file=None)
        print(f"Elapsed time: {time.time() - start}")
        import pdb; pdb.set_trace()
        pdf_dict[param_name] = {"pdf": pdf, "init_min:": param_seq.min(), "init_max": param_seq.max()}

    # TODO: Add .get() loop if parallelized to populate pdf_dict

    # # Calculate kernels
    # kernels_dict = {}
    # # TODO: parallelize??
    # for param_name, param_dict in pdf_dict.items():
    #     # Abstraction for easy parallelization
    #     # output_dir = f"{args.output_dir}/kernel/{param_name}.npy"
    #     kernel = get_kernel(param_dict, output_dir=None)
    #     kernels_dict[param_name] = kernel

    # # TODO: Add .get() loop if parallelized to populate kernels_dict
