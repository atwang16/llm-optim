import argparse
import multiprocessing as mp
import os
import pickle
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import partial
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
    model_state_dicts = [torch.load(ckpt_path) for ckpt_path in sorted(glob(f"{ckpts_path}/*"))]
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


def get_pdf(
    sequence: np.ndarray,
    llama: Llama,
    good_tokens: str,
    output_file: str = None,
    refinement_depth: int = 1,
    mode: str = "neighbor",
    str_delimiter: str = ",",
    continue_from: list[HierarchyPDF] = None,
    continue_idx: int = 0,
) -> list[HierarchyPDF]:
    """Compute hierarchical PDF for each step in sequence.

    :param sequence: np.ndarray of integer sequence states between 0 and 10^k (in practice, should be scaled to
    roughly 150 to 850).
    :param llama: LLM model for predicting sequences
    :param good_tokens: IDs of tokens to get probabilities for ("0123456789")
    :param output_file: path to output file for PDFs, defaults to None
    :param refinement_depth: depth of refinement for hierarchical PDF (for coarse bins), defaults to 1 (only top-level
    bin)
    :param mode: "neighbor" or "all" for refining coarse bins around main prediction, defaults to "neighbor"
    :param str_delimiter: delimiter for string version of sequence, defaults to ","
    :return: list of hierarchical PDFs for each step in sequence
    """
    # Get HierarchyPDF
    # If output_dir is not None, save probs into npy (this way we can split jobs for even more distributed computing)
    # build sequence string
    sequence_str = llama.tokenizer.to_string(sequence)
    delimiters = [i for i, char in enumerate(sequence_str) if char == str_delimiter]
    int_seq = str_seq_to_int(sequence_str)

    start_idx = 0
    pdf_list = [] if continue_from is None else continue_from

    if continue_from is not None:
        int_seq = int_seq[continue_idx+1:]
        start_idx = delimiters[continue_idx] + 1
        delimiters = delimiters[continue_idx+1:]

    probs, kv_cache, _ = llama.forward_probs(sequence_str, good_tokens, use_cache=True)
    for idx, (num_list, delim_idx) in tqdm(enumerate(zip(int_seq, delimiters)), total=len(sequence)):
        pdf = HierarchyPDF.from_sample(num_list, probs[0, start_idx:delim_idx])
        rel_idx_from_end = delim_idx - len(sequence_str) - 1
        kv_cache_trimmed = kv_cache.trim(rel_idx_from_end)

        pdf.refine(
            llama,
            s_traj=sequence_str[:delim_idx],
            refinement_depth=refinement_depth,
            kv_cache=kv_cache_trimmed,
            good_tokens=good_tokens,
            mode=mode,
        )
        pdf_list.append(pdf)

        start_idx = delim_idx + 1

        if idx % 10 == 0 and output_file is not None:
            output_parent_dir = os.path.join(os.path.dirname(output_file), "intermediate")
            os.makedirs(output_parent_dir, exist_ok=True)
            intermediate_output_path = os.path.join(output_parent_dir, f"pdfs_{idx}.pkl")
            with open(intermediate_output_path, "wb") as intermediate_output_file:
                pickle.dump({"pdf": pdf_list}, intermediate_output_file)

    if output_file is not None:
        with open(output_file, "wb") as output_file:
            pickle.dump({"pdf": pdf_list}, output_file)

    return pdf_list


def get_kernel(pdfs: list[HierarchyPDF], sequence: np.ndarray, init_min: float, init_max: float, output_file: str = None):
    """Compute transition kernel from hierarchical PDFs for each parameter

    :param pdfs: list of HierarchyPDF objects representing P^{(i,i)}(X_{t+1}|X_t = sequence[t])
    :param sequence: np.ndarray of integer sequence states between 0 and 10^k (in practice, should be scaled to
    roughly 150 to 850).
    :param output_dir: _description_, defaults to None
    :return: _description_
    """
    # FIXME: may have an off by 1 error in aligning PDFs to the sequence?
    assert len(pdfs) == sequence.shape[0]

    # Get kernel
    pdf_unrolled = []
    for pdf in pdfs:
        pdf_unrolled.append(pdf.get_prob())  # (10^prec,)
    pdf_unrolled = np.array(pdf_unrolled)  # (len(sequence), 10^prec)
    assert pdf_unrolled.shape[0] == sequence.shape[0]

    # construct sparse probability matrix
    sparse_prob = np.zeros((pdf_unrolled.shape[1], pdf_unrolled.shape[1]))
    sparse_prob[sequence, :] = pdf_unrolled

    kernel = fill_rows(sparse_prob)
    if output_file is not None:
        np.savez(output_file, kernel=kernel, init_min=init_min, init_max=init_max)
    return kernel


def load_dummy(pkl_path):
    pkl = np.load(pkl_path, allow_pickle=True)
    sequence = pkl["full_series"]
    # Back to floats
    sequence = np.array([int(num) for num in sequence.split(",")[:-1]])
    sequence = sequence[:100]
    return {"full_series": sequence}


def compute_pdf_parallel(param_name, param_seq, llama, good_tokens, output_dir, load_existing, continue_existing, init_seq, refinement_depth):
    pdf_path = os.path.join(output_dir, "pdfs", f"{param_name}.pkl")
    if continue_existing:
        pdf_paths = sorted(glob(f"{output_dir}/pdfs/intermediate/*.pkl"))
        if pdf_paths:
            with open(pdf_paths[-1], "rb") as pdf_file:
                pdfs = pickle.load(pdf_file)["pdf"]
            continue_idx = pdf_paths[-1].split("/")[-1].split(".")[0].split("_")[-1]
            pdfs = get_pdf(param_seq, llama, good_tokens, output_file=pdf_path, continue_from=pdfs if continue_existing else None, continue_idx=continue_idx, refinement_depth=refinement_depth)
    elif load_existing and os.path.exists(pdf_path):
        with open(pdf_path, "rb") as pdf_file:
            pdfs = pickle.load(pdf_file)["pdf"]
    else:
        pdfs = get_pdf(param_seq, llama, good_tokens, output_file=pdf_path)
    return param_name, {
        "pdf": pdfs,
        "states": param_seq,
        "init_min": init_seq.min(),
        "init_max": init_seq.max(),
    }


# Parallelized kernel computation
def compute_kernel_parallel(param_name, param_dict, output_dir):
    output_file = os.path.join(output_dir, "kernel", f"{param_name}.npz")
    kernel = get_kernel(
        param_dict["pdf"],
        param_dict["states"],
        param_dict["init_min"],
        param_dict["init_max"],
        output_file=output_file,
    )
    return param_name, kernel


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpts_path", type=str, required=False, help="Path to directory containing the checkpoints")
    parser.add_argument("--llama_v", choices=[2, 3], type=int, required=True, help="Version of Llama model")
    parser.add_argument("--output_dir", type=str, required=True, help="Path to directory to save inferred PDFs and kernels")
    parser.add_argument("--load", action="store_true", help="if true, load PDFs from output_dir if possible")
    parser.add_argument("--continue", action="store_true", help="Continue from existing PDFs", dest="_continue")
    parser.add_argument("--use-dummy", dest="use_dummy", action="store_true", help="Use dummy data for testing")
    parser.add_argument("--n_threads", type=int, default=1, help="Number of threads for parallel processing")
    parser.add_argument("--gpu_idx", type=int, default=0, help="GPU index to use")
    parser.add_argument("--depth", type=int, default=1, help="Depth of refinement for hierarchical PDF")
    args = parser.parse_args()

    llama = Llama(llama_v=args.llama_v, device=f"cuda:{args.gpu_idx}")

    if args.use_dummy:
        sequences = load_dummy("tmp/brownian_motion_0.pkl")
        rescaled_sequences = sequences
    else:
        sequences = load_ckpts_into_seq(args.ckpts_path)
        rescaled_sequences = {
            param_name: llama.tokenizer.rescale(param_seq) for param_name, param_seq in sequences.items()
        }

    good_tokens_str = list("0123456789")
    good_tokens = [llama.llama_tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

    # Calculate PDFs in parallel
    pdf_dict = {}
    os.makedirs(os.path.join(args.output_dir, "pdfs"), exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = [
            executor.submit(compute_pdf_parallel, param_name, param_seq, llama, good_tokens,
                            args.output_dir, args.load, args._continue, init_seq, args.depth)
            for (param_name, param_seq), init_seq in zip(rescaled_sequences.items(), sequences.values())
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing PDFs"):
            param_name, result = future.result()
            pdf_dict[param_name] = result

    # Calculate kernels in parallel
    kernels_dict = {}
    os.makedirs(os.path.join(args.output_dir, "kernel"), exist_ok=True)

    with ThreadPoolExecutor(max_workers=args.n_threads) as executor:
        futures = [
            executor.submit(compute_kernel_parallel, param_name, param_dict, args.output_dir)
            for param_name, param_dict in pdf_dict.items()
        ]

        for future in tqdm(as_completed(futures), total=len(futures), desc="Computing Kernels"):
            param_name, kernel = future.result()
            kernels_dict[param_name] = kernel
