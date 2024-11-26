import sys

import bitsandbytes as bnb
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    LlamaForCausalLM,
)

sys.path.append("../")
from llmoptim.hierarchy_pdf import HierarchyCache, HierarchyPDF
from llmoptim.tokenizer import Tokenizer


class Llama(nn.Module):
    def __init__(
        self,
        _4bit: bool = True,
        _8bit: bool = False,
        flash_attention: bool = False,
        dtype=torch.float16,
        llama_v: [2, 3] = 3,
    ):
        super().__init__()
        if _4bit:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=dtype,
                bnb_4bit_use_double_quant=True,
            )
        elif _8bit:
            bnb_config = BitsAndBytesConfig(
                load_in_8bit=True,
                bnb_8bit_quant_type="nf8",
                bnb_8bit_compute_dtype=dtype,
                bnb_8bit_use_double_quant=True,
            )
        if flash_attention:
            attn_type = "flash_attention_2"
        else:
            # Default value will be used
            attn_type = None
        if llama_v == 3:
            self.model = LlamaForCausalLM.from_pretrained(
                "NousResearch/Hermes-3-Llama-3.1-8B",
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
                attn_implementation=attn_type,
            )
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                "NousResearch/Hermes-3-Llama-3.1-8B", trust_remote_code=True
            )
        elif llama_v == 2:
            self.model = AutoModelForCausalLM.from_pretrained(
                "NousResearch/Nous-Hermes-llama-2-7b",
                torch_dtype=dtype,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
                attn_implementation=attn_type,
            )
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                "NousResearch/Nous-Hermes-llama-2-7b", trust_remote_code=True
            )

        self.tokenizer = Tokenizer(self.llama_tokenizer)

    def forward_llama(
        self, prompt, max_new_tokens=128, temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0, do_sample=True
    ):
        """
        This method is for text generation, not llmoptim
        """
        input_ids = self.llama_tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            eos_token_id=self.llama_tokenizer.eos_token_id,
        )
        return (
            self.llama_tokenizer.decode(
                generated_ids[0][input_ids.shape[-1] :], skip_special_tokens=True, clean_up_tokenization_space=True
            ),
            generated_ids,
        )

    def forward_response(
        self, prompt, max_new_tokens=128, temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0, do_sample=True
    ):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        generated_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
            do_sample=do_sample,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return (
            self.tokenizer.decode(
                generated_ids[0][input_ids.shape[-1] :], skip_special_tokens=True, clean_up_tokenization_space=True
            ),
            generated_ids,
        )

    def forward_state(self, prompt, good_tokens, state_len, temperature=1.0, kv_cache=None, use_cache=False):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        new_state = ""
        for _ in range(state_len):
            with torch.no_grad():
                out = self.model(
                    input_ids,
                    use_cache=use_cache,
                    past_key_values=(
                        tuple(tuple(x.cuda() for x in sub_tuple) for sub_tuple in kv_cache)
                        if kv_cache is not None and use_cache
                        else None
                    ),
                )
            logit_mat = out["logits"]
            kv_cache = out["past_key_values"] if kv_cache is not None and use_cache else None
            probs = torch.nn.functional.softmax(
                logit_mat[0, -1, good_tokens].clone().cpu().to(torch.float32), dim=0
            ).numpy()
            next_token = good_tokens[np.argmax(probs)]
            new_state += self.tokenizer.decode([next_token])[0]
            input_ids = torch.cat([input_ids, torch.tensor([[next_token]], dtype=torch.long).to("cuda")], dim=-1)
        return (
            new_state,
            (
                tuple(tuple(x.cpu() for x in sub_tuple) for sub_tuple in kv_cache)
                if kv_cache is not None and use_cache
                else None
            ),
        )

    def forward_probs(self, prompt, good_tokens, kv_cache=None, use_cache=False):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to("cuda")
        with torch.no_grad():
            out = self.model(
                input_ids,
                use_cache=use_cache,
                past_key_values=(
                    tuple(tuple(x.cuda() for x in sub_tuple) for sub_tuple in kv_cache)
                    if kv_cache is not None and use_cache
                    else None
                ),
            )
        logit_mat = out["logits"]
        kv_cache = out["past_key_values"]
        probs = torch.nn.functional.softmax(
            logit_mat[0, -1, good_tokens].clone().cpu().to(torch.float32), dim=0
        ).numpy()
        good_logits = logit_mat[0, -1, good_tokens].clone().cpu().to(torch.float32).numpy()
        return probs, kv_cache, good_logits


if __name__ == "__main__":
    import numpy as np

    # prompts = ["""123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678,123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678,123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678,123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678,123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678"""]

    prompts = [
        np.array(
            [
                1.23,
                4.56,
                7.89,
                2.34,
                5.67,
                8.90,
                3.45,
                6.78,
                9.01,
                4.56,
                7.89,
                0.12,
                5.67,
                8.90,
                1.23,
                6.78,
                9.01,
                2.34,
                7.89,
                0.12,
                3.45,
                8.90,
                1.23,
                4.56,
                9.01,
                2.34,
                5.67,
                0.12,
                3.45,
                6.78,
                1.23,
                4.56,
                7.89,
                2.34,
                5.67,
                8.90,
                3.45,
                6.78,
                9.01,
                4.56,
                7.89,
                0.12,
                5.67,
                8.90,
                1.23,
                6.78,
                9.01,
                2.34,
                7.89,
                0.12,
                3.45,
                8.90,
                1.23,
                4.56,
                9.01,
                2.34,
                5.67,
                0.12,
                3.45,
                6.78,
                1.23,
                4.56,
                7.89,
                2.34,
                5.67,
                8.90,
                3.45,
                6.78,
                9.01,
                4.56,
                7.89,
                0.12,
                5.67,
                8.90,
                1.23,
                6.78,
                9.01,
                2.34,
                7.89,
                0.12,
                3.45,
                8.90,
                1.23,
                4.56,
                9.01,
                2.34,
                5.67,
                0.12,
                3.45,
                6.78,
                1.23,
                4.56,
                7.89,
                2.34,
                5.67,
                8.90,
                3.45,
                6.78,
                9.01,
                4.56,
                7.89,
                0.12,
                5.67,
                8.90,
            ]
        )
    ]
    # prompts.append(prompts[0])

    llama = Llama(llama_v=2)

    """for prompt in prompts:
        import time
        start = time.time()
        response, tokens = llama.forward_response(prompt, max_new_tokens=1024)
        time_taken = time.time() - start
        print(f"Generated response: {response}")
        print(f"Response took {time_taken} seconds")
        print(f"With the total of {len(tokens[0])} tokens")
        print(f"For token/sec of {len(tokens[0]) / time_taken}")"""

    good_tokens_str = list("0123456789")
    good_tokens = [llama.llama_tokenizer.convert_tokens_to_ids(token) for token in good_tokens_str]

    import time

    n_states = 50
    kv_cache = None
    """start = time.time()
    for _ in range(n_states):
        next_state, kv_cache = llama.forward_state(prompts[0], good_tokens, 3, use_cache=False, kv_cache=kv_cache)
        next_state = float(next_state[:1] + "." + next_state[1:])
        # Append the next state to the prompt
        prompts[0] = np.append(prompts[0], next_state)
    time_taken = time.time() - start
    print(f"Generated next state: {next_state}")
    print(f"Next state took {time_taken} seconds")"""
    pkl = np.load("/local-scratch2/cmpt981/llmICL/generated_series/geometric_brownian_motion_0.pkl", allow_pickle=True)
    seq = pkl["full_series"].split(",")[:-1][:940]
    import pdb

    pdb.set_trace()
    # All to floats
    prompt = np.zeros(len(seq))
    for i in range(len(seq)):
        prompt[i] = float(seq[i]) / 100

    probs, kv_cache, logits = llama.forward_probs(prompt, good_tokens, kv_cache=kv_cache, use_cache=False)
    import pdb

    pdb.set_trace()
    kv_cache = HierarchyCache(kv_cache)

    pdf = HierarchyPDF(True, 10, probs)
    import pdb

    pdb.set_trace()
    pdf.refine(llama, tokenizer=llama.tokenizer, s_traj=seq, kv_cache=kv_cache, good_tokens=good_tokens, mode="pdf")
    # Plot
    import matplotlib.pyplot as plt

    plt.plot(prompt)
    plt.show()
    import pdb

    pdb.set_trace()
