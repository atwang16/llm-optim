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
from llmoptim.tokenizer import Tokenizer


class Llama(nn.Module):
    def __init__(self, _4bit: bool = True, _8bit: bool = False, flash_attention: bool = False, dtype=torch.float16):
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
        self.model = LlamaForCausalLM.from_pretrained(
            "NousResearch/Hermes-3-Llama-3.1-8B",
            torch_dtype=dtype,
            device_map="auto",
            quantization_config=bnb_config,
            trust_remote_code=True,
            attn_implementation=attn_type,
        )
        self.llama_tokenizer = AutoTokenizer.from_pretrained("NousResearch/Hermes-3-Llama-3.1-8B",
                                                             trust_remote_code=True)
        self.tokenizer = Tokenizer(self.llama_tokenizer)

    def forward_llama(self, prompt, max_new_tokens=128, temperature=1.0, top_p=1.0, top_k=50, repetition_penalty=1.0, do_sample=True):
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
        return (self.llama_tokenizer.decode(generated_ids[0][input_ids.shape[-1]:], skip_special_tokens=True, clean_up_tokenization_space=True), generated_ids)


if __name__ == "__main__":
    prompts = ["""123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678,123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678,123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678,123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678,123,456,789,234,567,890,345,678,901,456,789,012,567,890,123,678,901,234,789,012,345,890,123,456,901,234,567,012,345,678"""]

    llama = Llama()
    for prompt in prompts:
        import time
        start = time.time()
        response, tokens = llama.forward_llama(prompt, max_new_tokens=1024, temperature=0.2)
        time_taken = time.time() - start
        print(f"Generated response: {response}")
        print(f"Response took {time_taken} seconds")
        print(f"With the total of {len(tokens[0])} tokens")
        print(f"For token/sec of {len(tokens[0]) / time_taken}")
