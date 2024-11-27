import numpy as np
import torch
from transformers.tokenization_utils_base import BatchEncoding


class Tokenizer:
    def __init__(self, tokenizer, n_digits: int = 3, min_: float = 1.50, max_: float = 8.50) -> None:
        self.tokenizer = tokenizer
        self.min = min_
        self.max = max_
        self.n_digits = n_digits
        self.eos_token_id = self.tokenizer.eos_token_id if tokenizer else None

    def decode(self, generated_ids, skip_special_tokens=True, clean_up_tokenization_space=True):
        return self.tokenizer.decode(
            generated_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_space=clean_up_tokenization_space,
        )

    def rescale(self, data: np.ndarray):
        new_range = (data - np.min(data)) / (np.max(data) - np.min(data)) * (self.max - self.min) + self.min
        return (new_range * 10 ** (self.n_digits - 1)).astype(int)

    def to_string(self, data: np.ndarray):
        # def to_string_num(num: float) -> str:
        #     num *= 10 ** (self.n_digits - 1)
        #     return "".join(letter for letter in f"{int(num):0{self.n_digits}d}")
        return ",".join(str(value) for value in data) + ","

    def __call__(self, sequence: str, return_tensors: str = None, rescale: bool = True):
        if rescale:
            data = np.array(list(map(float, sequence.split(",")[:-1])))
            data = self.rescale(data / 100)
            data = np.round(data, self.n_digits - 1)
            data_string = self.to_string(data)
        else:
            data_string = sequence
        # data_tensor = torch.tensor([self.tokenizer(data_string[i], return_tensors=return_tensors).input_ids[0][1] for i in range(len(data_string))], dtype=torch.long)
        # data_tensor = torch.cat([torch.tensor([self.tokenizer.bos_token_id], dtype=torch.long), data_tensor], dim=0)
        data_tensor = self.tokenizer(data_string, return_tensors=return_tensors).input_ids[0]
        return BatchEncoding(
            {"input_ids": data_tensor.unsqueeze(0), "attention_mask": torch.ones_like(data_tensor).unsqueeze(0)}
        )
