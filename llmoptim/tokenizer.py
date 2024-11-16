import numpy as np


class Tokenizer:
    def __init__(self, tokenizer, n_digits: int = 3, min_: float = 1.50, max_: float = 8.50) -> None:
        self.tokenizer = tokenizer
        self.min = min_
        self.max = max_
        self.n_digits = n_digits

    def _rescale(self, data: np.ndarray):
        return (data - np.min(data)) / (np.max(data) - np.min(data)) * (self.max - self.min) + self.min

    def _to_string(self, data: np.ndarray):
        def to_string_num(num: float) -> str:
            num *= 10**self.n_digits
            return " ".join(letter for letter in str(int(num)))

        return " , ".join(to_string_num(value) for value in data)

    def __call__(self, data: np.ndarray):
        data = self._rescale(data)
        data = np.round(data, self.n_digits)
        return self.tokenizer(self._to_string(data))
