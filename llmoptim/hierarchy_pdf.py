from typing import Self

import numpy as np
import torch
import torch.nn as nn

from llmoptim.utils import str_seq_to_int, int_seq_to_str


class HierarchyCache:
    def __init__(self, num_levels: int):
        self.num_levels = num_levels  # last layer is array
        self.cache = {}

    def __contains__(self, key: list[int]) -> bool:
        cache = self.cache
        for k in key:
            try:
                cache = cache[k]
            except KeyError:
                return False
        return "logits" in cache

    def __getitem__(self, key: list[int]) -> float:
        cache = self.cache
        for k in key[:-1]:
            try:
                cache = cache[k]
            except KeyError:
                raise KeyError(f"Could not find cached probability for key: {key}")
        return cache["logits"][key[-1]]

    def __setitem__(self, key: list[int], value: np.ndarray):
        if len(key) == self.num_levels - 1:
            cache = self.cache
            for k in key:
                if k in cache:
                    cache = cache[k]
                else:
                    cache[k] = {}
                    cache = cache[k]
            cache["logits"] = value
        else:
            raise ValueError(
                f"Key length must be one less than the number of levels (last level is an array), but got: {key}"
            )


class HierarchyPDF:
    """Implementation of algorithm from llmICL.

    This is a recreation of https://github.com/AntonioLiu97/llmICL/blob/master/models/ICL.py using a recursive class
    rather than a functional implementation.
    """

    def __init__(self, is_head: bool, n_states: int, init_prob: np.ndarray = None) -> None:
        self.is_head = is_head
        self.n_states = n_states

        # initialize PDF
        self.prob = init_prob

        # add hierarchical level
        self.states: list[HierarchyPDF] = [None for _ in range(n_states)]

    @property
    def is_leaf(self) -> bool:
        return self.n_levels == 1

    def refine(
        self, model: nn.Module, tokenizer, s_traj: str, kv_cache: HierarchyCache, mode: str = "neighbor"
    ) -> Self:
        """Implementation of algorithm 2 of llmICL to refine hierarchy PDF and replication of recursive_refiner() in
        https://github.com/AntonioLiu97/llmICL/blob/master/models/ICL.py

        :param s_traj: a string representing a sampled stochastic trajectory whose states are separated by commas
        :param kv_cache: key-value cache of running model.forward(S_traj)
        :param mode: either "neighbor" (update only the branches around the main branch) or "all" (update all branches)
        """
        inp = []
        for state in str_seq_to_int(s_traj):
            inp.append(state)
            self._recursive_refine(model, tokenizer, True, len(state), inp, kv_cache, mode=mode)

        return self

    def _recursive_refine(
        self,
        model: nn.Module,
        tokenizer,
        is_main_branch: bool,
        refinement_depth: int,
        sequence: list[list[int]],
        kv_cache: HierarchyCache,
        mode: str = "neighbor",
    ):
        if refinement_depth == 0:
            return

        curr_state = sequence[-refinement_depth]

        # is_main_branch because we have already calculated and cached these logits
        if is_main_branch:
            # refine neighboring states at current level
            if mode == "neighbor":
                new_states = [ns for ns in [curr_state - 1, curr_state + 1] if ns >= 0 and ns < self.n_states]

            elif mode == "all":
                new_states = [ns for ns in range(self.n_states) if ns != curr_state]

            else:
                raise ValueError(f"Invalid mode. Expected 'neighbor' or 'all', but got: {mode}")

            for new_state in new_states:
                new_sequence = [*sequence[: -self.n_levels], new_state]
                self._recursive_refine(model, tokenizer, False, refinement_depth, new_sequence, kv_cache, mode=mode)

            # iterate at next level
            if refinement_depth > 1:  # recurse to refine down another level
                pdf: HierarchyPDF = self.states[curr_state]
                pdf._recursive_refine(model, tokenizer, True, refinement_depth - 1, sequence, kv_cache, mode=mode)

        else:
            # collect refined logits
            new_logits, kv_cache_new = self._next_token_probs(model, tokenizer, sequence, kv_cache)
            last_digit_pdf = HierarchyPDF.from_sample(sequence, new_logits)
            self.define_branch(sequence[-1], last_digit_pdf)

            if refinement_depth > 1:
                l_new = [[*sequence, i] for i in range(self.n_states)]  # form 10 new sequences by appending digits
                for new_sequence, pdf in zip(l_new, self.states):
                    pdf._recursive_refine(
                        model, tokenizer, False, refinement_depth - 1, new_sequence, kv_cache_new, mode=mode
                    )

    def _next_token_probs(
        self,
        model: nn.Module,
        tokenizer,
        states: list[list[int]],
        kv_cache: HierarchyCache,
        good_tokens: list = None,
        load_cache_to_cpu: bool = False,
    ) -> np.ndarray:
        """Calculate the probability of the next token given the current state.

        :param state: a list of integers representing the current state
        :param kv_cache: key-value cache of running model.forward(S_traj)
        """
        s_traj = int_seq_to_str(states)
        batch = tokenizer([s_traj], return_tensors="pt", add_special_tokens=True)

        if load_cache_to_cpu:
            kv_cache = tuple(tuple(x.cuda() for x in sub_tuple) for sub_tuple in kv_cache)
        with torch.no_grad():
            out = model(batch["input_ids"][:, -1:].cuda(), use_cache=True, past_key_values=kv_cache)

        logit_mat = out["logits"]
        if load_cache_to_cpu:
            kv_cache_new = tuple(tuple(x.cpu() for x in sub_tuple) for sub_tuple in out["past_key_values"])
        else:
            kv_cache_new = out["past_key_values"]
        probs = torch.nn.functional.softmax(logit_mat[0, -1, good_tokens].clone().cpu(), dim=0).numpy()
        return (probs, kv_cache_new)

    def update(self, pdf: Self) -> Self:
        if self.prob is None:
            self.prob = pdf.prob
        elif pdf.prob is None:
            pass
        else:  # neither is None
            for i in range(self.n_states):
                if self.states[i] is None:
                    self.states[i] = pdf.states[i]
                else:
                    self.states[i].update(pdf.states[i])

    def define_branch(self, branch: int, pdf: Self) -> Self:
        assert self.states[branch] is None
        self.states[branch] = pdf

    @classmethod
    def from_sample(cls, digits: list[int], probs: np.ndarray) -> Self:
        assert len(digits) == probs.shape[0]
        prob = probs[0, :]

        pdf = HierarchyPDF(True, probs.shape[1], init_prob=prob)
        if len(digits) > 1:
            pdf.define_branch(digits[0], cls.from_sample(digits[1:], probs[1:, :]))

        return pdf
