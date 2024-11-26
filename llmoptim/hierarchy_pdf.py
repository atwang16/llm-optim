from typing import Self

import numpy as np
import torch
import torch.nn as nn

from llmoptim.utils import str_seq_to_int, int_seq_to_str


class HierarchyCache:
    """Class wrapper around cache implementation from llmICL.

    The cache represents a tuple of tuples, where each nested tuple isa layer in the transformer containing the key and value states.
    """

    def __init__(self, past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] = None) -> None:
        self.past_key_values: tuple[tuple[torch.Tensor, torch.Tensor], ...] = past_key_values

    @property
    def is_empty(self) -> bool:
        return self.past_key_values is None

    def trim(self, length: int, inplace: bool = False) -> Self:
        trimmed_past_key_values = []
        for layer_past in self.past_key_values:
            key_states, value_states = layer_past
            new_layer_past = (key_states[..., :length, :], value_states[..., :length, :])
            trimmed_past_key_values.append(new_layer_past)

        if inplace:
            self.past_key_values = tuple(trimmed_past_key_values)
            return self
        return HierarchyCache(tuple(trimmed_past_key_values))

    def to(self, device: str) -> Self:
        self.past_key_values = tuple(tuple(x.to(device) for x in sub_tuple) for sub_tuple in self.past_key_values)
        return self

    def cuda(self) -> Self:
        return self.to("cuda")

    def cpu(self) -> Self:
        return self.to("cpu")

    def to_tuple(self) -> tuple[tuple[torch.Tensor, torch.Tensor], ...]:
        return self.past_key_values


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

    def refine(self, model: nn.Module, s_traj: str, kv_cache: HierarchyCache, mode: str = "neighbor") -> Self:
        """Implementation of algorithm 2 of llmICL to refine hierarchy PDF and replication of recursive_refiner() in
        https://github.com/AntonioLiu97/llmICL/blob/master/models/ICL.py

        :param s_traj: a string representing a sampled stochastic trajectory whose states are separated by commas
        :param kv_cache: key-value cache of running model.forward(S_traj)
        :param mode: either "neighbor" (update only the branches around the main branch) or "all" (update all branches)
        """
        inp = []
        for state in str_seq_to_int(s_traj):
            inp.append(state)
            self._recursive_refine(model, True, len(state), inp, kv_cache, mode=mode)

        return self

    def _recursive_refine(
        self,
        model: nn.Module,
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
                if refinement_depth > 1:
                    trimmed_kv_cache = kv_cache.trim(-refinement_depth + 1)
                new_sequence = [*sequence[:-refinement_depth], new_state]
                self._recursive_refine(model, False, refinement_depth, new_sequence, trimmed_kv_cache, mode=mode)

            # iterate at next level
            if refinement_depth > 1:  # recurse to refine down another level
                pdf: HierarchyPDF = self.states[curr_state]
                pdf._recursive_refine(model, True, refinement_depth - 1, sequence, kv_cache, mode=mode)

        else:
            # collect refined logits
            new_logits, kv_cache_new = self._next_token_probs(model, sequence, kv_cache)
            last_digit_pdf = HierarchyPDF.from_sample(sequence, new_logits)
            self.define_branch(sequence[-1], last_digit_pdf)

            if refinement_depth > 1:
                l_new = [[*sequence, i] for i in range(self.n_states)]  # form 10 new sequences by appending digits
                for new_sequence, pdf in zip(l_new, self.states):
                    pdf._recursive_refine(model, False, refinement_depth - 1, new_sequence, kv_cache_new, mode=mode)

    def _next_token_probs(
        self,
        model,
        states: list[list[int]],
        kv_cache: HierarchyCache,
        good_tokens: list = None,
    ) -> tuple[np.ndarray, HierarchyCache]:
        """Calculate the probability of the next token given the current state.

        :param state: a list of integers representing the current state
        :param kv_cache: key-value cache of running model.forward(S_traj)
        """
        s_traj = int_seq_to_str(states)

        with torch.no_grad():
            probs, kv_cache_new, _ = model.forward_probs(s_traj, good_tokens, kv_cache=kv_cache, use_cache=True)

        return probs, kv_cache_new

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
