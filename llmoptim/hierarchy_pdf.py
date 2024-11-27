from copy import deepcopy

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

    @property
    def num_layers(self) -> int:
        return len(self.past_key_values)

    def __len__(self) -> int:
        return self.past_key_values[0][0].shape[2]

    def trim(self, length: int, inplace: bool = False) -> "HierarchyCache":
        trimmed_past_key_values = []
        for layer_past in self.past_key_values:
            key_states, value_states = layer_past
            new_layer_past = (key_states[..., :length, :], value_states[..., :length, :])
            trimmed_past_key_values.append(new_layer_past)

        if inplace:
            self.past_key_values = tuple(trimmed_past_key_values)
            return self
        return HierarchyCache(tuple(trimmed_past_key_values))

    def to(self, device: str) -> "HierarchyCache":
        self.past_key_values = tuple(tuple(x.to(device) for x in sub_tuple) for sub_tuple in self.past_key_values)
        return self

    def cuda(self) -> "HierarchyCache":
        return self.to("cuda")

    def cpu(self) -> "HierarchyCache":
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

    def get_prob(self) -> np.ndarray:
        if self.prob is None:
            return None

        if all(state is None for state in self.states):
            return self.prob

        next_level_states = []
        for state in self.states:
            if state is not None:
                prob_state = state.get_prob()
            else:
                prob_state = None
            next_level_states.append(prob_state)
        
        # set missing distributions as uniform
        max_dim = max(state.shape[0] for state in next_level_states if state is not None)
        for i, state in enumerate(next_level_states):
            if state is None:
                next_level_states[i] = np.ones((max_dim, )) / max_dim
        
        next_level_states = np.stack(next_level_states)

        unraveled_probability = (self.prob[:, None] * next_level_states).flatten()
        return unraveled_probability

    def refine(
        self,
        model: nn.Module,
        s_traj: str,
        kv_cache: HierarchyCache,
        mode: str = "neighbor",
        refinement_depth: int = None,
        good_tokens: list[int] = None,
    ) -> "HierarchyPDF":
        """Implementation of algorithm 2 of llmICL to refine hierarchy PDF and replication of recursive_refiner() in
        https://github.com/AntonioLiu97/llmICL/blob/master/models/ICL.py

        :param s_traj: a string representing a sampled stochastic trajectory whose states are separated by commas
        :param kv_cache: key-value cache of running model.forward(S_traj)
        :param mode: either "neighbor" (update only the branches around the main branch) or "all" (update all branches)
        """
        inp = str_seq_to_int(s_traj)
        if refinement_depth is None:
            refinement_depth = len(inp[0])

        self._recursive_refine(model, True, 0, refinement_depth, inp, kv_cache, mode=mode, good_tokens=good_tokens)

        return self

    def _recursive_refine(
        self,
        model: nn.Module,
        is_main_branch: bool,
        current_pos: int,
        refinement_depth: int,
        sequence: list[list[int]],
        kv_cache: HierarchyCache,
        mode: str = "neighbor",
        good_tokens: list[int] = None,
    ):
        """_summary_

        :param model: _description_
        :param is_main_branch: _description_
        :param current_pos: index into last number in sequence
        :param refinement_depth: _description_
        :param sequence: _description_
        :param kv_cache: _description_
        :param mode: _description_, defaults to "neighbor"
        :param good_tokens: _description_, defaults to None
        :raises ValueError: _description_
        """
        if current_pos == refinement_depth:
            return

        # is_main_branch because we have already calculated and cached these logits
        if is_main_branch:
            curr_state = sequence[-1][current_pos]

            # refine neighboring states at current level
            if mode == "neighbor":
                new_states = [ns for ns in [curr_state - 1, curr_state + 1] if ns >= 0 and ns < self.n_states]

            elif mode == "all":
                new_states = [ns for ns in range(self.n_states) if ns != curr_state]

            else:
                raise ValueError(f"Invalid mode. Expected 'neighbor' or 'all', but got: {mode}")

            for new_state in new_states:
                trimmed_kv_cache = (
                    kv_cache.trim(current_pos - len(sequence[-1])) if current_pos < len(sequence[-1]) - 1 else kv_cache
                )
                trimmed_sequence = deepcopy(sequence)
                trimmed_sequence[-1] = trimmed_sequence[-1][:current_pos]
                trimmed_sequence[-1].append(new_state)
                self._recursive_refine(
                    model,
                    False,
                    current_pos,
                    refinement_depth,
                    trimmed_sequence,
                    trimmed_kv_cache,
                    mode=mode,
                    good_tokens=good_tokens,
                )

            # iterate at next level
            if current_pos < refinement_depth - 1:  # recurse to refine down another level
                pdf: HierarchyPDF = self.states[curr_state]
                pdf._recursive_refine(
                    model,
                    True,
                    current_pos + 1,
                    refinement_depth,
                    sequence,
                    kv_cache,
                    mode=mode,
                    good_tokens=good_tokens,
                )

        else:
            # collect refined logits
            new_logits, kv_cache_new = self._next_token_probs(model, sequence, kv_cache, good_tokens=good_tokens)
            last_digit_pdf = HierarchyPDF.from_sample(sequence[-1], new_logits[0, : len(sequence[-1])])
            self.update(last_digit_pdf)

            if current_pos < refinement_depth - 1:
                for i, pdf in enumerate(self.states):
                    if pdf is None:
                        self.states[i] = HierarchyPDF(True, self.n_states)
                        pdf = self.states[i]
                    new_sequence = deepcopy(sequence)
                    new_sequence[-1].append(i)
                    pdf._recursive_refine(
                        model,
                        False,
                        current_pos + 1,
                        refinement_depth,
                        new_sequence,
                        kv_cache_new,
                        mode=mode,
                        good_tokens=good_tokens,
                    )

    def _next_token_probs(
        self,
        model,
        states: list[list[int]],
        kv_cache: HierarchyCache,
        good_tokens: list[int] = None,
    ) -> tuple[np.ndarray, HierarchyCache]:
        """Calculate the probability of the next token given the current state.

        :param state: a list of integers representing the current state
        :param kv_cache: key-value cache of running model.forward(S_traj)
        """
        s_traj = int_seq_to_str(states)

        with torch.no_grad():
            probs, kv_cache_new, _ = model.forward_probs(s_traj, good_tokens, kv_cache=kv_cache, use_cache=True)

        return probs, kv_cache_new

    def update(self, pdf: "HierarchyPDF") -> "HierarchyPDF":
        if self.prob is None:
            self.prob = pdf.prob
        elif pdf.prob is None:
            pass
        else:  # neither is None
            for i in range(self.n_states):
                if self.states[i] is None:
                    self.states[i] = pdf.states[i]
                elif pdf.states[i] is not None:
                    self.states[i].update(pdf.states[i])

    def define_branch(self, branch: int, pdf: "HierarchyPDF") -> "HierarchyPDF":
        assert self.states[branch] is None
        self.states[branch] = pdf

    @classmethod
    def from_sample(cls, digits: list[int], probs: np.ndarray) -> "HierarchyPDF":
        assert len(digits) == probs.shape[0]
        prob = probs[0, :]

        pdf = HierarchyPDF(True, prob.shape[0], init_prob=prob)
        if len(digits) > 1:
            pdf.define_branch(digits[0], cls.from_sample(digits[1:], probs[1:, :]))

        return pdf
