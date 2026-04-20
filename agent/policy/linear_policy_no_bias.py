import numpy as np
from agent.policy.base_policy import Policy
from decimal import Decimal


class LinearPolicy(Policy):
    def __init__(
        self,
        dim_states,
        dim_actions,
        matrix_init_mode="near_zero",
        near_zero_init_scale=0.15,
        near_zero_init_min_abs=0.02,
        near_zero_init_decimals=2,
    ):
        super().__init__(dim_states, dim_actions)

        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.matrix_init_mode = self._normalize_init_mode(matrix_init_mode)
        self.near_zero_init_scale = float(near_zero_init_scale)
        self.near_zero_init_min_abs = float(near_zero_init_min_abs)
        self.near_zero_init_decimals = int(near_zero_init_decimals)

        if self.near_zero_init_scale <= 0:
            raise ValueError(f"near_zero_init_scale must be > 0, got: {self.near_zero_init_scale}")
        if self.near_zero_init_min_abs <= 0:
            raise ValueError(f"near_zero_init_min_abs must be > 0, got: {self.near_zero_init_min_abs}")
        if self.near_zero_init_decimals < 0:
            raise ValueError(
                f"near_zero_init_decimals must be >= 0, got: {self.near_zero_init_decimals}"
            )

        self.initialize_policy()

    def _normalize_init_mode(self, matrix_init_mode):
        mode = str(matrix_init_mode).strip().lower()
        if mode == "zero":
            return "near_zero"
        if mode not in {"random", "near_zero"}:
            raise ValueError(
                f"matrix_init_mode must be 'near_zero', 'random', or 'zero' (alias), got: {matrix_init_mode!r}"
            )
        return mode

    def _enforce_nonzero_values(self, values, source_values=None):
        adjusted = np.array(values, copy=True)
        zero_mask = adjusted == 0.0
        if not np.any(zero_mask):
            return adjusted

        min_abs = self.near_zero_init_min_abs
        reference = adjusted if source_values is None else np.asarray(source_values)
        reference_sign = np.where(reference >= 0.0, 1.0, -1.0)
        replacement = reference_sign[zero_mask] * min_abs

        remaining_zero_mask = replacement == 0.0
        if np.any(remaining_zero_mask):
            replacement[remaining_zero_mask] = np.random.choice(
                [-min_abs, min_abs],
                size=int(np.sum(remaining_zero_mask)),
            )

        adjusted[zero_mask] = replacement
        return adjusted

    def _sample_near_zero_nonzero(self, shape):
        scale = self.near_zero_init_scale
        min_abs = self.near_zero_init_min_abs
        decimals = self.near_zero_init_decimals

        values = np.random.uniform(-scale, scale, size=shape)
        signs = np.where(values >= 0.0, 1.0, -1.0)
        values = np.where(np.abs(values) < min_abs, signs * min_abs, values)
        values = np.round(values, decimals)

        return self._enforce_nonzero_values(values)

    def initialize_policy(self):
        if self.matrix_init_mode == "near_zero":
            self.weight = self._sample_near_zero_nonzero((self.dim_states, self.dim_actions))
        else:
            raw = (np.random.rand(self.dim_states, self.dim_actions) - 0.5) * 6.0
            rounded = np.round(raw, 2)
            self.weight = self._enforce_nonzero_values(rounded, source_values=raw)

    def get_action(self, state):
        state = state.T
        # print(state.shape, self.weight.shape, self.bias.shape)
        # print(np.matmul(state, self.weight).shape, (np.matmul(state, self.weight) + self.bias).shape)
        # print((np.matmul(state, self.weight) + self.bias).shape)
        # print()
        # print(self.weight.shape, state.shape)
        return np.matmul(state, self.weight)

    def __str__(self):
        output = "Weights:\n"
        for w in self.weight:
            output += ", ".join([str(i) for i in w])
            output += "\n"

        return output

    def update_policy(self, weight_and_bias_list):
        if weight_and_bias_list is None:
            return
        self.weight = np.array(weight_and_bias_list)
        self.weight = self.weight.reshape(-1)
        for i in range(len(self.weight)):
            self.weight[i] = Decimal(self.weight[i]).normalize()
        
        self.weight = self.weight.reshape(
            self.dim_states, self.dim_actions
        )
    
    def get_parameters(self):
        return self.weight
