import numpy as np
from agent.policy.base_policy import Policy
from decimal import Decimal


class LinearPolicy(Policy):
    def __init__(self, dim_states, dim_actions):
        super().__init__(dim_states, dim_actions)

        self.dim_states = dim_states
        self.dim_actions = dim_actions

        self.weight = self._sample_near_zero_nonzero((self.dim_states, self.dim_actions))

    def _sample_near_zero_nonzero(self, shape, scale=0.15, min_abs=0.02, decimals=2):
        values = np.random.uniform(-scale, scale, size=shape)
        signs = np.where(values >= 0.0, 1.0, -1.0)
        values = np.where(np.abs(values) < min_abs, signs * min_abs, values)
        values = np.round(values, decimals)

        zero_mask = values == 0.0
        if np.any(zero_mask):
            replacement = np.random.choice([-min_abs, min_abs], size=int(np.sum(zero_mask)))
            values[zero_mask] = replacement

        return values

    def initialize_policy(self):
        self.weight = self._sample_near_zero_nonzero((self.dim_states, self.dim_actions))

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
