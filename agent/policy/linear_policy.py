import numpy as np
from agent.policy.base_policy import Policy


class LinearPolicy(Policy):
    def __init__(self, dim_states, dim_actions, use_factorized_policy=False, factor_rank=None):
        super().__init__(dim_states, dim_actions)

        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.use_factorized_policy = use_factorized_policy

        # Determine factor rank (inner dimension for L @ U = policy)
        if use_factorized_policy:
            if factor_rank is None:
                # Default: use min(dim_states, dim_actions) // 2 for reduced representation
                self.factor_rank = max(1, min(dim_states, dim_actions) // 2)
            else:
                self.factor_rank = min(factor_rank, min(dim_states, dim_actions))
        else:
            self.factor_rank = None

        self.weight = np.random.rand(self.dim_states, self.dim_actions)
        self.bias = np.random.rand(1, self.dim_actions)

        # Two-matrix factor components: policy = L @ U
        self.L = None  # shape: (dim_states, factor_rank)
        self.U = None  # shape: (factor_rank, dim_actions)

    def initialize_policy(self):
        # self.weight = np.round((np.random.rand(self.dim_states, self.dim_actions)) * 1, 1)
        # self.bias = np.round((np.random.rand(1, self.dim_actions) - 0.) * 1, 1)

        self.weight = np.round(np.random.normal(0., 3., size=(self.dim_states, self.dim_actions)), 1)
        self.bias = np.round(np.random.normal(0., 3., size=(1, self.dim_actions)), 1)

        # self.weight = np.round(np.random.uniform(-3., 3., size=(self.dim_states, self.dim_actions)), 1)
        # self.bias = np.round(np.random.uniform(-3., 3., size=(1, self.dim_actions)), 1)
        
        # If using factorized policy, initialize L and U randomly
        if self.use_factorized_policy:
            self.L = np.round(np.random.normal(0., 1., size=(self.dim_states, self.factor_rank)), 2)
            self.U = np.round(np.random.normal(0., 1., size=(self.factor_rank, self.dim_actions)), 2)
            self.reconstruct_weight_from_factors()

    def reconstruct_weight_from_factors(self):
        """Reconstruct policy weight matrix from L and U: policy = L @ U."""
        if not self.use_factorized_policy:
            return
        
        # policy weight = L @ U (simple matrix multiplication)
        self.weight = np.round(self.L @ self.U, 2)
    
    def get_action(self, state):
        state = state.T
        # print(state.shape, self.weight.shape, self.bias.shape)
        # print(np.matmul(state, self.weight).shape, (np.matmul(state, self.weight) + self.bias).shape)
        # print((np.matmul(state, self.weight) + self.bias).shape)
        # print()
        # return np.matmul(state, self.weight + np.array([[2], [1]])) + self.bias + np.array([[-1]])
        return np.matmul(state, self.weight) + self.bias

    def __str__(self):
        if self.use_factorized_policy and self.L is not None:
            # Show factorized form: policy = L @ U
            output = "Factorized Policy (weight = L @ U):\n\n"
            output += "L matrix:\n"
            for row in self.L:
                output += ", ".join([str(i) for i in row])
                output += "\n"
            
            output += "\nU matrix:\n"
            for row in self.U:
                output += ", ".join([str(i) for i in row])
                output += "\n"
            
            output += "\nBias:\n"
            for b in self.bias:
                output += ", ".join([str(i) for i in b])
                output += "\n"
        else:
            # Show full weight matrix
            output = "Weights:\n"
            for w in self.weight:
                output += ", ".join([str(i) for i in w])
                output += "\n"

            output += "Bias:\n"
            for b in self.bias:
                output += ", ".join([str(i) for i in b])
                output += "\n"

        return output

    def update_policy(self, weight_and_bias_list=None, factor_components=None):
        """Update policy with either full parameters or L/U factor components."""
        if self.use_factorized_policy and factor_components is not None:
            # Update L, U, bias and reconstruct policy weight = L @ U
            self.L = factor_components['L']
            self.U = factor_components['U']
            self.bias = factor_components['bias']
            self.reconstruct_weight_from_factors()
        elif weight_and_bias_list is not None:
            weight_and_bias_list = np.array(weight_and_bias_list).reshape(self.dim_states + 1, self.dim_actions)
            self.weight = np.array(weight_and_bias_list[:-1])
            self.bias = np.expand_dims(np.array(weight_and_bias_list[-1]), axis=0)

    
    def get_parameters(self, return_factors=None):
        """Return parameters in full or factorized form."""
        if return_factors is None:
            return_factors = self.use_factorized_policy

        if return_factors and self.L is not None:
            # Return L, U factor components
            return {
                'L': self.L,
                'U': self.U,
                'bias': self.bias,
                'factor_rank': self.factor_rank
            }
        else:
            # Return full parameters
            parameters = np.concatenate((self.weight, self.bias), axis=0)
            return parameters
