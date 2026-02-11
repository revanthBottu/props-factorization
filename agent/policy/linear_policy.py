import numpy as np
from agent.policy.base_policy import Policy
from scipy.linalg import svd


class LinearPolicy(Policy):
    def __init__(self, dim_states, dim_actions, svd_rank=None, use_svd=False):
        super().__init__(dim_states, dim_actions)

        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.use_svd = use_svd
        
        # Determine SVD rank (number of components to keep)
        if use_svd:
            if svd_rank is None:
                # Default: use min(dim_states, dim_actions) // 2
                self.svd_rank = max(1, min(dim_states, dim_actions) // 2)
            else:
                self.svd_rank = min(svd_rank, min(dim_states, dim_actions))
        else:
            self.svd_rank = None

        self.weight = np.random.rand(self.dim_states, self.dim_actions)
        self.bias = np.random.rand(1, self.dim_actions)
        
        # SVD components: weight ≈ U @ S @ Vt
        self.U = None  # shape: (dim_states, svd_rank)
        self.S = None  # shape: (svd_rank,)
        self.Vt = None  # shape: (svd_rank, dim_actions)

    def initialize_policy(self):
        # self.weight = np.round((np.random.rand(self.dim_states, self.dim_actions)) * 1, 1)
        # self.bias = np.round((np.random.rand(1, self.dim_actions) - 0.) * 1, 1)

        self.weight = np.round(np.random.normal(0., 3., size=(self.dim_states, self.dim_actions)), 1)
        self.bias = np.round(np.random.normal(0., 3., size=(1, self.dim_actions)), 1)

        # self.weight = np.round(np.random.uniform(-3., 3., size=(self.dim_states, self.dim_actions)), 1)
        # self.bias = np.round(np.random.uniform(-3., 3., size=(1, self.dim_actions)), 1)
        
        # If using SVD, factorize the initial weight matrix
        if self.use_svd:
            self.factorize_weight()

    def factorize_weight(self):
        """Perform truncated SVD on the weight matrix."""
        if not self.use_svd:
            return
        
        # Perform SVD: weight = U @ S @ Vt
        U_full, S_full, Vt_full = svd(self.weight, full_matrices=False)
        
        # Truncate to keep only top svd_rank components
        self.U = np.round(U_full[:, :self.svd_rank], 2)
        self.S = np.round(S_full[:self.svd_rank], 2)
        self.Vt = np.round(Vt_full[:self.svd_rank, :], 2)
    
    def reconstruct_weight(self):
        """Reconstruct weight matrix from SVD components."""
        if not self.use_svd:
            return
        
        # Reconstruct: weight = U @ diag(S) @ Vt
        self.weight = np.round(self.U @ np.diag(self.S) @ self.Vt, 1)
    
    def get_action(self, state):
        state = state.T
        # print(state.shape, self.weight.shape, self.bias.shape)
        # print(np.matmul(state, self.weight).shape, (np.matmul(state, self.weight) + self.bias).shape)
        # print((np.matmul(state, self.weight) + self.bias).shape)
        # print()
        # return np.matmul(state, self.weight + np.array([[2], [1]])) + self.bias + np.array([[-1]])
        return np.matmul(state, self.weight) + self.bias

    def __str__(self):
        if self.use_svd and self.U is not None:
            # Show factorized form for SVD
            output = "SVD Factorization (weight = U @ S @ Vt):\n\n"
            output += "U matrix:\n"
            for row in self.U:
                output += ", ".join([str(i) for i in row])
                output += "\n"
            
            output += "\nS vector (singular values):\n"
            output += ", ".join([str(i) for i in self.S])
            output += "\n"
            
            output += "\nVt matrix:\n"
            for row in self.Vt:
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

    def update_policy(self, weight_and_bias_list=None, svd_components=None):
        """Update policy with either full parameters or SVD components."""
        if self.use_svd and svd_components is not None:
            # Update with SVD components: (U, S, Vt, bias)
            self.U = svd_components['U']
            self.S = svd_components['S']
            self.Vt = svd_components['Vt']
            self.bias = svd_components['bias']
            # Reconstruct weight matrix
            self.reconstruct_weight()
        elif weight_and_bias_list is not None:
            weight_and_bias_list = np.array(weight_and_bias_list).reshape(self.dim_states + 1, self.dim_actions)
            self.weight = np.array(weight_and_bias_list[:-1])
            self.bias = np.expand_dims(np.array(weight_and_bias_list[-1]), axis=0)
            # If using SVD, factorize the new weight
            if self.use_svd:
                self.factorize_weight()
    
    def get_parameters(self, return_svd=None):
        """Return parameters in full or factorized form."""
        if return_svd is None:
            return_svd = self.use_svd
            
        if return_svd and self.U is not None:
            # Return SVD components flattened
            return {
                'U': self.U,
                'S': self.S,
                'Vt': self.Vt,
                'bias': self.bias,
                'svd_rank': self.svd_rank
            }
        else:
            # Return full parameters
            parameters = np.concatenate((self.weight, self.bias), axis=0)
            return parameters
