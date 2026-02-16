import numpy as np
from agent.policy.base_policy import Policy
from scipy.linalg import svd


class LinearPolicy(Policy):
    def __init__(self, dim_states, dim_actions, svd_rank=None, use_svd=False, use_lu_factorization=False, lu_rank=None):
        super().__init__(dim_states, dim_actions)

        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.use_svd = use_svd
        self.use_lu_factorization = use_lu_factorization
        
        # Determine SVD rank (number of components to keep)
        if use_svd:
            if svd_rank is None:
                # Default: use min(dim_states, dim_actions) // 2
                self.svd_rank = max(1, min(dim_states, dim_actions) // 2)
            else:
                self.svd_rank = min(svd_rank, min(dim_states, dim_actions))
        else:
            self.svd_rank = None

        # Determine LU rank (inner dimension for L @ U factorization)
        if use_lu_factorization:
            if lu_rank is None:
                # Default: use min(dim_states, dim_actions) // 2 for reduced representation
                self.lu_rank = max(1, min(dim_states, dim_actions) // 2)
            else:
                self.lu_rank = min(lu_rank, min(dim_states, dim_actions))
        else:
            self.lu_rank = None

        self.weight = np.random.rand(self.dim_states, self.dim_actions)
        self.bias = np.random.rand(1, self.dim_actions)
        
        # SVD components: weight ≈ U @ S @ Vt
        self.U = None  # shape: (dim_states, svd_rank)
        self.S = None  # shape: (svd_rank,)
        self.Vt = None  # shape: (svd_rank, dim_actions)
        
        # LU factorization components: weight = L @ U
        self.L = None  # shape: (dim_states, lu_rank)
        self.U_matrix = None  # shape: (lu_rank, dim_actions)

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
        
        # If using LU factorization, initialize L and U randomly
        if self.use_lu_factorization:
            self.L = np.round(np.random.normal(0., 1., size=(self.dim_states, self.lu_rank)), 1)
            self.U_matrix = np.round(np.random.normal(0., 1., size=(self.lu_rank, self.dim_actions)), 1)
            self.reconstruct_weight_from_lu()

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
    
    def reconstruct_weight_from_lu(self):
        """Reconstruct weight matrix from L and U matrices."""
        if not self.use_lu_factorization:
            return
        
        # Reconstruct: weight = L @ U_matrix
        self.weight = np.round(self.L @ self.U_matrix, 1)
    
    def get_action(self, state):
        state = state.T
        # print(state.shape, self.weight.shape, self.bias.shape)
        # print(np.matmul(state, self.weight).shape, (np.matmul(state, self.weight) + self.bias).shape)
        # print((np.matmul(state, self.weight) + self.bias).shape)
        # print()
        # return np.matmul(state, self.weight + np.array([[2], [1]])) + self.bias + np.array([[-1]])
        return np.matmul(state, self.weight) + self.bias

    def __str__(self):
        if self.use_lu_factorization and self.L is not None:
            # Show LU factorization form
            output = "LU Factorization (weight = L @ U):\n\n"
            output += "L matrix:\n"
            for row in self.L:
                output += ", ".join([str(i) for i in row])
                output += "\n"
            
            output += "\nU matrix:\n"
            for row in self.U_matrix:
                output += ", ".join([str(i) for i in row])
                output += "\n"
            
            output += "\nBias:\n"
            for b in self.bias:
                output += ", ".join([str(i) for i in b])
                output += "\n"
        elif self.use_svd and self.U is not None:
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

    def update_policy(self, weight_and_bias_list=None, svd_components=None, lu_components=None):
        """Update policy with either full parameters, SVD components, or LU components."""
        if self.use_lu_factorization and lu_components is not None:
            # Update with LU components: (L, U, bias)
            self.L = lu_components['L']
            self.U_matrix = lu_components['U']
            self.bias = lu_components['bias']
            # Reconstruct weight matrix from L @ U
            self.reconstruct_weight_from_lu()
        elif self.use_svd and svd_components is not None:
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
    
    def get_parameters(self, return_svd=None, return_lu=None):
        """Return parameters in full or factorized form."""
        if return_lu is None:
            return_lu = self.use_lu_factorization
        if return_svd is None:
            return_svd = self.use_svd
        
        if return_lu and self.L is not None:
            # Return LU components
            return {
                'L': self.L,
                'U': self.U_matrix,
                'bias': self.bias,
                'lu_rank': self.lu_rank
            }
        elif return_svd and self.U is not None:
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
