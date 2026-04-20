import numpy as np
from agent.policy.base_policy import Policy


class LinearPolicy(Policy):
    VALID_DECOMPOSITIONS = ("lu", "qr", "svd")
    VALID_MATRIX_INIT_MODES = ("near_zero", "random", "zero")

    def __init__(
        self,
        dim_states,
        dim_actions,
        use_factorized_policy=False,
        factor_rank=None,
        decomposition_type="lu",
        matrix_init_mode="near_zero",
        near_zero_init_scale=0.15,
        near_zero_init_min_abs=0.02,
        near_zero_init_decimals=2,
    ):
        super().__init__(dim_states, dim_actions)

        self.dim_states = dim_states
        self.dim_actions = dim_actions
        self.use_factorized_policy = use_factorized_policy
        self.decomposition_type = str(decomposition_type or "lu").lower()

        if self.decomposition_type not in self.VALID_DECOMPOSITIONS:
            raise ValueError(
                f"decomposition_type must be one of {self.VALID_DECOMPOSITIONS}, "
                f"got: {decomposition_type!r}"
            )

        # Determine factor rank (inner dimension for L @ U = policy)
        max_rank = min(dim_states, dim_actions)
        if use_factorized_policy:
            if factor_rank is None:
                # Default: use min(dim_states, dim_actions) // 2 for reduced representation
                self.factor_rank = max(1, max_rank // 2)
            else:
                self.factor_rank = int(factor_rank)

            if self.factor_rank <= 0:
                raise ValueError(f"factor_rank must be positive, got: {self.factor_rank}")
            if self.factor_rank > max_rank:
                raise ValueError(
                    f"factor_rank ({self.factor_rank}) must be <= min(dim_states, dim_actions) ({max_rank})"
                )
        else:
            self.factor_rank = None

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

        # Factor components (decomposition-specific)
        self.factor_names = self._factor_names_for_decomposition(self.decomposition_type) if self.use_factorized_policy else []
        self.factors = {}
        self._sync_named_factor_attrs()

        self.initialize_policy()

    def _normalize_init_mode(self, matrix_init_mode):
        mode = str(matrix_init_mode).strip().lower()
        if mode == "zero":
            return "near_zero"
        if mode not in {"near_zero", "random"}:
            raise ValueError(
                "matrix_init_mode must be 'near_zero', 'random', or 'zero' (alias), "
                f"got: {matrix_init_mode!r}"
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

    def _sample_random_values(self, shape, std=1.0, decimals=2):
        raw = np.random.normal(0.0, std, size=shape)
        rounded = np.round(raw, decimals)
        return self._enforce_nonzero_values(rounded, source_values=raw)

    def _sample_weight_bias_values(self, shape):
        if self.matrix_init_mode == "near_zero":
            return self._sample_near_zero_nonzero(shape)
        return self._sample_random_values(shape, std=3.0, decimals=1)

    def _sample_factor_values(self, shape):
        if self.matrix_init_mode == "near_zero":
            return self._sample_near_zero_nonzero(shape)
        return self._sample_random_values(shape, std=1.0, decimals=2)

    def _factor_names_for_decomposition(self, decomposition_type):
        if decomposition_type == "lu":
            return ["L", "U"]
        if decomposition_type == "qr":
            return ["Q", "R"]
        # svd
        return ["U", "S", "Vt"]

    def get_factor_names(self):
        return list(self.factor_names)

    def get_factor_equation(self):
        equation_map = {
            "lu": "A = L @ U",
            "qr": "A = Q @ R",
            "svd": "A = U @ diag(S) @ Vt",
        }
        return equation_map.get(self.decomposition_type, "A = factors")

    def get_factor_shapes(self):
        if not self.use_factorized_policy:
            return {}

        m, n, k = self.dim_states, self.dim_actions, self.factor_rank
        if self.decomposition_type == "lu":
            return {"L": (m, k), "U": (k, n)}
        if self.decomposition_type == "qr":
            return {"Q": (m, k), "R": (k, n)}
        return {"U": (m, k), "S": (k,), "Vt": (k, n)}

    def canonicalize_factor_name(self, name):
        if name is None:
            return None

        normalized = str(name).strip().lower()
        normalized = normalized.replace("`", "").replace("*", "")
        normalized = normalized.replace(" ", "").replace("_", "")
        normalized = normalized.replace("^", "")
        normalized = normalized.replace("matrix", "")
        normalized = normalized.replace("vector", "")
        normalized = normalized.replace("factor", "")
        normalized = normalized.replace(":", "")
        normalized = normalized.strip()

        mapping = {
            "lu": {
                "l": "L",
                "u": "U",
            },
            "qr": {
                "q": "Q",
                "r": "R",
            },
            "svd": {
                "a": "U",
                "rawa": "U",
                "rawu": "U",
                "left": "U",
                "u": "U",
                "s": "S",
                "sigma": "S",
                "singularvalues": "S",
                "singularvalue": "S",
                "b": "Vt",
                "rawb": "Vt",
                "rawv": "Vt",
                "right": "Vt",
                "vt": "Vt",
                "vtranspose": "Vt",
                "vtransposed": "Vt",
                "v": "Vt",
            },
        }
        return mapping.get(self.decomposition_type, {}).get(normalized)

    def _sync_named_factor_attrs(self):
        # Keep legacy attribute names for compatibility with existing code paths.
        for attr_name in ("L", "U", "Q", "R", "S", "Vt"):
            setattr(self, attr_name, None)

        for factor_name, factor_value in self.factors.items():
            setattr(self, factor_name, factor_value)

    def _initialize_factor_components(self):
        m, n, k = self.dim_states, self.dim_actions, self.factor_rank

        if self.decomposition_type == "lu":
            self.factors = {
                "L": self._sample_factor_values((m, k)),
                "U": self._sample_factor_values((k, n)),
            }
        elif self.decomposition_type == "qr":
            q_raw = self._sample_factor_values((m, k))
            q_factor, _ = np.linalg.qr(q_raw, mode="reduced")
            r_factor = np.triu(self._sample_factor_values((k, n)))
            self.factors = {
                "Q": self._enforce_nonzero_values(np.round(q_factor, 2), source_values=q_factor),
                "R": self._enforce_nonzero_values(r_factor),
            }
        else:
            # SVD factors: A = U @ diag(S) @ Vt
            u_raw = self._sample_factor_values((m, k))
            u_factor, _ = np.linalg.qr(u_raw, mode="reduced")

            v_raw = self._sample_factor_values((n, k))
            v_factor, _ = np.linalg.qr(v_raw, mode="reduced")

            singular_values = np.sort(np.abs(self._sample_factor_values((k,))))[::-1]
            self.factors = {
                "U": self._enforce_nonzero_values(np.round(u_factor, 2), source_values=u_factor),
                "S": self._enforce_nonzero_values(singular_values),
                "Vt": self._enforce_nonzero_values(np.round(v_factor.T, 2), source_values=v_factor.T),
            }

        self._sync_named_factor_attrs()

    def normalize_factor_components(self, factor_components):
        if not self.use_factorized_policy:
            return {}

        expected_shapes = self.get_factor_shapes()
        normalized = {}

        for factor_name, expected_shape in expected_shapes.items():
            if factor_name not in factor_components:
                raise ValueError(f"Missing factor '{factor_name}' in factor_components")

            factor_array = np.asarray(factor_components[factor_name], dtype=float)
            expected_ndim = 1 if len(expected_shape) == 1 else 2
            if factor_array.ndim != expected_ndim:
                raise ValueError(
                    f"Factor '{factor_name}' must have {expected_ndim} dimensions, got {factor_array.ndim}"
                )

            if expected_ndim == 1:
                factor_array = factor_array.reshape(-1)

            if factor_array.shape != expected_shape:
                # Accept transposed Vt if the model outputs V instead.
                if factor_name == "Vt" and factor_array.ndim == 2 and factor_array.T.shape == expected_shape:
                    factor_array = factor_array.T
                else:
                    raise ValueError(
                        f"Factor '{factor_name}' has shape {factor_array.shape}, expected {expected_shape}"
                    )

            normalized[factor_name] = factor_array

        if self.decomposition_type == "qr":
            q_factor, _ = np.linalg.qr(normalized["Q"], mode="reduced")
            r_factor = np.triu(normalized["R"])
            normalized["Q"] = q_factor
            normalized["R"] = r_factor
        elif self.decomposition_type == "svd":
            # Accept raw SVD precursor factors from the LLM and convert them
            # into valid decomposition factors used by the policy.
            u_factor, _ = np.linalg.qr(normalized["U"], mode="reduced")
            v_factor, _ = np.linalg.qr(normalized["Vt"].T, mode="reduced")

            singular_values = np.abs(normalized["S"])
            order = np.argsort(-singular_values)

            normalized["S"] = singular_values[order]
            normalized["U"] = u_factor[:, order]
            normalized["Vt"] = v_factor.T[order, :]

        for factor_name, factor_value in normalized.items():
            normalized[factor_name] = np.round(factor_value, 2)

        return normalized

    def initialize_policy(self):
        # self.weight = np.round((np.random.rand(self.dim_states, self.dim_actions)) * 1, 1)
        # self.bias = np.round((np.random.rand(1, self.dim_actions) - 0.) * 1, 1)

        self.weight = self._sample_weight_bias_values((self.dim_states, self.dim_actions))
        self.bias = self._sample_weight_bias_values((1, self.dim_actions))

        # self.weight = np.round(np.random.uniform(-3., 3., size=(self.dim_states, self.dim_actions)), 1)
        # self.bias = np.round(np.random.uniform(-3., 3., size=(1, self.dim_actions)), 1)

        if self.use_factorized_policy:
            self._initialize_factor_components()
            self.reconstruct_weight_from_factors()

    def reconstruct_weight_from_factors(self):
        """Reconstruct policy weight matrix from decomposition factors."""
        if not self.use_factorized_policy:
            return

        if not self.factors:
            return

        self.factors = self.normalize_factor_components(self.factors)
        self._sync_named_factor_attrs()

        if self.decomposition_type == "lu":
            self.weight = np.round(self.factors["L"] @ self.factors["U"], 2)
        elif self.decomposition_type == "qr":
            self.weight = np.round(self.factors["Q"] @ self.factors["R"], 2)
        else:
            self.weight = np.round(
                self.factors["U"] @ np.diag(self.factors["S"]) @ self.factors["Vt"],
                2,
            )
    
    def get_action(self, state):
        state = state.T
        # print(state.shape, self.weight.shape, self.bias.shape)
        # print(np.matmul(state, self.weight).shape, (np.matmul(state, self.weight) + self.bias).shape)
        # print((np.matmul(state, self.weight) + self.bias).shape)
        # print()
        # return np.matmul(state, self.weight + np.array([[2], [1]])) + self.bias + np.array([[-1]])
        return np.matmul(state, self.weight) + self.bias

    def __str__(self):
        if self.use_factorized_policy and self.factors:
            output = (
                f"Factorized Policy ({self.decomposition_type.upper()} decomposition) "
                f"[{self.get_factor_equation()}]:\n\n"
            )

            for factor_name in self.factor_names:
                factor_value = self.factors[factor_name]
                output += f"{factor_name} factor:\n"
                if factor_value.ndim == 1:
                    output += ", ".join([str(i) for i in factor_value]) + "\n"
                else:
                    for row in factor_value:
                        output += ", ".join([str(i) for i in row])
                        output += "\n"
                output += "\n"

            output += "Bias:\n"
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
        """Update policy with either full parameters or decomposition factors."""
        if self.use_factorized_policy and factor_components is not None:
            self.factors = self.normalize_factor_components(factor_components)
            self._sync_named_factor_attrs()
            if "bias" in factor_components:
                self.bias = np.array(factor_components["bias"], dtype=float).reshape(1, self.dim_actions)
            self.reconstruct_weight_from_factors()
        elif weight_and_bias_list is not None:
            weight_and_bias_list = np.array(weight_and_bias_list).reshape(self.dim_states + 1, self.dim_actions)
            self.weight = np.array(weight_and_bias_list[:-1])
            self.bias = np.expand_dims(np.array(weight_and_bias_list[-1]), axis=0)

    
    def get_parameters(self, return_factors=None):
        """Return parameters in full or factorized form."""
        if return_factors is None:
            return_factors = self.use_factorized_policy

        if return_factors and self.use_factorized_policy and not self.factors:
            self._initialize_factor_components()
            self.reconstruct_weight_from_factors()

        if return_factors and self.factors:
            factor_payload = {
                factor_name: np.array(self.factors[factor_name], copy=True)
                for factor_name in self.factor_names
            }
            factor_payload.update(
                {
                    "bias": np.array(self.bias, copy=True),
                    "factor_rank": self.factor_rank,
                    "decomposition_type": self.decomposition_type,
                }
            )
            return factor_payload
        else:
            # Return full parameters
            parameters = np.concatenate((self.weight, self.bias), axis=0)
            return parameters
