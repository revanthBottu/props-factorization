import numpy as np

from agent.policy.linear_policy import LinearPolicy


def _assert_reconstruction_close(actual, expected, atol=1e-6):
    max_diff = float(np.max(np.abs(actual - expected)))
    assert np.allclose(actual, expected, atol=atol), f"reconstruction mismatch (max diff={max_diff})"


def test_lu_reconstruction():
    policy = LinearPolicy(
        dim_states=4,
        dim_actions=2,
        use_factorized_policy=True,
        factor_rank=2,
        decomposition_type="lu",
    )
    policy.initialize_policy()

    expected = np.round(policy.L @ policy.U, 2)
    _assert_reconstruction_close(policy.weight, expected)


def test_qr_constraints_and_reconstruction():
    policy = LinearPolicy(
        dim_states=5,
        dim_actions=3,
        use_factorized_policy=True,
        factor_rank=3,
        decomposition_type="qr",
    )

    proposed_components = {
        "Q": np.random.randn(5, 3),
        "R": np.random.randn(3, 3),
        "bias": np.zeros((1, 3)),
    }
    policy.update_policy(factor_components=proposed_components)

    # Q should be column-orthonormal after normalization.
    _assert_reconstruction_close(policy.Q.T @ policy.Q, np.eye(policy.factor_rank), atol=2e-2)

    # R should be upper-triangular.
    _assert_reconstruction_close(policy.R, np.triu(policy.R), atol=1e-8)

    expected = np.round(policy.Q @ policy.R, 2)
    _assert_reconstruction_close(policy.weight, expected)


def test_svd_constraints_and_reconstruction():
    policy = LinearPolicy(
        dim_states=6,
        dim_actions=4,
        use_factorized_policy=True,
        factor_rank=3,
        decomposition_type="svd",
    )

    proposed_components = {
        "U": np.random.randn(6, 3),
        "S": np.array([-0.2, 1.1, -2.0]),
        "Vt": np.random.randn(3, 4),
        "bias": np.zeros((1, 4)),
    }
    policy.update_policy(factor_components=proposed_components)

    # U columns should be orthonormal.
    _assert_reconstruction_close(policy.U.T @ policy.U, np.eye(policy.factor_rank), atol=2e-2)

    # Vt rows should be orthonormal.
    _assert_reconstruction_close(policy.Vt @ policy.Vt.T, np.eye(policy.factor_rank), atol=2e-2)

    # S is normalized to non-negative descending values.
    assert np.all(policy.S >= -1e-8), f"S has negative values: {policy.S}"
    assert np.all(policy.S[:-1] >= policy.S[1:] - 1e-8), f"S is not sorted descending: {policy.S}"

    expected = np.round(policy.U @ np.diag(policy.S) @ policy.Vt, 2)
    _assert_reconstruction_close(policy.weight, expected)
