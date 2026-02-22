"""
Test script to verify factorized policy functionality and visualization.

This demonstrates how the LLM generates L and U matrices that are
multiplied together (policy = L @ U) to form the policy weight matrix, and shows the
visualization capabilities for tracking progress.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from agent.policy.linear_policy import LinearPolicy

def test_lu_factorization():
    """Test factorized policy (L @ U) in LinearPolicy."""
    
    # Create a policy with factorized representation
    dim_states = 4
    dim_actions = 2
    factor_rank = 2  # Inner dimension for L @ U
    
    policy = LinearPolicy(
        dim_states=dim_states,
        dim_actions=dim_actions,
        use_factorized_policy=True,
        factor_rank=factor_rank
    )
    
    # Initialize the policy
    policy.initialize_policy()
    
    print("=" * 60)
    print("Testing Factorized Policy (L @ U) in LinearPolicy")
    print("=" * 60)
    print(f"\nPolicy dimensions:")
    print(f"  States: {dim_states}")
    print(f"  Actions: {dim_actions}")
    print(f"  Factor Rank: {factor_rank}")
    
    print(f"\nL matrix shape: {policy.L.shape}")
    print(f"L matrix:\n{policy.L}")
    
    print(f"\nU matrix shape: {policy.U.shape}")
    print(f"U matrix:\n{policy.U}")
    
    print(f"\nPolicy weight matrix (L @ U) shape: {policy.weight.shape}")
    print(f"Weight matrix:\n{policy.weight}")
    
    # Verify that weight = L @ U
    manual_reconstruction = policy.L @ policy.U
    print(f"\nManual reconstruction check:")
    print(f"Max difference: {np.max(np.abs(manual_reconstruction - policy.weight))}")
    
    # Test updating with new L and U matrices
    print("\n" + "=" * 60)
    print("Testing update with new L and U matrices")
    print("=" * 60)
    
    new_L = np.array([[1.0, 0.5], [0.5, 1.0], [1.5, 0.3], [0.2, 1.2]])
    new_U = np.array([[2.0, 1.0], [1.0, 2.0]])
    new_bias = np.array([[0.5, -0.5]])
    
    print(f"\nNew L matrix:\n{new_L}")
    print(f"\nNew U matrix:\n{new_U}")
    
    factor_components = {
        'L': new_L,
        'U': new_U,
        'bias': new_bias
    }
    
    policy.update_policy(factor_components=factor_components)
    
    print(f"\nUpdated weight matrix (L @ U):\n{policy.weight}")
    print(f"Updated bias:\n{policy.bias}")
    
    # Verify reconstruction
    expected_weight = new_L @ new_U
    print(f"\nExpected weight (new_L @ new_U):\n{expected_weight}")
    print(f"Max difference: {np.max(np.abs(expected_weight - policy.weight))}")
    
    # Test get_parameters
    print("\n" + "=" * 60)
    print("Testing get_parameters with factorized mode")
    print("=" * 60)
    
    params = policy.get_parameters()
    print(f"\nReturned parameters type: {type(params)}")
    if isinstance(params, dict):
        print(f"Keys: {params.keys()}")
        print(f"L shape: {params['L'].shape}")
        print(f"U shape: {params['U'].shape}")
        print(f"bias shape: {params['bias'].shape}")
    
    print("\n" + "=" * 60)
    print("Test completed successfully!")
    print("=" * 60)

def test_visualization():
    """Test visualization of policy matrices and reward progress."""
    print("\n" + "=" * 60)
    print("Testing Visualization")
    print("=" * 60)
    
    # Create test directory
    test_dir = "test_visualizations"
    os.makedirs(test_dir, exist_ok=True)
    
    # Create a policy with factorized representation
    dim_states = 4
    dim_actions = 2
    factor_rank = 2
    
    policy = LinearPolicy(
        dim_states=dim_states,
        dim_actions=dim_actions,
        use_factorized_policy=True,
        factor_rank=factor_rank
    )
    policy.initialize_policy()
    
    # Test 1: Reward progress plot
    print("\n1. Creating reward progress plot...")
    episodes = list(range(10))
    rewards = [10 + i + np.random.randn() * 2 for i in range(10)]
    
    plt.figure(figsize=(10, 6))
    plt.plot(episodes, rewards, 'b-', marker='o', markersize=4, linewidth=2)
    plt.xlabel('Training Episode', fontsize=12)
    plt.ylabel('Reward', fontsize=12)
    plt.title('Training Reward Progress', fontsize=14)
    plt.grid(True, alpha=0.3)
    
    reward_plot_file = f"{test_dir}/test_reward_progress.png"
    plt.savefig(reward_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to {reward_plot_file}")
    
    # Test 2: Policy weight heatmap
    print("\n2. Creating policy weight heatmap...")
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        policy.weight,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        cbar_kws={'label': 'Weight Value'},
        linewidths=0.5,
        linecolor='gray'
    )
    plt.xlabel('Action Dimension', fontsize=12)
    plt.ylabel('State Dimension', fontsize=12)
    plt.title('Policy Weight Matrix', fontsize=14)
    
    weight_plot_file = f"{test_dir}/test_policy_heatmap.png"
    plt.savefig(weight_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to {weight_plot_file}")
    
    # Test 3: Factorized policy heatmap
    print("\n3. Creating factorized policy (L, U) heatmap...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # L matrix
    sns.heatmap(
        policy.L,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        ax=axes[0],
        cbar_kws={'label': 'Value'}
    )
    axes[0].set_title('L Matrix')
    axes[0].set_xlabel('Factor Rank Dimension')
    axes[0].set_ylabel('State Dimension')
    
    # U matrix
    sns.heatmap(
        policy.U,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        ax=axes[1],
        cbar_kws={'label': 'Value'}
    )
    axes[1].set_title('U Matrix')
    axes[1].set_xlabel('Action Dimension')
    axes[1].set_ylabel('Factor Rank Dimension')
    
    # Policy weight (L @ U)
    sns.heatmap(
        policy.weight,
        annot=True,
        fmt='.2f',
        cmap='viridis',
        ax=axes[2],
        cbar_kws={'label': 'Value'}
    )
    axes[2].set_title('Policy Weight (L @ U)')
    axes[2].set_xlabel('Action Dimension')
    axes[2].set_ylabel('State Dimension')
    
    plt.suptitle('Factorized Policy Matrices', fontsize=16)
    plt.tight_layout()
    
    factor_plot_file = f"{test_dir}/test_factorized_policy_heatmap.png"
    plt.savefig(factor_plot_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   Saved to {factor_plot_file}")
    
    print("\n" + "=" * 60)
    print("Visualization test completed!")
    print(f"All plots saved to {test_dir}/")
    print("=" * 60)

if __name__ == "__main__":
    test_lu_factorization()
    test_visualization()
