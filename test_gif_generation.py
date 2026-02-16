"""
Test script for heatmap GIF generation functionality.
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from utils.create_heatmap_gif import create_heatmap_gif, create_both_gifs


def test_gif_creation():
    """Test GIF creation on existing log directory."""
    print("=" * 70)
    print("Testing Heatmap GIF Generation")
    print("=" * 70)
    
    # Check if there's an existing log directory
    log_base = "logs"
    if not os.path.exists(log_base):
        print(f"\nWarning: {log_base} directory not found.")
        print("Run a training session first to generate heatmap images.")
        return
    
    # Find subdirectories in logs
    log_dirs = [d for d in os.listdir(log_base) 
                if os.path.isdir(os.path.join(log_base, d))]
    
    if not log_dirs:
        print(f"\nWarning: No subdirectories found in {log_base}.")
        print("Run a training session first to generate heatmap images.")
        return
    
    # Use the first available log directory
    test_dir = os.path.join(log_base, log_dirs[0])
    
    print(f"\nTesting with log directory: {test_dir}")
    print("-" * 70)
    
    # Test creating both GIFs
    print("\nTest 1: Creating both policy and LU heatmap GIFs...")
    policy_gif, lu_gif = create_both_gifs(
        logdir=test_dir,
        duration=500,
        loop=0
    )
    
    if policy_gif:
        print(f"✓ Policy heatmap GIF created successfully: {policy_gif}")
    else:
        print("✗ Policy heatmap GIF creation failed or no images found")
    
    if lu_gif:
        print(f"✓ LU factorization GIF created successfully: {lu_gif}")
    else:
        print("✗ LU factorization GIF creation failed or no images found")
    
    print("\n" + "=" * 70)
    print("Test Complete!")
    print("=" * 70)
    
    if policy_gif or lu_gif:
        print("\nYou can view the generated GIF(s) in the log directory.")
    else:
        print("\nNo GIFs were created. Make sure heatmap images exist in the log directory.")


if __name__ == "__main__":
    test_gif_creation()
