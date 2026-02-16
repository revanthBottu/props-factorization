"""
Utility to create animated GIFs from policy heatmap images.

This script collects heatmap images generated during training episodes
and combines them into an animated GIF to visualize the progression
of the policy over time.
"""

import os
import re
from pathlib import Path
from PIL import Image
import argparse


def create_heatmap_gif(logdir, output_filename="policy_heatmaps.gif", 
                       heatmap_type="policy_lu_heatmap", duration=500, loop=0):
    """
    Create an animated GIF from policy heatmap images.
    
    Args:
        logdir (str): Directory containing the heatmap images
        output_filename (str): Name of the output GIF file
        heatmap_type (str): Type of heatmap to collect. Options:
                           - "policy_heatmap" : Single policy weight heatmap
                           - "policy_lu_heatmap" : Combined L, U, and reconstructed weight heatmap
        duration (int): Duration of each frame in milliseconds (default 500ms)
        loop (int): Number of times to loop (0 = infinite loop)
    
    Returns:
        str: Path to the generated GIF file, or None if failed
    """
    logdir_path = Path(logdir)
    
    if not logdir_path.exists():
        print(f"Error: Directory {logdir} does not exist")
        return None
    
    # Find all heatmap images - check both main dir and episode subdirs
    heatmap_files = []
    
    # Pattern to match heatmap files with episode numbers
    pattern = f"{heatmap_type}_ep(\\d+)\\.png"
    
    # Search in main directory
    for file in logdir_path.glob(f"{heatmap_type}_ep*.png"):
        match = re.search(pattern, file.name)
        if match:
            episode_num = int(match.group(1))
            heatmap_files.append((episode_num, file))
    
    # Search in episode subdirectories
    for episode_dir in logdir_path.glob("episode_*"):
        if episode_dir.is_dir():
            for file in episode_dir.glob(f"{heatmap_type}_ep*.png"):
                match = re.search(pattern, file.name)
                if match:
                    episode_num = int(match.group(1))
                    heatmap_files.append((episode_num, file))
    
    if not heatmap_files:
        print(f"Warning: No {heatmap_type} images found in {logdir} or its episode subdirectories")
        return None
    
    # Sort by episode number
    heatmap_files.sort(key=lambda x: x[0])
    
    print(f"Found {len(heatmap_files)} heatmap images")
    print(f"Episode range: {heatmap_files[0][0]} to {heatmap_files[-1][0]}")
    
    # Load images
    images = []
    for episode_num, filepath in heatmap_files:
        try:
            img = Image.open(filepath)
            images.append(img)
        except Exception as e:
            print(f"Warning: Could not load {filepath}: {e}")
    
    if not images:
        print("Error: No images could be loaded")
        return None
    
    # Save as GIF
    output_path = logdir_path / output_filename
    
    try:
        images[0].save(
            output_path,
            save_all=True,
            append_images=images[1:],
            duration=duration,
            loop=loop,
            optimize=False
        )
        print(f"\nSuccessfully created GIF: {output_path}")
        print(f"Total frames: {len(images)}")
        print(f"Frame duration: {duration}ms")
        return str(output_path)
    
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return None


def create_both_gifs(logdir, duration=500, loop=0):
    """
    Create GIFs for both policy heatmap and LU factorization heatmap.
    
    Args:
        logdir (str): Directory containing the heatmap images
        duration (int): Duration of each frame in milliseconds
        loop (int): Number of times to loop (0 = infinite loop)
    
    Returns:
        tuple: Paths to (policy_gif, lu_gif) or None for failed ones
    """
    print("=" * 60)
    print("Creating Policy Weight Heatmap GIF...")
    print("=" * 60)
    policy_gif = create_heatmap_gif(
        logdir, 
        output_filename="policy_heatmaps.gif",
        heatmap_type="policy_heatmap",
        duration=duration,
        loop=loop
    )
    
    print("\n" + "=" * 60)
    print("Creating LU Factorization Heatmap GIF...")
    print("=" * 60)
    lu_gif = create_heatmap_gif(
        logdir,
        output_filename="policy_lu_heatmaps.gif",
        heatmap_type="policy_lu_heatmap",
        duration=duration,
        loop=loop
    )
    
    return policy_gif, lu_gif


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIFs from policy heatmap images"
    )
    parser.add_argument(
        "logdir",
        type=str,
        help="Directory containing the heatmap images"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="both",
        choices=["policy", "lu", "both"],
        help="Type of heatmap to create GIF for (default: both)"
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=500,
        help="Duration of each frame in milliseconds (default: 500)"
    )
    parser.add_argument(
        "--loop",
        type=int,
        default=0,
        help="Number of times to loop (0 = infinite, default: 0)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Custom output filename (only for single type, not 'both')"
    )
    
    args = parser.parse_args()
    
    if args.type == "both":
        if args.output:
            print("Warning: --output is ignored when --type is 'both'")
        create_both_gifs(args.logdir, args.duration, args.loop)
    
    elif args.type == "policy":
        output_filename = args.output or "policy_heatmaps.gif"
        create_heatmap_gif(
            args.logdir,
            output_filename=output_filename,
            heatmap_type="policy_heatmap",
            duration=args.duration,
            loop=args.loop
        )
    
    elif args.type == "lu":
        output_filename = args.output or "policy_lu_heatmaps.gif"
        create_heatmap_gif(
            args.logdir,
            output_filename=output_filename,
            heatmap_type="policy_lu_heatmap",
            duration=args.duration,
            loop=args.loop
        )


if __name__ == "__main__":
    main()
