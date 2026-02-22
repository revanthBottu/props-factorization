"""
Create GIF for a specific range of episodes.
"""

import os
import re
from pathlib import Path
from PIL import Image
import argparse


def create_heatmap_gif_range(logdir, start_episode, end_episode, 
                              output_filename="policy_heatmaps_range.gif", 
                              heatmap_type="policy_factor_heatmap", 
                              duration=500, loop=0):
    """
    Create an animated GIF from policy heatmap images for a specific episode range.
    
    Args:
        logdir (str): Directory containing the heatmap images
        start_episode (int): Starting episode number (inclusive)
        end_episode (int): Ending episode number (inclusive)
        output_filename (str): Name of the output GIF file
        heatmap_type (str): Type of heatmap to collect
        duration (int): Duration of each frame in milliseconds
        loop (int): Number of times to loop (0 = infinite loop)
    
    Returns:
        str: Path to the generated GIF file, or None if failed
    """
    logdir_path = Path(logdir)
    
    if not logdir_path.exists():
        print(f"Error: Directory {logdir} does not exist")
        return None
    
    print(f"Collecting heatmaps for episodes {start_episode} to {end_episode}...")
    
    # Find all heatmap images within the episode range
    heatmap_files = []
    pattern = f"{heatmap_type}_ep(\\d+)\\.png"
    
    # Search in episode subdirectories
    for episode_num in range(start_episode, end_episode + 1):
        episode_dir = logdir_path / f"episode_{episode_num}"
        
        if episode_dir.is_dir():
            # Look for heatmap file
            for file in episode_dir.glob(f"{heatmap_type}_ep{episode_num}.png"):
                heatmap_files.append((episode_num, file))
                break  # Only take one file per episode
    
    # Also search in main directory for any direct files
    for file in logdir_path.glob(f"{heatmap_type}_ep*.png"):
        match = re.search(pattern, file.name)
        if match:
            episode_num = int(match.group(1))
            if start_episode <= episode_num <= end_episode:
                heatmap_files.append((episode_num, file))
    
    if not heatmap_files:
        print(f"Warning: No {heatmap_type} images found in episode range {start_episode}-{end_episode}")
        return None
    
    # Sort by episode number and remove duplicates
    heatmap_files.sort(key=lambda x: x[0])
    seen_episodes = set()
    unique_files = []
    for episode_num, filepath in heatmap_files:
        if episode_num not in seen_episodes:
            seen_episodes.add(episode_num)
            unique_files.append((episode_num, filepath))
    
    heatmap_files = unique_files
    
    print(f"Found {len(heatmap_files)} heatmap images")
    if heatmap_files:
        print(f"Episode range: {heatmap_files[0][0]} to {heatmap_files[-1][0]}")
    
    # Load images
    images = []
    for episode_num, filepath in heatmap_files:
        try:
            img = Image.open(filepath)
            # Add episode number to the image
            images.append(img)
            if (episode_num - start_episode) % 10 == 0:
                print(f"  Loaded episode {episode_num}")
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
        print(f"\n✓ Successfully created GIF: {output_path}")
        print(f"  Total frames: {len(images)}")
        print(f"  Frame duration: {duration}ms")
        print(f"  Total duration: {len(images) * duration / 1000:.1f} seconds")
        return str(output_path)
    
    except Exception as e:
        print(f"Error creating GIF: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(
        description="Create animated GIF from policy heatmap images for a specific episode range"
    )
    parser.add_argument(
        "logdir",
        type=str,
        help="Directory containing the heatmap images"
    )
    parser.add_argument(
        "--start",
        type=int,
        required=True,
        help="Starting episode number (inclusive)"
    )
    parser.add_argument(
        "--end",
        type=int,
        required=True,
        help="Ending episode number (inclusive)"
    )
    parser.add_argument(
        "--type",
        type=str,
        default="factor",
        choices=["policy", "factor"],
        help="Type of heatmap: 'policy' for weight matrix, 'factor' for factorized policy L/U matrices (default: factor)"
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
        help="Custom output filename (default: auto-generated based on type and range)"
    )
    
    args = parser.parse_args()
    
    # Determine heatmap type
    heatmap_type = "policy_factor_heatmap" if args.type == "factor" else "policy_heatmap"
    
    # Generate default output filename if not provided
    if args.output is None:
        args.output = f"{args.type}_heatmaps_ep{args.start}-{args.end}.gif"
    
    create_heatmap_gif_range(
        args.logdir,
        args.start,
        args.end,
        output_filename=args.output,
        heatmap_type=heatmap_type,
        duration=args.duration,
        loop=args.loop
    )


if __name__ == "__main__":
    main()
