#!/usr/bin/env python
"""
Verify and visualize processed handwriting data.

This script checks the quality of processed data and visualizes
samples to ensure everything looks correct before training.

Usage:
    python verify_data.py [--data_dir DATA_DIR] [--num_samples N]
"""

from __future__ import print_function
import os
import sys
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt

import drawing


def load_processed_data(data_dir):
    """Load processed numpy arrays."""
    required_files = ['x.npy', 'x_len.npy', 'c.npy', 'c_len.npy']

    print(f"Loading data from {data_dir}...")

    data = {}
    for filename in required_files:
        filepath = os.path.join(data_dir, filename)
        if not os.path.exists(filepath):
            print(f"Error: Required file not found: {filepath}")
            print("\nMake sure to run convert_custom_data.py first!")
            sys.exit(1)
        data[filename.replace('.npy', '')] = np.load(filepath)

    print(f"✓ Successfully loaded all data files")
    return data


def print_statistics(data):
    """Print dataset statistics."""
    num_samples = len(data['x'])

    print(f"\n{'='*60}")
    print("Dataset Statistics")
    print(f"{'='*60}")

    print(f"\nBasic Info:")
    print(f"  Total samples: {num_samples}")
    print(f"  Data shapes:")
    print(f"    x (strokes):  {data['x'].shape}")
    print(f"    x_len:        {data['x_len'].shape}")
    print(f"    c (text):     {data['c'].shape}")
    print(f"    c_len:        {data['c_len'].shape}")

    print(f"\nStroke Statistics:")
    print(f"  Average length: {np.mean(data['x_len']):.1f} points")
    print(f"  Min length:     {np.min(data['x_len'])} points")
    print(f"  Max length:     {np.max(data['x_len'])} points")
    print(f"  Std deviation:  {np.std(data['x_len']):.1f} points")

    print(f"\nText Statistics:")
    print(f"  Average length: {np.mean(data['c_len']):.1f} characters")
    print(f"  Min length:     {np.min(data['c_len'])} characters")
    print(f"  Max length:     {np.max(data['c_len'])} characters")
    print(f"  Std deviation:  {np.std(data['c_len']):.1f} characters")

    # Check for potential issues
    print(f"\nData Quality Checks:")

    # Check for very short strokes
    very_short = np.sum(data['x_len'] < 10)
    if very_short > 0:
        print(f"  ⚠ Warning: {very_short} samples with < 10 points (may be too short)")
    else:
        print(f"  ✓ No samples with insufficient stroke points")

    # Check for very short text
    very_short_text = np.sum(data['c_len'] < 3)
    if very_short_text > 0:
        print(f"  ⚠ Warning: {very_short_text} samples with < 3 characters")
    else:
        print(f"  ✓ No samples with very short text")

    # Check for NaN or Inf values
    has_nan = np.any(np.isnan(data['x']))
    has_inf = np.any(np.isinf(data['x']))
    if has_nan or has_inf:
        print(f"  ✗ Error: Data contains NaN or Inf values!")
    else:
        print(f"  ✓ No NaN or Inf values detected")

    # Check stroke magnitudes
    stroke_norms = np.linalg.norm(data['x'][:, :, :2], axis=2)
    max_norm = np.max(stroke_norms)
    if max_norm > 100:
        print(f"  ⚠ Warning: Very large stroke offsets detected (max: {max_norm:.1f})")
    else:
        print(f"  ✓ Stroke magnitudes look reasonable (max: {max_norm:.1f})")

    # Dataset size recommendations
    print(f"\nDataset Size Assessment:")
    if num_samples < 50:
        print(f"  ⚠ Small dataset ({num_samples} samples)")
        print(f"     Recommendation: Collect more samples or use fine-tuning")
    elif num_samples < 200:
        print(f"  ✓ Moderate dataset ({num_samples} samples)")
        print(f"     Recommendation: Fine-tuning recommended")
    else:
        print(f"  ✓ Large dataset ({num_samples} samples)")
        print(f"     Recommendation: Can train from scratch or fine-tune")


def decode_text(encoded_text, text_len):
    """Decode encoded text back to string."""
    chars = []
    for i in range(text_len):
        char_code = encoded_text[i]
        if char_code < len(drawing.alphabet):
            chars.append(drawing.alphabet[char_code])
    return ''.join(chars)


def visualize_samples(data, num_samples=5, output_dir='verification_output'):
    """Visualize sample strokes."""
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Visualizing {num_samples} random samples")
    print(f"{'='*60}\n")

    # Select random samples
    num_total = len(data['x'])
    num_samples = min(num_samples, num_total)
    sample_indices = np.random.choice(num_total, num_samples, replace=False)

    for idx, sample_idx in enumerate(sample_indices):
        # Get stroke and text data
        stroke = data['x'][sample_idx]
        stroke_len = data['x_len'][sample_idx]
        text_encoded = data['c'][sample_idx]
        text_len = data['c_len'][sample_idx]

        # Decode text
        text = decode_text(text_encoded, text_len)

        # Visualize
        print(f"Sample {idx + 1}/{num_samples}:")
        print(f"  Text: \"{text}\"")
        print(f"  Stroke length: {stroke_len} points")

        # Create visualization
        fig, ax = plt.subplots(figsize=(12, 3))

        # Convert offsets to coordinates for plotting
        coords = drawing.offsets_to_coords(stroke[:stroke_len])

        # Plot each stroke
        current_stroke = []
        for x, y, eos in coords:
            current_stroke.append((x, y))
            if eos == 1:
                if len(current_stroke) > 1:
                    xs, ys = zip(*current_stroke)
                    ax.plot(xs, ys, 'k-', linewidth=2)
                current_stroke = []

        # Plot remaining stroke if any
        if len(current_stroke) > 1:
            xs, ys = zip(*current_stroke)
            ax.plot(xs, ys, 'k-', linewidth=2)

        ax.set_aspect('equal')
        ax.set_title(f'Sample {idx + 1}: "{text}"', fontsize=12)
        ax.grid(True, alpha=0.3)

        # Save figure
        output_path = os.path.join(output_dir, f'sample_{idx + 1:02d}.png')
        plt.savefig(output_path, dpi=100, bbox_inches='tight')
        plt.close()

        print(f"  ✓ Saved visualization to {output_path}")

    print(f"\n✓ All visualizations saved to {output_dir}/")


def check_text_coverage(data):
    """Check character coverage in the dataset."""
    print(f"\n{'='*60}")
    print("Character Coverage Analysis")
    print(f"{'='*60}\n")

    # Collect all unique characters
    all_chars = set()
    for i in range(len(data['c'])):
        text_len = data['c_len'][i]
        for j in range(text_len):
            char_idx = data['c'][i][j]
            if char_idx < len(drawing.alphabet):
                all_chars.add(drawing.alphabet[char_idx])

    # Remove null character
    all_chars.discard('\x00')

    print(f"Unique characters in dataset: {len(all_chars)}")
    print(f"Characters: {sorted(all_chars)}")

    # Check for common missing characters
    common_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
    missing_common = common_chars - all_chars

    if missing_common:
        print(f"\n⚠ Missing common characters: {sorted(missing_common)}")
        print(f"  Consider collecting samples with these characters for better coverage")
    else:
        print(f"\n✓ All common letters covered!")


def main():
    parser = argparse.ArgumentParser(
        description="Verify and visualize processed handwriting data"
    )
    parser.add_argument('--data_dir', type=str, default='data/processed',
                       help='Directory containing processed numpy arrays')
    parser.add_argument('--num_samples', type=int, default=5,
                       help='Number of samples to visualize')
    parser.add_argument('--output_dir', type=str, default='verification_output',
                       help='Directory to save visualizations')
    parser.add_argument('--no_visualize', action='store_true',
                       help='Skip visualization (only print statistics)')

    args = parser.parse_args()

    print(f"\n{'='*60}")
    print("Handwriting Data Verification")
    print(f"{'='*60}\n")

    # Load data
    data = load_processed_data(args.data_dir)

    # Print statistics
    print_statistics(data)

    # Check character coverage
    check_text_coverage(data)

    # Visualize samples
    if not args.no_visualize:
        visualize_samples(data, args.num_samples, args.output_dir)

    print(f"\n{'='*60}")
    print("Verification Complete!")
    print(f"{'='*60}\n")

    # Final recommendations
    num_samples = len(data['x'])
    print("Next Steps:")
    if num_samples < 50:
        print("  1. Consider collecting more samples (current: {})".format(num_samples))
        print("  2. Or use fine-tuning: python rnn.py --warm_start_init_step=17900")
    elif num_samples < 200:
        print("  1. Ready for fine-tuning: python rnn.py --warm_start_init_step=17900")
        print("  2. Or collect more samples for training from scratch")
    else:
        print("  1. Ready for training: python rnn.py")
        print("  2. Or fine-tune for faster results: python rnn.py --warm_start_init_step=17900")
    print()


if __name__ == '__main__':
    main()
