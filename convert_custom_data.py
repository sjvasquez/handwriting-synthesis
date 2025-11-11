#!/usr/bin/env python
"""
Convert collected handwriting samples to training format.

This script takes the raw stroke data collected by collect_handwriting.py
and converts it to the numpy format required by rnn.py for training.

Usage:
    python convert_custom_data.py [--input INPUT_DIR] [--output OUTPUT_DIR]
"""

from __future__ import print_function
import os
import sys
import argparse
import json

import numpy as np
import drawing


def load_collected_samples(input_dir):
    """
    Load samples collected by collect_handwriting.py.

    Returns:
        List of (stroke_data, transcription) tuples
    """
    metadata_path = os.path.join(input_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        print(f"Error: No metadata.json found in {input_dir}")
        print("Make sure to run collect_handwriting.py first!")
        sys.exit(1)

    with open(metadata_path, 'r') as f:
        metadata = json.load(f)

    samples = []
    strokes_dir = os.path.join(input_dir, "strokes")

    print(f"Loading {len(metadata['samples'])} samples from {input_dir}...")

    for sample_info in metadata['samples']:
        stroke_file = os.path.join(strokes_dir, sample_info['filename'])
        if not os.path.exists(stroke_file):
            print(f"Warning: Missing stroke file {stroke_file}, skipping...")
            continue

        stroke_data = np.load(stroke_file)
        transcription = sample_info['transcription']

        samples.append((stroke_data, transcription))

    print(f"Successfully loaded {len(samples)} samples")
    return samples


def preprocess_stroke(coords):
    """
    Apply preprocessing to stroke coordinates.

    Steps:
    1. Convert absolute coordinates to offsets
    2. Align (correct for slant/offset)
    3. Denoise (smooth the strokes)
    4. Normalize (scale to unit norm)
    5. Clip to max length

    Args:
        coords: numpy array of shape [num_points, 3] with [x, y, end_of_stroke]

    Returns:
        Preprocessed offsets array
    """
    # Convert coordinates to offsets
    offsets = drawing.coords_to_offsets(coords)

    # Note: We skip align() and denoise() here because the collected data
    # is already in screen coordinates and relatively clean.
    # If your data is noisy, uncomment these lines:
    #
    # coords_aligned = drawing.align(coords)
    # coords_denoised = drawing.denoise(coords_aligned)
    # offsets = drawing.coords_to_offsets(coords_denoised)

    # Clip to maximum stroke length
    offsets = offsets[:drawing.MAX_STROKE_LEN]

    # Normalize
    offsets = drawing.normalize(offsets)

    return offsets


def convert_samples(samples, output_dir):
    """
    Convert samples to training format and save as numpy arrays.

    Args:
        samples: List of (stroke_data, transcription) tuples
        output_dir: Directory to save processed numpy arrays
    """
    if len(samples) == 0:
        print("Error: No samples to convert!")
        sys.exit(1)

    print(f"\nConverting {len(samples)} samples to training format...")

    # Initialize arrays
    x = np.zeros([len(samples), drawing.MAX_STROKE_LEN, 3], dtype=np.float32)
    x_len = np.zeros([len(samples)], dtype=np.int16)
    c = np.zeros([len(samples), drawing.MAX_CHAR_LEN], dtype=np.int8)
    c_len = np.zeros([len(samples)], dtype=np.int8)
    valid_mask = np.zeros([len(samples)], dtype=np.bool)

    # Process each sample
    for i, (stroke_data, transcription) in enumerate(samples):
        if (i + 1) % 10 == 0:
            print(f"Processing sample {i + 1}/{len(samples)}...")

        # Preprocess strokes
        try:
            offsets = preprocess_stroke(stroke_data)

            # Check if stroke is valid (not too large movements)
            # This filters out erratic/invalid strokes
            valid_mask[i] = ~np.any(np.linalg.norm(offsets[:, :2], axis=1) > 60)

            # Store stroke data
            x[i, :len(offsets), :] = offsets
            x_len[i] = len(offsets)

            # Encode transcription
            encoded_text = drawing.encode_ascii(transcription)[:drawing.MAX_CHAR_LEN]
            c[i, :len(encoded_text)] = encoded_text
            c_len[i] = len(encoded_text)

        except Exception as e:
            print(f"Warning: Error processing sample {i}: {e}")
            valid_mask[i] = False

    # Filter to valid samples only
    num_valid = np.sum(valid_mask)
    num_invalid = len(samples) - num_valid

    print(f"\nProcessing complete:")
    print(f"  Valid samples: {num_valid}")
    print(f"  Invalid samples: {num_invalid}")

    if num_valid == 0:
        print("Error: No valid samples after preprocessing!")
        sys.exit(1)

    # Save to output directory
    os.makedirs(output_dir, exist_ok=True)

    print(f"\nSaving processed data to {output_dir}...")

    np.save(os.path.join(output_dir, 'x.npy'), x[valid_mask])
    np.save(os.path.join(output_dir, 'x_len.npy'), x_len[valid_mask])
    np.save(os.path.join(output_dir, 'c.npy'), c[valid_mask])
    np.save(os.path.join(output_dir, 'c_len.npy'), c_len[valid_mask])

    # Create dummy writer IDs (all same writer since it's your handwriting)
    w_id = np.zeros([num_valid], dtype=np.int16)
    np.save(os.path.join(output_dir, 'w_id.npy'), w_id)

    print(f"✓ Saved x.npy: shape {x[valid_mask].shape}")
    print(f"✓ Saved x_len.npy: shape {x_len[valid_mask].shape}")
    print(f"✓ Saved c.npy: shape {c[valid_mask].shape}")
    print(f"✓ Saved c_len.npy: shape {c_len[valid_mask].shape}")
    print(f"✓ Saved w_id.npy: shape {w_id.shape}")

    # Print statistics
    print(f"\nDataset Statistics:")
    print(f"  Total samples: {num_valid}")
    print(f"  Average stroke length: {np.mean(x_len[valid_mask]):.1f} points")
    print(f"  Max stroke length: {np.max(x_len[valid_mask])} points")
    print(f"  Average text length: {np.mean(c_len[valid_mask]):.1f} characters")
    print(f"  Max text length: {np.max(c_len[valid_mask])} characters")

    return num_valid


def main():
    parser = argparse.ArgumentParser(
        description="Convert collected handwriting to training format"
    )
    parser.add_argument('--input', type=str, default='data/custom',
                       help='Input directory containing collected samples')
    parser.add_argument('--output', type=str, default='data/processed',
                       help='Output directory for processed numpy arrays')
    parser.add_argument('--append', action='store_true',
                       help='Append to existing processed data (not recommended for first run)')

    args = parser.parse_args()

    # Check if output directory exists and has data
    if os.path.exists(args.output) and not args.append:
        existing_files = os.listdir(args.output)
        if any(f.endswith('.npy') for f in existing_files):
            print(f"\nWarning: Output directory {args.output} already contains .npy files!")
            response = input("Do you want to overwrite? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Aborted. Use --output to specify a different directory.")
                sys.exit(0)

    # Load collected samples
    samples = load_collected_samples(args.input)

    if len(samples) < 10:
        print(f"\nWarning: Only {len(samples)} samples collected.")
        print("For good results, collect at least 50-100 samples.")
        print("More samples = better quality handwriting synthesis!")
        response = input("\nContinue anyway? (yes/no): ")
        if response.lower() not in ['yes', 'y']:
            print("Aborted. Run collect_handwriting.py to collect more samples.")
            sys.exit(0)

    # Convert to training format
    num_converted = convert_samples(samples, args.output)

    print(f"\n{'='*60}")
    print(f"Conversion complete!")
    print(f"{'='*60}")
    print(f"\nProcessed data saved to: {args.output}")
    print(f"Total samples ready for training: {num_converted}")
    print(f"\nNext steps:")
    print(f"  1. Verify data quality: python verify_data.py")
    print(f"  2. Train model: python rnn.py")
    print(f"     (or fine-tune: python rnn.py --warm_start_init_step=17900)")
    print()


if __name__ == '__main__':
    main()
