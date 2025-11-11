#!/usr/bin/env python
"""
Interactive handwriting collection tool.

Usage:
    python collect_handwriting.py [--samples SAMPLE_FILE] [--output OUTPUT_DIR]

Instructions:
    - Write the prompted text using your mouse/stylus
    - Press ENTER to save and move to next sample
    - Press BACKSPACE to clear and retry current sample
    - Press ESC to finish and exit
    - Press SPACE to skip current prompt
"""

from __future__ import print_function
import os
import sys
import argparse
import json
from datetime import datetime

try:
    import pygame
    import numpy as np
except ImportError as e:
    print("Error: Missing required libraries.")
    print("Install with: pip install pygame numpy")
    sys.exit(1)


# Default sample texts to write
DEFAULT_SAMPLES = [
    "The quick brown fox jumps over the lazy dog",
    "Hello world, this is my handwriting",
    "Machine learning is fascinating",
    "Pack my box with five dozen liquor jugs",
    "How vexingly quick daft zebras jump",
    "The five boxing wizards jump quickly",
    "Sphinx of black quartz, judge my vow",
    "Two driven jocks help fax my big quiz",
    "Five quacking zephyrs jolt my wax bed",
    "The jay, pig, fox, zebra and my wolves quack",
    "I have 123 apples and 456 oranges",
    "Python programming language",
    "Deep learning neural networks",
    "Artificial intelligence research",
    "Natural language processing",
]


class HandwritingCollector:
    def __init__(self, output_dir="data/custom", sample_texts=None):
        self.output_dir = output_dir
        self.sample_texts = sample_texts or DEFAULT_SAMPLES
        self.current_sample_idx = 0

        # Create output directories
        self.strokes_dir = os.path.join(output_dir, "strokes")
        os.makedirs(self.strokes_dir, exist_ok=True)

        # Initialize pygame
        pygame.init()
        self.width, self.height = 1200, 400
        self.screen = pygame.display.set_mode((self.width, self.height))
        pygame.display.set_caption("Handwriting Collection Tool")

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.GRAY = (200, 200, 200)
        self.BLUE = (0, 100, 255)
        self.GREEN = (0, 200, 0)
        self.RED = (255, 0, 0)

        # Fonts
        self.font_large = pygame.font.Font(None, 36)
        self.font_medium = pygame.font.Font(None, 28)
        self.font_small = pygame.font.Font(None, 20)

        # Drawing state
        self.drawing = False
        self.current_stroke = []
        self.all_strokes = []
        self.stroke_data = []  # List of [x, y, end_of_stroke]

        # Collection metadata
        self.metadata = {
            'collection_date': datetime.now().isoformat(),
            'samples': []
        }
        self.samples_collected = 0

        # Drawing area
        self.draw_area_top = 120
        self.draw_area_height = 200

    def get_current_prompt(self):
        if self.current_sample_idx < len(self.sample_texts):
            return self.sample_texts[self.current_sample_idx]
        return None

    def draw_ui(self):
        self.screen.fill(self.WHITE)

        # Title
        title = self.font_large.render("Handwriting Collection Tool", True, self.BLACK)
        self.screen.blit(title, (20, 20))

        # Progress
        progress_text = f"Sample {self.samples_collected + 1} | Total collected: {self.samples_collected}"
        progress = self.font_small.render(progress_text, True, self.GRAY)
        self.screen.blit(progress, (self.width - 350, 25))

        # Current prompt
        prompt = self.get_current_prompt()
        if prompt:
            prompt_label = self.font_small.render("Write this:", True, self.GRAY)
            self.screen.blit(prompt_label, (20, 70))

            prompt_text = self.font_medium.render(f'"{prompt}"', True, self.BLUE)
            self.screen.blit(prompt_text, (20, 90))
        else:
            done_text = self.font_large.render("All prompts completed! Press ESC to exit.", True, self.GREEN)
            self.screen.blit(done_text, (20, 70))

        # Drawing area border
        pygame.draw.rect(self.screen, self.GRAY,
                        (10, self.draw_area_top, self.width - 20, self.draw_area_height), 2)

        # Draw collected strokes
        for stroke in self.all_strokes:
            if len(stroke) > 1:
                pygame.draw.lines(self.screen, self.BLACK, False, stroke, 3)

        # Draw current stroke being drawn
        if len(self.current_stroke) > 1:
            pygame.draw.lines(self.screen, self.BLACK, False, self.current_stroke, 3)

        # Instructions
        instructions = [
            "ENTER: Save sample | BACKSPACE: Clear | SPACE: Skip | ESC: Exit"
        ]
        y_offset = self.height - 50
        for instruction in instructions:
            inst_text = self.font_small.render(instruction, True, self.GRAY)
            self.screen.blit(inst_text, (20, y_offset))
            y_offset += 25

        pygame.display.flip()

    def clear_drawing(self):
        self.current_stroke = []
        self.all_strokes = []
        self.stroke_data = []

    def save_current_sample(self):
        prompt = self.get_current_prompt()
        if not prompt or len(self.stroke_data) == 0:
            print("No strokes to save!")
            return False

        # Convert to numpy array
        stroke_array = np.array(self.stroke_data, dtype=np.float32)

        # Save stroke data
        sample_filename = f"sample_{self.samples_collected:04d}.npy"
        sample_path = os.path.join(self.strokes_dir, sample_filename)
        np.save(sample_path, stroke_array)

        # Save metadata
        self.metadata['samples'].append({
            'id': self.samples_collected,
            'filename': sample_filename,
            'transcription': prompt,
            'num_strokes': len(self.all_strokes),
            'num_points': len(self.stroke_data)
        })

        self.samples_collected += 1
        print(f"Saved sample {self.samples_collected}: '{prompt}' ({len(self.stroke_data)} points)")

        return True

    def save_metadata(self):
        metadata_path = os.path.join(self.output_dir, "metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(self.metadata, f, indent=2)

        # Also save transcriptions in simple text format
        transcriptions_path = os.path.join(self.output_dir, "transcriptions.txt")
        with open(transcriptions_path, 'w') as f:
            for sample in self.metadata['samples']:
                f.write(f"{sample['transcription']}\n")

        print(f"\nMetadata saved to {metadata_path}")
        print(f"Transcriptions saved to {transcriptions_path}")

    def handle_mouse_down(self, pos):
        # Check if click is in drawing area
        if self.draw_area_top <= pos[1] <= self.draw_area_top + self.draw_area_height:
            self.drawing = True
            self.current_stroke = [pos]
            # Add first point of new stroke
            self.stroke_data.append([pos[0], pos[1], 0])

    def handle_mouse_motion(self, pos):
        if self.drawing:
            # Check if still in drawing area
            if self.draw_area_top <= pos[1] <= self.draw_area_top + self.draw_area_height:
                self.current_stroke.append(pos)
                # Add intermediate point
                self.stroke_data.append([pos[0], pos[1], 0])

    def handle_mouse_up(self, pos):
        if self.drawing:
            self.drawing = False
            if len(self.current_stroke) > 0:
                # Mark last point as end of stroke
                if len(self.stroke_data) > 0:
                    self.stroke_data[-1][2] = 1
                self.all_strokes.append(self.current_stroke[:])
                self.current_stroke = []

    def run(self):
        print("\n" + "="*60)
        print("Handwriting Collection Tool")
        print("="*60)
        print(f"\nOutput directory: {self.output_dir}")
        print(f"Number of prompts: {len(self.sample_texts)}")
        print("\nInstructions:")
        print("  - Write the prompted text using your mouse/stylus")
        print("  - Press ENTER to save and move to next sample")
        print("  - Press BACKSPACE to clear and retry")
        print("  - Press SPACE to skip current prompt")
        print("  - Press ESC to finish and exit")
        print("\nStarting collection...\n")

        clock = pygame.time.Clock()
        running = True

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self.handle_mouse_down(event.pos)

                elif event.type == pygame.MOUSEMOTION:
                    self.handle_mouse_motion(event.pos)

                elif event.type == pygame.MOUSEBUTTONUP:
                    self.handle_mouse_up(event.pos)

                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

                    elif event.key == pygame.K_RETURN:
                        # Save current sample and move to next
                        if self.save_current_sample():
                            self.clear_drawing()
                            self.current_sample_idx += 1

                    elif event.key == pygame.K_BACKSPACE:
                        # Clear current drawing
                        self.clear_drawing()
                        print("Drawing cleared")

                    elif event.key == pygame.K_SPACE:
                        # Skip to next prompt
                        self.clear_drawing()
                        self.current_sample_idx += 1
                        print("Skipped to next prompt")

            self.draw_ui()
            clock.tick(60)  # 60 FPS

        # Save metadata before exiting
        if self.samples_collected > 0:
            self.save_metadata()
            print(f"\n{'='*60}")
            print(f"Collection complete! Total samples: {self.samples_collected}")
            print(f"Data saved to: {self.output_dir}")
            print(f"{'='*60}\n")
        else:
            print("\nNo samples collected.")

        pygame.quit()


def load_sample_texts(filepath):
    """Load sample texts from a file (one per line)."""
    if not os.path.exists(filepath):
        print(f"Warning: Sample file '{filepath}' not found. Using default samples.")
        return None

    with open(filepath, 'r') as f:
        samples = [line.strip() for line in f if line.strip()]

    print(f"Loaded {len(samples)} sample texts from {filepath}")
    return samples


def main():
    parser = argparse.ArgumentParser(description="Collect handwriting samples")
    parser.add_argument('--samples', type=str, default=None,
                       help='Path to text file with sample texts (one per line)')
    parser.add_argument('--output', type=str, default='data/custom',
                       help='Output directory for collected samples')

    args = parser.parse_args()

    # Load sample texts if provided
    sample_texts = None
    if args.samples:
        sample_texts = load_sample_texts(args.samples)

    # Create and run collector
    collector = HandwritingCollector(
        output_dir=args.output,
        sample_texts=sample_texts
    )
    collector.run()


if __name__ == '__main__':
    main()
