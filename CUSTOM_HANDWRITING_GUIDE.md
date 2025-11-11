# Guide: Training on Your Own Handwriting

This guide explains how to collect and convert your own handwriting samples to train a personalized handwriting synthesis model.

## Table of Contents
1. [Overview](#overview)
2. [Data Collection Methods](#data-collection-methods)
3. [Required Data Format](#required-data-format)
4. [Conversion Process](#conversion-process)
5. [Training Your Model](#training-your-model)

## Overview

To train on your own handwriting, you need:
- **Stroke data**: Sequential x,y coordinates of your pen movements
- **Text transcriptions**: What text each sample represents
- **Sufficient samples**: Ideally 100+ sentences for good results

The model learns from the **temporal sequence** of pen movements, not static images.

## Data Collection Methods

### Method 1: Digital Tablet/Stylus (RECOMMENDED)

**Best option for quality data collection.**

#### Hardware Options:
- **Wacom tablets** (Intuos, Bamboo)
- **iPad + Apple Pencil** with appropriate app
- **Microsoft Surface** with Surface Pen
- **Android tablet** with stylus support
- **Graphic tablets** (any pressure-sensitive tablet)

#### Software/Apps to Capture Strokes:

##### For Tablets (iPad/Android):
- **MyScript Nebo** - exports to JSON/text
- **Inkodo** - stroke recording
- **GoodNotes** or **Notability** - can export with some processing

##### For Desktop:
- Use the provided `collect_handwriting.py` script (see below)
- Python libraries: `pygame`, `pynput`, or `tkinter` with mouse events

### Method 2: Mouse/Trackpad

**Less natural but accessible without special hardware.**

- Slower and less natural than stylus
- Works with the collection script provided
- May require more practice for smooth writing

### Method 3: Convert from Paper

**Most challenging, lower quality.**

- Scan handwritten samples
- Use image processing to extract stroke paths
- Requires significant post-processing
- Not recommended unless you have existing samples

## Required Data Format

The model expects data in this structure:

### Stroke Format
Each handwriting sample is a sequence of points:
```
[[x1, y1, end_of_stroke],
 [x2, y2, end_of_stroke],
 [x3, y3, end_of_stroke],
 ...]
```

Where:
- `x, y`: Coordinate of pen position (integers)
- `end_of_stroke`: `1` if this is the last point in a stroke (pen lift), `0` otherwise

### Example:
Writing "hi" might look like:
```python
# Letter 'h'
[100, 200, 0],  # Start of vertical line
[100, 150, 0],  # Top of vertical
[100, 175, 0],  # Middle
[110, 175, 0],  # Curve to right
[120, 180, 1],  # End of 'h' - pen lifts (1)

# Letter 'i'
[130, 180, 0],  # Start vertical
[130, 175, 1],  # End vertical - pen lifts
[130, 165, 1],  # Dot - single point with pen lift
```

### Text Transcription
Each stroke sequence needs corresponding text:
- Plain ASCII text
- What you actually wrote
- Used to train the text-to-handwriting mapping

### Directory Structure

The conversion script will create:
```
data/
├── custom/
│   ├── strokes/
│   │   ├── sample_001.npy
│   │   ├── sample_002.npy
│   │   └── ...
│   ├── transcriptions.txt
│   └── metadata.json
├── processed/
│   ├── x.npy
│   ├── x_len.npy
│   ├── c.npy
│   ├── c_len.npy
│   └── w_id.npy
```

## Conversion Process

### Step 1: Collect Your Handwriting

Use the `collect_handwriting.py` script (provided in this guide):

```bash
python collect_handwriting.py
```

**Instructions:**
1. A window will open
2. You'll see a prompt with text to write
3. Write the text using your stylus/mouse
4. Press ENTER when done with that sample
5. Press ESC to finish collection session

**Tips for collection:**
- Write at a natural speed
- Include variety: different sentence structures, words
- Aim for 100-500 samples minimum
- Write the same text 2-3 times for variation
- Include common words and phrases

### Step 2: Prepare Sample Texts

Create a file `sample_texts.txt` with sentences to write:

```
The quick brown fox jumps over the lazy dog
Hello world, this is my handwriting
Machine learning is fascinating
Pack my box with five dozen liquor jugs
How vexingly quick daft zebras jump
```

Include:
- Common phrases
- Pangrams (use all letters)
- Different sentence lengths
- Punctuation variety
- Numbers if desired: "I have 123 apples"

### Step 3: Convert to Training Format

Run the conversion script:

```bash
python convert_custom_data.py --input data/custom --output data/processed
```

This will:
1. Load your stroke data
2. Apply preprocessing (alignment, denoising, normalization)
3. Encode text transcriptions
4. Create numpy arrays in the format the model expects

### Step 4: Verify Data

Check the processed data:

```bash
python verify_data.py
```

This will:
- Show statistics about your dataset
- Visualize sample strokes
- Check for any data quality issues

## Training Your Model

### Option 1: Train from Scratch

```bash
python rnn.py
```

**Note:** Training from scratch requires:
- Significant data (500+ samples recommended)
- GPU recommended (2-3 days on Tesla K80)
- May need to adjust hyperparameters for smaller datasets

### Option 2: Fine-tune Pretrained Model

```bash
python rnn.py --warm_start_init_step=17900 --learning_rate=0.00001
```

**Better for smaller datasets:**
- Start with pretrained model
- Adapt to your handwriting style
- Requires less data (50-100 samples)
- Faster training (hours instead of days)

### Training Configuration

Edit parameters in `rnn.py` for your dataset size:

**For small datasets (<200 samples):**
```python
batch_sizes=[16, 16, 16]  # Smaller batches
learning_rates=[.00005, .00002, .00001]  # Lower learning rates
patiences=[500, 300, 200]  # Less patience
```

**For large datasets (>500 samples):**
```python
batch_sizes=[32, 64, 64]  # Standard
learning_rates=[.0001, .00005, .00002]
patiences=[1500, 1000, 500]
```

## Tips for Best Results

### Data Collection
1. **Consistency**: Write in your natural style
2. **Variety**: Include different words, sentences, lengths
3. **Quality over quantity**: Clean strokes better than many messy ones
4. **Multiple sessions**: Collect over several days to capture natural variation

### Data Quality
- Smooth, continuous strokes
- Avoid jittery/shaky lines (unless that's your style)
- Clear pen lifts between letters/strokes
- Accurate transcriptions

### Training
- Monitor validation loss
- Start with fine-tuning if data is limited
- Experiment with bias parameter during generation
- Save checkpoints frequently

## Troubleshooting

### "Not enough data" error
- Collect more samples (minimum 50, ideally 200+)
- Or use fine-tuning approach instead of training from scratch

### Generated handwriting looks random
- Need more training data
- Train for more epochs
- Check data quality with `verify_data.py`

### Model doesn't converge
- Reduce learning rate
- Reduce batch size
- Check for data preprocessing issues

### Generated handwriting doesn't match your style
- Collect more diverse samples
- Include more character combinations
- Try fine-tuning instead of training from scratch
- Increase training time

## Next Steps

1. Run `collect_handwriting.py` to start gathering samples
2. Create your `sample_texts.txt` file
3. Collect 100+ samples of your handwriting
4. Run `convert_custom_data.py` to process the data
5. Train your model with `rnn.py`
6. Generate samples with `demo.py`

## Example Workflow

```bash
# 1. Collect handwriting samples
python collect_handwriting.py

# 2. Convert to training format
python convert_custom_data.py

# 3. Verify data quality
python verify_data.py

# 4. Train (or fine-tune)
python rnn.py --warm_start_init_step=17900

# 5. Generate samples
python demo.py
```

## Required Scripts

The following helper scripts are provided:
- `collect_handwriting.py` - Interactive data collection tool
- `convert_custom_data.py` - Convert collected data to training format
- `verify_data.py` - Check data quality and visualize samples

See the next sections for these script implementations.
