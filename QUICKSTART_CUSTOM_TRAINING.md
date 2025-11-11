# Quick Start: Train on Your Own Handwriting

A simple 5-step guide to train this model on your handwriting.

## Prerequisites

```bash
pip install pygame numpy tensorflow matplotlib scipy
```

## Step 1: Collect Handwriting Samples

```bash
python collect_handwriting.py --samples sample_texts.txt
```

**What this does:**
- Opens a window where you write with mouse/stylus
- Shows prompts from `sample_texts.txt`
- Saves your strokes to `data/custom/`

**Controls:**
- **ENTER**: Save current sample and move to next
- **BACKSPACE**: Clear and retry
- **SPACE**: Skip to next prompt
- **ESC**: Finish and exit

**How many samples?**
- Minimum: 50 samples
- Recommended: 100-200 samples
- More is better!

## Step 2: Convert to Training Format

```bash
python convert_custom_data.py
```

**What this does:**
- Reads collected strokes from `data/custom/`
- Processes and normalizes the data
- Saves training files to `data/processed/`

## Step 3: Verify Data Quality

```bash
python verify_data.py
```

**What this does:**
- Shows statistics about your dataset
- Checks for data quality issues
- Creates visualizations in `verification_output/`

**Look for:**
- ✓ No NaN or Inf values
- ✓ Reasonable stroke magnitudes
- Dataset size assessment

## Step 4: Train Your Model

### Option A: Fine-tune (Recommended for <200 samples)

```bash
python rnn.py --warm_start_init_step=17900
```

**Advantages:**
- Faster (hours instead of days)
- Works with less data
- Adapts pretrained model to your style

### Option B: Train from scratch (For >500 samples)

```bash
python rnn.py
```

**Advantages:**
- Purely your handwriting style
- No influence from pretrained model
- Requires more data and time

## Step 5: Generate Handwriting

```python
from demo import Hand

hand = Hand()
hand.write(
    filename='output.svg',
    lines=["This is my handwriting!"],
    biases=[0.5],
    styles=[0]
)
```

**Parameters:**
- `biases`: Lower = messier (0.1), Higher = neater (1.5)
- `styles`: Different writing variations (0-11)

## Troubleshooting

### "Not enough data"
- Collect more samples (at least 50)
- Or use fine-tuning instead of training from scratch

### Generated handwriting looks random
- Collect more varied samples
- Train for more epochs
- Check data quality with `verify_data.py`

### Collection window doesn't open
- Make sure pygame is installed: `pip install pygame`
- Try running with `python` instead of `python3`

### Training is too slow
- Use GPU if available
- Reduce batch size in `rnn.py`
- Use fine-tuning instead of training from scratch

## File Structure

```
handwriting-synthesis/
├── collect_handwriting.py       # Collect your handwriting
├── convert_custom_data.py       # Convert to training format
├── verify_data.py               # Check data quality
├── sample_texts.txt             # Example texts to write
├── data/
│   ├── custom/                  # Your collected samples
│   │   ├── strokes/
│   │   ├── metadata.json
│   │   └── transcriptions.txt
│   └── processed/               # Converted training data
│       ├── x.npy
│       ├── x_len.npy
│       ├── c.npy
│       ├── c_len.npy
│       └── w_id.npy
├── checkpoints/                 # Model saves here
└── verification_output/         # Sample visualizations
```

## Tips for Best Results

1. **Write naturally** - Don't try to write perfectly
2. **Be consistent** - Write in your normal style
3. **Include variety** - Different words and sentence structures
4. **Multiple sessions** - Collect over several days
5. **Check quality** - Use `verify_data.py` before training
6. **Start with fine-tuning** - Unless you have 500+ samples

## Complete Example Workflow

```bash
# Install dependencies
pip install pygame numpy tensorflow matplotlib scipy

# Collect 100 samples
python collect_handwriting.py --samples sample_texts.txt
# (Write samples until you have ~100)

# Convert data
python convert_custom_data.py

# Verify quality
python verify_data.py

# Fine-tune model (recommended)
python rnn.py --warm_start_init_step=17900

# Generate sample
python -c "
from demo import Hand
hand = Hand()
hand.write(
    filename='my_handwriting.svg',
    lines=['Hello, this is my handwriting!'],
    biases=[0.75],
    styles=[0]
)
"
```

## More Information

For detailed information, see [CUSTOM_HANDWRITING_GUIDE.md](CUSTOM_HANDWRITING_GUIDE.md)
