## Model Training Instructions

In order to train a model, data must be downloaded and placed in this directory.

Follow the download instructions here http://www.fki.inf.unibe.ch/databases/iam-on-line-handwriting-database.

Only a subset of the downloaded data is required.  Move the relevant download data so the directory structure is as folllows:

```
data/
├── raw/
│   ├── ascii/
│   ├── lineStrokes/
│   ├── original/
|   blacklist.npy
```

Once this is completed, run `prepare_data.py` extract the data and dump it to numpy files.

To train the model, run `rnn.py`.  This takes a couple days on a single Tesla K80.

