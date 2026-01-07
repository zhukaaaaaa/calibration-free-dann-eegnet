This repository uses the BCI Competition IV dataset 2a.

Download the preprocessed epochs from:
- http://bnci-horizon-2020.eu/database/data-sets
- Or use MOABB to download and preprocess automatically.

Expected structure:
./data/
  S01-epo.fif
  S02-epo.fif
  ...
  S09-epo.fif

These are MNE Epochs files (already epoched, no further artifact rejection).
