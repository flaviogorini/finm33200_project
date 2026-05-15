# Manually-Created Data

This folder is used to hold manually created data that cannot be easily replicated.
You may keep data here and keep it under version control if it is small enough.
Keeping this data under version control can provide some peace of mind
that the data is not inadvertently modified.
Also, keep in mind that Git LFS is a good option if the data is large.

## Loughran-McDonald master dictionary (for the 10-Q signals pipeline)

`src/score_sec_10q_text.py` looks for the Loughran-McDonald financial
sentiment master dictionary at `data_manual/lm_master_dictionary.csv`.
The file is not redistributed in this repo (license + size); download it
manually from <https://sraf.nd.edu/loughranmcdonald-master-dictionary/>
and save the CSV at the path above.

If the file is missing the scorer falls back to a small built-in word
list so the pipeline still runs end-to-end — feature values are noisier
but the panel builds. A "Using ... dictionary" message at the top of the
scorer's output reports which path was used.