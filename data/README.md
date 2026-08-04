# Data

The dataset is **not committed** to this repository. It is downloaded on demand from Kaggle.
This keeps the repo small and avoids redistributing a large file.

## Dataset

**All Lending Club loan data** (`wordsforthewise/lending-club`): every loan issued through Lending Club from 2007 to 2018 Q4,
with borrower, loan, and credit-bureau attributes plus the final loan status.

## Download

1. Install the Kaggle CLI and authenticate (one time):

   ```bash
   pip install kaggle
   # Place your API token at ~/.kaggle/kaggle.json
   # (Kaggle > Account > Settings > Create New Token)
   ```

2. Download and unzip into this folder (run from the repository root):

   ```bash
   kaggle datasets download -d wordsforthewise/lending-club -p data --unzip
   ```

After unzipping you should have:

```
data/
  accepted_2007_to_2018Q4.csv.gz    # the file this project uses
  rejected_2007_to_2018Q4.csv.gz    # not used (rejected applications)
```

Only `accepted_2007_to_2018Q4.csv.gz` is used. Its path is configured in
`src/credit_risk/config.py` (`RAW_ACCEPTED_FILE`).
