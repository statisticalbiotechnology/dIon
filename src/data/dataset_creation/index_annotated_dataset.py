from depthcharge.data import SpectrumDataset
import pyarrow as pa
from pathlib import Path
from sklearn.model_selection import train_test_split

# Define constants
SOURCE_DATA_ROOT_DIR = "/path/to/data/bacteria_PXD010000__PXD010613/bacteria_PXD010613_annotated_preprocessed/"
DEST_DATA_DIR = "/path/to/data/bacteria_PXD010000__PXD010613/annotated_v2/"
FILE_TYPE_SUFFIX = ".mgf"
CREATE_VAL_SPLIT = False
VAL_SPLIT_PERCENT = 0  # Percentage of files to use for validation
RANDOM_SEED = 42  # Seed for reproducibility

# Create a Path object for the source directory
source_dir = Path(SOURCE_DATA_ROOT_DIR)

# Define the destination directories for training and validation datasets
TRAIN_DATA_DIR = Path(DEST_DATA_DIR) / "test.lance"
VAL_DATA_DIR = Path(DEST_DATA_DIR) / "val.lance"

# Find all .mzML files in the directory recursively
mzml_files = list(source_dir.rglob(f'*{FILE_TYPE_SUFFIX}'))

# Log the files found (optional but useful for verification)
print(f"Found {len(mzml_files)} files to process.")

# Split the files into training and validation sets if required
if CREATE_VAL_SPLIT:
    train_files, val_files = train_test_split(
        mzml_files, test_size=VAL_SPLIT_PERCENT / 100, random_state=RANDOM_SEED
    )
    print(f"Training files: {len(train_files)}")
    print(f"Validation files: {len(val_files)}")
else:
    train_files = mzml_files
    val_files = []

# Create the annotated dataset for training
annotated_train_dataset = SpectrumDataset(
    spectra=train_files,
    path=TRAIN_DATA_DIR, batch_size=1000, min_peaks=10, max_precursor_charge=5,
    custom_fields=[pa.field("seq", pa.string())],
)

# Create the annotated dataset for validation if split is enabled
if CREATE_VAL_SPLIT and val_files:
    annotated_val_dataset = SpectrumDataset(
        spectra=val_files,
        path=VAL_DATA_DIR, batch_size=1000, min_peaks=10, max_precursor_charge=5,
        custom_fields=[pa.field("seq", pa.string())],
    )

# Example code to read and display some batches from the dataset
# import lance
# lance_dataset = lance.dataset(TRAIN_DATA_DIR)
# # lance_generator = lance_dataset.to_batches(batch_size=1)
# lance_generator = lance_dataset.take([i for i in range(20)]).to_batches()
# for batch in lance_generator:
#     print(batch)
#     break
