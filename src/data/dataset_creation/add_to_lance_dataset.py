from depthcharge.data import SpectrumDataset
import pyarrow as pa
from pathlib import Path

# Define the root directory and the destination directory
SOURCE_DATA_DIR = "/path/to/data/foundational_dataset/PretrainV2/"
EXCLUDE_DATA_DIR = "/path/to/data/foundational_dataset/msconvert/"
LANCE_DIR = "/path/to/data/foundational_dataset/combined.lance"
FILE_TYPE_SUFFIX = ".mgf"
# custom_fields = [pa.field("sequence", pa.string())] # Use this to grab peptide annotations
custom_fields = None

# Set to store filenames to exclude
exclude_files = {file.name for file in Path(EXCLUDE_DATA_DIR).rglob(f'*{FILE_TYPE_SUFFIX}')}

# Dictionary to store unique files from the source directory
unique_files = {}

# Collect unique files from the source directory not in exclude files
source_dir = Path(SOURCE_DATA_DIR)
for file_path in source_dir.rglob(f'*{FILE_TYPE_SUFFIX}'):
    if file_path.name not in exclude_files:
        unique_files[file_path.name] = file_path

# Convert unique files to a list for processing
unique_mgf_files = list(unique_files.values())

# Log the files found (optional but useful for verification)
print(f"Found {len(unique_mgf_files)} unique files to process after excluding duplicates.")


annotated_dataset = SpectrumDataset(
    spectra=None,
    path=LANCE_DIR
)
annotated_dataset.add_spectra(
    spectra=unique_mgf_files, batch_size=1000, min_peaks=10,
    custom_fields=custom_fields
)

# import lance
# lance_dataset = lance.dataset(LANCE_DIR)
# # lance_generator = lance_dataset.to_batches(batch_size=1)
# # lance_generator = lance_dataset.take([i for i in range(20)]).to_batches()
# # for batch in lance_generator:
# #     print(batch)
# #     break
# print(lance_dataset.count_rows())
