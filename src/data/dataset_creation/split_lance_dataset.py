import os
import numpy as np
import lance
from tqdm import tqdm

# Define your Lance dataset directory
ROOT_DIR = "/path/to/data/foundational_dataset/combined/"
LANCE_DIR = os.path.join(ROOT_DIR, "combined_full.lance")
CHUNK_SIZE = 1000

# Set the sizes for validation and test splits as a percentage of the total dataset
VAL_PERCENT = 1.0  # 1% for validation
TEST_PERCENT = 0  # 0.25% for testing, set to 0 if no test split is desired

ds = lance.dataset(LANCE_DIR)

# Calculate the sizes for each dataset split
total_rows = ds.count_rows()
val_size = int(VAL_PERCENT / 100 * total_rows)
test_size = int(TEST_PERCENT / 100 * total_rows)

train_size = total_rows - val_size - test_size if test_size > 0 else total_rows - val_size

# Generate and shuffle indices
indices = np.arange(total_rows)
np.random.shuffle(indices)

# Split indices based on whether a test set is included
train_indices = indices[:train_size]
val_indices = indices[train_size:train_size + val_size]
test_indices = indices[train_size + val_size:] if test_size > 0 else []

def generate_record_batches(dataset, indices, chunk_size=1000):
    """
    Yields RecordBatches for specified indices in chunks, with a progress bar.
    """
    # Prepare tqdm to show progress over the chunks
    total_chunks = (len(indices) + chunk_size - 1) // chunk_size

    for i in tqdm(
        range(0, len(indices), chunk_size),
        total=total_chunks,
        desc="Processing batches",
    ):
        chunk_indices = indices[i : i + chunk_size]
        # Ensure the last chunk is processed even if it's smaller than chunk_size
        batches = dataset.take(chunk_indices, batch_readahead=0).to_batches()
        for batch in batches:
            yield batch

# Paths for the split datasets using os.path.join for compatibility
train_dataset_path = os.path.join(ROOT_DIR, "train.lance")
val_dataset_path = os.path.join(ROOT_DIR, "val.lance")
test_dataset_path = os.path.join(ROOT_DIR, "test.lance") if test_size > 0 else None

# Writing the datasets with progress bars
lance.write_dataset(
    generate_record_batches(ds, train_indices, chunk_size=CHUNK_SIZE), train_dataset_path, schema=ds.schema
)
lance.write_dataset(
    generate_record_batches(ds, val_indices, chunk_size=CHUNK_SIZE), val_dataset_path, schema=ds.schema
)

if test_size > 0:
    lance.write_dataset(
        generate_record_batches(ds, test_indices, chunk_size=CHUNK_SIZE), test_dataset_path, schema=ds.schema
    )

message = "Dataset has been split into train and validation sets."
if test_size > 0:
    message += " and test sets."
print(message)
