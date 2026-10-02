import os
import numpy as np
import lance
from tqdm import tqdm

# Define your Lance dataset directory
ROOT_DIR = "/path/to/data/foundational_dataset/combined/test"
LANCE_DIR = os.path.join(ROOT_DIR, "debug.lance")
CHUNK_SIZE = 1000
DELETION_CHUNK_SIZE = 10

# Set the sizes for validation and test splits as a percentage of the total dataset
VAL_PERCENT = 1.0  # 1% for validation
TEST_PERCENT = 0  # 0.25% for testing

ds = lance.dataset(LANCE_DIR)

# Calculate the sizes for each dataset split
total_rows = ds.count_rows()
print(f"Source dataset #datapoints: {total_rows}")
val_size = int(VAL_PERCENT / 100 * total_rows)
test_size = int(TEST_PERCENT / 100 * total_rows)

# Generate and shuffle indices
indices = np.arange(total_rows)
np.random.shuffle(indices)

# Split indices for validation and test datasets
val_indices = indices[:val_size]
test_indices = indices[val_size : val_size + test_size] if TEST_PERCENT > 0 else []


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


# Write validation and test datasets
val_dataset_path = os.path.join(ROOT_DIR, "val.lance")
test_dataset_path = os.path.join(ROOT_DIR, "test.lance") if TEST_PERCENT > 0 else None

lance.write_dataset(
    generate_record_batches(ds, val_indices, CHUNK_SIZE),
    val_dataset_path,
    schema=ds.schema,
)

if TEST_PERCENT > 0:
    lance.write_dataset(
        generate_record_batches(ds, test_indices, CHUNK_SIZE),
        test_dataset_path,
        schema=ds.schema,
    )


def extract_keys_for_indices(dataset, indices):
    keys_table = dataset.take(indices, columns=["peak_file", "scan_id"]).to_pydict()
    return list(zip(keys_table["peak_file"], keys_table["scan_id"]))


def create_deletion_predicate(keys):
    # Here we ensure that `scan_id` is treated as an integer and not quoted
    conditions = [
        f"(peak_file = '{peak_file}' AND scan_id = {scan_id})"
        for peak_file, scan_id in keys
    ]
    return " OR ".join(conditions)

try:
    val_keys = extract_keys_for_indices(ds, val_indices)
    test_keys = extract_keys_for_indices(ds, test_indices) if TEST_PERCENT > 0 else []

    _split_keys = [('val', val_keys), ('test', test_keys)] if TEST_PERCENT > 0 else [('val', val_keys)]

    for dataset_label, keys in _split_keys:
        for i in tqdm(range(0, len(keys), DELETION_CHUNK_SIZE), desc=f"Deleting the {len(keys)} {dataset_label} entries from source dataset ... "):
            batch_keys = keys[i:i + DELETION_CHUNK_SIZE]
            if batch_keys:
                deletion_predicate = create_deletion_predicate(batch_keys)
                
                ds.delete(deletion_predicate)

except Exception as e:
    print("Error during deletion:", e)

# Print row counts and paths
val_ds = lance.dataset(val_dataset_path)
test_ds = lance.dataset(test_dataset_path) if TEST_PERCENT > 0 else None

message = "Dataset has been split:\n"
message += f"\tTraining set:\t{ds.count_rows()} rows (was {total_rows} rows before splitting) at {LANCE_DIR}\n"
message += f"\tValidation set:\t{val_ds.count_rows()} rows at {val_dataset_path}\n"
if TEST_PERCENT > 0:
    message += f"\tTest set:\t{test_ds.count_rows()} rows at {test_dataset_path}\n"

print(message)
