import os
import lance
import pyarrow.compute as pc
from tqdm import tqdm

ROOT_IN = "/path/to/data/foundational_dataset/combined"
ROOT_OUT = "/path/to/data/foundational_dataset/combined_V2"

INPUT_TRAIN = f"{ROOT_IN}/train.lance"
INPUT_VAL = f"{ROOT_IN}/val.lance"

OUTPUT_TRAIN = f"{ROOT_OUT}/train.lance"
OUTPUT_VAL = f"{ROOT_OUT}/val.lance"

MIN_PEAKS = 20
CHUNK_SIZE = 40000


def process_one(input_path, output_path, name):
    ds = lance.dataset(input_path)
    schema = ds.schema
    total_rows = ds.count_rows()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    if os.path.exists(output_path):
        os.remove(output_path)

    scanner = ds.scanner(batch_size=CHUNK_SIZE)

    first = True
    kept_total = 0
    drop_total = 0

    with tqdm(total=total_rows, desc=name) as pbar:
        for rb in scanner.to_batches():

            lengths = pc.list_value_length(rb["mz_array"])
            mask = pc.greater_equal(lengths, MIN_PEAKS)

            n = len(rb)
            k = int(mask.sum().as_py())
            d = n - k

            kept_total += k
            drop_total += d

            if k > 0:
                filtered = rb.filter(mask)

                if first:
                    lance.write_dataset(
                        [filtered], output_path, schema=schema, mode="create"
                    )
                    first = False
                else:
                    lance.write_dataset(
                        [filtered], output_path, schema=schema, mode="append"
                    )

            pbar.set_postfix(kept=kept_total, drops=drop_total)
            pbar.update(n)


if __name__ == "__main__":
    process_one(INPUT_TRAIN, OUTPUT_TRAIN, "train.lance")
    process_one(INPUT_VAL, OUTPUT_VAL, "val.lance")
