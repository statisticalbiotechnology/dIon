import os
import re

root = "/path/to/data/foundational_dataset/PretrainV2/metadata"

instrument_types = set()
sample_origins = set()
collision_energies = set()

instrument_re = re.compile(r"Instrument model=.*?,\s*(.*?)\]")
origin_re = re.compile(r"Sample id=(.*)")
ce_re = re.compile(r"(?:Fragmentation types|Collision energy)=([^;\n]+)")

for fname in os.listdir(root):
    if not fname.endswith(".txt"):
        continue
    path = os.path.join(root, fname)
    with open(path) as f:
        for line in f:
            m = instrument_re.search(line)
            if m:
                instrument_types.add(m.group(1).strip())
            m = origin_re.search(line)
            if m:
                sample_origins.add(m.group(1).strip())
            m = ce_re.search(line)
            if m:
                collision_energies.add(m.group(1).strip())

with open("metadata_stats.txt", "w") as f:
    f.write("Instrument types:\n")
    for x in sorted(instrument_types):
        f.write(f"  {x}\n")

    f.write("\nSample origins:\n")
    for x in sorted(sample_origins):
        f.write(f"  {x}\n")

    f.write("\nCollision energies:\n")
    for x in sorted(collision_energies):
        f.write(f"  {x}\n")
