import csv
import os


class MztabWriter:
    def __init__(self, output_file):
        self.output_file = output_file
        self.metadata = []
        self.psm_columns = [
            "sequence",
            "PSM_ID",
            "accession",
            "unique",
            "database",
            "database_version",
            "search_engine",
            "search_engine_score[1]",
            "modifications",
            "retention_time",
            "charge",
            "exp_mass_to_charge",
            "calc_mass_to_charge",
            "spectra_ref",
            "pre",
            "post",
            "start",
            "end",
            "opt_ms_run[1]_aa_scores",
            "opt_ms_run[1]_ground_truth_sequence",
        ]
        self.file_handle = None
        self.writer = None
        self.psm_id_counter = 1

    def set_metadata(self, global_args):
        self.metadata = [
            ("mzTab-version", "1.0.0"),
            ("mzTab-mode", "Summary"),
            ("mzTab-type", "Identification"),
            (
                "description",
                f"CustomModel identification file {os.path.basename(self.output_file)}",
            ),
            ("software[1]", "[MS, MS:1001456, CustomModel, 1.0]"),
        ]
        for key, value in global_args.items():
            self.metadata.append((f"software[1]-setting[{key}]", f"{key} = {value}"))

    def open_file(self):
        self.file_handle = open(self.output_file, "w", newline="")
        self.writer = csv.writer(self.file_handle, delimiter="\t")
        for key, value in self.metadata:
            self.writer.writerow(["MTD", key, value])

    def write_headers(self):
        header = ["PSH"] + self.psm_columns
        self.writer.writerow(header)

    def write_psm(self, psm_entry):
        # Assign PSM_ID if not provided
        if psm_entry.get("PSM_ID") is None:
            psm_entry["PSM_ID"] = self.psm_id_counter
            self.psm_id_counter += 1

        # Prepare the row using predefined columns
        row = [
            self._format_field(psm_entry.get(col, "null")) for col in self.psm_columns
        ]
        self.writer.writerow(["PSM"] + row)

    def _format_field(self, value):
        if isinstance(value, list):
            return ",".join(value)
        elif isinstance(value, float):
            return f"{value:.6f}"
        else:
            return str(value)
        
    def flush(self):
        if self.file_handle:
            self.file_handle.flush()
            os.fsync(self.file_handle.fileno())

    def close_file(self):
        if self.file_handle:
            self.file_handle.close()
            self.file_handle = None
