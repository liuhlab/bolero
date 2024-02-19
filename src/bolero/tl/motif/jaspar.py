import pathlib
import subprocess
import tempfile

import joblib
import numpy as np
import pandas as pd
from Bio import motifs

# get pkg_data path from package root
import bolero

from bolero.pp.seq import DEFAULT_ONE_HOT_ORDER

PKG_DATA_PATH = pathlib.Path(bolero.__file__).parent / "pkg_data"
JASPAR_MTOFI_DBS = {
    "_".join(p.name.split("_")[:3]): p
    for p in pathlib.Path(PKG_DATA_PATH).glob("jaspar/*.motif_pwm.dict")
}

# The above motif_pwm.dict files are generated from the JASPAR 2024 CORE motif database using dump_jaspar_motif_pwm_dict
# JASPAR 2024 CORE motif database
_JASPAR_URL_BASE = "https://jaspar.elixir.no/download/data/2024/CORE"
JASPAR_URLS = {
    "JASPAR2024_CORE_nematodes": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_nematodes_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_diatoms": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_diatoms_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_insects": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_insects_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_vertebrates": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_vertebrates_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_fungi": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_fungi_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_plants": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_plants_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_urochordates": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_urochordates_non-redundant_pfms_jaspar.zip",
    "JASPAR2024_CORE_ALL": f"{_JASPAR_URL_BASE}/JASPAR2024_CORE_non-redundant_pfms_jaspar.zip",
}


def dump_jaspar_motif_pwm_dict(db, output_dir="."):
    """
    Download JASPAR motif database and dump the PWMs into a dictionary.

    Parameters
    ----------
    jaspar_url : str
        URL to the JASPAR database.
    output_dir : str
        Directory to save the motif PWMs.
    """
    jaspar_url = JASPAR_URLS[db]
    db_name = jaspar_url.split("/")[-1].split(".")[0]
    with tempfile.TemporaryDirectory() as tmp_dir:
        db_name = jaspar_url.split("/")[-1].split(".")[0]
        subprocess.run(
            f"wget {jaspar_url} -P {tmp_dir}",
            shell=True,
            check=True,
        )
        subprocess.run(
            f"unzip {tmp_dir}/{db_name}.zip -d {tmp_dir}",
            shell=True,
            check=True,
        )
        jaspar_paths = list(pathlib.Path(tmp_dir).glob("*.jaspar"))

        motif_pwms = {}
        for p in jaspar_paths:
            with open(p) as handle:
                motif_list = motifs.parse(handle, "jaspar")
                for motif in motif_list:
                    pwm = pd.DataFrame(motif.pwm)
                    motif_pwms[(motif.matrix_id, motif.name)] = pwm

    output_dir = pathlib.Path(output_dir).absolute()
    joblib.dump(motif_pwms, f"{db_name}.motif_pwm.dict", compress=1)
    return


def _calc_row_entropy(row):
    row = row[row > 0]
    e = -np.sum(row * np.log2(row))
    return e


class JASPARMotif:
    def __init__(self, motif_id, motif_name, pwm, base_order=DEFAULT_ONE_HOT_ORDER):
        self.motif_id = motif_id
        self.motif_name = motif_name
        self.pwm = pwm.loc[:, list(base_order)].copy()

    def __len__(self):
        return self.pwm.shape[0]

    def pwm_entropy(self):
        entropy = self.pwm.apply(_calc_row_entropy, axis=1)
        return entropy

    def clip_pwm_by_entropy(self, max_length=24):
        """Clip the PWM by removing the end positions with the highest entropy."""
        cur_length = len(self)
        if cur_length <= max_length:
            return

        # calculate the entropy at each position
        pwm = self.pwm.copy()
        entropy = self.pwm_entropy()
        while cur_length > max_length:
            start_e = entropy.values[0]
            end_e = entropy.values[-1]
            if start_e > end_e:
                entropy = entropy.iloc[1:]
                pwm = pwm.iloc[1:]
            else:
                entropy = entropy.iloc[:-1]
                pwm = pwm.iloc[:-1]
            cur_length -= 1
        self.pwm = pwm.copy()
        return


class JASPARMotifDatabase:

    @classmethod
    def available_databases(cls):
        return set(JASPAR_MTOFI_DBS.keys())

    def __init__(self, db="JASPAR2024_CORE_vertebrates", max_length=24, base_order=DEFAULT_ONE_HOT_ORDER):
        # check if db is valid using class method
        if db not in self.available_databases():
            raise ValueError(f"Invalid JASPAR database: {db}")

        self.db = db
        self.motif_pwms = joblib.load(JASPAR_MTOFI_DBS[db])

        self.motifs = []
        for (motif_id, motif_name), pwm in self.motif_pwms.items():
            motif = JASPARMotif(motif_id, motif_name, pwm, base_order=DEFAULT_ONE_HOT_ORDER)

            motif.clip_pwm_by_entropy(max_length)
            self.motifs.append(motif)
        return
    
