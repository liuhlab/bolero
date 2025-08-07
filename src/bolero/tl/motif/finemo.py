import pathlib
import warnings

import numpy as np
import pandas as pd
import polars as pl
from finemo.data_io import HITS_DTYPES, load_modisco_motifs
from finemo.hitcaller import fit_contribs

from bolero import Genome
from bolero.tl.motif.modisco import ModiscoHDF


def annotate_hits(hits_df, peaks_df, motifs_df, qc_df, motif_width):
    """Annotate hits with peak and motif information."""
    data_all = (
        hits_df.lazy()
        .join(peaks_df.lazy(), on="peak_id", how="inner")
        .join(qc_df.lazy(), on="peak_id", how="inner")
        .join(motifs_df.lazy(), on="motif_id", how="inner")
        .select(
            chr_id=pl.col("chr_id"),
            chr=pl.col("chr"),
            start=pl.col("peak_region_start")
            + pl.col("hit_start")
            + pl.col("motif_start"),
            end=pl.col("peak_region_start") + pl.col("hit_start") + pl.col("motif_end"),
            start_untrimmed=pl.col("peak_region_start") + pl.col("hit_start"),
            end_untrimmed=pl.col("peak_region_start")
            + pl.col("hit_start")
            + motif_width,
            motif_name=pl.col("motif_name"),
            hit_coefficient=pl.col("hit_coefficient"),
            hit_coefficient_global=pl.col("hit_coefficient")
            * (pl.col("global_scale") ** 2),
            hit_similarity=pl.col("hit_similarity"),
            hit_correlation=pl.col("hit_similarity"),
            hit_importance=pl.col("hit_importance") * pl.col("global_scale"),
            hit_importance_sq=pl.col("hit_importance_sq")
            * (pl.col("global_scale") ** 2),
            strand=pl.col("strand"),
            peak_name=pl.col("peak_name"),
            peak_id=pl.col("peak_id"),
            motif_id=pl.col("motif_id"),
            motif_lambda=pl.col("lambda"),
        )
        .sort(["chr_id", "start"])
        .select(list(HITS_DTYPES.keys()) + ["motif_id"])
    )

    data_unique = data_all.unique(
        subset=["chr", "start", "motif_name", "strand"], maintain_order=True
    )

    data_unique = data_unique.collect().to_pandas()
    return data_unique


class Finemo:
    def __init__(
        self,
        genome,
        modisco_dir,
        cwm_trim_threshold=0.3,
        global_lambda=0.7,
        batch_size=2000,
    ):
        self.genome = Genome(genome) if isinstance(genome, str) else genome

        self.modisco_dir = pathlib.Path(modisco_dir)
        self.modisco_hdf5_path = f"{self.modisco_dir}/all_patterns.filtered.h5"
        self.modisco_annot_path = (
            f"{self.modisco_dir}/all_patterns_annot.filtered.feather"
        )

        self.cwm_trim_threshold = cwm_trim_threshold
        self.global_lambda = global_lambda
        self.mode = "hp"  # fix mode to "hp" here
        self.motif_type = "cwm" if self.mode[0] == "p" else "hcwm"
        self.use_hypothetical_contribs = False if self.mode[1] == "p" else True
        self.batch_size = batch_size

        self.pattern_dict, self.pattern_annot = self._load_patterns()
        (
            self._finemo_motifs_df,
            self._finemo_cwms,
            self._finemo_trim_masks,
            self._finemo_motif_names,
        ) = self._finemo_load_motifs()

    def _load_patterns(self):
        modisco_annot = pd.read_feather(self.modisco_annot_path)
        modisco_annot["sign"] = modisco_annot.index.map(lambda n: n.split(".")[1])
        modisco_patterns = ModiscoHDF(self.modisco_hdf5_path)
        pattern_names = modisco_annot.query("sign == 'pos'").index
        for pat in modisco_patterns.pos_patterns:
            pat_idx = int(pat.name.split("_")[1])
            pat.name = pattern_names[pat_idx]
        pattern_names = modisco_annot.query("sign == 'neg'").index
        for pat in modisco_patterns.neg_patterns:
            pat_idx = int(pat.name.split("_")[1])
            pat.name = pattern_names[pat_idx]

        pattern_dict = {}
        for pat in modisco_patterns.pos_patterns + modisco_patterns.neg_patterns:
            pattern_dict[pat.name] = pat
        return pattern_dict, modisco_annot

    def _finemo_load_motifs(self):
        # get modisco pattern names
        all_name_map = {}
        for sign, sign_df in self.pattern_annot.groupby("sign"):
            name_map = {
                f"{sign}_patterns.pattern_{idx}": name
                for idx, name in enumerate(sign_df.index)
            }
            all_name_map.update(name_map)

        # load modisco patterns
        motifs_df, cwms, trim_masks, motif_names = load_modisco_motifs(
            modisco_h5_path=self.modisco_hdf5_path,
            trim_threshold=self.cwm_trim_threshold,
            motif_type=self.motif_type,
            motifs_include=None,
            motif_name_map=all_name_map,
            motif_lambdas=None,
            motif_lambda_default=self.global_lambda,
            include_rc=True,
        )
        return motifs_df, cwms, trim_masks, motif_names

    def plot_pattern(self, name, **kwargs):
        """Plot a modisco pattern by name."""
        self.pattern_dict[name].to_seqlet().plot(**kwargs)

    def _load_npz(self, npz_path):
        data = np.load(npz_path, allow_pickle=True)
        sequences = data["seq"].astype("int8")
        contribs = data["attr"].astype("float16")
        attr_region = data["attr_region"]
        return sequences, contribs, attr_region

    def _single_finemo(self, sequences, contribs, attr_region):
        motifs_df = self._finemo_motifs_df
        cwms = self._finemo_cwms
        trim_masks = self._finemo_trim_masks

        # project contribs
        contribs = (contribs * sequences).sum(axis=1).astype("float16")
        num_regions = contribs.shape[0]

        motif_width = cwms.shape[2]
        lambdas = self._finemo_motifs_df.get_column("lambda").to_numpy(writable=True)

        _contribs = np.pad(
            contribs, pad_width=((0, 0), (10, 10)), mode="constant", constant_values=0
        )
        _sequences = np.pad(
            sequences,
            pad_width=((0, 0), (0, 0), (10, 10)),
            mode="constant",
            constant_values=0,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hits_df, qc_df = fit_contribs(
                cwms=cwms,
                contribs=_contribs,
                sequences=_sequences,
                cwm_trim_mask=trim_masks,
                use_hypothetical=self.use_hypothetical_contribs,
                lambdas=lambdas,
                step_size_max=3,
                step_size_min=0.08,
                sqrt_transform=False,
                convergence_tol=0.0005,
                max_steps=100,
                batch_size=self.batch_size,
                step_adjust=0.7,
                post_filter=True,
                device=None,
                compile_optimizer=True,
            )

        peak_data = {
            "chr": np.array(["NA"] * num_regions, dtype="U"),
            "chr_id": np.arange(num_regions, dtype=np.uint32),
            "peak_region_start": np.zeros(num_regions, dtype=np.int32),
            "peak_id": np.arange(num_regions, dtype=np.uint32),
            "peak_name": attr_region,
        }
        peaks_df = pl.DataFrame(peak_data)

        all_hits = annotate_hits(hits_df, peaks_df, motifs_df, qc_df, motif_width)

        all_hits["start"] -= 10
        all_hits["end"] -= 10
        return all_hits
