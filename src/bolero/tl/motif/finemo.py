import warnings

import numpy as np
import pandas as pd
import polars as pl

try:
    from finemo.data_io import HITS_DTYPES, load_modisco_motifs
    from finemo.hitcaller import fit_contribs
except ModuleNotFoundError as e:  # pragma: no cover - optional git-only backend
    raise ModuleNotFoundError(
        "bolero.tl.motif.finemo requires the git-only 'finemo' package (finemo-gpu). "
        "Install it with "
        "`pip install git+https://github.com/austintwang/finemo_gpu.git`, "
        "or use the pixi environment (which declares it). See docs/installation.md."
    ) from e

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
        modisco_hdf5_path,
        cwm_trim_threshold=0.3,
        global_lambda=0.7,
        batch_size=2000,
        use_patterns=None,
    ):
        self.modisco_hdf5_path = modisco_hdf5_path

        self.cwm_trim_threshold = cwm_trim_threshold
        self.global_lambda = global_lambda
        self.mode = "hp"  # fix mode to "hp" here
        self.motif_type = "cwm" if self.mode[0] == "p" else "hcwm"
        self.use_hypothetical_contribs = False if self.mode[1] == "p" else True
        self.batch_size = batch_size

        self.pattern_dict = self._load_patterns()
        _motifs_df, _cwms, _trim_masks = self._finemo_load_motifs(use_patterns)
        self._finemo_motifs_df: pl.DataFrame = (
            _motifs_df  # (n_patterns, 8) with columns:
        )
        # 'motif_id', 'motif_name', 'motif_name_orig', 'strand', 'motif_start', 'motif_end', 'motif_scale', 'lambda'
        self._finemo_cwms: np.ndarray = _cwms  # (n_patterns, 4, motif_width)
        self._finemo_trim_masks: np.ndarray = _trim_masks  # (n_patterns, motif_width)

    def _load_patterns(self):
        modisco_patterns = ModiscoHDF(self.modisco_hdf5_path)

        pattern_dict = {}
        for pat in modisco_patterns.pos_patterns:
            pattern_dict[pat.name + ".pos"] = pat
        for pat in modisco_patterns.neg_patterns:
            pattern_dict[pat.name + ".neg"] = pat
        return pattern_dict

    def _finemo_load_motifs(self, use_patterns=None):
        # load modisco patterns
        motifs_df, cwms, trim_masks, _ = load_modisco_motifs(
            modisco_h5_path=self.modisco_hdf5_path,
            trim_threshold=self.cwm_trim_threshold,
            motif_type=self.motif_type,
            motifs_include=None,
            motif_name_map=None,
            motif_lambdas=None,
            motif_lambda_default=self.global_lambda,
            include_rc=True,
        )

        if use_patterns is not None:
            motif_sel = motifs_df["motif_name"].is_in(use_patterns).to_numpy()
            motifs_df = motifs_df.filter(motif_sel)
            cwms = cwms[motif_sel]
            trim_masks = trim_masks[motif_sel]
        return motifs_df, cwms, trim_masks

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

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            hits_df, qc_df = fit_contribs(
                cwms=cwms,
                contribs=contribs,
                sequences=sequences,
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
        return all_hits

    def run(self, all_sequences, all_contribs, all_attr_region, max_round=3):
        """Run the finemo pipeline."""
        sequences = all_sequences.copy()
        contribs = all_contribs.copy()
        attr_region = all_attr_region.copy()

        total_hits = []
        for round_id in range(max_round):
            all_hits = self._single_finemo(sequences, contribs, attr_region)
            sel_hits = all_hits.query(
                "(hit_coefficient_global > 5) & (start > 0)"
            ).copy()
            if sel_hits.shape[0] < 1:
                break
            sel_hits["scan_round"] = round_id
            total_hits.append(sel_hits)

            # mask hit region motif
            for _, row in sel_hits.iterrows():
                mstart = row["start"]
                mend = row["end"]
                region_idx = row["peak_id"]
                contribs[region_idx, :, mstart:mend] = 0

        total_hits = pd.concat(total_hits)
        return total_hits
