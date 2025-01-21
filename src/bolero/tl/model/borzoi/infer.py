import pathlib
import warnings

import numpy as np
import pandas as pd
import pyranges as pr
import torch
import xarray as xr
from captum.attr import InputXGradient
from tqdm import tqdm

from bolero.pp.genome import Genome
from bolero.utils import understand_regions


class GeneDataSummary:
    def __init__(
        self,
        data_keys,
        genome: Genome,
        region_key="region",
        resolution=32,
        seq_len=16352,
        feature_types=("gene", "exon", "intron"),
        merge_gene_features=True,
    ):
        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome

        self.data_keys = data_keys
        self.seq_len = seq_len
        self.resolution = resolution
        self.seq_len_bp = self.seq_len * resolution

        self.region_key = region_key
        self.feature_types = tuple(feature_types)
        self.merge_gene_features = merge_gene_features
        return

    def get_regions(self, data_dict):
        """Get regions from data_dict"""
        region_df = self.genome.parse_global_coords(data_dict[self.region_key])
        crop_adjust = ((region_df["End"] - region_df["Start"]) - self.seq_len_bp) // 2
        region_df["Start"] += crop_adjust
        region_df["End"] -= crop_adjust
        return region_df

    def get_region_features_and_masks(self, regions: pd.DataFrame):
        """Get features and masks for each region"""
        total_bins = self.seq_len_bp // self.resolution

        features_col = []
        feature_masks = []
        for _, (chrom, start, end, *_) in regions.iterrows():
            region = chrom, start, end
            features = self.genome.gtf_db.find_region_features(
                region, feature_types=self.feature_types, return_bed=True
            )

            if self.merge_gene_features and (features.shape[0] > 0):
                gene_dfs = []
                for (gene, feature_type), gene_df in features.groupby(
                    ["GeneID", "FeatureType"]
                ):
                    gene_bed = pr.PyRanges(gene_df)
                    gene_bed = gene_bed.merge().df
                    gene_bed["FeatureType"] = feature_type
                    gene_bed["GeneID"] = gene
                    gene_dfs.append(gene_bed)
                features = pd.concat(gene_dfs)

            features["StartBin"] = (features["Start"] - start) // self.resolution
            features["StartBin"] = features["StartBin"].clip(0, total_bins)
            features["EndBin"] = (features["End"] - start) // self.resolution
            features["EndBin"] = features["EndBin"].clip(0, total_bins)

            feature_mask = np.zeros((features.shape[0], total_bins), dtype="bool")
            for i, (start, end) in enumerate(features[["StartBin", "EndBin"]].values):
                feature_mask[i, start:end] = True

            features_col.append(features)
            feature_masks.append(feature_mask)

        feature_masks = np.array(feature_masks)  # (bs, n_features, n_bins)
        return features_col, feature_masks

    def get_feature_level_data(self, data, feature_masks):
        """Get feature level data from raw data"""
        result = np.einsum("bcs, bfs -> bcf", data, feature_masks)
        return result

    def __call__(self, data_dict, suffix="gene"):
        """Summarize gene data"""
        regions = self.get_regions(data_dict)
        features, feature_masks = self.get_region_features_and_masks(regions)
        feature_data_dict = {"features": features}
        for key in self.data_keys:
            data = data_dict[key]
            feature_data = self.get_feature_level_data(data, feature_masks)
            feature_data_dict[f"{key}_{suffix}"] = feature_data
        data_dict.update(feature_data_dict)
        return data_dict


class PeakDataSummary(GeneDataSummary):
    def __init__(
        self,
        data_keys,
        genome: Genome,
        peak_bed,
        region_key="region",
        resolution=32,
        seq_len=16352,
    ):
        super().__init__(
            data_keys=data_keys,
            genome=genome,
            region_key=region_key,
            resolution=resolution,
            seq_len=seq_len,
        )

        if isinstance(peak_bed, str):
            peak_bed = pr.read_bed(peak_bed).df
        if "Name" not in peak_bed.columns:
            peak_bed["Name"] = (
                peak_bed["Chromosome"]
                + ":"
                + peak_bed["Start"].astype(str)
                + "-"
                + peak_bed["End"].astype(str)
            )

        self.peak_bed: pd.DataFrame = peak_bed
        return

    def select_peaks(self, chrom, start, end):
        """Select peaks within the region"""
        peaks = self.peak_bed.loc[
            (self.peak_bed["Chromosome"] == chrom)
            & (self.peak_bed["Start"] >= start)
            & (self.peak_bed["End"] <= end)
        ].copy()
        return peaks

    def get_region_features_and_masks(self, regions: pd.DataFrame):
        """Get features and masks for each region"""
        total_bins = self.seq_len_bp // self.resolution

        features_col = []
        feature_masks = []
        for _, (chrom, start, end, *_) in regions.iterrows():
            features = self.select_peaks(chrom, start, end)
            features["StartBin"] = (features["Start"] - start) // self.resolution
            features["StartBin"] = features["StartBin"].clip(0, total_bins)
            features["EndBin"] = (features["End"] - start) // self.resolution
            features["EndBin"] = features["EndBin"].clip(0, total_bins)

            feature_mask = np.zeros((features.shape[0], total_bins), dtype="bool")
            for i, (start, end) in enumerate(features[["StartBin", "EndBin"]].values):
                feature_mask[i, start:end] = True

            features_col.append(features)
            feature_masks.append(feature_mask)

        feature_masks = np.array(feature_masks)  # (bs, n_features, n_bins)
        return features_col, feature_masks

    def __call__(self, data_dict, suffix="peak"):
        """Summarize peak data"""
        return super().__call__(data_dict, suffix=suffix)


def _clip_at_center(data, clip_length):
    seq_len = data.shape[-1]
    radius = clip_length // 2
    start = seq_len // 2 - radius
    end = start + clip_length
    return data[..., start:end]


def _clip_bed_region_at_center(bed, clip_length):
    seq_len = bed["End"] - bed["Start"]
    center = bed["Start"] + seq_len // 2
    radius = clip_length // 2
    bed["Start"] = center - radius
    bed["End"] = bed["Start"] + clip_length
    return bed


def _project_attr(attr, dna_one_hot):
    # (bs, 1, seq_len) or (sample, bs, 1, seq_len)
    attr = (attr * dna_one_hot).sum(-2)
    return attr


class BorzoiInferencer:
    def __init__(
        self,
        genome,
        checkpoint_path,
        peak_length=512,
        attr_length=1024,
        batch_size=8,
    ):
        if isinstance(genome, str):
            genome = Genome(genome)
        self.genome = genome
        _ = self.genome.genome_one_hot

        self.model = self._load_model(checkpoint_path)

        self.model_dna_length = 524288
        self.model_resolution = 32
        self.peak_length = peak_length
        self.peak_bins = peak_length // self.model_resolution
        self.attr_length = attr_length
        self.batch_size = batch_size
        return

    def _load_model(self, path):
        """Load model for inference."""
        model = torch.load(path, weights_only=False).cuda()
        model = model.eval()
        return model

    def _prepare_bed(self, bed):
        """Prepare peak bed centered with borzoi region length."""
        bed = understand_regions(bed)
        attr_bed = self.genome.standard_region_length(
            bed,
            length=self.model_dna_length,
            keep_original=True,
        )
        return attr_bed

    def _collapse_model(self, emb):
        """Collapse model for inference given an embedding vector."""
        if isinstance(emb, pd.Series):
            emb = torch.from_numpy(emb.values)
        elif isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb)
        emb = emb.cuda().unsqueeze(0)

        emb_model = self.model.collapse_lora(emb)
        emb_model = emb_model.eval().cuda()
        return emb_model

    def _get_dna_one_hot(self, bed) -> torch.Tensor:
        dna = self.genome.get_regions_one_hot(bed)
        dna = torch.from_numpy(dna).cuda().swapaxes(1, 2)
        dna = dna.half()
        dna.requires_grad = True
        return dna

    def _single_attribute(self, model, bed):
        """Run attribution for a single embedding."""

        def _forward_hook(dna):
            with torch.amp.autocast("cuda"):
                outputs = model(dna)  # Shape: (batch_size, 1, 16352)
            # Shape: (batch_size, )
            outputs = _clip_at_center(outputs, self.peak_bins)
            peak_outputs = outputs.sum(dim=-1)
            return peak_outputs

        input_x_gradient = InputXGradient(_forward_hook)

        all_data = []
        for i in range(0, bed.shape[0], self.batch_size):
            batch_bed = bed.iloc[i : i + self.batch_size]
            batch_dna = self._get_dna_one_hot(batch_bed)
            # shape: (batch_size, 4, 524288)
            data = input_x_gradient.attribute(inputs=batch_dna)
            data = _clip_at_center(data, self.attr_length)

            all_data.append(data.detach().cpu())
        all_data = torch.cat(all_data, dim=0)
        return all_data

    @torch.no_grad()
    def _single_predict(self, model, bed):
        all_data = []
        for i in range(0, bed.shape[0], self.batch_size):
            batch_bed = bed.iloc[i : i + self.batch_size]
            batch_dna = self._get_dna_one_hot(batch_bed)
            with torch.amp.autocast("cuda"):
                outputs = model(batch_dna)
            outputs = _clip_at_center(outputs, self.peak_bins)
            all_data.append(outputs.cpu())
        all_data = torch.cat(all_data, dim=0)
        return all_data

    def infer(
        self, embedding: pd.DataFrame, bed: str, mode="attr", progress_bar=True
    ) -> xr.Dataset:
        """Inference for given embedding and bed."""
        bed = self._prepare_bed(bed)

        if mode == "attr":
            infer_func = self._single_attribute
        elif mode == "pred":
            infer_func = self._single_predict
        else:
            raise ValueError(f"Unknown mode {mode}")

        final_data = []
        for _, single_emb in tqdm(embedding.iterrows(), disable=not progress_bar):
            emb_model = self._collapse_model(single_emb)
            data = infer_func(emb_model, bed)
            final_data.append(data)

            del emb_model
            torch.cuda.empty_cache()

        # concat along the sample dim, resulting (sample, bs, c, seq_len)
        final_data = torch.stack(final_data, dim=0).numpy()

        # prepare xarray
        region_index = pd.Index(bed["Original_Name"].values)
        bed = bed.set_index("Original_Name")
        bed["Name"] = bed.index
        bed.index.name = "region"

        da = xr.DataArray(
            final_data,
            dims=["sample", "region", "channel", "pos"],
            coords={
                "sample": embedding.index,
                "region": region_index,
            },
        )
        bed = _clip_bed_region_at_center(bed, self.attr_length)
        for col, v in bed.items():
            if col == "Chromosome":
                v = v.astype(str)
            da.coords[col] = ("region", v.values)
        ds = da.to_dataset(name=mode)

        ds = BorzoiInferenceDataset(ds, self.genome)
        return ds

    def infer_offline(self, embedding, bed, output_dir, emb_chunk=10, mode="attr"):
        """Inference for given embedding and bed, save to output_dir."""
        output_dir = pathlib.Path(output_dir) / mode
        output_dir.mkdir(parents=True, exist_ok=True)
        success_flag = output_dir / ".success"
        if success_flag.exists():
            print(f"Output directory {output_dir} already success. Skipping.")

        for emb_chunk_i in range(0, len(embedding), emb_chunk):
            emb_chunk_df = embedding.iloc[emb_chunk_i : emb_chunk_i + emb_chunk]

            chunk_out_path = output_dir / f"emb_{emb_chunk_i}.zarr"
            if chunk_out_path.exists():
                continue

            # attribute chunk
            ds = self.infer(emb_chunk_df, bed, mode=mode, progress_bar=False)
            ds = ds.dataset

            # save chunk
            temp_out_path = pathlib.Path(f"{chunk_out_path}.temp")

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=xr.SerializationWarning)
                ds = ds.chunk(ds.sizes)
                ds.to_zarr(temp_out_path, mode="w")
            temp_out_path.rename(chunk_out_path)

        success_flag.touch()
        return


class BorzoiInferenceDataset:
    def __init__(self, dataset, genome):
        self.dataset = dataset
        if isinstance(genome, str):
            genome = Genome(genome)
            _ = genome.genome_one_hot
        self.genome = genome

        self.regions = dataset.get_index("region")
        self.region_bed = self._get_region_bed()
        self.samples = dataset.get_index("sample")
        return

    def _get_region_bed(self):
        region_bed = pd.DataFrame(
            {k: self.dataset.coords[k].values for k in ["Chromosome", "Start", "End"]}
        )
        region_bed.index = self.regions
        return region_bed

    def get_attr_and_seq(self, region=None, sample=None, project=True):
        """
        Get attribution and sequence for given regions and samples.

        Parameters
        ----------
        region : str or list of str, optional
            Region name. Default is None.
        sample : str or list of str, optional
            Sample name. Default is None.
        project : bool, optional
            Project attribution to 1D sequence. Default is True.
        """
        sel_dict = {}
        if region is not None:
            sel_dict["region"] = region
        if sample is not None:
            sel_dict["sample"] = sample
        da = self.dataset["attr"].sel(**sel_dict)

        # region and one hot
        regions = da.get_index("region")
        # shape: (n_sample, n_regions, 4, seq_len)
        dna_one_hot = self.genome.get_regions_one_hot(
            self.region_bed.loc[regions]
        ).swapaxes(1, 2)

        # attr
        # shape: (n_sample, n_regions, 4, attr_len)
        attr = da.values
        if project:
            # shape: (n_sample, n_regions, 1, attr_len)
            attr = _project_attr(attr, dna_one_hot[np.newaxis, ...])
        return attr, dna_one_hot

    def __getitem__(self, key):
        return self.dataset[key]

    def __repr__(self):
        return self.dataset.__repr__()

    def to_zarr(self, *args, **kwargs):
        """Save dataset to zarr."""
        return self.dataset.to_zarr(*args, **kwargs)

    @staticmethod
    def select_motifs_by_attr(
        motifs, attr_score, attr_cutoff=3, overlap_ratio_cutoff=0.3
    ):
        """Select top motifs by attribution score."""
        if attr_score.ndim == 2:
            attr_score = np.abs(attr_score).max(axis=0)

        motifs["attr_score"] = motifs.apply(
            lambda row: attr_score[row["start"] : row["end"]].mean(), axis=1
        )
        motifs = motifs[motifs["attr_score"] > attr_cutoff].sort_values(
            "attr_score", ascending=False
        )

        selected_motifs = []
        pos_min = motifs["start"].min()
        pos_max = motifs["end"].max()
        covered_positions = np.zeros(pos_max - pos_min)

        for _, motif in motifs.iterrows():
            start, end = motif["start"], motif["end"]
            start = start - pos_min
            end = end - pos_min

            covered_bp = covered_positions[start:end].sum()
            covered_ratio = covered_bp / (end - start)
            if covered_ratio < overlap_ratio_cutoff:
                selected_motifs.append(motif)
                covered_positions[start:end] += 1

        selected_motifs_df = pd.DataFrame(selected_motifs)
        return selected_motifs_df
