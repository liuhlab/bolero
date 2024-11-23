import numpy as np
import pandas as pd
import pyranges as pr

from bolero.pp.genome import Genome


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
