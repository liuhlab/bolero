import pathlib
import warnings

import numpy as np
import pandas as pd
import pyranges as pr
import torch
import xarray as xr
from captum.attr import InputXGradient
from einops import rearrange
from tangermeme.ersatz import dinucleotide_shuffle, multisubstitute, substitute
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


def _clip_at_center(data, clip_length, modality="atac"):
    if isinstance(data, dict):
        data = data[modality]
        seq_len = data.shape[-1]
        radius = clip_length // 2
        start = seq_len // 2 - radius
        end = start + clip_length
        return data[..., start:end]

    else:
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


def prepare_marginalize_one_hot(
    genome,
    bed,
    n_regions,
    shuffle_size,
    n_shuffle=1,
    random_state=0,
    motifs=None,
    distances=None,
    seq_fold=16,
):
    """
    Take sequence from real genome regions, then perform dinucleotide shuffle and insert motif(s).

    Parameters
    ----------
    genome : Genome
        Genome object.
    bed : pd.DataFrame
        Bed file to sample regions to prepare dna one hot.
    n_regions : int
        Number of regions to sample.
    shuffle_size : int
        Only dinuc shuffle this size of the region in the middle.
    n_shuffle : int, optional
        Number of shuffle per region. Default is 1.
    random_state : int, optional
        Random state. Default is 0.
    motifs : str or list of str, optional
        Motif to insert. Default is None.
    distances : int or list of int, optional
        Distance between multiple motifs. Default is None.
    seq_fold : int, optional
        Fold sequence and then shuffle and insert motifs. Default is 16.
    """
    # get region one hot
    # dinucleotide shuffle the middle shuffle_size region
    bed = bed.sample(n_regions, random_state=random_state)
    rand_one_hot = genome.get_regions_one_hot(bed)
    rand_one_hot = rand_one_hot.swapaxes(1, 2).astype("float16")
    rand_one_hot = torch.from_numpy(rand_one_hot).half()

    if seq_fold > 1:
        rand_one_hot = rearrange(rand_one_hot, "b c (f l) -> (b f) c l", f=seq_fold)

    center = rand_one_hot.shape[-1] // 2
    shuffle_radius = shuffle_size // 2
    shuffle_range = (center - shuffle_radius, center + shuffle_radius)

    # select part of the sequence to shuffle
    if shuffle_range is None:
        to_shuffle = rand_one_hot
    else:
        ss, se = shuffle_range
        to_shuffle = rand_one_hot[..., ss:se].clone()
    to_shuffle = fill_n_with_random_base(to_shuffle)
    shuffle_rand_one_hot = dinucleotide_shuffle(
        to_shuffle, n=n_shuffle, random_state=random_state
    ).view(-1, 4, to_shuffle.shape[-1])

    # insert motif into the center of the region
    if motifs is not None:
        if isinstance(motifs, str):
            # substitute single motif into region center
            substitute_start = shuffle_rand_one_hot.shape[-1] // 2 - len(motifs) // 2
            shuffle_rand_one_hot = substitute(
                shuffle_rand_one_hot, motifs, start=substitute_start
            )
        else:
            if isinstance(distances, int):
                distances = [distances for _ in range(len(motifs) - 1)]
            motif_lengths = [
                len(motif) if isinstance(motif, str) else motif.shape[-1]
                for motif in motifs
            ]
            total_substitute_length = sum(distances) + sum(motif_lengths)
            substitute_start = (
                shuffle_rand_one_hot.shape[-1] // 2 - total_substitute_length // 2
            )
            shuffle_rand_one_hot = multisubstitute(
                shuffle_rand_one_hot, motifs, distances, start=substitute_start
            )

    # put the shuffled sequence back to the original position
    if shuffle_range is None:
        rand_one_hot = shuffle_rand_one_hot
    else:
        ss, se = shuffle_range
        if n_shuffle > 1:
            rand_one_hot = rand_one_hot.repeat(n_shuffle, 1)
        rand_one_hot[..., ss:se] = shuffle_rand_one_hot

    if seq_fold > 1:
        rand_one_hot = rearrange(
            rand_one_hot, "(b f ns) c l -> (b ns) c (f l)", f=seq_fold, ns=n_shuffle
        )

    # shape (n_region * n_shuffle, 4, region_length)
    return rand_one_hot


def fill_n_with_random_base(one_hot):
    """
    Fill "N" positions with random bases.

    input shape (bs, 4, seq_len)
    """
    mask = one_hot.sum(dim=1) == 0  # Find "N" positions (batch_size, seq_len)
    if mask.any():
        batch_indices, seq_indices = torch.where(mask)  # Get batch & sequence indices
        random_bases = torch.randint(
            0, 4, (batch_indices.numel(),), device=one_hot.device
        )  # Choose random bases
        one_hot[batch_indices, random_bases, seq_indices] = 1  # Assign random bases
    return one_hot


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
        # model = torch.load(path, weights_only=False, map_location=torch.device('cpu'))
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
        # emb = emb.cpu().unsqueeze(0)

        emb_model = self.model.collapse_lora(emb)
        emb_model = emb_model.eval().cuda()
        # emb_model = emb_model.eval().cpu()
        return emb_model

    def _get_dna_one_hot(self, bed, snp_mode=False) -> torch.Tensor:
        dna = self.genome.get_regions_one_hot(bed, snp_mode=snp_mode)
        #returned ref and alt allele one-hots
        if isinstance(dna, dict):
            for k, v in dna.items():
                dna[k] = torch.from_numpy(v).cuda().swapaxes(1, 2)
                dna[k] = dna[k].half()
                dna[k].requires_grad = True
        else:
            dna = torch.from_numpy(dna).cuda().swapaxes(1, 2)
            # dna = torch.from_numpy(dna).cpu().swapaxes(1, 2)
            dna = dna.half()
            dna.requires_grad = True
        return dna


    def _single_attribute(self, model, bed):
        """Run attribution for a single embedding."""

        def _forward_hook(dna):
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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

    def _single_predict(self, model, bed):
        all_data = []
        for i in range(0, bed.shape[0], self.batch_size):
            batch_bed = bed.iloc[i : i + self.batch_size]
            batch_dna = self._get_dna_one_hot(batch_bed)

            outputs = self._forward_pass(model, batch_dna)
            outputs = _clip_at_center(outputs, self.peak_bins)
            all_data.append(outputs.cpu())
        all_data = torch.cat(all_data, dim=0)
        return all_data

    def _snp_predict(self, model, bed, modality='atac'):
        #1. batch ref batch alt
        #2. Concatenate
        #3. create dictionary of concats
        all_data = {'ref': [], 'alt': []}
        for i in range(0, bed.shape[0], self.batch_size):
            batch_bed = bed.iloc[i : i + self.batch_size]

            batch = self._get_dna_one_hot(batch_bed, snp_mode=True)
            outputs_ref = self._forward_pass(model, batch['ref'])
            outputs_alt = self._forward_pass(model, batch['alt'])
            
            # import pdb; breakpoint()
            # outputs = _clip_at_center(outputs, self.peak_bins)
            all_data['ref'].append(outputs_ref[modality].cpu())
            all_data['alt'].append(outputs_alt[modality].cpu())
        all_data['ref'] = torch.cat(all_data['ref'], dim=0)
        all_data['alt'] = torch.cat(all_data['alt'], dim=0)

        return all_data

    @torch.no_grad()
    def _forward_pass(self, model, data):

        # Determine device from data
        device = data.device
        
        if device.type == "cuda":
            # Use mixed precision for CUDA
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(data)
        else:
            # Regular precision for CPU
            outputs = model(data)
        
        return outputs

    def _marginalize(
        self,
        model,
        motifs,
        distances,
        region_bed,
        n_regions,
        random_state=0,
        shuffle_size=2048,
        reduce=True,
        seq_fold=16,
    ):
        n_batches = max(1, n_regions // self.batch_size // seq_fold)

        null_results = []
        motif_results = []
        for idx in range(n_batches):
            _random_state = random_state + idx
            for _insert in [False, True]:
                one_hot_data = prepare_marginalize_one_hot(
                    genome=self.genome,
                    bed=region_bed,
                    n_regions=self.batch_size,
                    shuffle_size=shuffle_size,
                    n_shuffle=1,
                    random_state=_random_state,
                    motifs=motifs if _insert else None,
                    distances=distances if _insert else None,
                    seq_fold=seq_fold,
                )
                one_hot_data = one_hot_data.cuda()
                result = self._forward_pass(model, one_hot_data)
                if seq_fold > 1:
                    result = rearrange(result, "b c (f l) -> (b f) c l", f=seq_fold)
                result = _clip_at_center(result, self.peak_bins)
                if _insert:
                    motif_results.append(result)
                else:
                    null_results.append(result)
        null_results = torch.cat(null_results, dim=0)
        motif_results = torch.cat(motif_results, dim=0)
        if reduce:
            # sum bins to get region level score
            null_results = null_results.sum(dim=(1, 2))
            motif_results = motif_results.sum(dim=(1, 2))
        return null_results, motif_results

    @staticmethod
    def seqlet_to_seq(seqlet, **kwargs):
        """Convert seqlet to sequence."""
        if isinstance(seqlet, str):
            if kwargs.get("rc", False):
                seqlet = seqlet.translate(str.maketrans("ACGT", "TGCA"))[::-1]
            return seqlet
        else:
            return seqlet.get_consensus_sequence(**kwargs)

    def marginalize_seqlet_pair(
        self,
        embedding,
        region_bed,
        seqlet1,
        seqlet2=None,
        distances="default",
        n_regions=100,
        random_state=0,
        shuffle_size=20000,
        trim_threshold=0.3,
        seq_fold=16,
    ):
        """Marginalize seqlet pair for given embedding and region bed."""
        # prepare seqlet and distances
        if seqlet2 is None:
            seqlet2 = seqlet1

        if distances == "default":
            # 22 distances * 100 regions * 3 orientation = 6600 samples
            distances = (
                list(range(0, 10)) + list(range(10, 60, 10)) + list(range(60, 151, 15))
            )

        # prepare region and model
        bed = understand_regions(region_bed)
        region_bed = self.genome.standard_region_length(
            bed, self.model_dna_length, remove_blacklist=True
        )
        emb_model = self._collapse_model(embedding)

        total_results = {}

        # run single motif
        seq1 = self.seqlet_to_seq(seqlet1, trim_threshold=trim_threshold)
        seq2 = self.seqlet_to_seq(seqlet2, trim_threshold=trim_threshold)
        for idx, seq in enumerate([seq1, seq2]):
            if (idx == 1) and (seq1 == seq2):
                total_results["single:seqlet1"] = total_results["single:seqlet0"]
                continue

            null_results, motif_results = self._marginalize(
                model=emb_model,
                motifs=seq,
                distances=None,
                region_bed=region_bed,
                n_regions=n_regions * 2,
                random_state=random_state,
                shuffle_size=shuffle_size,
                seq_fold=seq_fold,
            )
            total_results[f"single:seqlet{idx}"] = {
                "null": null_results.cpu().numpy(),
                "motif": motif_results.cpu().numpy(),
            }

        # run seqlet pair
        seq_strands = [(0, 0), (0, 1), (1, 0)]
        for strand1, strand2 in seq_strands:
            motifs = [
                self.seqlet_to_seq(
                    seqlet1, trim_threshold=trim_threshold, rc=strand1 == 1
                ),
                self.seqlet_to_seq(
                    seqlet2, trim_threshold=trim_threshold, rc=strand2 == 1
                ),
            ]
            for distance in distances:
                null_results, motif_results = self._marginalize(
                    model=emb_model,
                    motifs=motifs,
                    distances=distance,
                    region_bed=region_bed,
                    n_regions=n_regions * 2,
                    random_state=random_state,
                    shuffle_size=shuffle_size,
                )
                total_results[
                    f"pair:seqlet1_{strand1}:seqlet2_{strand2}:{distance}"
                ] = {
                    "null": null_results.cpu().numpy(),
                    "motif": motif_results.cpu().numpy(),
                }
        return total_results

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

    def infer_snp(
        self, embedding: pd.DataFrame, bed: str, mode="attr", progress_bar=True
    ) -> xr.Dataset:
        """Inference of variant effect for given embedding and bed."""

        bed = pd.read_csv(bed, header=0, sep="\t")
        
        # Standardize columns
        _columns = ["Chromosome", "Start", "End", "peak-start", "peak-end", "ref", "snp", "pos2start", "beta", "id"]
        bed.columns = _columns
        
        final_data = []
        for _, single_emb in tqdm(embedding.iterrows(), disable=not progress_bar):
            emb_model = self._collapse_model(single_emb)
            data = self._snp_predict(emb_model, bed)
            final_data.append(data)

            del emb_model
            torch.cuda.empty_cache()

        # concat along the sample dim, resulting (sample, bs, c, seq_len)
        # final_data = torch.stack(final_data, dim=0).numpy()
        # final_data = final_data.numpy()
        # return final_data
        # import pdb;breakpoint()
        # Stack ref and alt predictions separately.
        ref_data = torch.stack([data['ref'] for data in final_data], dim=0).cpu().numpy()  # shape: (sample, region, channel, pos)
        alt_data = torch.stack([data['alt'] for data in final_data], dim=0).cpu().numpy()  # shape: (sample, region, channel, pos)
        
        # Create a region index from the BED file (using its row order).
        region_index = np.arange(len(bed))
        
        # Build an xarray Dataset with separate DataArrays for ref and alt predictions.
        ds = xr.Dataset(
            {
                "ref_prediction": (["sample", "region", "channel", "pos"], ref_data),
                "alt_prediction": (["sample", "region", "channel", "pos"], alt_data)
            },
            coords={
                "sample": embedding.index,
                "region": region_index
            },
            attrs={
                "description": "Variant effect predictions with region information from BED file"
            }
        )
        
        # Attach region metadata from the BED file as coordinates.
        # For each column in the BED file, add a 1D coordinate with dimension "region".
        for col_name in bed.columns:
            col_values = bed[col_name].values
            if col_name == "Chromosome":
                col_values = col_values.astype(str)
            ds.coords[col_name] = (("region",), col_values)
        
        # Optionally, you can also add an "allele" coordinate if you prefer to combine ref and alt 
        # in a single DataArray (here they are stored separately).
        ds = BorzoiInferenceDataset(ds, self.genome)
        return ds


    def infer_offline(self, embedding, bed, output_dir, emb_chunk=10, mode="attr", use_snp=False):
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
            if use_snp:
                ds = self.infer_snp(emb_chunk_df, bed, mode=mode, progress_bar=False)
            else:
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

    def random_activation_iter(
        self,
        embedding,
        bed,
        batch_size=8,
        shuffle=True,
        region_chunk=1000,
        emb_chunk=100,
        pred_cutoff=0.03,
        clip_to=None,
        reshape=True,
    ):
        """
        Get random activation for given embedding and bed for training SAE.

        Parameters
        ----------
        embedding : pd.DataFrame
            Embedding dataframe.
        bed : pd.DataFrame
            Bed file to sample regions to prepare dna one hot.
        batch_size : int, optional
            Batch size for borzoi inference. Default is 8.
        shuffle : bool, optional
            Shuffle embedding and bed. Default is True.
        region_chunk : int, optional
            Region chunk size. Default is 1000.
        emb_chunk : int, optional
            Embedding chunk size. Default is 100.
        pred_cutoff : float, optional
            Output head prediction cutoff. Default is 0.03.
        clip_to : int, optional
            Clip final seq dim to this size. Default is None.
        reshape : bool, optional
            Reshape output to (n_token, act_dim). Default is True.
        """
        bed = self._prepare_bed(bed)
        if shuffle:
            embedding = embedding.sample(frac=1).reset_index(drop=True)
            bed = bed.sample(frac=1).reset_index(drop=True)
        n_embedding = embedding.shape[0]
        n_region = bed.shape[0]

        if n_embedding > emb_chunk or n_region > region_chunk:
            for emb_chunk_i in range(0, n_embedding, emb_chunk):
                for region_chunk_i in range(0, n_region, region_chunk):
                    emb_chunk_df = embedding.iloc[emb_chunk_i : emb_chunk_i + emb_chunk]
                    region_chunk_df = bed.iloc[
                        region_chunk_i : region_chunk_i + region_chunk
                    ]
                    yield from self.random_activation_iter(
                        emb_chunk_df,
                        region_chunk_df,
                        batch_size,
                        shuffle=False,
                        region_chunk=region_chunk,
                        emb_chunk=emb_chunk,
                        pred_cutoff=pred_cutoff,
                        clip_to=clip_to,
                        reshape=reshape,
                    )
        else:
            # create an random shuffle without replace of emb and region combination
            rand_idx = np.random.permutation(n_embedding * n_region)
            rand_region_idx = rand_idx // n_embedding
            rand_emb_idx = rand_idx % n_embedding

            for batch_start in range(0, rand_idx.size, batch_size):
                batch_end = min(batch_start + batch_size, rand_idx.size)
                batch_region_idx = rand_region_idx[batch_start:batch_end]
                batch_emb_idx = rand_emb_idx[batch_start:batch_end]

                emb_batch = embedding.iloc[batch_emb_idx].values
                emb_batch = torch.from_numpy(emb_batch).cuda()
                region_batch = bed.iloc[batch_region_idx]
                dna_one_hot = self._get_dna_one_hot(region_batch)
                dna_one_hot.requires_grad = False
                act = self._model_activation(
                    dna_one_hot, emb_batch, pred_cutoff, clip_to, reshape
                )
                yield act

    def fixed_activation_iter(
        self,
        embedding,
        bed,
        batch_size=8,
        clip_to=None,
    ):
        """
        Get fixed activation for given embedding and bed for inference SAE.

        Parameters
        ----------
        embedding : pd.DataFrame
            Embedding dataframe.
        bed : pd.DataFrame
            Bed file to sample regions to prepare dna one hot.
        batch_size : int, optional
            Batch size for borzoi inference. Default is 8.
        pred_cutoff : float, optional
            Output head prediction cutoff. Default is 0.03.
        clip_to : int, optional
            Clip final seq dim to this size. Default is None.
        """
        bed = self._prepare_bed(bed)
        n_embedding = embedding.shape[0]
        n_region = bed.shape[0]

        for emb_i in range(0, n_embedding):
            for region_chunk_i in range(0, n_region, batch_size):
                region_chunk_df = bed.iloc[region_chunk_i : region_chunk_i + batch_size]
                emb_i = np.array([emb_i] * region_chunk_df.shape[0])
                emb_batch = embedding.iloc[emb_i].values
                emb_batch = torch.from_numpy(emb_batch).cuda()
                dna_one_hot = self._get_dna_one_hot(region_chunk_df)
                dna_one_hot.requires_grad = False
                pred, act = self._model_activation(
                    dna=dna_one_hot, emb=emb_batch, clip_to=clip_to, reshape=False
                )
                data = {
                    "pred": pred,  # tensor (bs, 1, clip_to)
                    "act": act,  # tensor (bs, 1920, clip_to)
                    "region": region_chunk_df,  # pd.DataFrame (bs, 3)
                    "emb": emb_batch.cpu().numpy(),  # np.array (bs, emb_dim)
                    "emb_id": emb_i,  # np.array (bs, )
                }
                yield data

    @torch.inference_mode()
    def _model_activation(self, dna, emb, pred_cutoff=0.1, clip_to=None, reshape=True):
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # pred shape (bs, 1, 16352)
            # act shape (bs, 1920, 16352)
            pred, act = self.model(x=dna, embedding=emb, return_dna_embedding=True)

            if clip_to is not None:
                pred = _clip_at_center(pred, clip_to)
                act = _clip_at_center(act, clip_to)

            if reshape:
                mask = rearrange((pred > pred_cutoff).any(dim=1), "b l -> (b l)")
                act = rearrange(act, "b c l -> (b l) c")

                # keep activation if any channel is above pred_cutoff
                use_act = act[mask]
                return use_act
            else:
                return pred, act


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
