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

    @torch.no_grad()
    def _forward_pass(self, model, data):

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
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





class BorzoiSNPInferencer(BorzoiInferencer):
    def __init__(
        self,
        genome,
        checkpoint_path,
        peak_length=512,
        attr_length=1024,
        batch_size=8,
    ):
        super().__init__(
            genome=genome,
            checkpoint_path=checkpoint_path,
            peak_length=peak_length,
            attr_length=attr_length,
            batch_size=batch_size,
        )

    def _load_model(self, path):
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        model = torch.load(path, weights_only=False, map_location=torch.device(device))
        model = model.eval()
        return model


    #TODO: use tangermeme package for snp 
    #1. take dna tensor as input
    #2. mutation info as input 
    
    #do mutation after fetching genome
    #IN FUTURE
    #bedfile, mutations vcf format --> dna one hot output
    #mutation overlap, then do tangermeme substitution 
    #vcf file parser exist
    def _get_snp_dna_one_hot(self, bed):

        """Get reference and alternate allele one-hot encodings for SNPs.
        
        Parameters
        ----------
        bed : pd.DataFrame
            Bed file with variant information.
        snp_mode : bool, optional
            Whether to return both reference and alternate alleles. Default is False.
            
        Returns
        -------
        torch.Tensor or dict of torch.Tensor
            One-hot encoded DNA sequences.
        """
        
        # Get reference allele sequences
        
        ref_dna = self.genome.get_regions_one_hot(bed)

        snp_info = bed[["ref", "snp", "pos2start"]]
        
        nucleotide_map = {'A': 0, 'C': 1, 'G': 2, 'T': 3} 
        
        # Create a copy of the one-hot encoding to modify
        alt_dna = ref_dna.copy()
        
        assert len(snp_info) == ref_dna.shape[0], f'Regions != Batch size: {len(snp_info)}, {ref_dna.shape[0]}'

        # Process each sample in the batch with its specific SNP info
        for batch_idx in range(len(snp_info)):

            # Get the SNP info for this specific batch item
            sample_row = snp_info.iloc[batch_idx]
            ref, alt_allele, relative_pos = sample_row
            
            # Convert position to integer if needed
            relative_pos = int(relative_pos)

            # Validate alternate nucleotide
            if alt_allele not in nucleotide_map:
                raise ValueError(f"Unknown nucleotide '{alt_allele}' in SNP data for batch item {batch_idx}")
                
            # For alt allele, set the alternate base
            # Clear the position for this batch item
            alt_dna[batch_idx, relative_pos-1, :] = 0
            # Set the alternate nucleotide
            alt_dna[batch_idx, relative_pos-1, nucleotide_map[alt_allele]] = 1


        def transform_tensor(dna):
            """
            dna is of shape (bs, seqlen, 4)
            outputs dna of shape (bs, 4, seqlen) with correct precision etc. 
            """
            if torch.cuda.is_available():
                dna = torch.from_numpy(dna).cuda().swapaxes(1, 2)
            
            else:
                dna =  torch.from_numpy(dna).cpu().swapaxes(1, 2)
            dna = dna.half()
            dna.requires_grad = True
            return dna
        
        ref_dna = transform_tensor(ref_dna)
        alt_dna = transform_tensor(alt_dna)

        
        return {'ref': ref_dna, 'alt': alt_dna}

    def _snp_predict(self, model, bed, modality='atac', mode='peak'):
        """Run prediction for SNP variants.
        
        Parameters
        ----------
        model : torch.nn.Module
            Collapsed model for prediction.
        bed : pd.DataFrame
            Bed file with variant information.
        modality : str, optional
            Modality to extract from model output. Default is 'atac'.
            
        Returns
        -------
        dict
            Dictionary with reference and alternate predictions.
        """
        all_data = {'ref': [], 'alt': []}
        peak_effects = []
        for i in tqdm(range(0, bed.shape[0], self.batch_size)):
            batch_bed = bed.iloc[i : i + self.batch_size]
            
            batch = self._get_snp_dna_one_hot(batch_bed)
            outputs_ref = self._forward_pass(model, batch['ref'])
            outputs_alt = self._forward_pass(model, batch['alt'])

            
            if mode == 'snp':
                all_data['ref'].append(outputs_ref[modality].cpu())
                all_data['alt'].append(outputs_alt[modality].cpu())


            elif mode == 'peak':
                effect = outputs_alt[modality] - outputs_ref[modality]
                # Extract only the peak regions for each batch item
                batch_peak_effects = []
                for j in range(len(batch_bed)):
                    row = batch_bed.iloc[j]
                    # Calculate relative peak positions
                    # Convert from bp to 32bp bins and account for the 512bp padding
                    relative_peak_start = max(0, (row["peak-start"] - row["Start"] + 512) // 32)
                    relative_peak_end = min(effect.shape[-1] - 2, (row["peak-end"] - row["Start"] + 512) // 32)
                    
                    # Extract peak region effects
                    peak_effect = torch.mean(effect[j, :, relative_peak_start:relative_peak_end+1],axis=-1)
                    batch_peak_effects.append(peak_effect)

                # Stack batch results
                if batch_peak_effects:
                    peak_effects.append(torch.stack(batch_peak_effects, dim=0).cpu())
            
            else:
                raise ValueError(f"Unknown mode {mode}")

        if mode == 'snp':
            all_data['ref'] = torch.cat(all_data['ref'], dim=0)
            all_data['alt'] = torch.cat(all_data['alt'], dim=0)
            return all_data
        elif mode == 'peak':
            return torch.cat(peak_effects, dim=0)
        
    
    def infer_snp(
        self, celltype: str, embedding: np.array, bed_path: str, progress_bar=True, mode='peak'
    ) -> xr.Dataset:
        """Inference of variant effect for given embedding and bed files.
        
        Parameters
        ----------
        celltype : str
            The cell type identifier (e.g., 'ASC')
        embedding : np.array
            Embedding array for the cell type
        bed_path : str
            Path to the bed file with variant information
        progress_bar : bool, optional
            Show progress bar. Default is True.
            
        Returns
        -------
        BorzoiInferenceDataset
            Dataset with SNP effect predictions.
        """
        final_data = []

        # Read and process the BED file
        bed = pd.read_csv(bed_path, header=0, sep="\t")
        
        # Standardize columns
        _columns = ["Chromosome", "Start", "End", "peak-start", "peak-end", "ref", "snp", "pos2start", "beta", "id"]
        bed.columns = _columns
        
        # Process the embedding
        emb_model = self._collapse_model(embedding)
        data = self._snp_predict(emb_model, bed, mode=mode)
        final_data.append(data)
        
        del emb_model
        torch.cuda.empty_cache()
        
        if mode == 'snp':
            # Stack ref and alt predictions separately
            ref_data = torch.stack([data['ref'] for data in final_data], dim=0).cpu().numpy()
            alt_data = torch.stack([data['alt'] for data in final_data], dim=0).cpu().numpy()
            
            # Create a region index from the BED file
            region_index = np.arange(len(bed))
            
            # Build an xarray Dataset
            ds = xr.Dataset(
                {
                    "ref_prediction": (["sample", "region", "channel", "pos"], ref_data),
                    "alt_prediction": (["sample", "region", "channel", "pos"], alt_data)
                },
                coords={
                    "sample": [celltype],  # List with single cell type
                    "region": region_index
                },
                attrs={
                    "description": f"Variant effect predictions for {celltype}"
                }
            )

            assert ref_data.shape[1] == len(bed), "Mismatch between prediction regions and bed file rows"

        elif mode == 'peak':
            peak_effects_np = data.numpy()
            # Get dimensions
            n_regions, n_peaks = peak_effects_np.shape

            # Create a region index from the BED file
            region_index = np.arange(len(bed))
            
            # Build an xarray Dataset
            ds = xr.Dataset(
                {
                    "peak_effect": (["region", "peak"], peak_effects_np),
                },
                coords={
                    "sample": [celltype],  # List with single cell type
                    "region": region_index
                },
                attrs={
                    "description": f"Peak effect predictions for {celltype}"
                }
            )
        
        # Attach region metadata
        for col_name in bed.columns:
            col_values = bed[col_name].values
            if col_name == "Chromosome":
                col_values = col_values.astype(str)
            ds.coords[col_name] = (("region",), col_values)

        if mode == 'snp':
            # Calculate effect scores (alternate - reference)
            ds["effect"] = ds["alt_prediction"] - ds["ref_prediction"]
            ds = self.calculate_peak_effect(ds)

        return BorzoiInferenceDataset(ds, self.genome)
        
    def infer_snp_offline(
        self, 
        embedding_bed_dict,
        output_dir,
        experiment_name,
        progress_bar=True,
        mode='peak',
    ):
        """Inference for SNP variants, saving results to output_dir for each cell type.
        
        Parameters
        ----------
        embedding_bed_dict : dict
            Dictionary where:
            - Keys are cell types (e.g., 'ASC')
            - Values are dictionaries with:
                - 'path': path to the bed file with variant information
                - 'embedding': embedding dataframe for that cell type
        output_dir : str
            Path to output directory.
        experiment_name : str
            Name of the experiment for the subfolder.
        progress_bar : bool, optional
            Show progress bar. Default is True.
        """
        # Create main output directory with experiment name
        output_dir = pathlib.Path(output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a success flag for the entire process
        # success_flag = output_dir / ".success"
        # if success_flag.exists():
        #     print(f"Output directory {output_dir} already has success flag. Skipping.")
        #     return

        # Process each cell type
        for cell_type, data_dict in tqdm(embedding_bed_dict.items(), disable=not progress_bar, 
                                        desc="Processing cell types"):
            
            # Create cell type specific directory
            cell_type_dir = output_dir / cell_type
            cell_type_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this cell type has already been processed
            # cell_type_success = cell_type_dir / ".success"
            # if cell_type_success.exists():
            #     print(f"Cell type {cell_type} already processed. Skipping.")
            #     continue
            
            embedding = data_dict['embedding']
            bed_path = data_dict['path']
            try:
                # Infer SNP effects for this cell type
                ds = self.infer_snp(cell_type, embedding, bed_path, progress_bar=progress_bar, mode=mode)
                
                # Get the xarray dataset
                ds = ds.dataset
                
                # Output path for this cell type
                zarr_path = cell_type_dir / f"{cell_type}_{mode}.zarr"
                temp_out_path = pathlib.Path(f"{zarr_path}.temp")
                
                # Save dataset to zarr format
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=xr.SerializationWarning)
                    # Chunk the dataset for efficient storage and access
                    ds = ds.chunk(ds.sizes)
                    ds.to_zarr(temp_out_path, mode="w")
                
                # Rename the temp file to final path
                temp_out_path.rename(zarr_path)
                
                # Mark this cell type as successfully processed
                # cell_type_success.touch()
                
                print(f"Successfully processed cell type: {cell_type}")
                
            except Exception as e:
                print(f"Error processing cell type {cell_type}: {str(e)}")
                # Continue with other cell types even if one fails
                continue
            
            # Clear memory
            torch.cuda.empty_cache()
        
        # Mark the entire process as complete
        # success_flag.touch()
        print(f"Completed processing all cell types for experiment: {experiment_name}")
        return
    
    def calculate_peak_effect(self, ds, peak_length=16):
        """Calculate effect differences specifically at peak regions.
        
        This method takes a dataset generated by infer_snp and adds two new variables:
        1. peak_effect: average effect across the entire peak region
        2. snp_effect: effect specifically at the SNP position
        
        Parameters
        ----------
        ds : xr.Dataset or BorzoiInferenceDataset
            Dataset with variant effect predictions.
            
        Returns
        -------
        xr.Dataset or BorzoiInferenceDataset
            Dataset with added peak_effect and snp_effect variables.
        """
        # If input is a BorzoiInferenceDataset, extract the xarray dataset
        if hasattr(ds, 'dataset'):
            is_borzoi_dataset = True
            xr_ds = ds.dataset
            genome = ds.genome
        else:
            is_borzoi_dataset = False
            xr_ds = ds
        
        
        # Extract region start positions
        region_starts = xr_ds.coords["Start"].values
        
        # Extract peak region coordinates
        peak_starts = xr_ds.coords["peak-start"].values
        peak_ends = xr_ds.coords["peak-end"].values
        
        # Calculate relative peak positions within each region
        # First, make peak positions relative to the region start position
        relative_peak_starts = (peak_starts - (region_starts + 512)) // 32
        relative_peak_ends = (peak_ends - (region_starts + 512)) // 32
        
        # Then convert to 32bp resolution, accounting for the 512bp padding
        # 512bp is added to the actual region start position
        peak_start_indices = np.maximum(0, relative_peak_starts)
        peak_end_indices = np.minimum(16350, relative_peak_ends)
        
        num_samples = xr_ds.dims["sample"]
        num_regions = xr_ds.dims["region"]
        num_channels = xr_ds.dims["channel"]
        
        # Create array to store peak region effects
        # Using numpy array first as it's easier to fill
        peak_values = np.zeros((num_samples, num_regions, num_channels, peak_length))
        
        
        # Extract peak effects for each region
        for region_idx in range(num_regions):
            start_idx = peak_start_indices[region_idx]
            end_idx = peak_end_indices[region_idx]

            # Get effect values from dataset (using numpy for speed)
            region_effect = xr_ds.effect.isel(region=region_idx).values

            # Extract just the peak region
            peak_effect = region_effect[:, :, start_idx:end_idx+1]
            
            peak_values[:, region_idx, :, :] = peak_effect

        # Add the peak region effects to dataset
        xr_ds["peak_effect_values"] = (
            ["sample", "region", "channel", "peak_pos"], 
            peak_values,
            {"description": "Effect values within peak regions"}
        )

        # Return the appropriate type
        if is_borzoi_dataset:
            return BorzoiInferenceDataset(xr_ds, genome)
        else:
            return xr_ds


    def _peak_predict(self, model, bed, modality='atac'):
        """Run prediction for peak regions only.
        
        Parameters
        ----------
        model : torch.nn.Module
            Collapsed model for prediction.
        bed : pd.DataFrame
            Bed file with variant information.
        modality : str, optional
            Modality to extract from model output. Default is 'atac'.
            
        Returns
        -------
        torch.Tensor
            Tensor containing only the peak effect values (alt - ref).
        """
        peak_effects = []
        
        for i in tqdm(range(0, bed.shape[0], self.batch_size)):
            batch_bed = bed.iloc[i : i + self.batch_size]
            
            # Get reference and alternate sequences
            batch = self._get_snp_dna_one_hot(batch_bed)
            
            # Get predictions
            outputs_ref = self._forward_pass(model, batch['ref'])
            outputs_alt = self._forward_pass(model, batch['alt'])
            
            # Calculate effect (alt - ref)
            effect = outputs_alt[modality] - outputs_ref[modality]
            
            # Extract only the peak regions for each batch item
            batch_peak_effects = []
            for j in range(len(batch_bed)):
                row = batch_bed.iloc[j]
                # Calculate relative peak positions
                # Convert from bp to 32bp bins and account for the 512bp padding
                relative_peak_start = max(0, (row["peak-start"] - row["Start"] + 512) // 32)
                relative_peak_end = min(effect.shape[-1] - 2, (row["peak-end"] - row["Start"] + 512) // 32)
                
                # Extract peak region effects
                peak_effect = torch.mean(effect[j, :, relative_peak_start:relative_peak_end+1],axis=-1)
                batch_peak_effects.append(peak_effect)
            
            # Stack batch results
            if batch_peak_effects:
                peak_effects.append(torch.stack(batch_peak_effects, dim=0).cpu())
        
        # Combine all batch results
        if peak_effects:
            return torch.cat(peak_effects, dim=0)
        else:
            return None


    def _snp_predict_with_peak_effect(self, model, bed, modality='atac', output_dir=None, celltype=None, checkpoint_interval=10000):
        """Run prediction for SNP variants, calculating and saving peak effects and beta values.
        
        Parameters
        ----------
        model : torch.nn.Module
            Collapsed model for prediction.
        bed : pd.DataFrame
            Bed file with variant information.
        modality : str, optional
            Modality to extract from model output. Default is 'atac'.
        output_dir : str or pathlib.Path, optional
            Directory to save checkpoints. If None, checkpoints are not saved.
        celltype : str, optional
            Cell type identifier for checkpoint filenames.
        checkpoint_interval : int, optional
            Number of regions to process before saving a checkpoint. Default is 10000.
            
        Returns
        -------
        dict
            Dictionary with 'peak_values' and 'beta' arrays, both in the same order.
        """
        import pickle
        import pathlib
        
        # Initialize our simple result structure
        result = {
            'peak_values': [],
            'beta': [],
            'dist_to_peak': [],
            'peak_size': []
        }
        
        # For checkpointing
        regions_processed = 0
        checkpoint_count = 0
        
        # Prepare checkpoint directory if provided
        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a checkpoint-specific directory
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create or load progress tracker
            progress_file = checkpoint_dir / f"{celltype or 'unknown'}_progress.txt"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    regions_processed = int(f.read().strip())
                    print(f"Resuming from {regions_processed} processed regions")
        
        # Skip already processed regions
        if regions_processed > 0:
            print(f"Skipping first {regions_processed} regions (already processed)")
        
        for i in tqdm(range(0, bed.shape[0], self.batch_size)):
            # Skip if we've already processed these regions
            if i < regions_processed:
                continue
                
            batch_bed = bed.iloc[i : i + self.batch_size]
            
            # Get reference and alternate sequences
            batch = self._get_snp_dna_one_hot(batch_bed)
            
            # Get predictions
            outputs_ref = self._forward_pass(model, batch['ref'])
            outputs_alt = self._forward_pass(model, batch['alt'])
            
            # Calculate effect (alt - ref) immediately
            batch_effect = outputs_alt[modality] - outputs_ref[modality]
            
            # Process each sample in the batch
            for j in range(len(batch_bed)):
                row = batch_bed.iloc[j]
                
                # Calculate relative peak positions within the sequence
                # Convert from bp to 32bp bins and account for the 512bp padding
                relative_peak_start = max(0, (row["peak-start"] - row["Start"] + 512) // 32)
                relative_peak_end = min(batch_effect.shape[-1] - 1, (row["peak-end"] - row["Start"] + 512) // 32)
                
                # Extract peak region effects and calculate the mean
                # This preserves all channels but averages across the peak positions
                peak_effect = torch.mean(batch_effect[j, :, relative_peak_start:relative_peak_end+1], dim=-1)
                
                # Store peak effect and corresponding beta value
                result['peak_values'].append(peak_effect.cpu().detach().numpy())
                result['beta'].append(row['beta'])
                result['peak_size'].append(row["peak-end"] - row["peak-start"])
                midpoint = np.abs((row["peak-end"] - row["peak-start"]) // 2 + row["peak-start"])
                result['dist_to_peak'].append(np.abs(midpoint - (row['pos2start']+row["Start"])))
                
                # Increment regions processed counter
                regions_processed += 1
            
            # Save checkpoint if interval reached
            if output_dir is not None and regions_processed // checkpoint_interval > checkpoint_count:
                checkpoint_count = regions_processed // checkpoint_interval
                
                # Create checkpoint filename
                checkpoint_file = checkpoint_dir / f"{celltype or 'unknown'}_checkpoint_{regions_processed}.pkl"
                
                # Convert current results to numpy arrays for saving
                temp_result = {
                    'peak_values': np.array(result['peak_values']),
                    'beta': np.array(result['beta']),
                    'peak_size': np.array(result['peak_size']),
                    'dist_to_peak': np.array(result['dist_to_peak']),
                    'regions_processed': regions_processed,
                    'celltype': celltype
                }
                
                # Save checkpoint
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(temp_result, f)
                
                # Update progress file
                with open(progress_file, 'w') as f:
                    f.write(str(regions_processed))
                    
                print(f"Checkpoint saved at {regions_processed} regions")
                
                # Clear memory for tensors that have been saved
                torch.cuda.empty_cache()
        
        # Convert lists to numpy arrays for easier handling
        result['peak_values'] = np.array(result['peak_values'])
        result['beta'] = np.array(result['beta'])
        result['dist_to_peak'] = np.array(result['dist_to_peak'])
        result['peak_size'] = np.array(result['peak_size'])
        
        # Save final result if checkpointing was enabled
        if output_dir is not None:
            final_file = output_dir / f"{celltype or 'unknown'}_final_result.pkl"
            
            # Add metadata to result
            result['regions_processed'] = regions_processed
            result['celltype'] = celltype
            
            # Save final result
            with open(final_file, 'wb') as f:
                pickle.dump(result, f)
            
            print(f"Final results saved to {final_file}")
        
        return result


    def infer_snp_simple(
        self, celltype: str, embedding: np.array, bed_path: str, output_dir=None, progress_bar=True, checkpoint_interval=10000
    ) -> dict:
        """Simplified inference of variant effect that returns a dictionary with peak values and beta values.
        
        Parameters
        ----------
        celltype : str
            The cell type identifier (e.g., 'ASC')
        embedding : np.array
            Embedding array for the cell type
        bed_path : str
            Path to the bed file with variant information
        output_dir : str or pathlib.Path, optional
            Directory to save checkpoints. If None, checkpoints are not saved.
        progress_bar : bool, optional
            Show progress bar. Default is True.
        checkpoint_interval : int, optional
            Number of regions to process before saving a checkpoint. Default is 10000.
            
        Returns
        -------
        dict
            Dictionary containing 'peak_values' and 'beta' arrays in corresponding order.
        """
        # Read and process the BED file
        bed = pd.read_csv(bed_path, header=0, sep="\t")
        
        # Standardize columns
        _columns = ["Chromosome", "Start", "End", "peak-start", "peak-end", "ref", "snp", "pos2start", "beta", "id"]
        bed.columns = _columns
        
        # Process the embedding
        emb_model = self._collapse_model(embedding)
        
        # Use the optimized SNP predict function that calculates peak effects during the forward pass
        # and returns a simple dictionary structure
        result = self._snp_predict_with_peak_effect(
            emb_model, 
            bed, 
            output_dir=output_dir,
            celltype=celltype,
            checkpoint_interval=checkpoint_interval
        )
        
        # Clean up
        del emb_model
        torch.cuda.empty_cache()
        
        # Add the celltype to the result for reference if not already added
        if 'celltype' not in result:
            result['celltype'] = celltype
        
        return result

    def infer_peak_effect_offline_chunked(
        self, 
        embedding_bed_dict,
        output_dir,
        experiment_name,
        progress_bar=True,
        checkpoint_batches=5,  # Save progress after processing this many batches
        resume=True  # Whether to resume from existing progress
    ):
        """Inference for peak effects with simple checkpointing aligned with batch processing.
        
        Parameters
        ----------
        embedding_bed_dict : dict
            Dictionary where:
            - Keys are cell types (e.g., 'ASC')
            - Values are dictionaries with:
                - 'path': path to the bed file with variant information
                - 'embedding': embedding dataframe for that cell type
        output_dir : str
            Path to output directory.
        experiment_name : str
            Name of the experiment for the subfolder.
        progress_bar : bool, optional
            Show progress bar. Default is True.
        checkpoint_batches : int, optional
            Save progress after processing this many batches. Default is 5.
        resume : bool, optional
            Whether to resume from existing progress. Default is True.
        """
        import json
        
        # Create main output directory with experiment name
        output_dir = pathlib.Path(output_dir) / f"{experiment_name}_peak_effects"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Create a success flag for the entire process
        success_flag = output_dir / ".success"
        if success_flag.exists() and resume:
            print(f"Output directory {output_dir} already has success flag. Skipping.")
            return

        # Process each cell type
        for cell_type, data_dict in tqdm(embedding_bed_dict.items(), disable=not progress_bar, 
                                        desc="Processing cell types for peak effects"):
            
            # Create cell type specific directory
            cell_type_dir = output_dir / cell_type
            cell_type_dir.mkdir(parents=True, exist_ok=True)
            
            # Check if this cell type has already been processed
            cell_type_success = cell_type_dir / ".success"
            if cell_type_success.exists() and resume:
                print(f"Cell type {cell_type} already processed. Skipping.")
                continue
            
            # Path for progress tracking and dataset
            progress_file = cell_type_dir / f"{cell_type}_progress.json"
            zarr_path = cell_type_dir / f"{cell_type}_peak_effects.zarr"
            
            embedding = data_dict['embedding']
            bed_path = data_dict['path']
            
            # Load bed file
            bed = pd.read_csv(bed_path, header=0, sep="\t")
            
            # Standardize columns
            _columns = ["Chromosome", "Start", "End", "peak-start", "peak-end", "ref", "snp", "pos2start", "beta", "id"]
            bed.columns = _columns
            
            # Initialize or load progress tracking
            processed_rows = 0
            if progress_file.exists() and resume and zarr_path.exists():
                try:
                    with open(progress_file, 'r') as f:
                        progress_data = json.load(f)
                        processed_rows = progress_data.get("processed_rows", 0)
                    print(f"Resuming from checkpoint: {processed_rows} rows processed out of {len(bed)}")
                except Exception as e:
                    print(f"Failed to load progress file: {str(e)}. Starting fresh.")
                    processed_rows = 0
            
            # Skip if all rows are processed
            if processed_rows >= len(bed):
                print(f"All rows already processed for {cell_type}.")
                cell_type_success.touch()
                continue
            
            try:
                # Process the embedding
                emb_model = self._collapse_model(embedding)
                
                # Initialize all_results dictionary to store accumulated results
                all_results = {
                    "peak_effect_mean": [],
                    "region_indices": [],
                    "metadata": {}
                }
                
                # Check if we need to load existing results to append to
                if processed_rows > 0 and zarr_path.exists():
                    try:
                        print(f"Loading existing dataset for {cell_type}...")
                        existing_ds = xr.open_zarr(zarr_path)
                        
                        # Extract existing data
                        all_results["peak_effect_mean"] = [existing_ds.peak_effect_mean.values]
                        all_results["region_indices"] = list(existing_ds.region.values)
                        
                        # Extract metadata
                        for col_name in _columns:
                            if col_name in existing_ds.coords:
                                all_results["metadata"][col_name] = list(existing_ds.coords[col_name].values)
                        
                        print(f"Loaded {len(all_results['region_indices'])} existing rows.")
                    except Exception as e:
                        print(f"Error loading existing dataset: {str(e)}. Starting fresh.")
                        processed_rows = 0
                        all_results = {
                            "peak_effect_mean": [],
                            "region_indices": [],
                            "metadata": {col: [] for col in _columns}
                        }
                
                # Process remaining rows in batches, respecting self.batch_size
                batches_since_checkpoint = 0
                total_batches = (len(bed) - processed_rows + self.batch_size - 1) // self.batch_size
                
                for i in tqdm(range(0, total_batches), 
                            desc=f"Processing {cell_type} in batches", 
                            disable=not progress_bar):
                    
                    # Calculate batch indices
                    start_idx = processed_rows + i * self.batch_size
                    end_idx = min(start_idx + self.batch_size, len(bed))
                    
                    # Skip if we've processed past this batch
                    if start_idx >= len(bed):
                        continue
                        
                    # Process this batch
                    batch_bed = bed.iloc[start_idx:end_idx]
                    batch_effects = self._peak_predict(emb_model, batch_bed)
                    batch_effects_np = batch_effects.numpy()
                    
                    # Append results
                    all_results["peak_effect_mean"].append(batch_effects_np)
                    all_results["region_indices"].extend(range(start_idx, end_idx))
                    
                    # Store metadata
                    for col_name in _columns:
                        if col_name not in all_results["metadata"]:
                            all_results["metadata"][col_name] = []
                        col_values = batch_bed[col_name].values
                        if col_name == "Chromosome":
                            col_values = col_values.astype(str)
                        all_results["metadata"][col_name].extend(col_values)
                    
                    # Update processed rows
                    processed_rows = end_idx
                    
                    # Update progress file after each batch
                    with open(progress_file, 'w') as f:
                        json.dump({"processed_rows": processed_rows, "total_rows": len(bed)}, f)
                    
                    # Increment batch counter
                    batches_since_checkpoint += 1
                    
                    # Save checkpoint after specified number of batches or at the end
                    if batches_since_checkpoint >= checkpoint_batches or end_idx == len(bed):
                        # Concatenate all collected results
                        combined_effects = np.concatenate(all_results["peak_effect_mean"], axis=0)
                        
                        # Create dataset
                        ds = xr.Dataset(
                            {
                                "peak_effect_mean": (["region", "channel"], combined_effects),
                            },
                            coords={
                                "sample": [cell_type],
                                "region": all_results["region_indices"]
                            }
                        )
                        
                        # Add metadata
                        for col_name, values in all_results["metadata"].items():
                            ds.coords[col_name] = (("region",), values)
                        
                        # Save dataset
                        temp_zarr_path = pathlib.Path(f"{zarr_path}.temp")
                        with warnings.catch_warnings():
                            warnings.filterwarnings("ignore", category=xr.SerializationWarning)
                            ds = ds.chunk(ds.sizes)
                            ds.to_zarr(temp_zarr_path, mode="w")
                        
                        # Rename to final path if successful
                        if temp_zarr_path.exists():
                            if zarr_path.exists():
                                import shutil
                                shutil.rmtree(zarr_path)
                            temp_zarr_path.rename(zarr_path)
                            print(f"Saved checkpoint at {processed_rows}/{len(bed)} rows for {cell_type}")
                        
                        # Reset batch counter
                        batches_since_checkpoint = 0
                
                # Clean up model
                del emb_model
                torch.cuda.empty_cache()
                
                # Mark as successful if all rows processed
                if processed_rows >= len(bed):
                    cell_type_success.touch()
                    print(f"Successfully processed all peak effects for cell type: {cell_type}")
                
            except Exception as e:
                print(f"Error processing peak effects for cell type {cell_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
        
        # Check if all cell types have been processed
        all_complete = True
        for cell_type in embedding_bed_dict.keys():
            cell_type_dir = output_dir / cell_type
            cell_type_success = cell_type_dir / ".success"
            if not cell_type_success.exists():
                all_complete = False
                break
        
        # Mark the entire process as complete if all cell types are done
        if all_complete:
            success_flag.touch()
            print(f"Completed processing peak effects for all cell types in experiment: {experiment_name}")
        else:
            print(f"Not all cell types were completed successfully. Re-run to continue processing.")
        
        return


    def _snp_predict_with_peak_effect(self, model, bed, modality='atac', output_dir=None, celltype=None, checkpoint_interval=10000):
        """Run prediction for SNP variants, calculating and saving peak effects and beta values.
        
        Parameters
        ----------
        model : torch.nn.Module
            Collapsed model for prediction.
        bed : pd.DataFrame
            Bed file with variant information.
        modality : str, optional
            Modality to extract from model output. Default is 'atac'.
        output_dir : str or pathlib.Path, optional
            Directory to save checkpoints. If None, checkpoints are not saved.
        celltype : str, optional
            Cell type identifier for checkpoint filenames.
        checkpoint_interval : int, optional
            Number of regions to process before saving a checkpoint. Default is 10000.
            
        Returns
        -------
        dict
            Dictionary with 'peak_values' and 'beta' arrays, both in the same order.
        """
        import pickle
        import pathlib
        
        # Initialize our simple result structure
        result = {
            'peak_values': [],
            'beta': [],
            'dist_to_peak': [],
            'peak_size': []
        }
        
        # For checkpointing
        regions_processed = 0
        checkpoint_count = 0
        
        # Prepare checkpoint directory if provided
        if output_dir is not None:
            output_dir = pathlib.Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a checkpoint-specific directory
            checkpoint_dir = output_dir / "checkpoints"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            
            # Create or load progress tracker
            progress_file = checkpoint_dir / f"{celltype or 'unknown'}_progress.txt"
            if progress_file.exists():
                with open(progress_file, 'r') as f:
                    regions_processed = int(f.read().strip())
                    print(f"Resuming from {regions_processed} processed regions")
        
        # Skip already processed regions
        if regions_processed > 0:
            print(f"Skipping first {regions_processed} regions (already processed)")
        
        for i in tqdm(range(0, bed.shape[0], self.batch_size)):
            # Skip if we've already processed these regions
            if i < regions_processed:
                continue
                
            batch_bed = bed.iloc[i : i + self.batch_size]
            
            # Get reference and alternate sequences
            batch = self._get_snp_dna_one_hot(batch_bed)
            
            # Get predictions
            outputs_ref = self._forward_pass(model, batch['ref'])
            outputs_alt = self._forward_pass(model, batch['alt'])
            
            # Calculate effect (alt - ref) immediately
            batch_effect = outputs_alt[modality] - outputs_ref[modality]
            
            # Process each sample in the batch
            for j in range(len(batch_bed)):
                row = batch_bed.iloc[j]
                
                # Calculate relative peak positions within the sequence
                # Convert from bp to 32bp bins and account for the 512bp padding
                relative_peak_start = max(0, (row["peak-start"] - row["Start"] + 512) // 32)
                relative_peak_end = min(batch_effect.shape[-1] - 1, (row["peak-end"] - row["Start"] + 512) // 32)
                
                # Extract peak region effects and calculate the mean
                # This preserves all channels but averages across the peak positions
                peak_effect = torch.mean(batch_effect[j, :, relative_peak_start:relative_peak_end+1], dim=-1)
                
                # Store peak effect and corresponding beta value
                result['peak_values'].append(peak_effect.cpu().detach().numpy())
                result['beta'].append(row['beta'])
                result['peak_size'].append(row["peak-end"] - row["peak-start"])
                midpoint = np.abs((row["peak-end"] - row["peak-start"]) // 2 + row["peak-start"])
                result['dist_to_peak'].append(np.abs(midpoint - (row['pos2start']+row["Start"])))
                
                # Increment regions processed counter
                regions_processed += 1
            
            # Save checkpoint if interval reached
            if output_dir is not None and regions_processed // checkpoint_interval > checkpoint_count:
                checkpoint_count = regions_processed // checkpoint_interval
                
                # Create checkpoint filename
                checkpoint_file = checkpoint_dir / f"{celltype or 'unknown'}_checkpoint_{regions_processed}.pkl"
                
                # Convert current results to numpy arrays for saving
                temp_result = {
                    'peak_values': np.array(result['peak_values']),
                    'beta': np.array(result['beta']),
                    'peak_size': np.array(result['peak_size']),
                    'dist_to_peak': np.array(result['dist_to_peak']),
                    'regions_processed': regions_processed,
                    'celltype': celltype
                }
                
                # Save checkpoint
                with open(checkpoint_file, 'wb') as f:
                    pickle.dump(temp_result, f)
                
                # Update progress file
                with open(progress_file, 'w') as f:
                    f.write(str(regions_processed))
                    
                print(f"Checkpoint saved at {regions_processed} regions")
                
                # Clear memory for tensors that have been saved
                torch.cuda.empty_cache()
        
        # Convert lists to numpy arrays for easier handling
        result['peak_values'] = np.array(result['peak_values'])
        result['beta'] = np.array(result['beta'])
        result['dist_to_peak'] = np.array(result['dist_to_peak'])
        result['peak_size'] = np.array(result['peak_size'])
        
        # Save final result if checkpointing was enabled
        if output_dir is not None:
            final_file = output_dir / f"{celltype or 'unknown'}_final_result.pkl"
            
            # Add metadata to result
            result['regions_processed'] = regions_processed
            result['celltype'] = celltype
            
            # Save final result
            with open(final_file, 'wb') as f:
                pickle.dump(result, f)
            
            print(f"Final results saved to {final_file}")
        
        return result


    def infer_snp_simple(
        self, celltype: str, embedding: np.array, bed_path: str, output_dir=None, progress_bar=True, checkpoint_interval=10000
    ) -> dict:
        """Simplified inference of variant effect that returns a dictionary with peak values and beta values.
        
        Parameters
        ----------
        celltype : str
            The cell type identifier (e.g., 'ASC')
        embedding : np.array
            Embedding array for the cell type
        bed_path : str
            Path to the bed file with variant information
        output_dir : str or pathlib.Path, optional
            Directory to save checkpoints. If None, checkpoints are not saved.
        progress_bar : bool, optional
            Show progress bar. Default is True.
        checkpoint_interval : int, optional
            Number of regions to process before saving a checkpoint. Default is 10000.
            
        Returns
        -------
        dict
            Dictionary containing 'peak_values' and 'beta' arrays in corresponding order.
        """
        # Read and process the BED file
        bed = pd.read_csv(bed_path, header=0, sep="\t")
        
        # Standardize columns
        _columns = ["Chromosome", "Start", "End", "peak-start", "peak-end", "ref", "snp", "pos2start", "beta", "id"]
        bed.columns = _columns
        
        # Process the embedding
        emb_model = self._collapse_model(embedding)
        
        # Use the optimized SNP predict function that calculates peak effects during the forward pass
        # and returns a simple dictionary structure
        result = self._snp_predict_with_peak_effect(
            emb_model, 
            bed, 
            output_dir=output_dir,
            celltype=celltype,
            checkpoint_interval=checkpoint_interval
        )
        
        # Clean up
        del emb_model
        torch.cuda.empty_cache()
        
        # Add the celltype to the result for reference if not already added
        if 'celltype' not in result:
            result['celltype'] = celltype
        
        return result

    def infer_snp_simple_offline(
        self, 
        embedding_bed_dict,
        output_dir,
        experiment_name,
        progress_bar=True,
        checkpoint_interval=10000,
        resume=True
    ):
        """Simplified offline inference for SNP variants, saving peak effects and beta values.
        
        Parameters
        ----------
        embedding_bed_dict : dict
            Dictionary where:
            - Keys are cell types (e.g., 'ASC')
            - Values are dictionaries with:
                - 'path': path to the bed file with variant information
                - 'embedding': embedding dataframe for that cell type
        output_dir : str
            Path to output directory.
        experiment_name : str
            Name of the experiment for the subfolder.
        progress_bar : bool, optional
            Show progress bar. Default is True.
        checkpoint_interval : int, optional
            Number of regions to process before saving a checkpoint. Default is 10000.
        resume : bool, optional
            Whether to resume from checkpoints if available. Default is True.
        """
        import pathlib
        
        # Create main output directory with experiment name
        output_dir = pathlib.Path(output_dir) / experiment_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Process each cell type
        for cell_type, data_dict in tqdm(embedding_bed_dict.items(), disable=not progress_bar, 
                                        desc="Processing cell types"):
            
            # Create cell type specific directory
            cell_type_dir = output_dir / cell_type
            cell_type_dir.mkdir(parents=True, exist_ok=True)
            
            # File to save results
            result_file = cell_type_dir / f"{cell_type}_results.pkl"
            
            # Check if already processed
            if result_file.exists() and resume:
                print(f"Cell type {cell_type} already processed. Skipping.")
                continue
            
            # Check if there are checkpoints to resume from
            checkpoint_dir = cell_type_dir / "checkpoints"
            if not resume and checkpoint_dir.exists():
                import shutil
                print(f"Removing existing checkpoints for {cell_type} since resume=False")
                shutil.rmtree(checkpoint_dir)
            
            embedding = data_dict['embedding']
            bed_path = data_dict['path']
            
            try:
                # Infer SNP effects with checkpointing
                result = self.infer_snp_simple(
                    cell_type, 
                    embedding, 
                    bed_path, 
                    output_dir=cell_type_dir,
                    progress_bar=progress_bar,
                    checkpoint_interval=checkpoint_interval
                )
                
                # The results are already saved by _snp_predict_with_peak_effect,
                # but we'll save them explicitly with our standard naming for consistency
                import pickle
                if not result_file.exists():
                    with open(result_file, 'wb') as f:
                        pickle.dump(result, f)
                    print(f"Results saved to {result_file}")
                
                print(f"Successfully processed cell type: {cell_type}")
                
            except Exception as e:
                print(f"Error processing cell type {cell_type}: {str(e)}")
                import traceback
                traceback.print_exc()
                # Continue with other cell types even if one fails
                continue
            
            # Clear memory
            torch.cuda.empty_cache()
        
        print(f"Completed processing all cell types for experiment: {experiment_name}")

    def merge_checkpoints(self, checkpoint_dir, output_file=None):
        """Merge multiple checkpoint files into a single result file.
        
        Parameters
        ----------
        checkpoint_dir : str or pathlib.Path
            Directory containing checkpoint files.
        output_file : str or pathlib.Path, optional
            Path to save the merged result. If None, will use the directory name.
        
        Returns
        -------
        dict
            Merged dictionary with peak values and beta values.
        """
        import pathlib
        import pickle
        import glob
        
        checkpoint_dir = pathlib.Path(checkpoint_dir)
        
        # Find all checkpoint files
        checkpoint_files = sorted(glob.glob(str(checkpoint_dir / "checkpoints" / "*_checkpoint_*.pkl")))
        
        if not checkpoint_files:
            raise ValueError(f"No checkpoint files found in {checkpoint_dir / 'checkpoints'}")
        
        # Initialize merged result
        merged_result = {
            'peak_values': [],
            'beta': []
        }
        
        # Load and merge checkpoints
        celltype = None
        
        for checkpoint_file in tqdm(checkpoint_files, desc="Merging checkpoints"):
            with open(checkpoint_file, 'rb') as f:
                checkpoint = pickle.load(f)
            
            merged_result['peak_values'].extend(checkpoint['peak_values'])
            merged_result['beta'].extend(checkpoint['beta'])
            
            # Get celltype from first checkpoint
            if celltype is None and 'celltype' in checkpoint:
                celltype = checkpoint['celltype']
        
        # Convert to numpy arrays
        merged_result['peak_values'] = np.array(merged_result['peak_values'])
        merged_result['beta'] = np.array(merged_result['beta'])
        
        # Add celltype if available
        if celltype:
            merged_result['celltype'] = celltype
        
        # Save merged result if output file is specified
        if output_file is None:
            output_file = checkpoint_dir / f"{celltype or checkpoint_dir.name}_merged_results.pkl"
        
        with open(output_file, 'wb') as f:
            pickle.dump(merged_result, f)
        
        print(f"Merged {len(checkpoint_files)} checkpoints into {output_file}")
        
        return merged_result