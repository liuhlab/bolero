import queue
import threading
from collections import OrderedDict, defaultdict
from pathlib import Path
from typing import Any, Generator

import joblib
import numpy as np
import pandas as pd
import pyBigWig
import pyranges as pr
import torch

from bolero.pp.genome import FastaOneHot, Genome
from bolero.tl.dataset.parquet_db import GenomeParquetDB

from .dna_gen import DNASynthesisFactory
from .utils import convert_np_to_torch, get_device


class BackgroundGenerator:
    def __init__(
        self, generator, max_prefetch=20, as_torch=True, device=None, collate_fn=None
    ):
        """
        The generator will run in a separate thread and will prefetch up to max_prefetch items.
        """
        self.queue = queue.Queue(max_prefetch)
        self.generator = generator
        self.thread = threading.Thread(target=self._worker)
        self.thread.daemon = True
        self.started = False
        self.as_torch = as_torch
        self.device = device
        if collate_fn is None:
            collate_fn = lambda x: x
        self.collate_fn = collate_fn

    def start(self):
        """Start the generator thread."""
        if not self.started:
            self.thread.start()
            self.started = True

    def _worker(self):
        for item in self.generator:
            self.queue.put(item)
        self.queue.put(None)  # Sentinel to mark end

    def __iter__(self):
        self.start()
        return self

    def __next__(self):
        self.start()
        item = self.queue.get()
        if item is None:
            raise StopIteration

        # convert numpy arrays to torch tensors
        if self.as_torch:
            item = convert_np_to_torch(item)
        for key, value in item.items():
            if isinstance(value, torch.Tensor):
                item[key] = value.to(self.device)

        # apply collate function
        item = self.collate_fn(item)

        return item


class PseudobulkRecord:
    pid: str
    row_names: dict[str, list[str]]
    n_frags: dict[int] | None
    cov_scale: dict[float] | None
    embedding: np.ndarray | None
    embedding_multi: np.ndarray | None
    sample_weight: float | None
    annotation: dict[str, Any] | None

    def __init__(
        self,
        pid: str,
        **kwargs,
    ):
        if "cluster_ids" in kwargs:
            kwargs["row_names"] = kwargs.pop("cluster_ids")

        self.pid = pid
        self.row_names = kwargs.pop("row_names")
        self.n_frags = kwargs.pop("n_frags", None)
        self.cov_scale = kwargs.pop("cov_scale", None)
        self.embedding = kwargs.pop("embedding", None)
        self.embedding_multi = kwargs.pop("embedding_multi", None)
        self.sample_weight = kwargs.pop("sample_weight", None)

        # remaining kwargs are considered as annotation
        self.annotation = kwargs
        self._validate()
        return

    def _validate(self):
        # one of embedding or embedding_multi must be set
        if self.embedding is not None:
            assert self.embedding.ndim == 1, "embedding must be 1D"
        elif self.embedding_multi is not None:
            assert self.embedding_multi.ndim == 2, "embedding_multi must be 2D"
        else:
            raise ValueError("Either embedding or embedding_multi must be set")
        return

    def __repr__(self):
        return f"PseudobulkRecord(pid={self.pid})"


class PseudobulkRecordManager:
    def __init__(
        self,
        pseudobulk_records: str | Path | dict[str, dict],
        annotation: pd.DataFrame = None,
    ):
        if isinstance(pseudobulk_records, (str, Path)):
            self.original_records = joblib.load(pseudobulk_records)
        else:
            self.original_records = pseudobulk_records

        if "pseudobulk_records" in self.original_records:
            self.original_records = self.original_records["pseudobulk_records"]

        self.pseudobulk_records: dict[str, PseudobulkRecord] = OrderedDict()
        for pid, pid_record in self.original_records.items():
            self.pseudobulk_records[pid] = PseudobulkRecord(pid, **pid_record)
        self.pseudobulk_ids = list(self.pseudobulk_records.keys())

        if annotation is not None:
            self.add_annotation("annotation", annotation)

        self.origin_id_to_pid_map = {}
        for pid, rec in self.pseudobulk_records.items():
            if "__pid__" in rec.annotation:
                orig_pid = rec.annotation["__pid__"]
            else:
                orig_pid = pid
            self.origin_id_to_pid_map[orig_pid] = pid
        self.pid_to_origin_id_map = {v: k for k, v in self.origin_id_to_pid_map.items()}
        return

    def get_pseudobulk_attrs(
        self, attr: str, pids: list = None, prefix: str = None, series: bool = True
    ) -> pd.Series | list:
        """
        Get pseudobulk attributes.
        """
        if pids is None:
            pids = self.pseudobulk_ids

        data = []
        for pid in pids:
            try:
                value = getattr(self.pseudobulk_records[pid], attr)
            except AttributeError:
                # try to get from annotation if not found in record
                try:
                    value = self.pseudobulk_records[pid].annotation[attr]
                except KeyError as e:
                    raise KeyError(
                        f"Attribute {attr} not found in pseudobulk record {pid} or its annotation"
                    ) from e
            if prefix is not None and isinstance(value, dict):
                value = value[prefix]
            data.append(value)
        if series:
            data = pd.Series(data, index=pids)
        return data

    def get_n_frags(self, pids: list = None, prefix: str = None) -> pd.Series:
        """
        Get the number of fragments for each pseudobulk.
        """
        return self.get_pseudobulk_attrs("n_frags", pids, prefix)

    def get_cov(self, pids: list = None, prefix: str = None) -> pd.Series:
        """
        Alias for get_n_frags.
        """
        return self.get_n_frags(pids, prefix)

    def get_cov_scale(self, pids: list = None, prefix: str = None) -> pd.Series:
        """
        Get the coverage scale for each pseudobulk.
        """
        return self.get_pseudobulk_attrs("cov_scale", pids, prefix)

    def get_embedding(self, pids: list = None) -> np.ndarray:
        """
        Get the embedding for each pseudobulk.
        """
        data = self.get_pseudobulk_attrs("embedding", pids, series=False)
        data = np.stack(data)
        return data

    def get_embedding_multi(self, pids: list = None) -> np.ndarray:
        """
        Get the multi-embedding for each pseudobulk.
        """
        data = self.get_pseudobulk_attrs("embedding_multi", pids, series=False)
        data = np.stack(data)
        return data

    def get_sample_weight(self, pids: list = None) -> np.ndarray:
        """
        Get the sample weight for each pseudobulk.
        """
        return self.get_pseudobulk_attrs("sample_weight", pids)

    def get_annotation(self, key, pids: list = None) -> pd.DataFrame:
        """
        Get the annotation for each pseudobulk.
        """
        return self.get_pseudobulk_attrs("annotation", pids=pids, prefix=key)

    def add_annotation(self, name: str, annotation: dict | pd.Series | pd.DataFrame):
        """
        Add annotation to the pseudobulk records.
        """
        if isinstance(annotation, pd.DataFrame):
            # assume each column is a separate annotation and the index is the pid
            for col, annot in annotation.items():
                self.add_annotation(col, annot)
            return

        for pid, annot in annotation.items():
            if pid not in self.pseudobulk_records:
                raise ValueError(f"PID {pid} not found in pseudobulk records")
            self.pseudobulk_records[pid].annotation[name] = annot
        return

    def get_merge_plan(self) -> dict[str, list[str]]:
        """
        Get the merge plan for the pseudobulk records.
        """
        merge_plan = {}
        for pid, record in self.pseudobulk_records.items():
            merge_plan[pid] = record.row_names
        return merge_plan

    def __getitem__(self, pid: str) -> PseudobulkRecord:
        """
        Get a pseudobulk record by its ID.
        """
        try:
            return self.pseudobulk_records[pid]
        except KeyError:
            try:
                return self.pseudobulk_records[self.origin_id_to_pid_map[pid]]
            except KeyError as e:
                raise KeyError(f"PID {pid} not found in pseudobulk records") from e

    def __len__(self) -> int:
        """
        Get the number of pseudobulk records.
        """
        return len(self.pseudobulk_records)

    def __iter__(self):
        """
        Iterate over the pseudobulk records.
        """
        for pid in self.pseudobulk_ids:
            yield self.pseudobulk_records[pid]

    def __repr__(self):
        return f"PseudobulkRecordManager({len(self.pseudobulk_records)} records)"

    def items(self):
        """
        Get the items of the pseudobulk records.
        """
        yield from self.pseudobulk_records.items()


class _RefBigwig:
    def __init__(self, bw_path, resolution):
        self.bw_path = bw_path
        self.bw_handle = pyBigWig.open(str(bw_path))
        self.resolution = resolution

    @staticmethod
    def _to_coords(region):
        chrom, coords = region.split(":")
        start, end = map(int, coords.split("-"))
        return chrom, start, end

    def fetch(self, regions):
        region_data = []
        for region in regions:
            chrom, start, end = self._to_coords(region)
            try:
                data = self.bw_handle.values(chrom, start, end, numpy=True)
            except Exception as e:
                print("Error when fetching region: ", region)
                print("BigWig path: ", self.bw_path)
                raise e
            region_data.append(data)
        region_data = np.nan_to_num(np.array(region_data), nan=0.0)
        region_data = region_data.reshape(
            -1, (region_data.shape[1] // self.resolution), self.resolution
        ).sum(axis=-1)
        return region_data


class GenericGenomeDataManager:
    def __init__(
        self,
        genome: str | dict[str, str] | Genome,
        device: str | None = None,
        genome_kwargs: dict[str, Any] = None,
    ):
        genome_kwargs = genome_kwargs or {}
        if isinstance(genome, dict):
            self.genome = DNASynthesisFactory(genome_fastas=genome, **genome_kwargs)
        else:
            self.genome = Genome(genome) if isinstance(genome, str) else genome

        # Pseudobulk records
        self.pseudobulk_manager = None

        # Signal dataset from parquet or bigwig
        self.datasets: dict[str, GenomeParquetDB] = {}
        self.bw_datasets: dict[str, _RefBigwig] = {}

        self.device = device if device is not None else get_device()

        self._onehot_encoder: FastaOneHot | DNASynthesisFactory = (
            self._create_one_hot_encoder()
        )

    def _create_one_hot_encoder(self):
        if isinstance(self.genome, DNASynthesisFactory):
            onehot_encoder = self.genome
        else:
            onehot_encoder = FastaOneHot(
                fasta_path=self.genome.fasta_path, device=self.device, parallel=8
            )
        return onehot_encoder

    def add_parquet_dataset(
        self,
        dataset_name,
        dataset_dir: str,
        parallel: int = 1,
        resolution: int | dict[int] = None,
    ):
        """
        Add a parquet dataset to the data manager.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        dataset_dir : str
            The directory path to the parquet dataset.
        parallel : int, optional
            The number of parallel processes to use for reading the dataset. Default is 1.
        merge_plan : dict, optional
            The pseudobulk merge plan to sum meta-cells (rows) into pseudobulks.
            Default is None, which means no pseudobulk merging.
            If a dict is provided, it should be in the format:
            {
                "pseudobulk1": ["cell1", "cell2", ...],
                "pseudobulk2": ["cell3", "cell4", ...],
                ...
            }, OR,
            {
                "pseudobulk1": {
                    "prefix1": ["cell1", "cell2", ...],
                    "prefix2": ["cell3", "cell4", ...]
                },
                "pseudobulk2": {
                    "prefix1": ["cell5", "cell6", ...],
                    "prefix2": ["cell7", "cell8", ...]
                },
                ...
            }
        resolution : int or dict, optional
            The resolution of the dataset.
            Default is None, which means resolution should be inferred from the dataset.
        """
        if self.pseudobulk_manager is None:
            raise ValueError(
                "No pseudobulk manager provided, call add_pseudobulk_records() first"
            )

        merge_plan = {
            pid: rec.row_names for pid, rec in self.pseudobulk_manager.items()
        }
        pseudobulk_ids = self.pseudobulk_manager.pseudobulk_ids

        self.datasets[dataset_name] = GenomeParquetDB(
            dataset_dir=dataset_dir,
            parallel=parallel,
            merge_plan=merge_plan,
            pseudobulk_ids=pseudobulk_ids,
            resolution=resolution,
        )

    def add_bigwig_dataset(
        self,
        dataset_name: str,
        dataset_path: str | Path,
        resolution: int,
    ):
        """
        Add a bigwig dataset to the data manager.

        Parameters
        ----------
        dataset_name : str
            The name of the dataset.
        dataset_path : str | Path
            The path to the bigwig file.
        resolution : int
            The resolution of the dataset.
        """
        self.bw_datasets[dataset_name] = _RefBigwig(
            bw_path=dataset_path, resolution=resolution
        )
        return

    def add_pseudobulk_records(
        self,
        pseudobulk_records: str | Path | dict[str, dict],
        annotation: pd.DataFrame = None,
    ):
        """
        Add pseudobulk records to the data manager.

        Parameters
        ----------
        pseudobulk_records : str | Path | dict
            The pseudobulk records to add. Can be a path to a file or a dictionary.
        annotation : pd.DataFrame, optional
            The annotation to add to the pseudobulk records.
        """
        self.pseudobulk_manager = PseudobulkRecordManager(
            pseudobulk_records=pseudobulk_records, annotation=annotation
        )

    def query_dna_onehot(
        self,
        regions: pd.DataFrame | pr.PyRanges | list[str],
        region_names: list[str],
        length_last: bool = True,
    ) -> np.ndarray:
        """
        Query the DNA one-hot encoding for the regions.

        Parameters
        ----------
        regions : pd.DataFrame | pr.PyRanges
            The regions to query. Can be a dataframe or a PyRanges object.
        length_last : bool, optional
            If True, the last dimension of the one-hot encoding will be the sequence length.

        Returns
        -------
        np.ndarray
            The one-hot encoding of the DNA sequence for the regions.
        """
        if isinstance(regions, pr.PyRanges):
            regions = regions.df

        if isinstance(self._onehot_encoder, DNASynthesisFactory):
            onehot = self._onehot_encoder.get_regions_onehot(regions, region_names)
        else:
            onehot = self._onehot_encoder.get_regions_onehot(regions)

        if length_last:
            onehot = onehot.permute(0, 2, 1)
            # shape is (n_regions, 4, seq_len)
        return onehot

    def _prepare_pseudobulk_info(
        self, info_keys: list[str], pids: list[str] = None
    ) -> dict[str, Any]:
        """
        Add pseudobulk information into the batch data.
        """
        if pids is None:
            pids = self.pseudobulk_manager.pseudobulk_ids

        data_col = defaultdict(list)
        for info_key in info_keys:
            data_list = self.pseudobulk_manager.get_pseudobulk_attrs(
                pids=pids, attr=info_key, series=False
            )
            for _data in data_list:
                if isinstance(_data, dict):
                    for k, v in _data.items():
                        data_col[f"{k}:{info_key}"].append(v)
                else:
                    data_col[info_key].append(_data)
        data_col = {k: np.array(v) for k, v in data_col.items()}
        return data_col

    def _iter_batches_chunk(
        self,
        regions: list[str],
        region_names: list[str] = None,
        batch_size: int = 32,
        add_dna: bool = True,
        add_data: bool = True,
        pseudobulk_subset: list[str] = None,
        pseudobulk_info_keys: list[str] = None,
    ) -> Generator:
        """
        Prepare batches for a list of regions.
        """
        _pseudobulk_subset_bool = {}
        if pseudobulk_subset is not None:
            for da_name, da in self.datasets.items():
                sel_bool = da.pseudobulk_ids.isin(pseudobulk_subset)
                _pseudobulk_subset_bool[da_name] = sel_bool

        if add_data:
            da_data_dict = {
                da_name: da.iter_batches(
                    regions,
                    batch_size=batch_size,
                    pseudobulk_subset_plan=_pseudobulk_subset_bool.get(da_name, None),
                )
                for da_name, da in self.datasets.items()
            }
        else:
            da_data_dict = {}

        if pseudobulk_info_keys is not None:
            pseudobulk_info = self._prepare_pseudobulk_info(
                info_keys=pseudobulk_info_keys,
                pids=pseudobulk_subset,
            )

        for cur_start in range(0, len(regions), batch_size):
            regions_ref = np.array(regions[cur_start : cur_start + batch_size])
            batch_data = {"region": regions_ref}
            if region_names is not None:
                batch_data["region_name"] = np.array(
                    region_names[cur_start : cur_start + batch_size]
                )

            # add bigwig data
            for da_name, bw in self.bw_datasets.items():
                batch_data[da_name] = bw.fetch(regions_ref)

            # add true data from each dataset
            for da_name, data_iter in da_data_dict.items():
                da_batch = next(data_iter)

                # make sure regions across datasets are matching
                da_regions = da_batch.pop("region")

                assert np.array_equal(
                    regions_ref, da_regions
                ), f"Regions do not match for dataset {da_name}, expected {regions_ref}, got {da_regions}"

                for key, value in da_batch.items():
                    assert (
                        key not in batch_data
                    ), f"Key {key} already exists in batch_data"
                    batch_data[key] = value

            # add dna one-hot encoding
            if add_dna:
                onehot = self.query_dna_onehot(regions_ref, region_names)
                batch_data["dna"] = onehot

            # add pseudobulk information
            if pseudobulk_info_keys is not None:
                batch_data.update(pseudobulk_info)

            yield batch_data

    def _iter_batches(
        self,
        regions: list[str],
        region_names: list[str] | None = None,
        batch_size: int = 32,
        add_dna: bool = True,
        add_data: bool = True,
        pseudobulk_subset: list[str] = None,
        pseudobulk_info_keys: list[str] = None,
        batches_per_chunk: int = 100,
    ) -> Generator:
        """
        Iterate over the chunked list of regions.

        To achieve asynchronous data loading, use the get_dataloader method.
        """
        region_chunk_size = batch_size * batches_per_chunk
        for cur_start in range(0, len(regions), region_chunk_size):
            cur_end = min(cur_start + region_chunk_size, len(regions))
            regions_chunk = regions[cur_start:cur_end]
            region_names_chunk = (
                region_names[cur_start:cur_end] if region_names else None
            )

            iterable = self._iter_batches_chunk(
                regions=regions_chunk,
                region_names=region_names_chunk,
                batch_size=batch_size,
                add_dna=add_dna,
                add_data=add_data,
                pseudobulk_subset=pseudobulk_subset,
                pseudobulk_info_keys=pseudobulk_info_keys,
            )
            yield from iterable

    def _get_data_prefixs(self) -> list[str]:
        data_prefix_names = []
        for da in self.datasets.values():
            data_prefix_names.extend(da.prefix_names)
        if len(data_prefix_names) == 0:
            data_prefix_names = ["__embedding_only__"]
        return data_prefix_names

    @staticmethod
    def apply_callbacks(batch, callbacks):
        """
        Apply the callbacks sequentially to the batch.
        """
        for callback in callbacks:
            batch = callback(batch)
        return batch

    def get_dataloader(
        self,
        regions: list[str],
        region_names: list[str] | None = None,
        batch_size: int = 32,
        add_dna: bool = True,
        add_data: bool = True,
        pseudobulk_subset: list[str] = None,
        pseudobulk_info_keys: list[str] = None,
        as_torch: bool = True,
        collate_fn: callable = None,
        **kwargs,
    ) -> BackgroundGenerator:
        """
        Get a dataloader for the data manager.

        Parameters
        ----------
        regions : list[str], optional
            The regions to query. If None, all regions will be used. Default is None.
        region_names : list[str], optional
            The names of the regions to query. If None, all region names will be used. Default is None.
        batch_size : int, optional
            The batch size to use. Default is 32.
        add_dna : bool, optional
            If True, the DNA one-hot encoding will be added to the batch. Default is True.
        add_data : bool, optional
            If True, the data from the parquet datasets will be added to the batch. Default is True.
        pseudobulk_info_keys : list[str], optional
            The keys of the pseudobulk information to add to the batch. Default is None, no pseudobulk info.
        max_prefetch : int, optional
            The maximum number of batches to prefetch. Default is 50.
        as_torch : bool, optional
            If True, the batches will be converted to torch tensors. Default is True.
        **kwargs : Any
            Additional arguments to pass to the iter_batches method.
        """
        iterable = self._iter_batches(
            regions=regions,
            region_names=region_names,
            batch_size=batch_size,
            add_dna=add_dna,
            add_data=add_data,
            pseudobulk_subset=pseudobulk_subset,
            pseudobulk_info_keys=pseudobulk_info_keys,
            **kwargs,
        )
        return BackgroundGenerator(
            iterable,
            as_torch=as_torch,
            device=self.device,
            collate_fn=collate_fn,
        )
