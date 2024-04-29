import numpy as np
import pandas as pd
import pyranges as pr
import xarray as xr

from .genome_dataset import GenomePositionZarr


def calculate_atac_norm_value(
    genome,
    peak_bed_path,
    cutsite_zarr_path,
    standard_size=500,
    sample_regions=5000,
):
    """
    Calculate the ATAC-seq normalization value for each sample.

    Two normalization metrics are calculated:
    - Peak value: the average value of the 99th to 99.9th percentile of the cutsite counts in the peak regions.
    - Non-peak value: the average value of the cutsite counts in the non-peak regions.

    The ATAC count value will then be normalized by:
    - NormalizedValue = (ATACCountValue - NonPeakValue) / PeakValue

    Parameters
    ----------
    genome : Genome
        Genome object.
    peak_bed_path : str
        Path to the peak regions in BED format.
    cutsite_zarr_path : str
        Path to the cutsite count Zarr file.
    standard_size : int, optional
        Standard size of the regions. Default is 500.
    sample_regions : int, optional
        Number of regions to sample. Default is 5000.
    """
    regions = pr.read_bed(peak_bed_path)
    peak_regions = genome.standard_region_length(regions, standard_size)
    non_peak_regions = genome.genome_bed.window(standard_size).subtract(
        peak_regions.extend(standard_size)
    )
    non_peak_regions = non_peak_regions[
        (non_peak_regions.End - non_peak_regions.Start) == standard_size
    ]

    if len(peak_regions) > sample_regions:
        peak_regions = peak_regions.sample(sample_regions)
    if len(non_peak_regions) > sample_regions:
        non_peak_regions = non_peak_regions.sample(sample_regions)

    ds = xr.open_zarr(cutsite_zarr_path)
    dataset = GenomePositionZarr(
        da=ds["site_count"], offsets=genome.chrom_offsets, load=False
    )

    _data = dataset.get_regions_data(peak_regions.df)
    peak_value_per_sample = np.percentile(
        _data.mean(axis=2), np.arange(99, 99.9, 0.1), axis=0
    ).mean(axis=0)

    _data = dataset.get_regions_data(non_peak_regions.df)
    non_peak_value_per_sample = _data.mean(axis=(0, 2)) + 1e-6

    data = pd.DataFrame(
        [peak_value_per_sample, non_peak_value_per_sample],
        columns=ds.get_index("category"),
        index=["peak_value", "non_peak_value"],
    ).T

    # save this data into the same zarr file
    data.index.name = "category"
    data.columns.name = "norm_value"
    ds["normalize"] = data
    ds[["normalize"]].to_zarr(cutsite_zarr_path, mode="a")
    return


def normalize_atac_batch(batch, norm_value):
    """
    Normalize ATAC raw count batch data with pre-calculated norm value.

    Parameters
    ----------
    batch : np.ndarray
        The batch data. The normalization happen along the second axis (axis=1).
    norm_value : pd.DataFrame
        The normalization value. The index is the sample name and the columns are "peak_value" and "non_peak_value".
    """
    peak_value = norm_value["peak_value"].values[None, :, None]
    non_peak_value = norm_value["non_peak_value"].values[None, :, None]

    norm_data = (batch - non_peak_value) / peak_value
    return norm_data


def _conv_signal(data, conv_size=50):
    return np.convolve(data, np.ones(shape=(conv_size,)), mode="same")


def convolve_data(batch, conv_size=50):
    """
    Convolve the batch data with a window of size conv_size.

    After convolution, the data will be divided by conv_size to bring the value back to the original scale.

    Parameters
    ----------
    batch : np.ndarray
        The batch data. The convolution happen along the third axis (axis=2).
    conv_size : int, optional
        The size of the convolution window. Default is 50.
    """
    # TODO, there will be boundary effect, DataLoader should load additional bases and trim the boundary or let convovle1d do the trimming
    _position_axis = 2
    conv_data = np.apply_along_axis(
        _conv_signal, _position_axis, batch, conv_size=conv_size
    )
    conv_data /= conv_size
    return conv_data
