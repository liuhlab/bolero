import numpy as np
import pandas as pd
import scipy
import scipy.ndimage


def get_peaks_df_pval_fp(
    fp: np.ndarray,
    scores: dict[str, np.ndarray],
    nuc_mode_cutoff: int = 20,
    height: float = 0.2,
    distance: int = 10,
    prominance: float = 0.01,
    fp_cutoff: float = 0.5,
    skip_bottom: int = 3,
) -> pd.DataFrame:
    """
    Get a dataframe of peaks from the scores and the footprints

    Parameters
    ----------
        fp (np.ndarray): The footprints array.
        scores (Dict[str, np.ndarray]): A dictionary of scores.
        nuc_mode_cutoff (int): The mode cutoff value for nucleosome peak.
        height (float): The minimum height of peaks, see scipy.signal.find_peaks for more information.
        distance (int): The minimum horizontal distance between peaks, see scipy.signal.find_peaks for more information.
        prominance (float): The minimum prominence of peaks, see scipy.signal.find_peaks for more information.
        fp_cutoff (float): The cutoff value for footprints at the peak location.
        skip_bottom (int): The number of small modes to skip from the bottom of the footprints.

    Returns
    -------
        pd.DataFrame: A dataframe containing the peaks information.
    """
    all_peaks = []
    all_modes = []
    all_values = []
    peak_types = []
    names = []
    for key, score in scores.items():
        peaks = (
            scipy.signal.find_peaks(
                score, height=height, distance=distance, prominence=prominance
            )[0]
            + 100
        )  # record peak pos using the footprint dim
        if key.endswith("_nucleosome_score"):
            use_fp = fp[nuc_mode_cutoff:, peaks]
            max_mode = np.argmax(use_fp, axis=0).ravel()
            max_mode_adj = nuc_mode_cutoff
            fp_value = use_fp[max_mode, np.arange(max_mode.size)].ravel()
            mask = fp_value > fp_cutoff
            peak_type = ["nucleosome"] * mask.sum()
        else:
            use_fp = fp[skip_bottom:nuc_mode_cutoff, peaks]
            max_mode = np.argmax(use_fp, axis=0).ravel()
            max_mode_adj = skip_bottom
            fp_value = use_fp[max_mode, np.arange(max_mode.size)].ravel()
            mask = fp_value > fp_cutoff
            peak_type = ["tf"] * mask.sum()
        name = [key] * mask.sum()
        names.extend(name)
        all_peaks.append(peaks[mask])
        all_modes.append(max_mode[mask] + max_mode_adj)
        all_values.append(fp_value[mask])
        peak_types.extend(peak_type)
    all_peaks = np.concatenate(all_peaks)
    all_modes = np.concatenate(all_modes)
    all_values = np.concatenate(all_values)
    peak_df = pd.DataFrame(
        {
            "peak_pos": all_peaks,
            "mode": all_modes,
            "peak_type": peak_types,
            "score_name": names,
            "fp_value": all_values,
        }
    )
    return peak_df


def get_masks(
    fp: np.ndarray,
    peaks_df: pd.DataFrame,
    mask_cutoff: float = 0.7,
    skip_bottom: int = 3,
) -> np.ndarray:
    """
    Generate masks based on the footprints and peaks dataframe.

    Parameters
    ----------
        fp (np.ndarray): The footprints array.
        peaks_df (pd.DataFrame): The dataframe containing the peaks information.
        mask_cutoff (float): The cutoff value for the mask.
        skip_bottom (int): The number of small modes to skip from the bottom of the footprints.

    Returns
    -------
        np.ndarray: The aggregated mask.
    """
    fp[:skip_bottom] = 0
    # masks.shape = (n_peaks, fp_modes, fp_length)
    masks = np.zeros([peaks_df.shape[0]] + list(fp.shape))
    for row_id, (_, (x, y, cutoff)) in enumerate(
        peaks_df[["peak_pos", "mode", "fp_value"]].iterrows()
    ):
        binary_fp = fp > cutoff * mask_cutoff
        labeled_array, _ = scipy.ndimage.label(binary_fp)
        this_cluster = labeled_array[int(y), int(x)]
        if this_cluster != 0:
            mask = labeled_array == labeled_array[int(y), int(x)]
            masks[row_id] = mask
    agg_mask = masks.any(axis=0)
    return agg_mask
