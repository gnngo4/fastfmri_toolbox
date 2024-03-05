from typing import Optional
from typing import List, Tuple, Union, Literal
from typing_extensions import Annotated
import typer

import nibabel as nib
import numpy as np
from pathlib import Path
import itertools
import glob
import shutil

import sys
sys.path.append('/opt/app')

from fastfmri_toolbox.modelling.design_matrix import DesignMatrix

def search(base_dir, wildcard, error=True):
    search_path = Path(base_dir) / wildcard
    files = glob.glob(str(search_path))

    if not files:
        if error:
            raise FileNotFoundError(f"No files were found in: {search_path}")
        else:
            return []

    return files

def get_ids(paths, _str):
    _ids = []
    for i in paths:
        if "tasklock" in i:
            continue
        assert _str in i, f"{_str} not in {i}"
        _id = i.split(_str)[-1].split('_')[0]
        _ids.append(_id)

    # Get unique
    return list(set(_ids))
    
def read_lines(txt_file, n_iterations):

    with open(txt_file) as f:
        lines = f.readlines()

    assert len(lines) == n_iterations, f"{txt_file} has {len(lines)} lines. [{n_iterations} expected]"

    return lines

def combine_bootstrapped_splits(txt_files, n_batches, n_iterations):

    assert len(txt_files) == n_batches, f"{len(txt_files)} txt_files in list. {n_batches} was expected."

    for ix, txt_file in enumerate(txt_files):
        if ix == 0:
            data_split = read_lines(txt_file, n_iterations)
        else:
            data_split += read_lines(txt_file, n_iterations)

    assert len(data_split) == n_batches * n_iterations, f"{len(data_split)} splits found in list. [{n_batches*n_iterations} expected]"

    return data_split

def read_splits(line):

    dtseries = [Path(i.strip()) for i in line.split(',')]
    for bold in dtseries:
        assert bold.exists(), f"{bold} does not exist."

    return dtseries

def get_frequencies_per_task(base_dir, base_bootstrap_dir_wc, sub_id, task_id):
    
    paths = search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/bootstrap/*{task_id}*")
    frequencies = get_ids(paths, "_f-")
    frequencies.sort()

    return [float(f) for f in frequencies]

def get_updated_task_id_name(base_dir, base_bootstrap_dir_wc, sub_id, task_id):
    
    paths = search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/bootstrap/*{task_id}*")
    all_tasks = get_ids(paths, "_task-")
    updated_task_id = None
    for task in all_tasks:
        if task.startswith(task_id):
            updated_task_id = task
            return updated_task_id 

    assert updated_task_id is not None, f"{task_id} could not be updated. All detected tasks: {all_tasks}"


def combine_bootstrapped_dtseries(dtseries, out_dtseries, n_batches):

    assert len(dtseries) == n_batches, f"{len(dtseries)} dtseries in list. {n_batches} was expected."

    for ix, _dtseries in enumerate(dtseries):
        if ix == 0:
            base_img = nib.load(_dtseries)
            X = base_img.get_fdata()
        else:
            X = np.concatenate(
                (X, nib.load(_dtseries).get_fdata())
            )
    
    merged_img = nib.Cifti2Image(X, header=base_img.header)
    merged_img.header.matrix[0].number_of_series_points = X.shape[0]
    nib.save(merged_img, out_dtseries)

    assert out_dtseries.exists(), f"{out_dtseries} does not exist."

    return out_dtseries

"""ROI
"""
def get_roi_from_binary_activation_maps(
    dtseries, 
    wholebrain, 
    fo_threshold, 
    n_batches, 
    n_iterations, 
    second_f=None, 
    z_score_f1=None,
):
    
    # Load data
    data = nib.load(dtseries).get_fdata()
    n_bootstraps = data.shape[0]
    data = data.sum(0, keepdims=True) / n_bootstraps
    data = (data >= fo_threshold).astype(int)

    # Load wholebrain data
    wb_data = nib.load(wholebrain).get_fdata()
    wb_data = wb_data.sum(0, keepdims=True)
    wb_data = (wb_data == wb_data.max()).astype(int)

    data *= wb_data

    if second_f is not None:
        second_f_data = nib.load(second_f).get_fdata()
        n_bootstraps = second_f_data.shape[0]
        second_f_data = second_f_data.sum(0, keepdims=True) / n_bootstraps
        second_f_data = (second_f_data >= fo_threshold).astype(int)
        if z_score_f1 is None:
            data *= second_f_data
        else:
            n_vertices = second_f_data.sum()
            z_score_data = nib.load(z_score_f1).get_fdata()
            z_score_data = z_score_data.mean(0, keepdims=True)
            z_score_data *= data
            if "Q1_" in str(second_f):
                z_score_data[0,29696:29696+29716] = 0
            else:
                z_score_data[0,:29696] = 0
            if (z_score_data > 0).astype(int).sum() >= n_vertices:
                thr_value = np.partition(z_score_data.flatten(), -n_vertices)[-n_vertices]
                z_score_data[z_score_data < thr_value] = 0

            z_score_data = (z_score_data > 0).astype(int)

            data *= z_score_data

    assert n_bootstraps == n_batches * n_iterations, f"{n_bootstraps} found in {dtseries}. {n_batches*n_iterations} is expected."

    return data

def get_rois(
    data_dict,
    task_id, 
    FO_THR,
    n_batches_expected,
    n_iterations,
    control_roi_size, 
):
    second_f = any(k.startswith("train_second_f-") for k in data_dict.keys())
    if second_f:
        second_f_key = [i for i in data_dict.keys() if i.startswith("train_second_f-")][0]
    roi_dict = {}
    ROI_KEYS = [i for i in data_dict.keys() if i.startswith("train_activation_") and "task-control" not in i]
    if control_roi_size:
        f1_z_score = [i for i in data_dict.keys() if i.startswith(f"train_z_score_") and "task-control" not in i][0]
    for roi_key in ROI_KEYS:
        if second_f:
            if control_roi_size:
                roi_bin = get_roi_from_binary_activation_maps(
                    data_dict[roi_key], 
                    data_dict["wholebrain_mask"],
                    FO_THR, 
                    n_batches_expected, 
                    n_iterations,
                    second_f=data_dict[second_f_key],
                    z_score_f1=data_dict[f1_z_score],
                )
            else:
                roi_bin = get_roi_from_binary_activation_maps(
                    data_dict[roi_key], 
                    data_dict["wholebrain_mask"],
                    FO_THR, 
                    n_batches_expected, 
                    n_iterations,
                    second_f=data_dict[second_f_key],
                )
        else:
            roi_bin = get_roi_from_binary_activation_maps(
                data_dict[roi_key], 
                data_dict["wholebrain_mask"],
                FO_THR, 
                n_batches_expected, 
                n_iterations,
            )
        roi_dict[roi_key] = roi_bin

    return roi_dict

"""DTSERIES handling
"""
def normalize_bold(data, normalize_type):
    bold_baseline = data.mean(0)
    if normalize_type == "pct":
        data = data - bold_baseline
        data = data / np.where(bold_baseline != 0, bold_baseline, 1)
        return data
    elif normalize_type == "z_score":
        bold_stdev = data.std(0)
        data = (data - bold_baseline) / bold_stdev
        return data
    else:
        raise ValueError(f"`normalize_type` must be set to pct or z_score")
        

def get_average_bold(dtseries, normalize_type = "z_score"):
    n_runs = len(dtseries)
    assert n_runs > 0, f"{n_runs} found. Expected more than 0."
    # Load the data for the first run
    try:
        data = nib.load(dtseries[0]).get_fdata()
    except Exception as e:
        raise Exception(f"Error loading data for run 0: {e}")
    # Accumulate data for the remaining runs
    for dt in dtseries[1:]:
        try:
            data += nib.load(dt).get_fdata()
        except Exception as e:
            raise Exception(f"Error loading data for a run: {e}")
    # Calculate the average
    data /= n_runs

    data = normalize_bold(data, normalize_type)

    return data

def create_bootstrapped_metric_map_from_roi_data(roi_data, roi_coords, dtseries_template, dtseries_out):
    """
    """
    n_vertices_in_roi, n_bootstraps_in_roi = roi_data.shape
    assert roi_coords.sum() == n_vertices_in_roi

    base_img = nib.load(dtseries_template)
    n_bootstraps_in_template, n_vertices_in_template = base_img.get_fdata().shape

    data = np.zeros((n_bootstraps_in_roi, n_vertices_in_template))
    for n in range(n_bootstraps_in_roi):
        data[n, roi_coords] = roi_data[:, n]
    
    new_img = nib.Cifti2Image(data, header=base_img.header)
    new_img.header.matrix[0].number_of_series_points = n_bootstraps_in_roi

    nib.save(new_img, dtseries_out)

def create_bootstrapped_power_metric_map_from_roi_data(bootstrapped_data, frequency, task_id, data_id, dtseries_template, dtseries_out):
    
    frequency_grid_key = f"data-{data_id}_task-{task_id}_roi_frequencies"
    roi_data = f"data-{data_id}_task-{task_id}_roi_power"

    frequency_grid = bootstrapped_data[frequency_grid_key]
    roi_data = bootstrapped_data[roi_data]
    roi_coords = bootstrapped_data["roi_coords"]

    abs_diff = np.abs(frequency_grid - frequency)
    closest_index = np.argmin(abs_diff)
    
    print(f"Frequency of interest: {frequency} [{frequency_grid[closest_index]}]")

    # Extract power spectrum data for the specified frquency
    roi_data = roi_data[closest_index, :, :]

    create_bootstrapped_metric_map_from_roi_data(roi_data, roi_coords, dtseries_template, dtseries_out)

    assert Path(dtseries_out).exists(), f"{dtseries_out} does not exist."

# Processing tools
def process_phase_delay(
    data: np.ndarray,
    stimulated_frequency: float,
):
    """
    """
    max_phasedelay = 1/float(stimulated_frequency) # in seconds
    max_indices = data == max_phasedelay
    non_max_indices = data != max_phasedelay
    data[max_indices] = 0 # Unphase max values: [0, max_phasedelay)
    data[non_max_indices] -= max_phasedelay / 2 # Move the stimulus to the right by pi/2 account for different between stimulus and sine wave
    data[non_max_indices] %= max_phasedelay # Rephase data: [0, max_phasedelay)

    return data

def interpolate_vertex_bold_after_phase_shift(
    roi_bold, 
    roi_phase_delay, 
    vertex_id,
    time_window,
    TR 
):

    from scipy.interpolate import CubicSpline

    ts = roi_bold[:,vertex_id] # vertex timeseries
    assert not np.any(np.isnan(ts)), f"nan value found\n{ts}"
    pd = roi_phase_delay[vertex_id] # vertex phase delay
    dm = DesignMatrix(time_window, None, TR=TR) # Initialize dm class
    # Get interpolated tp
    interp_tp = dm._get_time_points(time_window) # timepoints
    # This is a lazy fix. Basically, errors out when pd==0
    try:
        idx = (np.where(interp_tp<pd)[0][-1]) + 1 # idx before the phase delay value
    except:
        idx = 0
    interp_tp_copy = interp_tp.copy()
    interp_tp += idx*TR
    # Get phase delayed shifted tp
    pd_tp = dm._get_time_points(time_window) # timepoints
    pd_tp += pd
    # Interpolate timepoints to grid that matches the original sampling rate
    cs = CubicSpline(pd_tp, ts)
    interp_ts = cs(interp_tp)

    return interp_tp, interp_ts

def calculate_roi_phase_shifted_bold(roi_bold, roi_phase_delay, time_window, TR):

    n_coords = roi_phase_delay.shape[0]
    interp_roi_bold = np.zeros_like(roi_bold)
    pd_updated_tp = np.zeros_like(roi_bold)
    for vertex_id in range(n_coords):
        tp, ts = interpolate_vertex_bold_after_phase_shift(roi_bold, roi_phase_delay, vertex_id, time_window, TR)
        interp_roi_bold[:, vertex_id] = ts
        pd_updated_tp[:, vertex_id] = tp

    return pd_updated_tp, interp_roi_bold

def process_tps(tps, f, phased=False, precision=1e-2):
    if phased:
        period = 1/float(f)
        # Get precision accurate timepoints
        rounded_tps = np.round(np.fmod(tps,period)/precision)*precision
    else:
        rounded_tps = np.round(tps/precision)*precision
    # Get phased timepoints
    unique_tps = np.unique(rounded_tps)

    return rounded_tps, unique_tps

def calculate_mean_bold(roi_bold, tps, f, phased=False, precision=1e-2):

    rounded_tps, unique_tps = process_tps(tps, f, phased=phased, precision=precision)
    
    mean_bold = np.zeros_like(unique_tps)
    for ix, phased_tp in enumerate(unique_tps):
        bold_per_tp = roi_bold[rounded_tps == phased_tp].mean()
        mean_bold[ix] = bold_per_tp

    return unique_tps, mean_bold

def _calculate_frequency_spectrum(bold_data, TR):

    from scipy.signal import welch

    fs, power = welch(bold_data, fs = 1/TR)

    return fs, power

def calculate_frequency_spectrum(bold_data, TR):

    if len(bold_data.shape) == 1:
        fs, power = _calculate_frequency_spectrum(bold_data, TR)
        return fs, power

    else:
        n_vertices = bold_data.shape[-1]
        for vertex_ix, vertex_id in enumerate(range(n_vertices)):
            _bold_data = bold_data[:,vertex_id]
            if vertex_ix == 0:
                fs, power = _calculate_frequency_spectrum(_bold_data, TR)
                power = power[:, np.newaxis]
            else:
                _fs, _power = _calculate_frequency_spectrum(_bold_data, TR)
                n_same = (_fs==fs).sum()
                assert n_same == fs.shape[0], f"{n_same} != {fs.shape[0]}"
                power = np.concatenate((power, _power[:,np.newaxis]), axis=1)

        return fs, power

"""
Train-test reliability measures
"""
def _estimate_reliability_from_all_vertex_data(bootstrapped_data, train_key, test_key, circ_r=False):

    from astropy.stats import circcorrcoef
    from scipy.stats import pearsonr

    x = bootstrapped_data[train_key]
    y = bootstrapped_data[test_key]
    roi_coords = bootstrapped_data["roi_coords"]

    n_bootstraps = x.shape[0]

    reliability = np.zeros((n_bootstraps,))
    for n in range(n_bootstraps):
        if circ_r:
            reliability[n] = circcorrcoef(x[n, roi_coords], y[n, roi_coords])
        else:
            reliability[n], _ = pearsonr(x[n,roi_coords], y[n,roi_coords])

    return reliability

def _estimate_reliability_of_frequency_from_roi_data(bootstrapped_data, frequency_grid_key, train_key, test_key, frequency):

    from scipy.stats import pearsonr

    frequency_grid = bootstrapped_data[frequency_grid_key]
    x = bootstrapped_data[train_key]
    y = bootstrapped_data[test_key]

    abs_diff = np.abs(frequency_grid - frequency)
    closest_index = np.argmin(abs_diff)

    print(f"Frequency of interest: {frequency} [{frequency_grid[closest_index]}]")

    n_frequencies, n_vertex, n_bootstraps = x.shape

    reliability = np.zeros((n_bootstraps,))
    for n in range(n_bootstraps):
        reliability[n], _ = pearsonr(
            x[closest_index, :, n],
            y[closest_index, :, n],
        )

    return reliability

def train_test_reliability(bootstrapped_data, metric, frequency=None, task_id=None):

    METRICS = ["phasedelay", "zscore", "power"]
    assert metric in METRICS, f"{metric} must be in {METRICS}"

    if metric == "power":
        assert frequency is not None, f"Specify frequency for extracting power."
        assert task_id is not None, f"Specify task_id"

    if metric == "phasedelay":
        train_key = "data-train_roi_phase_delay"
        test_key = "data-test_roi_phase_delay"

        return _estimate_reliability_from_all_vertex_data(bootstrapped_data, train_key, test_key, circ_r=True)
    
    if metric == "zscore":
        train_key = "data-train_roi_z_score"
        test_key = "data-test_roi_z_score"

        return _estimate_reliability_from_all_vertex_data(bootstrapped_data, train_key, test_key, circ_r=False)

    if metric == "power":
        frequency_grid_key = f"data-train_task-{task_id}_roi_frequencies"
        train_key = f"data-train_task-{task_id}_roi_power"
        test_key = f"data-test_task-{task_id}_roi_power"

        return _estimate_reliability_of_frequency_from_roi_data(bootstrapped_data, frequency_grid_key, train_key, test_key, frequency)

"""
Other
"""
def get_timepoints(time_window, TR):
    dm = DesignMatrix(time_window, None, TR=TR)

    return dm._get_time_points(time_window)

def save_bootstrapped_data(out_dir, task_id, store_bootstrapped_data):
    
    import pickle

    pkl_out = out_dir / f"task-{task_id}_bootstrapped_data.pkl"
    with open(pkl_out, "wb") as f:
        pickle.dump(store_bootstrapped_data, f)

app = typer.Typer()

@app.command()
def run_merge_bootstrap(
    experiment_id: Annotated[str, typer.Option()],
    mri_id: Annotated[str, typer.Option()],
    sub_id: Annotated[str, typer.Option()],
    smooth_mm: Annotated[int, typer.Option()],
    desc_id: Annotated[str, typer.Option()],
    n_iterations: Annotated[int, typer.Option()],
    n_batches_expected: Annotated[int, typer.Option()],
    roi_task_id: Annotated[str, typer.Option()],
    roi_task_frequency: Annotated[float, typer.Option()],
    task_conditions: Annotated[List[str], typer.Option()],
    fractional_overlap_threshold: Annotated[float, typer.Option()],
    search_frequencies: Annotated[List[float], typer.Option()],
    time_window: Annotated[Tuple[int, int], typer.Option()],
    TR: Annotated[float, typer.Option()],
    base_dir: Annotated[str, typer.Option()],
    roi_task_frequency_intersect_with: Annotated[Optional[float], typer.Option()] = None,
    control_roi_size: Annotated[bool, typer.Option("--control_roi_size")] = False,
):

    TP_DECIMAL = 1
    TASK_CONDITIONS = task_conditions
    assert len(TASK_CONDITIONS) == 1
    _task_id = TASK_CONDITIONS[0]

    """Set-up
    """
    # Subject ID
    if not sub_id.startswith('sub-'):
        sub_id = f"sub-{sub_id}"
    # Set base directory
    base_dir = Path(base_dir)
    assert base_dir.exists(), f"{base_dir} does not exist."
    # Wildcard to find all bootstraps
    base_bootstrap_dir_wc = (
        f"experiment-{experiment_id}"
        f"_mri-{mri_id}_smooth-{smooth_mm}_truncate-{time_window[0]}-{time_window[1]}"
        f"_n-{n_iterations}_batch-*_desc-{desc_id}_bootstrap"
    )
    n_batches_found = len(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}"))
    assert n_batches_found == n_batches_expected, f"{n_batches_found} bootstrap directories were found for {sub_id}. [Expected: {n_batches_expected}]"

    # Set output directory
    if roi_task_frequency_intersect_with is None:
        out_dir_base = (
            f"experiment-{experiment_id}"
            f"_mri-{mri_id}_smooth-{smooth_mm}_truncate-{time_window[0]}-{time_window[1]}"
            f"_n-{n_iterations*n_batches_expected}_batch-merged_desc-{desc_id}_roi-{roi_task_id}-{roi_task_frequency}_fo-{fractional_overlap_threshold}_bootstrap"
        )
    else:
        if control_roi_size:
            assert roi_task_frequency_intersect_with > roi_task_frequency, f"Expect {roi_task_frequency_intersect_with} > {roi_task_frequency}"
            out_dir_base = (
                f"experiment-{experiment_id}"
                f"_mri-{mri_id}_smooth-{smooth_mm}_truncate-{time_window[0]}-{time_window[1]}"
                f"_n-{n_iterations*n_batches_expected}_batch-merged_desc-{desc_id}_roi-{roi_task_id}-{roi_task_frequency}_controlroisizetomatch-{roi_task_frequency_intersect_with}_fo-{fractional_overlap_threshold}_bootstrap"
            )
        else:
            out_dir_base = (
                f"experiment-{experiment_id}"
                f"_mri-{mri_id}_smooth-{smooth_mm}_truncate-{time_window[0]}-{time_window[1]}"
                f"_n-{n_iterations*n_batches_expected}_batch-merged_desc-{desc_id}_roi-{roi_task_id}-{roi_task_frequency}-{roi_task_frequency_intersect_with}_fo-{fractional_overlap_threshold}_bootstrap"
            )

    out_dir = base_dir / out_dir_base / sub_id / "bootstrap"
    if not out_dir.exists():
        out_dir.mkdir(parents=True, exist_ok=True)

    # Get timepoints for data
    tp = get_timepoints(time_window, TR)

    """Aggregate relevant batch-wise data
    """
    data_dict = {}
    # Check whether `roi_task_frequency` exists
    available_frequencies = get_frequencies_per_task(base_dir, base_bootstrap_dir_wc, sub_id, roi_task_id)
    assert roi_task_frequency in available_frequencies, f"{roi_task_frequency} not in {available_frequencies}"
    if roi_task_frequency_intersect_with is not None:
        assert roi_task_frequency_intersect_with in available_frequencies, f"{roi_task_frequency_intersect_with} not in {available_frequencies}"

    roi_task_id = get_updated_task_id_name(base_dir, base_bootstrap_dir_wc, sub_id, roi_task_id) # Add task_id suffix (i.e., Q1/Q2)
    for data_split_id in ["train", "test"]:
        # Store train/test splits across bootstraps for ROI task, and `TASK_CONDITIONS`
        data_dict[f"{data_split_id}_splits_task-{roi_task_id}"] = combine_bootstrapped_splits(
            sorted(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/task-{roi_task_id}*_{data_split_id}_splits.txt")),
            n_batches_expected,
            n_iterations
        )
        for task_id in TASK_CONDITIONS:
            task_id = get_updated_task_id_name(base_dir, base_bootstrap_dir_wc, sub_id, task_id)
            if roi_task_id == task_id:
                continue
            data_dict[f"{data_split_id}_splits_task-{task_id}"] = combine_bootstrapped_splits(
                sorted(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/task-{task_id}*_{data_split_id}_splits.txt")),
                n_batches_expected,
                n_iterations
            )
        # Store frequency-dependent bootstrapped files
        # Phase delay
        data_dict[f"{data_split_id}_phase_delay_task-{roi_task_id}_f-{roi_task_frequency}"] = combine_bootstrapped_dtseries(
            sorted(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/bootstrap/*task-{roi_task_id}*_f-{roi_task_frequency}_data-{data_split_id}_n-{n_iterations}_phasedelay.dtseries.nii")),
            out_dir / f"{sub_id}_ses-main_task-{roi_task_id}_f-{roi_task_frequency}_data-{data_split_id}_n-{n_batches_expected*n_iterations}_phasedelay.dtseries.nii",
            n_batches_expected
        )
        # Z score
        data_dict[f"{data_split_id}_z_score_task-{roi_task_id}_f-{roi_task_frequency}"] = combine_bootstrapped_dtseries(
            sorted(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/bootstrap/*task-{roi_task_id}*_f-{roi_task_frequency}_data-{data_split_id}_n-{n_iterations}_z_score.dtseries.nii")),
            out_dir / f"{sub_id}_ses-main_task-{roi_task_id}_f-{roi_task_frequency}_data-{data_split_id}_n-{n_batches_expected*n_iterations}_z_score.dtseries.nii",
            n_batches_expected
        )
        # Binary activation map
        data_dict[f"{data_split_id}_activation_task-{roi_task_id}_f-{roi_task_frequency}"] = combine_bootstrapped_dtseries(
            sorted(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/bootstrap/*task-{roi_task_id}*_f-{roi_task_frequency}_data-{data_split_id}_n-{n_iterations}_activations.dtseries.nii")),
            out_dir / f"{sub_id}_ses-main_task-{roi_task_id}_f-{roi_task_frequency}_data-{data_split_id}_n-{n_batches_expected*n_iterations}_activations.dtseries.nii",
            n_batches_expected
        )
        # Wholebrain map
        data_dict[f"{data_split_id}_wholebrain_task-{roi_task_id}_f-{roi_task_frequency}"] = combine_bootstrapped_dtseries(
            sorted(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/bootstrap/*task-{roi_task_id}*_f-{roi_task_frequency}_data-{data_split_id}_n-{n_iterations}_mask.dtseries.nii")),
            out_dir / f"{sub_id}_ses-main_task-{roi_task_id}_f-{roi_task_frequency}_data-{data_split_id}_n-{n_batches_expected*n_iterations}_mask.dtseries.nii",
            n_batches_expected
        )
    data_dict["wholebrain_mask"] = combine_bootstrapped_dtseries(
        sorted(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/bootstrap/*task-{_task_id}*_f-{search_frequencies[0]}_data-test_n-{n_iterations}_mask.dtseries.nii")),
        out_dir / f"{sub_id}_ses-main_task-{_task_id}_f-{search_frequencies[0]}_data-test_n-{n_batches_expected*n_iterations}_mask.dtseries.nii",
        n_batches_expected
    )
    if roi_task_frequency_intersect_with is not None:
        data_split_id = "train"
        data_dict[f"{data_split_id}_second_f-{roi_task_frequency_intersect_with}"] = combine_bootstrapped_dtseries(
            sorted(search(base_dir, f"{base_bootstrap_dir_wc}/{sub_id}/bootstrap/*task-{roi_task_id}*_f-{roi_task_frequency_intersect_with}_data-{data_split_id}_n-{n_iterations}_activations.dtseries.nii")),
            out_dir / f"{sub_id}_ses-main_task-{roi_task_id}_f-{roi_task_frequency_intersect_with}_data-{data_split_id}_n-{n_batches_expected*n_iterations}_activations.dtseries.nii",
            n_batches_expected
        )

    
    """Set data storage variable
    """
    store_bootstrapped_data = {}

    """Get ROIs based on z threshold and fractional overlap
    """
    roi_dict = get_rois(
        data_dict, _task_id, 
        fractional_overlap_threshold, 
        n_batches_expected, n_iterations,
        control_roi_size,
    )
    # Verbose
    for roi_key, v in roi_dict.items():
        print(f"sub-id: {sub_id}, roi-key: {roi_key}, n-voxels-in-roi: {v.sum()}") 
    assert len(roi_dict) == 1, f"Expect only 1 roi generated in `roi_dict` {len(roi_dict)} {roi_dict} {data_dict}"
    # Get ROI coordinates
    roi = roi_dict[roi_key][0, :]
    roi_coords = roi == 1
    n_vertices = roi_coords.sum()
    # Store `roi_coords`
    store_bootstrapped_data["roi_coords"] = roi_coords

    """Load metrics from ROI 
    - phase delays
    - z-scores
    """
    # Load phase delays and process
    train_phase_delay_across_bootstrap = nib.load(data_dict[f"train_phase_delay_task-{roi_task_id}_f-{roi_task_frequency}"]).get_fdata()
    train_phase_delay_across_bootstrap = process_phase_delay(train_phase_delay_across_bootstrap, roi_task_frequency)
    test_phase_delay_across_bootstrap = nib.load(data_dict[f"test_phase_delay_task-{roi_task_id}_f-{roi_task_frequency}"]).get_fdata()
    test_phase_delay_across_bootstrap = process_phase_delay(test_phase_delay_across_bootstrap, roi_task_frequency)
    # Load z-score
    train_z_score_across_bootstrap = nib.load(data_dict[f"train_z_score_task-{roi_task_id}_f-{roi_task_frequency}"]).get_fdata()
    test_z_score_across_bootstrap = nib.load(data_dict[f"test_z_score_task-{roi_task_id}_f-{roi_task_frequency}"]).get_fdata()
    # Store loaded metrics & ROI coordinates
    store_bootstrapped_data["data-train_roi_phase_delay"] = train_phase_delay_across_bootstrap
    store_bootstrapped_data["data-test_roi_phase_delay"] = test_phase_delay_across_bootstrap
    store_bootstrapped_data["data-train_roi_z_score"] = train_z_score_across_bootstrap
    store_bootstrapped_data["data-test_roi_z_score"] = test_z_score_across_bootstrap

    """ Loop over `TASK_CONDITIONS`
    """
    for _, task_id in enumerate(TASK_CONDITIONS):
        task_id = get_updated_task_id_name(base_dir, base_bootstrap_dir_wc, sub_id, task_id)

        """Loop over bootstraps
        """
        for test_ix, (train_split, test_split) in enumerate(
            zip(
                data_dict[f"train_splits_task-{task_id}"], 
                data_dict[f"test_splits_task-{task_id}"]
            )
        ):

            """Load BOLD
            """
            # Train
            train_split = read_splits(train_split)
            train_bold = get_average_bold(train_split, normalize_type="pct")
            train_roi_bold = train_bold[:, roi_coords] 
            #train_roi_tps = np.hstack([tp[:,np.newaxis]] * n_vertices)
            # Test
            test_split = read_splits(test_split)
            test_bold = get_average_bold(test_split, normalize_type="pct")
            test_roi_bold = test_bold[:, roi_coords]
            test_roi_tps = np.hstack([tp[:,np.newaxis]]* n_vertices)
            # Phase delay
            train_roi_phase_delay = train_phase_delay_across_bootstrap[test_ix, roi_coords]
            """
            # Z-score
            train_roi_z_score = train_z_score_across_bootstrap[test_ix, roi_coords]
            test_roi_z_score = test_z_score_across_bootstrap[test_ix, roi_coords]
            """
            # Get phase-shifted BOLD and associated timepoints
            pd_test_roi_tps, pd_test_roi_bold = calculate_roi_phase_shifted_bold(test_roi_bold, train_roi_phase_delay, time_window, TR)
            # Get rid of first volume (interpolation of phase-shifted case gives problem at first timepoint)
            pd_test_roi_tps = np.round(pd_test_roi_tps[1:, :], decimals=TP_DECIMAL)
            pd_test_roi_bold = pd_test_roi_bold[1:, :]
            test_roi_tps = np.round(test_roi_tps[1:, :], decimals=TP_DECIMAL)
            train_roi_bold = train_roi_bold[1:,:] # Train set
            test_roi_bold = test_roi_bold[1:, :] # Test set
            # Calculate vertex-wise frequency spectrum
            train_roi_fs, train_roi_power = calculate_frequency_spectrum(train_roi_bold, TR)
            test_roi_fs, test_roi_power = calculate_frequency_spectrum(test_roi_bold, TR)

            """Loop over search frequencies
            """
            for search_f in search_frequencies:
                # Logging
                print(f"{str(test_ix).zfill(4)}/{str(n_iterations*n_batches_expected).zfill(4)} || Search frequency: {search_f}")
                # Calculate mean phased timeseries (across roi)
                phased_tps, phased_bold = calculate_mean_bold(test_roi_bold, test_roi_tps, search_f, phased=True)
                phased_pd_tps, phased_pd_bold = calculate_mean_bold(pd_test_roi_bold, pd_test_roi_tps, search_f, phased=True)
                # Calculate mean unphased timeseries (across roi)
                unphased_tps, unphased_bold = calculate_mean_bold(test_roi_bold, test_roi_tps, search_f, phased=False)
                unphased_pd_tps, unphased_pd_bold = calculate_mean_bold(pd_test_roi_bold, pd_test_roi_tps, search_f, phased=False)
                # Matchs timepoints of unphased_pd_tps to unphased_tps
                keep_coords = np.where(
                    (unphased_pd_tps >= 1/roi_task_frequency) &
                    (unphased_pd_tps <= unphased_tps.max())
                )
                unphased_pd_tps = unphased_pd_tps[keep_coords]
                unphased_pd_bold = unphased_pd_bold[keep_coords]
                # Calculate mean power spectrum (across roi)
                unphased_fs, unphased_power = calculate_frequency_spectrum(unphased_bold, TR)
                unphased_pd_fs, unphased_pd_power = calculate_frequency_spectrum(unphased_pd_bold, TR)
                
                # store data: mean unphased frequency spectrum & mean unphased/phased timeseries
                if test_ix == 0:
                    # Store timepoint and frequency grids (this is consistent across bootstraps)
                    store_bootstrapped_data[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{search_f}_frequencies"] = unphased_fs
                    store_bootstrapped_data[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{search_f}_frequencies"] = unphased_pd_fs
                    store_bootstrapped_data[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{search_f}_timepoints"] = unphased_tps
                    store_bootstrapped_data[f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-no_f-{search_f}_timepoints"] = phased_tps
                    store_bootstrapped_data[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{search_f}_timepoints"] = unphased_pd_tps
                    store_bootstrapped_data[f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-yes_f-{search_f}_timepoints"] = phased_pd_tps
                    # Store magnitude and power
                    for _key, _bold in zip([
                        f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{search_f}_power",
                        f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{search_f}_power",
                        f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{search_f}_bold",
                        f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-no_f-{search_f}_bold",
                        f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{search_f}_bold",
                        f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-yes_f-{search_f}_bold",
                    ], [unphased_power, unphased_pd_power, unphased_bold, phased_bold, unphased_pd_bold, phased_pd_bold]):
                        store_bootstrapped_data[_key] = _bold[:,np.newaxis]
                else:
                    # Store magnitude and power
                    for _key, _bold in zip([
                        f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{search_f}_power",
                        f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{search_f}_power",
                        f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{search_f}_bold",
                        f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-no_f-{search_f}_bold",
                        f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{search_f}_bold",
                        f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-yes_f-{search_f}_bold",
                    ], [unphased_power, unphased_pd_power, unphased_bold, phased_bold, unphased_pd_bold, phased_pd_bold]):
                        store_bootstrapped_data[_key] = np.concatenate((store_bootstrapped_data[_key], _bold[:,np.newaxis]), axis=1)

            # store data: frequency spectrum of all vertices in ROI
            # store data: timeseries for all vertices in ROI
            if test_ix == 0:
                # PSD
                store_bootstrapped_data[f"data-train_task-{task_id}_roi_frequencies"] = train_roi_fs
                store_bootstrapped_data[f"data-train_task-{task_id}_roi_power"] = train_roi_power[:,:,np.newaxis]
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_frequencies"] = test_roi_fs
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_power"] = test_roi_power[:,:,np.newaxis]
                # Timeseries
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_timepoints"] = test_roi_tps # sample of all voxels' timeseries from the 0th bootstrap
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_bold"] = test_roi_bold[:,:,np.newaxis] # sample of all voxels' timeseries from the 0th bootstrap
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_phaseadjusted_timepoints"] = pd_test_roi_tps # sample of all phase-shifted voxels' timeseries from the 0th bootstrap
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_phaseadjusted_bold"] = pd_test_roi_bold[:,:,np.newaxis] # sample of all phase-shifted voxels' timeseries from the 0th bootstrap
            else:
                store_bootstrapped_data[f"data-train_task-{task_id}_roi_power"] = np.concatenate((store_bootstrapped_data[f"data-train_task-{task_id}_roi_power"], train_roi_power[:,:,np.newaxis]), axis=2) 
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_power"] = np.concatenate((store_bootstrapped_data[f"data-test_task-{task_id}_roi_power"], test_roi_power[:,:,np.newaxis]), axis=2) 
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_bold"] = np.concatenate((store_bootstrapped_data[f"data-test_task-{task_id}_roi_bold"], test_roi_bold[:,:,np.newaxis]), axis=2)
                store_bootstrapped_data[f"data-test_task-{task_id}_roi_phaseadjusted_bold"] = np.concatenate((store_bootstrapped_data[f"data-test_task-{task_id}_roi_phaseadjusted_bold"], pd_test_roi_bold[:,:,np.newaxis]), axis=2)

    # Save power spectrums (masked by ROI)
    dtseries_template = data_dict[f"train_z_score_task-{roi_task_id}_f-{roi_task_frequency}"]
    for f, task_id, data_split in itertools.product(
        search_frequencies, 
        TASK_CONDITIONS, 
        ["train", "test"],
    ):
        task_id = get_updated_task_id_name(base_dir, base_bootstrap_dir_wc, sub_id, task_id)
        dtseries_out = out_dir / f"{sub_id}_ses-main_task-{task_id}_f-{f}_data-{data_split}_n-{n_batches_expected*n_iterations}_power.dtseries.nii"
        create_bootstrapped_power_metric_map_from_roi_data(
            store_bootstrapped_data, f, task_id, data_split, dtseries_template, dtseries_out
        )

    # Save `store_bootstrapped_data` as pkl [ `out_dir`` / "bootstrapped_data.pkl"]
    save_bootstrapped_data(out_dir, _task_id, store_bootstrapped_data)

    print("Complete")

if __name__ == "__main__":
    app()