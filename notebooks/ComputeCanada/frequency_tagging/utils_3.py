import glob
from pathlib import Path
import pickle
import numpy as np

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

"""
Data handler
"""

class DataHandler:

    def __init__(
        self, 
        base_dir,
        sub_id,
        experiment_id,
        mri_id, 
        smooth_id,
        time_window,
        total_batch_size,
        desc_id, 
        roi_task_id,
        roi_task_frequency,
        z_threshold,
        fractional_overlap_threshold,
    ):

        self.base_dir = Path(base_dir)
        self.sub_id = self._get_sub_id(sub_id)
        self.experiment_id = experiment_id
        self.mri_id = mri_id
        self.smooth_id = smooth_id
        self.time_window = time_window
        self.total_batch_size = total_batch_size
        self.desc_id = desc_id
        self.roi_task_id = roi_task_id
        self.roi_task_frequency = float(roi_task_frequency)
        self.z_threshold = float(z_threshold)
        self.fractional_overlap_threshold = fractional_overlap_threshold
        self.merged_batch_dir = self._get_merged_batch_dir()
        self.entrain_id = self._get_updated_task_id(self.merged_batch_dir.parent, self.merged_batch_dir.stem, "entrain")
        self.control_id = self._get_updated_task_id(self.merged_batch_dir.parent, self.merged_batch_dir.stem, "control")

    def load_bootstrapped_data(self,):
        
        pkl_out = self.merged_batch_dir / "bootstrapped_data.pkl"
        assert pkl_out.exists(), f"{pkl_out} does not exist."
        with open(pkl_out, "rb") as f:
            store_bootstrapped_data = pickle.load(f)

        return store_bootstrapped_data

    def _get_sub_id(self, sub_id):

        if not sub_id.startswith('sub-'):
            sub_id = f"sub-{sub_id}"

        return sub_id

    def _get_updated_task_id(self, base_dir, wc, task_id):

        paths = search(base_dir, f"{wc}/*task-{task_id}*")
        all_tasks = get_ids(paths, "_task-")
        updated_task_id = None
        for task in all_tasks:
            if task.startswith(task_id):
                updated_task_id = task

                return updated_task_id

        assert updated_task_id is not None, f"{task_id} could not be updated. All detected tasks: {all_tasks}"

    def _get_merged_batch_dir(self):
        merged_batch_dir = (
            f"experiment-{self.experiment_id}_mri-{self.mri_id}"
            f"_smooth-{self.smooth_id}_truncate-{self.time_window[0]}-{self.time_window[1]}"
            f"_n-{self.total_batch_size}_batch-merged"
            f"_desc-{self.desc_id}"
            f"_roi-{self.roi_task_id}-{self.roi_task_frequency}"
            f"_z-{self.z_threshold}_fo-{self.fractional_overlap_threshold}"
            "_bootstrap"
        )

        abs_merged_batch_dir = self.base_dir / merged_batch_dir / self.sub_id / "bootstrap"
        assert abs_merged_batch_dir.exists(), f"{abs_merged_batch_dir} does not exist."

        return abs_merged_batch_dir

def setup_data_dict(search_frequencies):
    data_dict = {
        "z_threshold": [],
        "sub_id": [],
        "quadrant": [],
        "roi_task_id": [],
        "roi_task_frequency": [],
        "roi_map": [], # boolean array of ROIs
        "n_vertices": [], # Metric
        "metric-zscore_reliability": [], # Metric
        "metric-phasedelay_reliability": [], # Metric
    }
    for search_f in search_frequencies:
        data_dict[f"f-{search_f}_task-entrain_metric-powerspectrum_reliability"] = []
        data_dict[f"f-{search_f}_task-control_metric-powerspectrum_reliability"] = []

    return data_dict

def setup_roi_data_dict():
    data_dict = {
        "z_threshold": [],
        "sub_id": [],
        "quadrant": [],
        "roi_task_id": [],
        "roi_task_frequency": [],
        "hemi_label": [],
        "hcp_label": [],
        "n_vertices": [], 
        "n_vertices_normalized": [],
        "coordinates": [],
    }

    return data_dict

"""
Plotting
"""
def plot_hexbin_timeseries(
    data_store,
    task_id,
    phaseadjusted,
    fig,
    ax,
    CMAP="magma",
    GRIDSIZE=150,
    VMAX=50,
    ZORDER=1,
):
    if phaseadjusted:
        phaseadjusted_label = "shifted [+]"
        roi_tps, roi_bold = data_store[f"task-{task_id}_bootstrap_phaseadjusted_sample"]
    else:
        phaseadjusted_label = "shifted [-]"
        roi_tps, roi_bold = data_store[f"task-{task_id}_bootstrap_sample"]

    VMAX = roi_bold.shape[1] # Set max to voxel count
    
    hb = ax.hexbin(
        roi_tps.flatten(),
        roi_bold.flatten(),
        gridsize=GRIDSIZE,
        mincnt=1,
        zorder=ZORDER,
        cmap=CMAP,
        vmax=VMAX,
    )

    ax.set_title(f'{task_id}, {phaseadjusted_label}')
    cb = fig.colorbar(hb)

def _get_confidence_interval_from_timeseries(bold, confidence_level=.95):
    ends = ( 1 - .95 ) / 2
    upper_percentile = 1-ends
    lower_percentile = ends
    n_vertices, n_bootstraps = bold.shape
    lower_idx = np.round(lower_percentile*n_bootstraps)
    upper_idx = np.round(upper_percentile*n_bootstraps)
    lower_bound = np.zeros((n_vertices,))
    upper_bound = np.zeros((n_vertices,))
    for n in range(n_vertices):
        bold_per_tp = bold[n,:]
        bold_per_tp = np.sort(bold_per_tp)
        lower_bound[n] = bold_per_tp[int(lower_idx)]
        upper_bound[n] = bold_per_tp[int(upper_idx)]

    return lower_bound, upper_bound

def _get_mean_from_timeseries(bold):

    return bold.mean(axis=1)

def _process_bootstrapped_timeseries(bold, confidence_level=0.95):
    
    lower_bound, upper_bound = _get_confidence_interval_from_timeseries(bold, confidence_level=confidence_level)
    mean_bold = _get_mean_from_timeseries(bold)

    return mean_bold, lower_bound, upper_bound

def plot_unphased_timeseries(data_store, task_id, phaseadjusted, frequency, fig, ax, color, lw=.5, linestyle='-'):

    if phaseadjusted:
        phaseadjusted_label = "[+]"
        tps = data_store[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{frequency}_timepoints"]
        bold = data_store[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{frequency}_bold"]
    else:
        phaseadjusted_label = "[-]"
        tps = data_store[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{frequency}_timepoints"]
        bold = data_store[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{frequency}_bold"]
    mean_bold, lb, ub = _process_bootstrapped_timeseries(bold, confidence_level=0.95)
    ax.plot(tps, mean_bold, lw=lw, c=color, linestyle=linestyle)
    ax.fill_between(tps, lb, ub, alpha=.2, color=color, linewidth=0)

def plot_phased_timeseries(data_store, task_id, phaseadjusted, frequency, fig, ax, color, lw=.5, linestyle='-'):

    if phaseadjusted:
        phaseadjusted_label = "[+]"
        tps = data_store[f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-yes_f-{frequency}_timepoints"]
        bold = data_store[f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-yes_f-{frequency}_bold"]
    else:
        phaseadjusted_label = "[-]"
        tps = data_store[f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-no_f-{frequency}_timepoints"]
        bold = data_store[f"task-{task_id}_phasedtimeseries-yes_phaseadjusted-no_f-{frequency}_bold"]
    mean_bold, lb, ub = _process_bootstrapped_timeseries(bold, confidence_level=0.95)
    ax.plot(tps, mean_bold, lw=lw, c=color, linestyle=linestyle)
    ax.fill_between(tps, lb, ub, alpha=.2, color=color, linewidth=0)

def plot_frequency_spectrum(data_store, task_id, phaseadjusted, frequency, fig, ax, color, lw=.5, linestyle='-', search_frequencies = None):

    if phaseadjusted:
        phaseadjusted_label = "[+]"
        tps = data_store[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{frequency}_frequencies"]
        bold = data_store[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-yes_f-{frequency}_power"]
    else:
        phaseadjusted_label = "[-]"
        tps = data_store[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{frequency}_frequencies"]
        bold = data_store[f"task-{task_id}_phasedtimeseries-no_phaseadjusted-no_f-{frequency}_power"]
    mean_bold, lb, ub = _process_bootstrapped_timeseries(bold, confidence_level=0.95)
    ax.plot(tps, mean_bold, lw=lw, c=color, linestyle=linestyle, zorder = 3)
    ax.fill_between(tps, lb, ub, alpha=.2, color=color, linewidth=0, zorder = 2)
    if search_frequencies is not None:
        for f in search_frequencies:
            ax.axvline(x=f, color='k', linestyle='dotted', zorder=1, lw=.2, alpha=.8)



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