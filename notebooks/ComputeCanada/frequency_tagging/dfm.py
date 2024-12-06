from pathlib import Path
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import welch
from collections import defaultdict

"""Functions
"""
def read_pkl(datadir, n_bootstraps, sub_id, roi_task_id, roi_frequency, task_id, experiment_id="1_frequency_tagging", mri_id="7T", fo=.8, pval="uncp"):
    """No longer supporting `control_roi_size`, and `roi_frequency_1` inputs
    These pkls were generated using the following (now, unused) notebooks
        - 3_merge_bootstrap_controlroisize_basic.ipynb
        - 3_merge_bootstrap_intersection_basic.ipynb
    """

    import pickle

    bootstrap_pkl: Path = datadir / f"experiment-{experiment_id}_mri-{mri_id}_smooth-0_truncate-39-219_n-{n_bootstraps}_batch-merged_desc-basic_roi-{roi_task_id}-{roi_frequency}_pval-{pval}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap/task-{task_id}_bootstrapped_data.pkl"
    if not bootstrap_pkl.exists():
        print(f"Warning: {bootstrap_pkl} does not exist.\nReturn None")
        return None

    print(f"Reading: {bootstrap_pkl}")
    with open(bootstrap_pkl, 'rb') as f:
        data = pickle.load(f)

    return data

def find_raw_bold(i):
    
    import os
    
    experiment_id = str(i.parent).split("experiment-")[1].split('_mri-')[0]
    mri_id = str(i.parent).split("mri-")[1].split('_')[0]
    sub_id = i.stem.split('sub-')[1].split('_')[0]
    ses_id = i.stem.split('ses-')[1].split('_')[0]
    task_id = i.stem.split('task-')[1].split('_')[0]
    run_id = i.stem.split('run-')[1].split('_')[0]

    directory = f"/data/{experiment_id}/{mri_id}/bids/derivatives/oscprep_grayords_fmapless/bold_preproc/sub-{sub_id}/ses-{ses_id}/func"
    raw_bold = [f"{directory}/{file}" for file in os.listdir(directory) if f"run-{run_id}" in file and f"task-{task_id}" in file and file.endswith("bold.dtseries.nii")]
    assert len(raw_bold) == 1, f"Multiple raw bolds found: {raw_bold}"

    return Path(raw_bold[0])

def average_bold(bold_list):
    for bold_ix, bold in enumerate(bold_list):
        _bold_data = nib.load(bold).get_fdata()
        if bold_ix == 0:
            y_all = _bold_data.copy() 
        else:
            y_all += _bold_data.copy()
        
    y_all /= len(bold_list)
    y_all = (( y_all - y_all.mean(0)) / y_all.std(0) ).T

    return y_all

def read_bootstrap_txt(bootstrap_txt, bootstrap_idx):
    with open(bootstrap_txt, "r") as f:
        lines = f.readlines()

    fs = lines[bootstrap_idx]
    raw_bolds = []
    raw_windowed_bolds = []
    processed_bolds = []
    for i in fs.split(','):
        i = Path(i.strip())
        raw_windowed_bold = Path(str(i).replace("desc-denoised_bold.dtseries.nii","desc-windowed_bold.dtseries.nii"))
        raw_bold = find_raw_bold(Path(i))
        assert i.exists(), f"{i} not found."
        assert raw_windowed_bold.exists(), f"{raw_windowed_bold} not found"
        assert raw_bold.exists(), f"{raw_bold} not found."
        raw_bolds.append(raw_bold)
        raw_windowed_bolds.append(raw_windowed_bold)
        processed_bolds.append(i)

    raw_avg = average_bold(raw_bolds)
    raw_windowed_avg = average_bold(raw_windowed_bolds)
    processed_avg = average_bold(processed_bolds)

    #print(f"Raw: {raw_avg.shape}")
    #print(f"Raw: {raw_windowed_avg.shape}")
    #print(f"Raw: {processed_avg.shape}")

    return raw_avg, raw_windowed_avg, processed_avg

def extract_carpet_data(data, task_id, task_quadrant, bootstrap_id, phased_flag):

    if phased_flag:
        data_tps = data[f'data-test_task-{task_id}{task_quadrant}_roi_phaseadjusted_timepoints']
        data_bold = data[f'data-test_task-{task_id}{task_quadrant}_roi_phaseadjusted_bold'][:,:,bootstrap_id]
        updated_tps, mean_bold = [], []
        n_voxels = data_bold.shape[1]
        for single_tp in np.unique(data_tps):
            coords = (data_tps == single_tp)
            tp_all = np.all((coords).sum(0)) == 1
            if tp_all:
                updated_tps.append(single_tp)
                mean_bold.append(data_bold[coords])
        updated_tps = np.array(updated_tps)
        assert _check_difference(updated_tps)
        
        return updated_tps, np.vstack(mean_bold)

    else:
        # Select timeseries (timepoints x voxels x bootstraps)
        return data[f'data-test_task-{task_id}{task_quadrant}_roi_timepoints'], data[f'data-test_task-{task_id}{task_quadrant}_roi_bold'][:,:,bootstrap_id]

def find_quadrant_id_from_keys(_dict, task_id):
    for i in _dict.keys():
        if task_id in i:
            q_idx = i.find("Q")
            q_id = i[q_idx:q_idx+2]
            assert q_id in ['Q1', 'Q2']
            return q_id
    raise ValueError("No quadrant id found.")


def set_base_dir(basedir):
    basedir = Path(basedir)
    if not basedir.exists():
        basedir.mkdir(exist_ok=True, parents=True)

    return basedir

def extract_im_products(f1,f2):

    assert f2 > f1, f"{f2} <= {f1}"
    
    im_frequencies = {
        "f1": f1,
        "f2": f2,
        "f2-f1": round(f2-f1,10),
        "f1+f2": round(f1+f2, 10),
        "2f1": round(f1*2, 10),
        "2f2": round(f2*2, 10),
        "2f1-f2": round(2*f1-f2, 10),
        "2f2-f1": round(2*f2-f1, 10),
    }

    return im_frequencies

def store_data_in_dict(
    experiment_id,
    sub_id,
    roi_task_id,
    roi_pval,
    roi_fractional_overlap,
    roi_f_type,
    rephase,
    rephase_with,
    im_test_frequencies_map,
    data_dict,
    n_bootstraps,
    loaded_data_dict = None
):
    if loaded_data_dict is None:
        group_data_dict = defaultdict(list)
    else:
        group_data_dict = loaded_data_dict
    group_data_dict["experiment_id"].append(experiment_id)
    group_data_dict["sub_id"].append(sub_id)
    group_data_dict["roi_task_id"].append(roi_task_id)
    group_data_dict["roi_pval"].append(roi_pval)
    group_data_dict["roi_fractional_overlap"].append(roi_fractional_overlap)
    group_data_dict["roi_f_type"].append(roi_f_type)
    group_data_dict["rephase"].append(rephase)
    group_data_dict["rephase_with"].append(rephase_with)

    for metric_f_type, metric_f_value in im_test_frequencies_map.items():
        group_data_dict[metric_f_type].append(metric_f_value)
        group_data_dict[f"power_{metric_f_type}"].append(data_dict['observed_statistics'][metric_f_value])
        group_data_dict[f"pval_{metric_f_type}"].append(-np.log10(data_dict['p_values'][metric_f_value]))
        group_data_dict[f"bootstrap_power_{metric_f_type}"].append(np.median([i[1] for i in data_dict['bootstrapped_statistics'][f'test-{metric_f_value}']]))
        group_data_dict[f"bootstrap_pval_{metric_f_type}"].append(np.sum([-np.log10(i[0]) > -np.log10(.05) for i in data_dict['bootstrapped_statistics'][f'test-{metric_f_value}']]) / n_bootstraps)

    group_data_dict["power_spectrum"].append(data_dict["observed_power_spectrum"])

    return group_data_dict

def get_experiment_colour_codes():
    """deprecated?
    """
    c_dict = {
        "NORMAL_3T_CONTROL": np.array([i/256 for i in (58,148,42,256)])[np.newaxis,:],
        "NORMAL_3T": np.array([i/256 for i in (191,236,212,256)])[np.newaxis,:],
        "NORMAL_7T": np.array([i/256 for i in (160,210,3,256)])[np.newaxis,:],
        "VARY_3T": np.array([i/256 for i in (242,118,138,256)])[np.newaxis,:],
        "VARY_7T": np.array([i/256 for i in (215,72,48,256)])[np.newaxis,:],
    }

    return c_dict

def get_roi_colour_codes():
    c_dict = {
        "f1": np.array([i/256 for i in (242,51,42,256)])[np.newaxis,:],
        "f2": np.array([i/256 for i in (64,55,255,256)])[np.newaxis,:],
        "f1f2": np.array([i/256 for i in (251,243,64,256)])[np.newaxis,:],
        "f2-f1": np.array([i/256 for i in (53,152,98,256)])[np.newaxis,:],
        "f1+f2": np.array([i/256 for i in (124,19,88,256)])[np.newaxis,:],
        "2f1": np.array([i/256 for i in (132,230,129,256)])[np.newaxis,:],
        "2f2": np.array([i/256 for i in (83,54,146,256)])[np.newaxis,:],
        "2f1-f2": np.array([i/256 for i in (90,236,255,256)])[np.newaxis,:],
        "2f2-f1": np.array([i/256 for i in (244,121,29,256)])[np.newaxis,:],
    }

    return c_dict

def get_frequency_text_codes():
    text_dict = {
        "f1": f"$f_1$",
        "f2": f"$f_2$",
        "f1f2": f"$f_1\cap$$f_2$",
        "f2-f1": f"$f_2-f_1$",
        "f1+f2": f"$f_1+f_2$",
        "2f1": f"2$f_1$",
        "2f2": f"2$f_2$",
        "2f1-f2": f"2$f_1-f_2$",
        "2f2-f1": f"2$f_2-f_1$",
    }

    return text_dict

"""Classes
"""
class f1_f2_data:

    def __init__(self, datadir, n_bootstraps, sub_id, roi_task_id, roi_frequency_f1, roi_frequency_f2, task_id, experiment_id="1_frequency_tagging", mri_id="3T", fo=.8, pval="uncp"):
        if roi_frequency_f1 >= roi_frequency_f2:
            raise ValueError(f"$f_1$ must be less than $f_2$.")
        self.f_data = self._read_f1_f2_pkls(
            datadir, 
            n_bootstraps, 
            sub_id, 
            roi_task_id, 
            roi_frequency_f1, 
            roi_frequency_f2, 
            task_id, 
            experiment_id=experiment_id, 
            mri_id=mri_id, 
            fo=fo, 
            pval=pval,
        )
        self.task_id = self._read_task_id()
        self.roi_task_id = roi_task_id
        self.roi_f1 = roi_frequency_f1
        self.roi_f2 = roi_frequency_f2
        self.pval = pval
        self.n_bootstraps = n_bootstraps


    def extract_bootstrapped_mean_from_data(self, bootstrap_id, f_type, rephase=False, rephase_with=None, pure_f=True):
        f1_coords = self.f_data['f1']['roi_coords'].astype(int)
        f2_coords = self.f_data['f2']['roi_coords'].astype(int)
        intersection_roi_coords = f1_coords * f2_coords
        intersection_roi_coords_from_f1 = intersection_roi_coords[f1_coords==1]
        intersection_roi_coords_from_f2 = intersection_roi_coords[f2_coords==1] # redundant

        if rephase:
            f1_bold = self.f_data['f1'][f'data-test_{self.task_id}_roi_phaseadjusted_bold'][:,:,bootstrap_id]
            f1_tps = self.f_data['f1'][f'data-test_{self.task_id}_roi_phaseadjusted_timepoints']
            f2_bold = self.f_data['f2'][f'data-test_{self.task_id}_roi_phaseadjusted_bold'][:,:,bootstrap_id]
            f2_tps = self.f_data['f2'][f'data-test_{self.task_id}_roi_phaseadjusted_timepoints']
            intersection_f1f2_bold_from_f1 = f1_bold[:, intersection_roi_coords_from_f1==1]
            intersection_f1f2_tps_from_f1 = f1_tps[:, intersection_roi_coords_from_f1==1]
            intersection_f1f2_bold_from_f2 = f2_bold[:, intersection_roi_coords_from_f2==1]
            intersection_f1f2_tps_from_f2 = f2_tps[:, intersection_roi_coords_from_f2==1]
            if pure_f:
                pure_f1_coords = f1_coords - intersection_roi_coords
                pure_f1_coords = pure_f1_coords[f1_coords==1]
                f1_bold = f1_bold[:,pure_f1_coords==1]
                f1_tps = f1_tps[:,pure_f1_coords==1]
                pure_f2_coords = f2_coords - intersection_roi_coords
                pure_f2_coords = pure_f2_coords[f2_coords==1]
                f2_bold = f2_bold[:,pure_f2_coords==1]
                f2_tps = f2_tps[:,pure_f2_coords==1]
            if f_type == "f1":
                tps, bold = self._aggregate_timeseries(f1_tps, f1_bold, "mean", rephase=rephase)
            elif f_type == "f2":
                tps, bold = self._aggregate_timeseries(f2_tps, f2_bold, "mean", rephase=rephase)
            elif f_type == "f1f2":
                if rephase_with == "f1":
                    tps, bold = self._aggregate_timeseries(
                        intersection_f1f2_tps_from_f1, 
                        intersection_f1f2_bold_from_f1,
                        "mean",
                        rephase=rephase 
                    )
                elif rephase_with == "f2":
                    tps, bold = self._aggregate_timeseries(
                        intersection_f1f2_tps_from_f2, 
                        intersection_f1f2_bold_from_f2,
                        "mean",
                        rephase=rephase 
                    )
                else:
                    raise ValueError(f"Rephasing of timeseries is turned on, must set `rephase_with` to either f1 or f2.")
                
            return tps, bold

        else:
            f1_bold = self.f_data['f1'][f'data-test_{self.task_id}_roi_bold'][:,:,bootstrap_id]
            f1_tps = self.f_data['f1'][f'data-test_{self.task_id}_roi_timepoints']
            f2_bold = self.f_data['f2'][f'data-test_{self.task_id}_roi_bold'][:,:,bootstrap_id]
            f2_tps = self.f_data['f2'][f'data-test_{self.task_id}_roi_timepoints']
            intersection_f1f2_bold_from_f1 = f1_bold[:, intersection_roi_coords_from_f1==1]
            intersection_f1f2_tps_from_f1 = f1_tps[:, intersection_roi_coords_from_f1==1]
            intersection_f1f2_bold_from_f2 = f2_bold[:, intersection_roi_coords_from_f2==1]
            intersection_f1f2_tps_from_f2 = f2_tps[:, intersection_roi_coords_from_f2==1]
            if pure_f:
                pure_f1_coords = f1_coords - intersection_roi_coords
                pure_f1_coords = pure_f1_coords[f1_coords==1]
                f1_bold = f1_bold[:,pure_f1_coords==1]
                f1_tps = f1_tps[:,pure_f1_coords==1]
                pure_f2_coords = f2_coords - intersection_roi_coords
                pure_f2_coords = pure_f2_coords[f2_coords==1]
                f2_bold = f2_bold[:,pure_f2_coords==1]
                f2_tps = f2_tps[:,pure_f2_coords==1]
            if f_type == "f1":
                tps, bold = self._aggregate_timeseries(f1_tps, f1_bold, "mean", rephase=rephase)
            elif f_type == "f2":
                tps, bold = self._aggregate_timeseries(f2_tps, f2_bold, "mean", rephase=rephase)
            elif f_type == "f1f2":
                tps, bold = self._aggregate_timeseries(intersection_f1f2_tps_from_f1, intersection_f1f2_bold_from_f1, "mean", rephase=rephase)
            else:
                raise ValueError(f"{f_type} not supported")
            
            return tps, bold
        
    def _aggregate_timeseries(self, tps, bold, aggr_type, rephase=False):
        
        assert aggr_type == "mean"

        if rephase:
            updated_tps, mean_bold = [], []
            for single_tp in np.unique(tps):
                coords = (tps == single_tp)
                tp_all = np.all((coords).sum(0)) == 1
                if tp_all:
                    updated_tps.append(single_tp)
                    mean_bold.append(bold[coords].mean())

            updated_tps = np.array(updated_tps)
            mean_bold = np.array(mean_bold)
            assert _check_difference(updated_tps)
            assert mean_bold.shape == updated_tps.shape

            return updated_tps, mean_bold

        else:
            return tps.mean(1), bold.mean(1)

    def _read_f1_f2_pkls(self, datadir, n_bootstraps, sub_id, roi_task_id, roi_frequency_f1, roi_frequency_f2, task_id, experiment_id="1_frequency_tagging", mri_id="3T", fo=.8, pval="uncp"):

        import pickle
        
        f1_pkl: Path = datadir / f"experiment-{experiment_id}_mri-{mri_id}_smooth-0_truncate-39-219_n-{n_bootstraps}_batch-merged_desc-basic_roi-{roi_task_id}-{roi_frequency_f1}_pval-{pval}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap/task-{task_id}_bootstrapped_data.pkl"
        f2_pkl: Path = datadir / f"experiment-{experiment_id}_mri-{mri_id}_smooth-0_truncate-39-219_n-{n_bootstraps}_batch-merged_desc-basic_roi-{roi_task_id}-{roi_frequency_f2}_pval-{pval}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap/task-{task_id}_bootstrapped_data.pkl"

        assert f1_pkl.exists()
        assert f2_pkl.exists()
        f_data = {}
        for f_type, f_pkl in zip(["f1","f2"], [f1_pkl, f2_pkl]):
            with open(f_pkl, 'rb') as f:
                f_data[f_type] = pickle.load(f)

        return f_data
    
    def _read_task_id(self):
        for i in self.f_data['f1'].keys():
            if i.startswith("task-"):
                return i.split("_")[0]
            
        raise ValueError("No task_id found.")

# Run statistics on a timeseries
class TimeSeries:
    def __init__(self, ts, TR, n_permutations=5_000, nperseg=600):
        self.timeseries = ts
        self.fs = 1/TR
        self.nperseg = nperseg
        self.n_permutations = n_permutations
        self.frequencies = None
                
    def process(self, search_frequencies):
        p_values, observed_statistics = {}, {}
        for f in search_frequencies:
            observed_statistic, observed_power_spectrum = self.calculate_observed_statistic(f)
            observed_statistics[f] = observed_statistic
            null_statistics, null_power_spectrums = self.calculate_null_statistics(f)
            p_values[f] = (np.sum(null_statistics >= observed_statistic) + 1) / (self.n_permutations+1)

        return p_values, observed_statistics, observed_power_spectrum, null_power_spectrums

    def calculate_observed_statistic(self, f):
        if self.frequencies is None:
            self.frequencies, power_spectrum = self._estimate_power_spectrum(self.timeseries)
        else:
            _, power_spectrum = self._estimate_power_spectrum(self.timeseries)
        power = self._estimate_power(self.timeseries, f)

        return power, power_spectrum

    def calculate_null_statistics(self, f):
        null_power_spectrums = []
        null_statistics = []
        for i in range(self.n_permutations):
            y_shuffle = np.random.permutation(self.timeseries.copy())
            null_power_spectrums.append(self._estimate_power_spectrum(y_shuffle)[1])
            null_statistics.append(self._estimate_power(y_shuffle, f))

        return null_statistics, null_power_spectrums

    def _estimate_power_spectrum(self, ts):
        frequencies, power_spectrum = welch(ts, self.fs, nperseg=self.nperseg)

        return (frequencies, power_spectrum)

    def _estimate_power(self, ts, f):
        frequencies, power_spectrum = self._estimate_power_spectrum(ts)
        return np.interp(f, frequencies, power_spectrum)
    
"""Figures
"""
def plot_power_spectrum(frequency_grid, observed_power_spectrum, null_power_spectrums, n_permutations, frequencies, p_values, observed_statistics, add_im=False, sub_id=None, roi_frequency=None, close_figure=False, png_out=None):
    fig, ax = plt.subplots(figsize=(2,1), dpi=400)
    ax.plot(frequency_grid, observed_power_spectrum, c='k', zorder=2, lw=.5)
    null_power_spectrums = np.vstack(null_power_spectrums)
    null_power_spectrum = np.mean(null_power_spectrums, axis=0)
    std_dev_values = np.std(null_power_spectrum, axis=0)
    confidence_interval = 1.96 * std_dev_values / np.sqrt(n_permutations)
    ax.fill_between(
        frequency_grid, 
        null_power_spectrum - confidence_interval, null_power_spectrum + confidence_interval,
        color='r', 
        alpha=.8,
    )
    fig, ax = _decorate_fig_power_spectrum(fig, ax, frequencies, p_values, observed_statistics, add_im=add_im, sub_id=sub_id, roi_frequency=roi_frequency)

    fig.tight_layout()

    if png_out:
        fig.savefig(png_out,dpi='figure')

    if close_figure:
        plt.close()

def _decorate_fig_power_spectrum(fig, ax, frequencies, p_values, observed_statistics, add_im=False, sub_id=None, roi_frequency=None, fontsize=4):
    for f in frequencies:
        ax.text(f+.005, observed_statistics[f], f"p={-np.log10(p_values[f]):.3f}", fontsize=fontsize)
    if add_im:
        _frequencies = frequencies.copy()
        second_order_frequencies = [
            np.abs(frequencies[0]-frequencies[1]), 
            np.abs(frequencies[1]+frequencies[0]),
            frequencies[0]*2,
            frequencies[1]*2,
        ]
        third_order_frequencies = [
            np.abs(2*frequencies[0] - frequencies[1]),
            np.abs(2*frequencies[1] - frequencies[0]),
        ]
        _frequencies += second_order_frequencies
        _frequencies += third_order_frequencies
        #import pdb; pdb.set_trace()
    else:
        _frequencies = frequencies
    for f in _frequencies:
        if f not in second_order_frequencies and f not in third_order_frequencies:
            c = 'b'
        elif f in second_order_frequencies:
            c = 'cyan'
        elif f in third_order_frequencies:
            c = 'g'
        else: 
            raise ValueError(f"{f} not identified as a harmonic.")
        ax.axvline(x=f, c=c, linestyle=':', zorder=1, lw=.75)
    ax.set_xlim((0,.5))
    ax.set_ylabel("Power", fontsize=fontsize)
    ax.set_xlabel("Frequency", fontsize=fontsize)
    ax.tick_params(axis="both", length=0, labelsize=fontsize)
    for i in ("top", "right", "bottom", "left"):
        ax.spines[i].set_visible(False)
    ax.set_title(f"{sub_id}, roi-{roi_frequency}, {frequencies}", fontsize=fontsize)

    return fig, ax

def decorate_fig_carpetplot(fig, ax, im, f1, f2, n_f1, n_f2, n_f1f2, FONTSIZE=4, TR=.3):

    cbar = plt.colorbar(im, ax=ax, shrink=.5, drawedges=False)
    cbar.ax.set_title("Z-score", fontsize=FONTSIZE-2)
    cbar.ax.tick_params(axis="both", length=0, labelsize=FONTSIZE)
    cbar.outline.set_edgecolor('none')

    #ax.set_title(f"{sub_id}, roi-{task_id}, roi-frequency-{f}", fontsize=FONTSIZE)
    ax.title.set_position([.75,1.05])

    ax.set_ylabel("Voxel", fontsize=FONTSIZE)
    ax.set_yticks([])

    ax.set_xlabel("Acquisition Time (s)", fontsize=FONTSIZE)
    xticks = [i for i in ax.get_xticks()[1:]]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{i*TR:.2f}" for i in xticks], fontsize=FONTSIZE)
    ax.tick_params(axis="both", length=0)

    period_f1 = 1/f1
    period_f2 = 1/f2
    
    ax.plot([0,period_f1/TR], [-20,-20], c='white', zorder=1)
    ax.plot([0,period_f2/TR], [-50,-50], c='white', zorder=2)
    
    square_f1 = plt.Polygon([(-2, 0), (-15, 0), (-15, n_f1), (-2, n_f1)], closed=True, color='red', linewidth=0.)
    square_f1f2 = plt.Polygon([(-2, n_f1), (-15, n_f1), (-15, n_f1+n_f1f2), (-2, n_f1+n_f1f2)], closed=True, color='gold', linewidth=0.)
    square_f2 = plt.Polygon([(-2, n_f1+n_f1f2), (-15, n_f1+n_f1f2), (-15, n_f1+n_f1f2+n_f2), (-2, n_f1+n_f1f2+n_f2)], closed=True, color='blue', linewidth=0.)
    timescale_f1 = plt.Polygon([(0, -10), (period_f1/TR, -10), (period_f1/TR, -30), (0, -30)], closed=True, color='red', linewidth=0., zorder=10)
    timescale_f2 = plt.Polygon([(0, -40), (period_f2/TR, -40), (period_f2/TR, -60), (0, -60)], closed=True, color='blue', linewidth=0., zorder=10)

    for square in [square_f1, square_f1f2, square_f2, timescale_f1, timescale_f2]:
        ax.add_patch(square)

    for i in ("top", "right", "bottom", "left"):
        ax.spines[i].set_visible(False)

    return fig, ax

"""Analysis
"""
def analyze_rois(pkl_handler, f_type, test_frequencies, n_bootstraps, TR, n_permutations=1000, nperseg=580, rephase=False, frequency_grid=None, rephase_with=None, get_bootstrapped_psds=False,pure_f=True):
    """
    `f_type` is the type of roi [f1,f2,f1&f2]
    """

    from collections import defaultdict

    # Perform statistics on all bootstrapped-roi-level timeseries
    bootstrapped_means = []
    bootstrapped_psds = []
    bootstrapped_statistics = defaultdict(list)
    for bootstrap_id in range(n_bootstraps):
        tps, mean_bold = pkl_handler.extract_bootstrapped_mean_from_data(bootstrap_id, f_type, rephase=rephase, rephase_with=rephase_with, pure_f=pure_f)
        bootstrapped_means.append(mean_bold) # track
        ts = TimeSeries(mean_bold, TR, n_permutations=n_permutations, nperseg=nperseg)
        p_values, observed_statistics, observed_power_spectrum, null_power_spectrums = ts.process(test_frequencies)
        bootstrapped_psds.append(observed_power_spectrum)
        for test_f in test_frequencies:
            bootstrapped_statistics[f"test-{test_f}"].append((p_values[test_f], observed_statistics[test_f]))

    # Compute mean timeseries across all bootstrapped-roi-level timeseries
    bootstrapped_means = np.vstack(bootstrapped_means)
    y_bootstrapped_mean = bootstrapped_means.mean(0)
    # Compute statistics
    ts = TimeSeries(y_bootstrapped_mean, TR, n_permutations=n_permutations, nperseg=nperseg)
    p_values, observed_statistics, observed_power_spectrum, null_power_spectrums = ts.process(test_frequencies)

    if frequency_grid is None:
        frequency_grid = ts.frequencies
    else:
        assert np.allclose(frequency_grid, ts.frequencies, rtol=1e-05, atol=1e-08)

    if get_bootstrapped_psds:
        return (
            frequency_grid,
            np.vstack(bootstrapped_psds),
            observed_power_spectrum,
        )
    else:
        return (
            frequency_grid, 
            observed_statistics, 
            observed_power_spectrum, 
            null_power_spectrums, 
            p_values, 
            bootstrapped_statistics,
        )



"""Other
"""

def _check_difference(arr, diff=.3):
    differences = np.diff(arr)

    return np.all(np.isclose(differences,diff))

def change_font():
    # Change fontsize
    from matplotlib import font_manager

    font_dirs = ['/opt/app/notebooks/font_library']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        print(font_file)
        font_manager.fontManager.addfont(font_file)
    # set font
    plt.rcParams['font.family'] = 'aptos'