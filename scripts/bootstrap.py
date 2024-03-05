from typing import Optional
from typing import List, Tuple, Union, Literal
from typing_extensions import Annotated
import typer

from pathlib import Path
import itertools
import glob
import random
import subprocess
import nibabel as nib
import numpy as np
import os
import shutil
import statsmodels.stats.multitest as sm

import sys
sys.path.append("/opt/app")
from fastfmri_toolbox.modelling.design_matrix import (
    DesignMatrix, 
    FrequencyRegressors, 
)
from fastfmri_toolbox.modelling.first_level_analysis import FirstLevelAnalysis
from scripts.fla_utils import convert_niftis_to_ciftis

def search(base_dir, wildcard, error=True):
    search_path = Path(base_dir) / wildcard
    files = glob.glob(str(search_path))

    if not files:
        if error:
            raise FileNotFoundError(f"No files were found in: {search_path}")
        else:
            return []

    return files

def get_denoised_data(
    experiment_id,
    mri_id,
    smooth_mm,
    time_window,
    sub_id, 
    task_id_base,
    denoising_strategy = "00_experiment-min+motion24+wmcsfmean", 
    use_denoised = True,
):
    # Get windowed or denoised data
    if use_denoised:
        desc = "denoised"
    else:
        desc = "windowed"
    # Get output directory of denoised data (run `1_bold_run_denoising.ipynb` first)
    out_dir = Path(f"/scratch/fastfmri/experiment-{experiment_id}_mri-{mri_id}_smooth-{smooth_mm}_truncate-{time_window[0]}-{time_window[1]}_desc-denoised_bold")
    assert out_dir.exists(), f"Directory: {out_dir} does not exist."
    for child in [denoising_strategy, f"sub-{sub_id}"]:
        out_dir = out_dir / child
        assert out_dir.exists(), f"Directory: {out_dir} does not exist."
    # search all bold runs 
    runs = search(out_dir, f"*/task-{task_id_base}Q*/run-*/GLM/*desc-{desc}*")
    runs = [Path(r) for r in runs]
    runs.sort()
    
    return runs

def split_runs(lst):
    random.shuffle(lst)  # Shuffle the list in place
    midpoint = len(lst) // 2
    train, test = lst[:midpoint], lst[midpoint:]
    train.sort()
    test.sort()
    
    return train, test

def average_bold_runs(bold_runs, out_dtseries):
    # Check if all listed bold runs are CIFTI
    for i in bold_runs:
        assert str(i).split('.')[-2:] == ['dtseries', 'nii'], f"{i} does not end with [dtseries.nii]"
    command = [
        "wb_command",
        "-cifti-average",
        out_dtseries,
        "-cifti" 
    ] + (" -cifti ".join([str(i) for i in bold_runs])).split(' ')
    result = subprocess.run(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    out_dtseries = Path(out_dtseries)
    assert out_dtseries.exists(), f"{out_dtseries} was not generated."

    return out_dtseries

def get_mask_from_dtseries(dtseries, mask_nifti):

    assert str(dtseries).split('.')[-2:] == ['dtseries', 'nii']
    
    pseudo_nifti = str(dtseries).replace(".dtseries.nii", ".nii.gz")
    cmds = []
    # Convert to NIFTI
    _command = [
        "wb_command", "-cifti-convert", "-to-nifti", dtseries, pseudo_nifti,
    ]
    cmds.append(_command)
    # Create mask
    _command = [
        "fslmaths", pseudo_nifti, "-Tmean", "-bin", mask_nifti
    ]
    cmds.append(_command)
    # Run commands
    for command in cmds:
        result = subprocess.run(
            command, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True
        )
    # `pseudo_nifti` is 2D, enforce 3D so that other tools can run properly
    mask_img = nib.load(mask_nifti)
    mask_data = mask_img.get_fdata()[:,:,np.newaxis]
    mask_img = nib.Nifti1Image(mask_data, affine=mask_img.affine, header=mask_img.header)
    nib.save(mask_img, mask_nifti)
    assert Path(mask_nifti).exists(), f"{mask_nifti} was not generated."

    return mask_nifti

def get_design_matrix(time_window, search_frequencies, TR):
    # Design matrix
    dm = DesignMatrix(time_window, search_frequencies, TR=TR)
    dm.add_regressor(
        FrequencyRegressors(search_frequencies, dm.time_points)
    )
    design_matrix = dm.build_design_matrix()
    design_matrix['constant'] = 1

    return design_matrix

    
def get_fla_bold_base(sub_id, ses_id, task_id, run_id):

    return f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_run-{run_id}"

def convert_cifti_to_nifti(cifti, nifti):
    assert str(cifti).split('.')[-2:] == ['dtseries', 'nii']
    # Convert to NIFTI
    command = [
        "wb_command", "-cifti-convert", "-to-nifti", cifti, nifti,
    ]
    result = subprocess.run(
        command, 
        stdout=subprocess.PIPE, 
        stderr=subprocess.PIPE, 
        text=True
    )
    nifti = Path(nifti)
    assert nifti.exists(), f"{nifti} was not generated."

    return nifti

def run_fla(
    sub_id,
    ses_id,
    task_id,
    run_id,
    fla_dir,
    template_cifti,
    bold_nifti, 
    mask_nifti, 
    design_matrix, 
    time_window, 
    search_frequencies,
    TR,
):
    fla = FirstLevelAnalysis(
        derivatives_dir = fla_dir,
        bold_path = bold_nifti,
        mask_path = mask_nifti,
        design_matrix = design_matrix,
        time_window = time_window,
        search_frequencies = search_frequencies,
        TR = TR,
        data_windowed = True,
    )
    fla.run_frequency_glm(save_frequency_snrs=False)
    convert_niftis_to_ciftis(
        sub_id, 
        ses_id, 
        task_id, 
        run_id, 
        fla_dir, 
        template_cifti = template_cifti, 
        TR = TR
    )

def clean_up_files(lst):
    for file_path in lst:
        try:
            os.remove(file_path)
            print(f"{file_path} has been successfully removed.")
        except OSError as e:
            print(f"Error: {e.filename} - {e.strerror}")

def clean_up_directory(directory_path):
    try:
        shutil.rmtree(directory_path)
        print(f"{directory_path} has been successfully removed.")
    except OSError as e:
        print(f"Error: {e.filename} - {e.strerror}")

class analyze_bootstrap:
    def __init__(
        self, parent_fla_dir, iterations,
        search_frequencies,
        sub_id, ses_id, task_id,
        out_dir,
        template_dtseries,
    ):

        self.out_dir = Path(out_dir)
        self.parent_fla_dir = parent_fla_dir
        self.n_iters = iterations
        self.search_frequencies = search_frequencies
        self.sub_id = sub_id
        self.ses_id = ses_id
        self.task_id = task_id
        self.template_dtseries = template_dtseries
        self.prefix = f"sub-{sub_id}_ses-{ses_id}_task-{task_id}"
        # Set-up all outputs
        self._setup_outputs()
        # Assertions
        assert self.parent_fla_dir.exists(), f"{self.parent_fla_dir} does not exist."
        # Global var
        self.MANDATORY_METRICS = ["stat", "z_score", "phase_delay", "p_value"]
        self.z_increment = 0.5
        self.z_max = 10.1

    def add_train_test_set(
        self,
        iteration, 
        train_average,
        test_average,
    ):
        # Set-up variables 
        train_run_id = f"TRAIN{str(iteration).zfill(5)}"
        train_fla_dir = self.parent_fla_dir / f"sub-{self.sub_id}" / f"ses-{self.ses_id}" / f"task-{self.task_id}" / f"run-{train_run_id}" / "GLM"
        test_run_id = f"TEST{str(iteration).zfill(5)}"
        test_fla_dir = self.parent_fla_dir / f"sub-{self.sub_id}" / f"ses-{self.ses_id}" / f"task-{self.task_id}" / f"run-{test_run_id}" / "GLM"
        # Check if first level analysis were run for this iteration
        assert test_average.exists(), f"{test_average} does not exist."
        assert train_average.exists(), f"{train_average} does not exist."
        assert train_fla_dir.exists(), f"{train_fla_dir} does not exist."
        assert test_fla_dir.exists(), f"{test_fla_dir} does not exist."
        # Load
        self._process_multiple_metrics(self.train_stat, train_fla_dir, train_run_id, iteration, "stat")
        self._process_multiple_metrics(self.test_stat, test_fla_dir, test_run_id, iteration, "stat")
        self._process_multiple_metrics(self.train_z_scores, train_fla_dir, train_run_id, iteration, "z_score")
        self._process_multiple_metrics(self.test_z_scores, test_fla_dir, test_run_id, iteration, "z_score")
        self._process_multiple_metrics(self.train_phase_delays, train_fla_dir, train_run_id, iteration, "phase_delay")
        self._process_multiple_metrics(self.test_phase_delays, test_fla_dir, test_run_id, iteration, "phase_delay")
        #self._process_multiple_metrics(self.train_pSNR, train_fla_dir, train_run_id, iteration, "pSNR")
        #self._process_multiple_metrics(self.test_pSNR, test_fla_dir, test_run_id, iteration, "pSNR")
        self._process_multiple_metrics(self.train_p_value, train_fla_dir, train_run_id, iteration, "p_value")
        self._process_multiple_metrics(self.test_p_value, test_fla_dir, test_run_id, iteration, "p_value")
        self._process_tasklock_metric(train_average, test_average, iteration)

    def aggregate_metrics(self):
        for search_frequency in self.search_frequencies:
            # Mean & stdev
            self._calculate_mean_std_metric(self.train_stat[search_frequency], "_stat.")
            self._calculate_mean_std_metric(self.test_stat[search_frequency], "_stat.")
            self._calculate_mean_std_metric(self.train_z_scores[search_frequency], "_z_score.")
            self._calculate_mean_std_metric(self.test_z_scores[search_frequency], "_z_score.")
            self._calculate_mean_std_metric(self.train_phase_delays[search_frequency], "_phasedelay.")
            self._calculate_mean_std_metric(self.test_phase_delays[search_frequency], "_phasedelay.")
            #self._calculate_mean_std_metric(self.train_pSNR[search_frequency], "_pSNR.")
            #self._calculate_mean_std_metric(self.test_pSNR[search_frequency], "_pSNR.")
            self._calculate_mean_std_metric(self.train_p_value[search_frequency], "_p_value.")
            self._calculate_mean_std_metric(self.test_p_value[search_frequency], "_p_value.")
            # Overlap
            self._calculate_activation_overlap(self.train_stat[search_frequency], self.train_p_value[search_frequency])
            self._calculate_activation_overlap(self.test_stat[search_frequency], self.test_p_value[search_frequency])
        # Mean & stdev
        self._calculate_mean_std_metric(self.tasklock, "_tasklock.")

    def _calculate_mean_std_metric(self, dtseries, replace_str):
        # Load img and data
        img = nib.load(dtseries)
        data = img.get_fdata()
        # save mean
        mean = data.mean(0, keepdims=True)
        mean_img = nib.Cifti2Image(mean, header=img.header)
        mean_img.header.matrix[0].number_of_series_points = 1
        nib.save(mean_img, str(dtseries).replace(replace_str, f"_desc-mean{replace_str}"))
        # save stdev
        stdev = data.std(0, keepdims=True)
        stdev_img = nib.Cifti2Image(stdev, header=img.header)
        stdev_img.header.matrix[0].number_of_series_points = 1
        nib.save(stdev_img, str(dtseries).replace(replace_str, f"_desc-std{replace_str}"))

    def _calculate_activation_overlap(
        self, 
        stat_dtseries, 
        p_value_dtseries,
    ):
        # Load img and data
        stat_img = nib.load(stat_dtseries)
        stat_data = stat_img.get_fdata()
        pval_img = nib.load(p_value_dtseries)
        pval_data = pval_img.get_fdata()
        n_iters, _ = stat_data.shape
        for i in range(n_iters):
            _stat_data = stat_data[i,:]
            _pval_data = pval_data[i,:]
            _wholebrain_mask = _stat_data > 0
            n_vertices = _wholebrain_mask.sum()
            corrected_pvalue = np.zeros_like(_wholebrain_mask).astype(float)
            qvalues = sm.multipletests(
                _pval_data[_wholebrain_mask], method='fdr_bh'
            )[1]
            corrected_pvalue[_wholebrain_mask] = qvalues
            _corrected_pvalue_mask = ( (corrected_pvalue < .05) * _wholebrain_mask ).astype(float)[np.newaxis,:]
            _wholebrain_mask = _wholebrain_mask.astype(float)[np.newaxis,:]
            if i == 0:
                corrected_pvalue_mask = _corrected_pvalue_mask
                wholebrain_mask = _wholebrain_mask
            else:
                corrected_pvalue_mask = np.concatenate((corrected_pvalue_mask, _corrected_pvalue_mask), axis=0)
                wholebrain_mask = np.concatenate((wholebrain_mask, _wholebrain_mask), axis=0)
        mean_corrected_pvalue_mask_img = nib.Cifti2Image(corrected_pvalue_mask.mean(axis=0)[np.newaxis,:], header=stat_img.header)
        mean_corrected_pvalue_mask_img.header.matrix[0].number_of_series_points = 1
        nib.save(mean_corrected_pvalue_mask_img, str(stat_dtseries).replace("_stat.", "_desc-overlap_activations."))
        corrected_pvalue_mask_img = nib.Cifti2Image(corrected_pvalue_mask, header=stat_img.header)
        corrected_pvalue_mask_img.header.matrix[0].number_of_series_points = corrected_pvalue_mask.shape[0]
        nib.save(corrected_pvalue_mask_img, str(stat_dtseries).replace("_stat.", "_activations."))
        wholebrain_mask_img = nib.Cifti2Image(wholebrain_mask, header=stat_img.header)
        wholebrain_mask_img.header.matrix[0].number_of_series_points = wholebrain_mask.shape[0]
        nib.save(wholebrain_mask_img, str(stat_dtseries).replace("_stat.", "_mask."))

    def _process_multiple_metrics(self, metric_dict, fla_dir, run_id, iteration, metric_type):
        assert metric_type in self.MANDATORY_METRICS, f"metric type must be set to {self.MANDATORY_METRICS}"
        for search_frequency in self.search_frequencies:
            # Load merged data
            merged_img, merged_data = self._load_img_and_data(metric_dict[search_frequency])
            # Load bootstrap data
            if metric_type == "stat":
                bootstrap_metric_path = self._get_stat(fla_dir, run_id, search_frequency)
            if metric_type == "z_score":
                bootstrap_metric_path = self._get_z_score(fla_dir, run_id, search_frequency)
            if metric_type == "phase_delay":
                bootstrap_metric_path = self._get_phase_delay(fla_dir, run_id, search_frequency)
            """
            if metric_type == "pSNR":
                bootstrap_metric_path = self._get_pSNR(fla_dir, run_id, search_frequency)
            """
            if metric_type == "p_value":
                bootstrap_metric_path = self._get_p_value(fla_dir, run_id, search_frequency)
            bootstrap_data = nib.load(bootstrap_metric_path).get_fdata()
            merged_data[iteration,:] = bootstrap_data[0, :]
            merged_img = nib.Cifti2Image(merged_data, header=merged_img.header)
            metric_dict[search_frequency].unlink()  # Remove before re-saving
            nib.save(merged_img, metric_dict[search_frequency])

    def _load_img_and_data(self, dtseries):
        img = nib.load(dtseries)
        data = img.get_fdata()
        
        return img, data

    def _process_tasklock_metric(self, train_average, test_average, iteration):
        # Train
        train_data = nib.load(train_average).get_fdata()
        train_data -= train_data.mean(axis=0, keepdims=True)
        train_data /= train_data.std(axis=0, keepdims=True)
        train_data[np.isnan(train_data)] = 0 # 0-fill
        # Test
        test_data = nib.load(test_average).get_fdata()
        test_data -= test_data.mean(axis=0, keepdims=True)
        test_data /= test_data.std(axis=0, keepdims=True)
        test_data[np.isnan(test_data)] = 0 # 0-fill
        # Load merged data
        merged_img, merged_data = self._load_img_and_data(self.tasklock)
        # Loop over every vertex and compute inter-run vertex-locked cross correlation
        # This will tell us which timeseries are 'locked' between runs
        n_vertices = train_data.shape[-1]
        for v in range(n_vertices):
            merged_data[iteration, v] = np.corrcoef(train_data[:,v], test_data[:,v])[0,1]
        merged_data[np.isnan(merged_data)] = 0 # 0-fill
        merged_img = nib.Cifti2Image(merged_data, header = merged_img.header)
        self.tasklock.unlink() # Remove before re-saving
        nib.save(merged_img, self.tasklock)
        
    def _setup_outputs(
        self
    ):
        """
        Outputs:
        Multiple metrics (iterated over frequencies)
        - stat
        - Z-scores
        - phase delays
        - pSNR
        - p-value
        Single metric
        - task-locking
        """
        if not self.out_dir.exists():
            self.out_dir.mkdir(parents=True)

        # Multiple metrics
        # stat
        self.train_stat, self.test_stat = {}, {}
        # Z-scores
        self.train_z_scores, self.test_z_scores = {}, {}
        # Phase delays
        self.train_phase_delays, self.test_phase_delays = {}, {}
        # pSNR
        #self.train_pSNR, self.test_pSNR = {}, {}
        # p-value
        self.train_p_value, self.test_p_value = {}, {}
        # Create empty outputs
        for search_frequency in self.search_frequencies:
            self.train_stat[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-train_n-{self.n_iters}_stat.dtseries.nii"
            self.test_stat[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-test_n-{self.n_iters}_stat.dtseries.nii"
            self.train_z_scores[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-train_n-{self.n_iters}_z_score.dtseries.nii"
            self.test_z_scores[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-test_n-{self.n_iters}_z_score.dtseries.nii"
            self.train_phase_delays[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-train_n-{self.n_iters}_phasedelay.dtseries.nii"
            self.test_phase_delays[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-test_n-{self.n_iters}_phasedelay.dtseries.nii"
            #self.train_pSNR[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-train_n-{self.n_iters}_pSNR.dtseries.nii"
            #self.test_pSNR[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-test_n-{self.n_iters}_pSNR.dtseries.nii"
            self.train_p_value[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-train_n-{self.n_iters}_p_value.dtseries.nii"
            self.test_p_value[search_frequency] = self.out_dir / f"{self.prefix}_f-{search_frequency}_data-test_n-{self.n_iters}_p_value.dtseries.nii"
            self._set_up_empty_dtseries(self.train_stat[search_frequency])
            self._set_up_empty_dtseries(self.test_stat[search_frequency])
            self._set_up_empty_dtseries(self.train_z_scores[search_frequency])
            self._set_up_empty_dtseries(self.test_z_scores[search_frequency])
            self._set_up_empty_dtseries(self.train_phase_delays[search_frequency])
            self._set_up_empty_dtseries(self.test_phase_delays[search_frequency])
            #self._set_up_empty_dtseries(self.train_pSNR[search_frequency])
            #self._set_up_empty_dtseries(self.test_pSNR[search_frequency])
            self._set_up_empty_dtseries(self.train_p_value[search_frequency])
            self._set_up_empty_dtseries(self.test_p_value[search_frequency])
        self.tasklock = self.out_dir / f"{self.prefix}_n-{self.n_iters}_tasklock.dtseries.nii"
        self._set_up_empty_dtseries(self.tasklock)

    def _set_up_empty_dtseries(self, out_file: Path):
        """
        Create 0-filled dtseries with volumes == `n_iters`
        """
        assert str(out_file).endswith(".dtseries.nii")
        base_img = nib.load(self.template_dtseries)
        _, n_vertices = base_img.shape
        img = nib.Cifti2Image(
            np.zeros((self.n_iters, n_vertices)),
            header = base_img.header
        )
        img.header.matrix[0].number_of_series_points = self.n_iters
        nib.save(img, out_file)
    
    def _get_stat(self, fla_glm_dir, run_id, search_f):
        base = self._get_base(run_id, search_f)
        dscalar = fla_glm_dir / f"{base}_stat.dscalar.nii"
        assert dscalar.exists(), f"{dscalar} does not exist."
        
        return dscalar
        
    def _get_z_score(self, fla_glm_dir, run_id, search_f):
        base = self._get_base(run_id, search_f)
        dscalar = fla_glm_dir / f"{base}_z_score.dscalar.nii"
        assert dscalar.exists(), f"{dscalar} does not exist."
        
        return dscalar

    def _get_phase_delay(self, fla_glm_dir, run_id, search_f):
        base = self._get_base(run_id, search_f)
        dscalar = fla_glm_dir / f"{base}_phasedelay.dscalar.nii"
        assert dscalar.exists(), f"{dscalar} does not exist."
        
        return dscalar
    
    def _get_pSNR(self, fla_glm_dir, run_id, search_f):
        base = self._get_base(run_id, search_f)
        dscalar = fla_glm_dir / f"{base}_pSNR.dscalar.nii"
        assert dscalar.exists(), f"{dscalar} does not exist."
        
        return dscalar
    
    def _get_p_value(self, fla_glm_dir, run_id, search_f):
        base = self._get_base(run_id, search_f)
        dscalar = fla_glm_dir / f"{base}_p_value.dscalar.nii"
        assert dscalar.exists(), f"{dscalar} does not exist."
        
        return dscalar
        
    def _get_base(self, run_id, search_f):
        return f"sub-{self.sub_id}_ses-{self.ses_id}_task-{self.task_id}_run-{run_id}_frequency-{search_f}"

app = typer.Typer()

@app.command()
def run_bootstrap(
    experiment_id: Annotated[str, typer.Option()],
    mri_id: Annotated[str, typer.Option()],
    sub_id: Annotated[str, typer.Option()],
    task_id_base: Annotated[str, typer.Option()],
    smooth_mm: Annotated[int, typer.Option()], 
    search_frequencies: Annotated[List[float], typer.Option()],
    time_window: Annotated[Tuple[int, int], typer.Option()],
    TR: Annotated[float, typer.Option()],
    denoising_strategy: Annotated[str, typer.Option()],
    n_iterations: Annotated[int, typer.Option()], 
    base_out_dir: Annotated[str, typer.Option()],
    no_bootstrap: bool = typer.Option(False, "--no-bootstrap", help="Turn off bootstrapping"),
):

    # Logging
    print(
        f" Experiment ID: {experiment_id}\n",
        f"MRI ID: {mri_id}\n",
        f"subject ID: {sub_id}\n",
        f"task ID base: {task_id_base}\n", 
        f"smooth (mm): {smooth_mm}\n", 
        f"searchfrequencies: {search_frequencies} Hz\n", 
        f"time window: {time_window} secs\n",
        f"TR: {TR} secs\n",
        f"Denoising strategy: {denoising_strategy}\n",
        f"Number of bootstrap iterations: {n_iterations}\n",
        f"Output directory: {base_out_dir}\n",
    )

    # Get all bold_runs
    print(f"sub-{sub_id}, task-{task_id_base}")
    bold_runs = get_denoised_data(
        experiment_id, 
        mri_id, 
        smooth_mm, 
        time_window, 
        sub_id, 
        task_id_base,
        denoising_strategy = denoising_strategy,
        use_denoised=True,
    )

    """
    Get bold base for all bold runs
    """
    print(f"Processing average of all BOLD runs")
    ses_id = "main" # Placeholder session id
    task_id = str(bold_runs[0]).split('_task-')[1].split('_')[0]
    run_id = "ALL"
    # Check if all `bold_runs` have the same task_id
    for i in bold_runs:
        _task_id = str(i).split('_task-')[1].split('_')[0]
        assert _task_id == task_id, f"{i.stem} | {_task_id} != {task_id}"

    # Set first-level analysis directory
    fla_dir = Path(f"{base_out_dir}/first_level_analysis")

    """
    Run first-level analysis on ALL bold runs
    """

    # Make subject-level directory
    sub_dir = Path(base_out_dir) / f"sub-{sub_id}"
    if not sub_dir.exists():
        # Create the directory
        try:
            sub_dir.mkdir(parents=True)
        except:
            print("Exception: Already exists.") # This occurs when multiple batch scripts are running simultaneously

    check_dir = sub_dir / f"ses-{ses_id}" / f"task-{task_id}"
    if check_dir.exists():
        print("ALREADY RAN")
        return 0
    
    # Make text file to track bootstrapping iterations
    train_txt = sub_dir / f"task-{task_id}_train_splits.txt"
    test_txt = sub_dir / f"task-{task_id}_test_splits.txt"
    for _txt in [train_txt, test_txt]:
        with open(_txt, "w") as f:
            pass
        
    # Process all bold
    bold_base = get_fla_bold_base(
        sub_id, ses_id, task_id, run_id
    )
    # Average bold runs
    all_average = sub_dir / f"{bold_base}_bold.dtseries.nii"
    all_average = average_bold_runs(bold_runs, all_average)
    # Get mask from average bold_runs 
    # Note: This is a liberally thresholded mask
    mask_nifti = sub_dir / f"{bold_base}_mask.nii.gz"
    mask_nifti = get_mask_from_dtseries(all_average, mask_nifti)
    # Get design matrix
    design_matrix = get_design_matrix(time_window, search_frequencies, TR)
    # Convert bold average (to NIFTI format)
    # *FLA tools only processes NIFTIs
    bold_nifti = str(all_average).replace(".dtseries.nii", ".nii.gz")
    bold_nifti = convert_cifti_to_nifti(all_average, bold_nifti)
    # Run first-level analysis on averaged bold runs
    # *all of the FLA outputs will be converted back to cifti format
    run_fla(
        sub_id, ses_id, task_id, run_id, 
        fla_dir, bold_runs[0], 
        bold_nifti, mask_nifti, 
        design_matrix, time_window, search_frequencies, TR
    )
    
    """
    Run first-level analysis on ALL bold runs grouped by session id and run id
    """

    """ COMMENT OUT SESSION-LEVEL FLA
    # Group bold runs by session id
    print("Group bold runs by session id:")
    ses_id_bold_mappings = {}
    unique_ses_ids = list(set([i.stem.split("ses-")[1].split("_")[0] for i in bold_runs]))
    for _ses_id in unique_ses_ids:
        ses_id_bold_mappings[_ses_id] = []
        for _bold_run in bold_runs:
            if f"ses-{_ses_id}" in _bold_run.stem:
                print(f" - add {_bold_run} to {_ses_id}")
                ses_id_bold_mappings[_ses_id].append(_bold_run)
                # Run first-level analysis on each bold run -- added 20231101
                _run_id = _bold_run.stem.split("run-")[-1].split("_")[0]
                bold_base = get_fla_bold_base(
                    sub_id, ses_id, task_id, f"{_ses_id}X{_run_id}"
                )
                all_average = sub_dir / f"{bold_base}_bold.dtseries.nii"
                all_average = average_bold_runs([_bold_run], all_average)
                bold_nifti = str(all_average).replace(".dtseries.nii", ".nii.gz")
                bold_nifti = convert_cifti_to_nifti(all_average, bold_nifti)
                run_fla(
                    sub_id, ses_id, task_id, f"{_ses_id}X{_run_id}", 
                    fla_dir, bold_runs[0], 
                    bold_nifti, mask_nifti, 
                    design_matrix, time_window, search_frequencies, TR
                )
    # Loop over bold runs from each session id
    for _ses_id, _bold_runs in ses_id_bold_mappings.items():
        # Process all bold
        bold_base = get_fla_bold_base(
            sub_id, ses_id, task_id, _ses_id
        )
        # Average bold runs
        all_average = sub_dir / f"{bold_base}_bold.dtseries.nii"
        all_average = average_bold_runs(_bold_runs, all_average)
        # Convert bold average (to NIFTI format)
        # *FLA tools only processes NIFTIs
        bold_nifti = str(all_average).replace(".dtseries.nii", ".nii.gz")
        bold_nifti = convert_cifti_to_nifti(all_average, bold_nifti)
        # Run first-level analysis on averaged bold runs
        # *all of the FLA outputs will be converted back to cifti format
        run_fla(
            sub_id, ses_id, task_id, _ses_id, 
            fla_dir, bold_runs[0], 
            bold_nifti, mask_nifti, 
            design_matrix, time_window, search_frequencies, TR
        )
    """

    if no_bootstrap:
        print("--no-bootstrap set. Exiting.")
        sys.exit()

    """
    Set-up analysis class for bootstrapping
    """
    out_dir = sub_dir / "bootstrap"
    analyzer = analyze_bootstrap(
        fla_dir, n_iterations,
        search_frequencies,
        sub_id, ses_id, task_id,
        out_dir, all_average,
    )

    """
    Iterate first-level analysis on train/test split of the bold runs
    """
    for i in range(n_iterations):
        # Show progress
        print(f"Processing train/test split [{str(i+1).zfill(5)}/{str(n_iterations).zfill(5)}]")
        # Get data split
        train, test = split_runs(bold_runs)
        # Track bootstrap train/test splits
        with open(train_txt, "a") as f:
            f.write(",".join([str(i) for i in train]) + '\n')
        with open(test_txt, "a") as f:
            f.write(",".join([str(i) for i in test]) + '\n')
        
        # training set
        train_bold_base = get_fla_bold_base(
            sub_id, ses_id, task_id, f"TRAIN{str(i).zfill(5)}"
        )
        train_split_average = average_bold_runs(train, sub_dir / f"{train_bold_base}_bold.dtseries.nii")
        train_bold_nifti = str(train_split_average).replace(".dtseries.nii", ".nii.gz")
        train_bold_nifti = convert_cifti_to_nifti(train_split_average, train_bold_nifti)
        # first level analysis
        run_fla(
            sub_id, ses_id, task_id, f"TRAIN{str(i).zfill(5)}", 
            fla_dir, bold_runs[0], 
            train_bold_nifti, mask_nifti, 
            design_matrix, time_window, search_frequencies, TR
        )
        
        # test set
        test_bold_base = get_fla_bold_base(
            sub_id, ses_id, task_id, f"TEST{str(i).zfill(5)}"
        )
        test_split_average = average_bold_runs(test, sub_dir / f"{test_bold_base}_bold.dtseries.nii")
        test_bold_nifti = str(test_split_average).replace(".dtseries.nii", ".nii.gz")
        test_bold_nifti = convert_cifti_to_nifti(test_split_average, test_bold_nifti)
        # first level analysis
        run_fla(
            sub_id, ses_id, task_id, f"TEST{str(i).zfill(5)}", 
            fla_dir, bold_runs[0], 
            test_bold_nifti, mask_nifti, 
            design_matrix, time_window, search_frequencies, TR
        )

        # Compute bootstrap metric
        analyzer.add_train_test_set(i, train_split_average, test_split_average)
        
        # clean-up
        clean_up_files([
            train_split_average, train_bold_nifti, 
            test_split_average, test_bold_nifti, 
        ])
        # remove train/test FLA dir
        train_fla_dir = fla_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / f"task-{task_id}" / f"run-TRAIN{str(i).zfill(5)}"
        test_fla_dir = fla_dir / f"sub-{sub_id}" / f"ses-{ses_id}" / f"task-{task_id}" / f"run-TEST{str(i).zfill(5)}"
        clean_up_directory(train_fla_dir)
        clean_up_directory(test_fla_dir)

    analyzer.aggregate_metrics()

    # Clean-up
    """
    clean_up_files([
        all_average,
        bold_nifti,
        mask_nifti,
    ])
    """

    print("Complete")

if __name__ == "__main__":
    app()