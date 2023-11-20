import sys

sys.path.append("/opt/app")

from pathlib import Path
import os
import tempfile
import glob
import subprocess

import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt

from fastfmri_toolbox.modelling.design_matrix import (
    DesignMatrix,
    FrequencyRegressors,
    DriftRegressors,
    MotionParameters,
    MeanSignalRegressors,
    CompCorRegressors,
    ScrubbingRegressors,
)
from fastfmri_toolbox.modelling.first_level_analysis import FirstLevelAnalysis
from fastfmri_toolbox.visualize.base import PlotSlices

PREPROC_DIR_TYPES = ["OSCPREP", "NORDIC"]

"""
Processing functions
"""


def create_temp_file(suffix):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as temp_file:
        return temp_file.name


def search(base_dir, wildcard):
    search_path = Path(base_dir) / wildcard
    files = glob.glob(str(search_path))

    if not files:
        raise FileNotFoundError(f"No files were found in: {search_path}")

    return files


def convert_niftis_to_ciftis(
    sub_id,
    ses_id,
    task_id,
    run_id,
    root_directory, 
    template_cifti, 
    TR
):
    file_pattern = f"sub-{sub_id}/ses-{ses_id}/task-{task_id}/run-{run_id}/GLM/*.nii.gz"
    matching_files = glob.glob(
        os.path.join(root_directory, file_pattern), recursive=False
    )

    for nifti in matching_files:
        nifti_path = Path(nifti)
        output_file = (
            Path(nifti.replace(".nii.gz", ".dtseries.nii"))
            if nifti.endswith("bold.nii.gz")
            else Path(nifti.replace(".nii.gz", ".dscalar.nii"))
        )

        cmd = [
            "wb_command",
            "-cifti-convert",
            "-from-nifti",
            str(nifti_path),
            str(template_cifti),
            str(output_file),
        ]

        if nifti.endswith("bold.nii.gz"):
            cmd.extend(["-reset-timepoints", str(TR), "0"])
        else:
            cmd.extend(["-reset-scalars"])

        try:
            process = subprocess.Popen(cmd)  # Run the subprocess asynchronously
            process.communicate()  # Wait for subprocess to finish
            os.remove(nifti)
        except subprocess.CalledProcessError as e:
            print(f"Error converting {nifti}: {e}")
        except OSError as e:
            print(f"Error removing {nifti}: {e}")

"""
Path loader - automatically searches and grabs paths required
for first level analysis of oscprep tools.
"""


class PathLoader:
    def __init__(
        self, 
        oscprep_dir, 
        sub_id, 
        ses_id, 
        task_id, 
        run_id, 
        nordic_dir = 'None',
    ):
        self.oscprep_dir = oscprep_dir
        self.sub_id = sub_id
        self.ses_id = ses_id
        self.task_id = task_id
        self.run_id = run_id
        
        if nordic_dir == 'None':
            self.preproc_dir_type = "OSCPREP"
        else:
            self.preproc_dir_type = "NORDIC"
            self.nordic_dir = nordic_dir

        if self.preproc_dir_type == 'OSCPREP':
            self.bold_nifti = Path(
                self._search(
                    self._get_func_dir_oscprep(),
                    f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_*_run-{run_id}*desc-preproc_bold.nii.gz",
                )
            )
            self.bold_dtseries = Path(
                self._search(
                    self._get_func_dir_oscprep(),
                    f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_*_run-{run_id}*desc-preproc_bold.dtseries.nii",
                )
            )
        elif self.preproc_dir_type == 'NORDIC':
            self.bold_nifti_oscprep = Path(
                self._search(
                    self._get_func_dir_oscprep(),
                    f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_*_run-{run_id}*desc-preproc_bold.nii.gz",
                )
            )
            self.bold_nifti = Path(
                self._search(
                    self._get_func_dir_nordic(),
                    f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_*_run-{run_id}*desc-preproc_bold.nii.gz",
                )
            )
            print("WARNING: NORDIC directory does not output a dtseries.\nUsing dtseries from oscprep instead.")
            self.bold_dtseries = Path(
                self._search(
                    self._get_func_dir_oscprep(),
                    f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_*_run-{run_id}*desc-preproc_bold.dtseries.nii",
                )
            )
        else:
            assert self.preproc_dir_type in PREPROC_DIR_TYPES

        self.bold_reference = Path(
            self._search(
                self._get_func_dir_oscprep(),
                f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_*_run-{run_id}*_boldref.nii.gz",
            )
        )
        self.bold_brainmask = Path(
            self._search(
                self._get_func_dir_oscprep(),
                f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_*_run-{run_id}*_brainmask.nii.gz",
            )
        )
        self.t1w = Path(
            self._search(
                self._get_smriprep_dir(),
                f"ses-*/anat/sub-{sub_id}_*_desc-preproc_T1w.nii.gz",
            )
        )

        self.xfm_mni_to_t1w = Path(
            self._search(
                self._get_smriprep_dir(),
                f"ses-*/anat/sub-{sub_id}_*_from-MNI152NLin2009cAsym_*h5",
            )
        )

    def _get_func_dir_oscprep(self):
        return (
            f"{self.oscprep_dir}/bold_preproc/sub-{self.sub_id}/ses-{self.ses_id}/func"
        )

    def _get_func_dir_nordic(self):
        return (
            f"{self.nordic_dir}/sub-{self.sub_id}/ses-{self.ses_id}/func"
        )

    def _get_smriprep_dir(self):
        return f"{self.oscprep_dir}/smriprep/sub-{self.sub_id}"

    def _search(self, base_dir, wildcard):
        search_path = Path(base_dir) / wildcard
        files = glob.glob(str(search_path))

        if not files:
            raise FileNotFoundError(f"No files matching {search_path} were found.")
        if len(files) != 1:
            raise ValueError(
                f"Expected only 1 returned file. {len(files)} file(s) found."
            )

        return files[0]

"""
REGRESSOR_COMBINATIONS = {
    "min": [FrequencyRegressors, DriftRegressors],
    "min+motion6": [FrequencyRegressors, DriftRegressors, MotionParameters],
    "min+motion24": [FrequencyRegressors, DriftRegressors, MotionParameters],
    "min+motion6+wmcsf_mean": [
        FrequencyRegressors,
        DriftRegressors,
        MotionParameters,
        MeanSignalRegressors,
    ],
    "min+motion24+wmcsf_mean": [
        FrequencyRegressors,
        DriftRegressors,
        MotionParameters,
        MeanSignalRegressors,
    ],
    "min+motion6+wmcsf_compcor": [
        FrequencyRegressors,
        DriftRegressors,
        MotionParameters,
        CompCorRegressors,
    ],
    "min+motion24+wmcsf_compcor": [
        FrequencyRegressors,
        DriftRegressors,
        MotionParameters,
        CompCorRegressors,
    ],
    "min+motion24+wmcsf_compcor+scrub": [
        FrequencyRegressors,
        DriftRegressors,
        MotionParameters,
        CompCorRegressors,
        ScrubbingRegressors,
    ],
}
"""

REGRESSOR_COMBINATIONS_SOFT = {
    "min+motion24+wmcsfmean": [
        FrequencyRegressors,
        DriftRegressors,
        MotionParameters,
        MeanSignalRegressors,
    ],
}
"""
"min+motion24+wmcsfcompcor+scrub": [
    FrequencyRegressors,
    DriftRegressors,
    MotionParameters,
    CompCorRegressors,
    ScrubbingRegressors,
],
"""

REGRESSOR_COMBINATIONS_AGGR = {
    "min+motion24+wmcsfmean": [
        DriftRegressors,
        MotionParameters,
        MeanSignalRegressors,
    ],
}
"""
    "min+motion24+wmcsfcompcor": [
        DriftRegressors,
        MotionParameters,
        CompCorRegressors,
    ],
"""


def build_design_matrix(
    path_loader,
    time_window,
    search_frequencies,
    dm_type,
    show_flag=False,
    high_pass_threshold=0.01,
    add_constant=True,
    denoise_only=False,
    nordic_dir="None"
):

    if denoise_only:
        REGRESSOR_COMBINATIONS = REGRESSOR_COMBINATIONS_AGGR
    else:
        REGRESSOR_COMBINATIONS = REGRESSOR_COMBINATIONS_SOFT

    if dm_type not in REGRESSOR_COMBINATIONS:
        raise ValueError(f"{dm_type} is not a valid dm_type.")

    if nordic_dir == "None":
        dm = DesignMatrix(time_window, search_frequencies, bold_path=path_loader.bold_nifti)
        print(f"[RUN DESIGN MATRIX] {path_loader.bold_nifti}")
    else:
        dm = DesignMatrix(time_window, search_frequencies, bold_path=path_loader.bold_nifti_oscprep)
        print(f"[RUN DESIGN MATRIX] {path_loader.bold_nifti_oscprep}")

    print(f"REGRESSOR INFO: {REGRESSOR_COMBINATIONS}")
    for regressor_class in REGRESSOR_COMBINATIONS[dm_type]:
        if regressor_class is FrequencyRegressors:
            print("Adding frequency regressors.")
            dm.add_regressor(regressor_class(search_frequencies, dm.time_points))
        elif regressor_class is DriftRegressors:
            print("Adding drift regressors.")
            dm.add_regressor(
                regressor_class(
                    dm.time_points,
                    high_pass_threshold=high_pass_threshold,
                    add_constant=add_constant,
                )
            )
        elif regressor_class is MotionParameters:
            print("Adding motion regressors.")
            _mc_params = dm_type.split("motion")[1].split("+")[0]
            dm.add_regressor(
                regressor_class(
                    dm.get_time_indices(time_window),
                    mc_params=24 if _mc_params == "24" else 6,
                )
            )
        elif regressor_class is MeanSignalRegressors:
            if "wmcsfmean" in dm_type:
                print("Adding mean WM and CSF regressors.")
                for _regressor_type in ["WM", "CSF"]:
                    dm.add_regressor(
                        regressor_class(
                            dm.get_time_indices(time_window),
                            regressor_type=_regressor_type,
                            higher_order_flag=True,
                        )
                    )
        elif regressor_class is CompCorRegressors:
            if "wmcsfcompcor" in dm_type:
                print("Adding compcor WM and CSF regressors.")
                for _regressor_type in ["WM", "CSF"]:
                    dm.add_regressor(
                        regressor_class(
                            dm.get_time_indices(time_window),
                            regressor_type=_regressor_type,
                            variance_explained=.5,
                        )
                    )
        elif regressor_class is ScrubbingRegressors:
            print("Adding scrubbing regressors.")
            dm.add_regressor(
                regressor_class(
                    dm.get_time_indices(time_window),
                    movement_param="FD",
                    movement_threshold=0.2,
                )
            )
        else:
            raise ValueError(f"{str(regressor_class)} is not a valid regressor class.")

    design_matrix = dm.build_design_matrix(enforce_demean=True)
    n_regressors = design_matrix.shape[-1]
    fig = dm.plot_design_matrix(show_plot=show_flag, figsize=((n_regressors / 2), 4))

    return design_matrix, fig


def run_glm(
    path_loader,
    time_window,
    search_frequencies,
    image_type,
    design_matrix,
    out_dir,
    smooth_mm,
    surf_left="/scratch/surfaces/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii", 
    surf_right="/scratch/surfaces/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii",
):
    if image_type == "CIFTI":
        temp_files = []
        workflows = []

        # Create pseudo-nifti from cifti
        if smooth_mm > 0:
            for _surf in [surf_left, surf_right]:
                assert Path(_surf).exists(), f"{_surf} does not exist."
            tmp_bold_dtseries_path = create_temp_file(".dtseries.nii")
            _cmd = f"wb_command -cifti-smoothing {path_loader.bold_dtseries} {smooth_mm} {smooth_mm} COLUMN {tmp_bold_dtseries_path} -fwhm -left-surface {surf_left} -right-surface {surf_right}"
            workflows.append(_cmd)
            temp_files.append(tmp_bold_dtseries_path)

            tmp_bold_path = create_temp_file(".nii.gz")
            bids_tmp_bold_path = Path("/tmp") / Path(
                path_loader.bold_nifti.stem.replace("nii", tmp_bold_path.split("/")[-1])
            )
            _cmd = f"wb_command -cifti-convert -to-nifti {tmp_bold_dtseries_path} {bids_tmp_bold_path}"
            workflows.append(_cmd)
            temp_files.append(tmp_bold_path)
            temp_files.append(bids_tmp_bold_path)
        else:
            tmp_bold_path = create_temp_file(".nii.gz")
            bids_tmp_bold_path = Path("/tmp") / Path(
                path_loader.bold_nifti.stem.replace("nii", tmp_bold_path.split("/")[-1])
            )
            _cmd = f"wb_command -cifti-convert -to-nifti {path_loader.bold_dtseries} {bids_tmp_bold_path}"
            workflows.append(_cmd)
            temp_files.append(tmp_bold_path)
            temp_files.append(bids_tmp_bold_path)

        # Create pseudo-nifti brainmask
        mask_path = create_temp_file(".nii.gz")
        _cmd = f"fslmaths {bids_tmp_bold_path} -Tmean -bin {mask_path}"
        workflows.append(_cmd)
        temp_files.append(mask_path)

        # Run
        for cmd in workflows:
            subprocess.run(cmd, shell=True, check=True)

        # Add an empty dimension back to `mask_path`
        mask_img = nib.load(mask_path)
        mask_data = mask_img.get_fdata()[:, :, np.newaxis]
        mask_img = nib.Nifti1Image(
            mask_data, affine=mask_img.affine, header=mask_img.header
        )
        nib.save(mask_img, mask_path)

    # Load TR
    TR = float(nib.load(path_loader.bold_nifti).header.get_zooms()[-1])

    # Set-up first-level analysis
    print(f"[RUN FLA] {path_loader.bold_nifti}")
    fla = FirstLevelAnalysis(
        derivatives_dir=f"{out_dir}",
        bold_path=path_loader.bold_nifti
        if image_type == "NIFTI"
        else bids_tmp_bold_path,
        mask_path=path_loader.bold_brainmask if image_type == "NIFTI" else mask_path,
        design_matrix=design_matrix,
        time_window=time_window,
        search_frequencies=search_frequencies,
        TR=TR,
    )

    # Run
    fla.run_frequency_glm(
        save_windowed_bold=True, save_denoised_bold=True,
    )

    if image_type == "CIFTI":
        convert_niftis_to_ciftis(
            path_loader.sub_id,
            path_loader.ses_id,
            path_loader.task_id,
            path_loader.run_id,
            f"{out_dir}", 
            path_loader.bold_dtseries, 
            TR,
        )
        for f in temp_files:
            os.remove(f)


def extract_glm_metrics(
    path_loader,
    glm_dir,
    figures_dir,
    roi="/data/fastfmri_toolbox_test_data/rois/roi-calcarine_space-MNI152NLin2009cAsym.nii.gz",
    show_plots=False,
):
    if not Path(roi).exists():
        raise FileNotFoundError(f"{roi} does not exist.")

    cmds = []
    temp_files = []
    # 1 apply mni transform to convert roi to t1-space
    roi_t1 = create_temp_file(".nii.gz")
    _cmd = f"antsApplyTransforms -d 3 -i {roi} -r {path_loader.t1w} -o {roi_t1} -n NearestNeighbor -t {path_loader.xfm_mni_to_t1w}"
    temp_files.append(_cmd)
    cmds.append(_cmd)
    # 2 resample roi to bold-t1 space
    _cmd = f"flirt -in {roi_t1} -ref {path_loader.bold_nifti} -applyxfm -usesqform -out {roi_t1} -interp nearestneighbour"
    cmds.append(_cmd)

    # Run
    for cmd in cmds:
        subprocess.run(cmd, shell=True, check=True)

    # Plot
    z_score_paths = search(
        f"{glm_dir}",
        f"sub-{path_loader.sub_id}/ses-{path_loader.ses_id}/task-{path_loader.task_id}/run-{path_loader.run_id}/GLM/frequency*z_score.nii.gz",
    )
    for ix, z_score_path in enumerate(z_score_paths):
        plotter = PlotSlices(path_loader.bold_nifti, n_cols=12, z_slice_range=None)
        plotter.plot_base()
        plotter.add_contours(roi_t1, c="cyan", lw=0.5, zorder=3)
        plotter.add_overlay(z_score_path, cmap="Reds", threshold=2, vmax=4, zorder=2)
        plotter.fig.suptitle(
            f"sub-{path_loader.sub_id}_ses-{path_loader.ses_id}_task-{path_loader.task_id}_run-{path_loader.run_id}\n{Path(z_score_path).stem.split('.nii')[0]}",
            fontsize=30,
        )

        # Save
        plotter.fig.savefig(
            f"{figures_dir}/sub-{path_loader.sub_id}_ses-{path_loader.ses_id}_task-{path_loader.task_id}_run-{path_loader.run_id}_{Path(z_score_path).stem.split('.nii')[0]}.png"
        )

        # Option to view in notebook
        if not show_plots:
            plt.close()
