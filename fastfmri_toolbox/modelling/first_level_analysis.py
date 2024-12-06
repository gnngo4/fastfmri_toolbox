from pathlib import Path
from typing import Union, List, Dict, Tuple, Optional
import itertools

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.first_level import FirstLevelModel

REQUIRED_KEYS = ["sub", "ses", "task", "run"]
DIRECTORY_TREE_NAMES = ["GLM"]
PATH_STEMS = {
    "windowed": "desc-windowed_bold.nii.gz",
    "denoised": "desc-denoised_bold.nii.gz",
}
FREQUENCY_GLM_BASIS = [
    "sine",
    "cosine",
]
SAVE_SINGLE_CONTRASTS = [
    "basis-cosine",
    "basis-sine",
]


class FirstLevelAnalysis:
    def __init__(
        self,
        derivatives_dir: Union[str, Path],
        bold_path: Union[str, Path],
        mask_path: Union[str, Path],
        design_matrix: pd.DataFrame,
        time_window: Tuple[float, float],
        search_frequencies: List[float],
        TR: Optional[float] = None,
        data_windowed: bool = False,
        fsnr_offset: float = .025,
    ):
        # Set-up
        self.derivatives_dir = Path(derivatives_dir)
        self.bold_path = Path(bold_path).resolve()
        self.mask_path = Path(mask_path).resolve()
        self._get_path_info()
        self._make_directory_tree()
        self.design_matrix = design_matrix
        self.search_frequencies = search_frequencies
        self.fsnr_offset = fsnr_offset
        self.bids_base_str = f"sub-{self.bids_info['sub']}_ses-{self.bids_info['ses']}_task-{self.bids_info['task']}_run-{self.bids_info['run']}"
        if TR is None:
            self.TR: float = float(nib.load(self.bold_path).header.get_zooms()[-1])
        else:
            self.TR = TR
        self.data_windowed = data_windowed
        if not self.data_windowed:
            self.window_indices = self._get_time_indices(time_window)

        # Check columns exist in input design matrix
        self._design_matrix_validator()

    def run_frequency_glm(
        self,
        save_windowed_bold: bool = False,
        save_denoised_bold: bool = False,
        save_frequency_snrs: bool = False,
        save_additional_single_contrasts: Union[str, List[str], None] = None,
    ) -> None:
        self.save_single_contrast_list = self._get_single_contrast_list(
            save_additional_single_contrasts
        )

        store_contrasts, store_contrast_results = False, {}
        if save_denoised_bold:
            store_contrasts = True

        # Instantiate FirstLevelModel object
        flm = FirstLevelModel(
            t_r=self.TR,
            mask_img=nib.load(self.mask_path),
            signal_scaling=False,
            minimize_memory=True,
        )

        # Window the inputted `bold_path` using the `window_indices`
        # If the bold data is already windowed, then do not truncate further
        if self.data_windowed:
            windowed_bold_img = nib.load(self.bold_path)
        else:
            windowed_bold_img = self._window_image()
            if save_windowed_bold:
                nib.save(
                    windowed_bold_img,
                    f"{self.directory_tree['GLM']}/{self.bids_base_str}_{PATH_STEMS['windowed']}",
                )
            

        # Fit
        flm.fit(windowed_bold_img, design_matrices=self.design_matrix)

        # Compute single contrast for each predictor
        single_contrasts = self._make_contrasts()
        for regressor_label, single_contrast in single_contrasts.items():
            # Compute single variable contrast
            single_contrast_results = flm.compute_contrast(
                single_contrast, output_type="all"
            )
            if store_contrasts:
                store_contrast_results[regressor_label] = single_contrast_results
            # Save all outputs
            if any([i in regressor_label for i in self.save_single_contrast_list]):
                for k, v in single_contrast_results.items():
                    nib.save(
                        v, f"{self.directory_tree['GLM']}/{self.bids_base_str}_{regressor_label}_{k}.nii.gz"
                    )

        # Compute phaseshift between sine and cosine fits
        for f in self.search_frequencies:
            B = nib.load(
                f"{self.directory_tree['GLM']}/{self.bids_base_str}_basis-cosine_f-{f}_effect_size.nii.gz"
            )
            A = nib.load(
                f"{self.directory_tree['GLM']}/{self.bids_base_str}_basis-sine_f-{f}_effect_size.nii.gz"
            )
            phi = np.arctan2(B.get_fdata(), A.get_fdata())
            # Force all values from [-pi, pi] to [0, 2pi]
            phi = self._normalize_angle(phi)
            # Currently, phi shows the phase-shift of a sine wave to the left [EQN: y = sin(theta + phi)]
            # Convert phi to indicate the phase-shift of a sine wave to the right [EQN: y = sin(theta - phi)]
            # This is done by subtracting phi from 2pi...
            phi = (-1 * phi) + (2 * np.pi)
            # Convert to seconds: for a given `f`, `phi` indicates the time delay of a sine wave, 
            # Note: this time delay does not account for stimulus offsets 
            phi /= (2 * np.pi * f)
            # Save
            phi_img = nib.Nifti1Image(phi, affine=A.affine, header=A.header)
            nib.save(
                phi_img, f"{self.directory_tree['GLM']}/{self.bids_base_str}_frequency-{f}_phasedelay.nii.gz"
            )

        # Compute F-statistic contrast for each search frequency
        for f in self.search_frequencies:
            # Create F-stat contrast for search frequency, `f`
            F_contrast = np.vstack(
                (
                    single_contrasts[f"basis-sine_f-{f}"],
                    single_contrasts[f"basis-cosine_f-{f}"],
                )
            )
            # Compute F-stat contrast
            F_contrast_results = flm.compute_contrast(F_contrast, output_type="all")
            # Save all outputs
            for k, v in F_contrast_results.items():
                nib.save(v, f"{self.directory_tree['GLM']}/{self.bids_base_str}_frequency-{f}_{k}.nii.gz")

        if store_contrasts:
            denoised_data = windowed_bold_img.get_fdata()
            for regressor_label in single_contrasts.keys():
                if regressor_label.startswith('basis-sine_f') or regressor_label.startswith('basis-cosine_f') or regressor_label == "constant":
                    #print(f"[DENOISING] Skipping regressor, {regressor_label}.")
                    continue
                else:
                    #print(f"[DENOISING] Removing regressor, {regressor_label}.")
                    denoised_data -= (
                        store_contrast_results[regressor_label]["effect_size"].get_fdata()
                        * self.design_matrix[regressor_label].values[
                            np.newaxis, np.newaxis, np.newaxis, :
                        ]
                    )

            if save_denoised_bold:
                denoised_img = nib.Nifti1Image(
                    denoised_data,
                    affine=windowed_bold_img.affine,
                    header=windowed_bold_img.header,
                )
                nib.save(
                    denoised_img,
                    f"{self.directory_tree['GLM']}/{self.bids_base_str}_{PATH_STEMS['denoised']}",
                )

        # messy code for calculating SNR of a frequency's peak, it only will calculate power spectrum on windowed data...
        if save_frequency_snrs:
            if self.data_windowed:
                for search_f in self.search_frequencies:
                    fpower_outbase = f"frequency-{search_f}_power.nii.gz"
                    fpower_save_nifti = f"{self.directory_tree['GLM']}/{self.bids_base_str}_{fpower_outbase}"
                    fsnr_outbase = f"frequency-{search_f}_pSNR.nii.gz"
                    fsnr_save_nifti = f"{self.directory_tree['GLM']}/{self.bids_base_str}_{fsnr_outbase}"
                    bold_img = nib.load(self.bold_path)
                    bold_data = bold_img.get_fdata()
                    coords = bold_img.shape[:-1]
                    n_dims = len(coords)
                    indices = [0] * n_dims
                    fpower_data = np.zeros(coords)
                    fsnr_data = np.zeros(coords)
                    while indices[0] < coords[0]:
                        current_coordinate = tuple(indices)
                        ts_indices = current_coordinate + (slice(None),)
                        ts = bold_data[ts_indices]
                        # Convert `ts` to fractional percent change
                        ts_baseline = np.mean(ts)
                        ts = (ts - ts_baseline) / ts_baseline
                        fpower, fsnr = self._calculate_fsnr_welch(
                            ts, 
                            search_f, 
                            [search_f-self.fsnr_offset, search_f+self.fsnr_offset],
                        )
                        fpower_data[current_coordinate] = fpower
                        fsnr_data[current_coordinate] = fsnr
                        for i in range(n_dims-1,-1,-1):
                            indices[i]+=1
                            if indices[i] < coords[i]:
                                break
                            else:
                                indices[i] = 0
                        else:
                            break
                    fpower_img = nib.Nifti1Image(fpower_data, affine=bold_img.affine, header=bold_img.header)
                    nib.save(fpower_img, fpower_save_nifti)
                    fsnr_img = nib.Nifti1Image(fsnr_data, affine=bold_img.affine, header=bold_img.header)
                    nib.save(fsnr_img, fsnr_save_nifti)
            else:
                raise ValueError(f"Set `data_windowed=True` to calculate power spectrum on non-windowed data.")

    def _find_closest_index(self, array: np.ndarray, value: float):
        # Calculate the absolute differences between each element and the target value
        absolute_differences = np.abs(array - value)
        
        # Find the index with the minimum absolute difference
        closest_index = np.argmin(absolute_differences)
        
        return closest_index

    def _calculate_fsnr_welch(
        self,
        time_series: np.ndarray, 
        signal_frequency: float, 
        noise_frequency_range: List[float], 
    ) -> float:

        from scipy import signal 

        fs = 1/self.TR
        nperseg = len(time_series)
        frequencies, power_density = signal.welch(time_series, fs=fs, nperseg=nperseg)
        signal_bin = self._find_closest_index(frequencies, signal_frequency)
        noise_bins = [self._find_closest_index(frequencies, f) for f in noise_frequency_range]
        signal_power = power_density[signal_bin]
        noise_power = np.sum(power_density[noise_bins])
        fsnr = signal_power / noise_power

        return signal_power, fsnr

    def _get_single_contrast_list(
        self, save_additional_single_contrasts: Union[str, List[str], None]
    ) -> List[str]:
        if save_additional_single_contrasts is None:
            save_single_contrast_labels = SAVE_SINGLE_CONTRASTS
        elif isinstance(save_additional_single_contrasts, str):
            save_single_contrast_labels = SAVE_SINGLE_CONTRASTS + [
                save_additional_single_contrasts
            ]
        else:
            save_single_contrast_labels = (
                SAVE_SINGLE_CONTRASTS + save_additional_single_contrasts
            )

        return save_single_contrast_labels

    def _get_path_info(self) -> None:
        base_str = self.bold_path.name
        base_str = [i.split("-") for i in base_str.split("_") if "-" in i]
        for key_pair in base_str:
            assert (
                len(key_pair) == 2
            ), f"{key_pair} has {len(key_pair)} elements.\nExpects only 2."

        self.bids_info = dict(base_str)
        for _key in REQUIRED_KEYS:
            assert _key in self.bids_info.keys()

    def _make_directory_tree(self) -> None:
        _list = []
        for name in DIRECTORY_TREE_NAMES:
            _dir = Path(
                f"{self.derivatives_dir}/sub-{self.bids_info['sub']}/ses-{self.bids_info['ses']}/task-{self.bids_info['task']}/run-{self.bids_info['run']}/{name}"
            )
            _list.append(_dir)
            if not _dir.exists():
                _dir.mkdir(parents=True)

        self.directory_tree = dict(zip(DIRECTORY_TREE_NAMES, _list))

    def _get_time_indices(self, time_window: Tuple[float, float]) -> Tuple[int, int]:

        import math
        
        scaled_TR, int_scaling_factor = self._get_int_scaling_factor(self.TR)
        time_window = tuple(value * int_scaling_factor for value in time_window)

        idx_1 = int(math.floor(time_window[0] / scaled_TR))
        idx_2 = int(math.floor(time_window[1] / scaled_TR))

        return (idx_1, idx_2)
    
    def _get_int_scaling_factor(self, x: float, tolerance: float = 0.0001) -> Tuple[int, int]:

        if x == int(x):
            return 1  # No decimals needed, it's already an integer
        
        decimals = 0
        while x != int(x):
            x *= 10
            decimals += 1
            if abs(x-int(x)) <= tolerance:
                return int(round(x)), 10**decimals

    def _design_matrix_validator(self) -> None:
        for f, _basis in itertools.product(
            self.search_frequencies, FREQUENCY_GLM_BASIS
        ):
            assert (
                f"basis-{_basis}_f-{f}" in self.design_matrix.columns
            ), f"[basis-{_basis}_f-{f}] not found in the design matrix."

    def _make_contrasts(self) -> Dict[str, np.ndarray]:
        contrast_matrix = np.eye(self.design_matrix.shape[-1])
        contrasts = {
            col: contrast_matrix[i, :]
            for i, col in enumerate(self.design_matrix.columns)
        }

        return contrasts

    def _window_image(self) -> nib.Nifti1Image:
        bold_img = nib.load(self.bold_path)
        bold_data = bold_img.get_fdata()[
            :, :, :, self.window_indices[0] : self.window_indices[1] + 1
        ]

        return nib.Nifti1Image(
            bold_data, affine=bold_img.affine, header=bold_img.header
        )

    def _normalize_angle(self, theta: np.ndarray) -> np.ndarray:
        
        return theta % ( 2 * np.pi )
