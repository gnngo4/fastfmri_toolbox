from pathlib import Path
from typing import Union, List, Dict, Tuple
import itertools

import numpy as np
import pandas as pd
import nibabel as nib

from nilearn.glm.first_level import FirstLevelModel
        
REQUIRED_KEYS = ['sub','ses','task','run']
DIRECTORY_TREE_NAMES = ['GLM']
PATH_STEMS = {
    'predicted': 'predicted_bold.nii.gz',
    'residual': 'residual_bold.nii.gz',
    'windowed': 'windowed_bold.nii.gz',
}
FREQUENCY_GLM_BASIS = ['sine','cosine']

class FirstLevelAnalysis():
    
    def __init__(
        self,
        derivatives_dir: Union[str,Path],
        bold_path: Union[str,Path],
        mask_path: Union[str,Path],
        design_matrix: pd.DataFrame,
        time_window: Tuple[float, float],
        search_frequencies: List[float],
    ):
        
        # Set-up
        self.derivatives_dir = Path(derivatives_dir)
        self.bold_path = Path(bold_path).resolve()
        self.mask_path = Path(mask_path).resolve()
        self._get_path_info()
        self._make_directory_tree()
        
        self.TR = nib.load(self.bold_path).header.get_zooms()[-1]
        self.design_matrix = design_matrix
        self.search_frequencies = search_frequencies
        self.window_indices = self._get_time_indices(time_window)
        
        # Check columns exist in input design matrix
        self._design_matrix_validator()
        
        
    def run_frequency_glm(
        self,
        out_windowed_bold: bool = False,
        out_predicted: bool = False,
        out_residual: bool = False,
    ) -> None:
        
        # Instantiate FirstLevelModel object
        flm = FirstLevelModel(
            t_r = self.TR,
            mask_img=nib.load(self.mask_path),
            signal_scaling=False,
            minimize_memory=True,
        )
        
        # Window the inputted `bold_path` using the `window_indices`
        windowed_bold_img = self._window_image()
        if out_windowed_bold:
            nib.save(windowed_bold_img, f"{self.directory_tree['GLM']}/{PATH_STEMS['windowed']}")
        
        # Fit
        flm.fit(windowed_bold_img,design_matrices=self.design_matrix)
        
        # Compute single contrast for each predictor
        single_contrasts = self._make_contrasts()
        for regressor_label, single_contrast in single_contrasts.items():
            # Compute single variable contrast
            single_contrast_results = flm.compute_contrast(single_contrast, output_type='all')
            # Save all outputs
            for k, v in single_contrast_results.items():
                nib.save(v,f"{self.directory_tree['GLM']}/{regressor_label}_{k}.nii.gz")
                
        # Compute F-statistic contrast for each search frequency
        for f in self.search_frequencies:
            # Create F-stat contrast for search frequency, `f`
            F_contrast = np.vstack((single_contrasts[f'basis-sine_f-{f}'],single_contrasts[f'basis-cosine_f-{f}']))
            # Compute F-stat contrast
            F_contrast_results = flm.compute_contrast(F_contrast, output_type='all')    
            # Save all outputs
            for k, v in F_contrast_results.items():
                nib.save(v, f"{self.directory_tree['GLM']}/frequency-{f}_{k}.nii.gz")
                
        if out_predicted or out_residual:
            
            for ix, regressor_label in enumerate(single_contrasts.keys()):
                if ix == 0:
                    predicted_data = nib.load(f"{self.directory_tree['GLM']}/{regressor_label}_effect_size.nii.gz").get_fdata()[:,:,:,np.newaxis] * self.design_matrix[regressor_label].values[np.newaxis,np.newaxis,np.newaxis,:]
                else:
                    predicted_data += nib.load(f"{self.directory_tree['GLM']}/{regressor_label}_effect_size.nii.gz").get_fdata()[:,:,:,np.newaxis] * self.design_matrix[regressor_label].values[np.newaxis,np.newaxis,np.newaxis,:]
            
            if out_predicted:
                predicted_img = nib.Nifti1Image(
                    predicted_data, 
                    affine=windowed_bold_img.affine,
                    header=windowed_bold_img.header,
                )
                nib.save(predicted_img, f"{self.directory_tree['GLM']}/{PATH_STEMS['predicted']}")
                
            if out_residual:                
                residual_data = (windowed_bold_img.get_fdata() * nib.load(self.mask_path).get_fdata()[:,:,:,np.newaxis]) - predicted_data
                residual_img = nib.Nifti1Image(
                    residual_data,
                    affine=windowed_bold_img.affine,
                    header=windowed_bold_img.header
                )
                nib.save(residual_img, f"{self.directory_tree['GLM']}/{PATH_STEMS['residual']}")
        
    def _get_path_info(self) -> None:
        
        base_str = self.bold_path.name
        base_str = [i.split('-') for i in base_str.split('_') if '-' in i]
        for key_pair in base_str:
            assert len(key_pair) == 2, f"{key_pair} has {len(key_pair)} elements.\nExpects only 2."
            
        self.bids_info = dict(base_str)
        for _key in REQUIRED_KEYS:
            assert _key in self.bids_info.keys()
            
    def _make_directory_tree(self) -> None:
        
        _list = []
        for name in DIRECTORY_TREE_NAMES:
            _dir = Path(f"{self.derivatives_dir}/sub-{self.bids_info['sub']}/ses-{self.bids_info['ses']}/task-{self.bids_info['task']}/run-{self.bids_info['run']}/{name}")
            _list.append(_dir)
            if not _dir.exists():
                _dir.mkdir(parents=True)
                
        self.directory_tree = dict(zip(DIRECTORY_TREE_NAMES,_list))
    
    def _get_time_indices(self, time_window: Tuple[float, float]) -> Tuple[int, int]:

        import math 

        idx_1 = int(math.floor(time_window[0] / self.TR))
        idx_2 = int(math.floor(time_window[1] / self.TR))
        
        return (idx_1, idx_2)
        
    def _design_matrix_validator(self) -> None:
        
        for ix, (f, _basis) in enumerate(itertools.product(self.search_frequencies,FREQUENCY_GLM_BASIS)):
            assert f"basis-{_basis}_f-{f}" in self.design_matrix.columns, f"[basis-{_basis}_f-{f}] not found in the design matrix."
            
    def _make_contrasts(self) -> Dict[str,np.ndarray]:

        contrast_matrix = np.eye(self.design_matrix.shape[-1])
        contrasts = {col: contrast_matrix[i,:] for i, col in enumerate(self.design_matrix.columns)}

        return contrasts
    
    def _window_image(self) -> nib.Nifti1Image:
        
        bold_img = nib.load(self.bold_path)
        bold_data = bold_img.get_fdata()[:,:,:,self.window_indices[0]:self.window_indices[1]+1]
        
        return nib.Nifti1Image(bold_data,affine=bold_img.affine,header=bold_img.header)