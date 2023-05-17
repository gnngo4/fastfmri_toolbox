from pathlib import Path
from typing import Optional, Union, List, Tuple, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
import itertools

class DesignMatrixRegressors:
    def __init__(self):
        self.confounds_required = False

    def get(self, confounds: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError
    
class FrequencyRegressors(DesignMatrixRegressors):
    def __init__(
        self, 
        search_frequencies: List[float],
        time_points: Tuple[float, float],
    ):
        super().__init__()
        self.confounds_required = False
        self.search_frequencies = search_frequencies
        self.time_points = time_points
        self.n_tps = len(self.time_points)
    
    def get(self, confounds: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, List[str]]:
        
        column_names = []
        basis = ["sine", "cosine"]
        regressors = np.zeros((self.n_tps, len(basis) * len(self.search_frequencies)))
        for ix, (f, _basis) in enumerate(
            itertools.product(self.search_frequencies, basis)
        ):
            _tps = 2 * np.pi * f * self.time_points
            if _basis == "sine":
                regressors[:, ix] = np.sin(_tps)
            else:
                regressors[:, ix] = np.cos(_tps)
            column_names.append(f"basis-{_basis}_f-{f}")

        return (regressors, column_names)

class DriftRegressors(DesignMatrixRegressors):
    def __init__(
        self, 
        time_points: Tuple[float, float],
        high_pass_threshold: float = 0.01,
        add_constant: bool = True,
    ):
        super().__init__()
        self.confounds_required = False
        self.time_points = time_points
        self.high_pass_threshold = high_pass_threshold
        self.add_constant = add_constant
        self.n_tps = len(self.time_points)
    
    def get(self, confounds: Optional[pd.DataFrame] = None) -> Tuple[np.ndarray, List[str]]:
    
        n_times = np.arange(self.n_tps)
        dt = (self.time_points[-1] - self.time_points[0]) / (self.n_tps - 1)
        if self.high_pass_threshold * dt >= 0.5:
            warn(f"""
            High-pass filter will span all accessible frequencies
            and saturate the design matrix. You may want to reduce
            the high_pass _threshold value. 
            The provided value is {self.high_pass_threshold} Hz.
            """)
        order = np.minimum(
            self.n_tps - 1,
            int(np.floor(2 * self.n_tps * self.high_pass_threshold * dt)),
        )
        cosine_drift = np.zeros((self.n_tps, order + 1))
        normalizer = np.sqrt(2.0 / self.n_tps)

        for k in range(1, order + 1):
            cosine_drift[:, k - 1] = normalizer * np.cos(
                (np.pi / self.n_tps) * (n_times + 0.5) * k
            )

        column_names = []
        for ix in range(1, cosine_drift.shape[1]):
            column_names.append(f"drift-{str(ix).zfill(2)}")

        if self.add_constant:
            cosine_drift[:, -1] = 1
            return (cosine_drift, column_names + ["constant"])
        else:
            return (cosine_drift[:,:-1], column_names)
    
class MotionParameters(DesignMatrixRegressors):
    def __init__(
        self,
        time_indices: Tuple[int, int],
        mc_params: Literal[6,24],
    ):
        super().__init__()
        self.confounds_required = True
        self.time_indices = time_indices
        self.mc_params = mc_params

    def get(self, confounds: Optional[pd.DataFrame] = None):

        assert confounds is not None, f"`confounds` must be specified to use this class"
        col_names = self._get_column_names()
        idx_1, idx_2 = self.time_indices[0], self.time_indices[1] + 1
        regressors = confounds[col_names][idx_1:idx_2].values
        return (regressors, col_names)

    def _get_column_names(self):
        
        MOTION_BASE: List[str] = ['trans', 'rot']
        MOTION_DIR: List[str] = ['x', 'y', 'z']
        MOTION_SUFFIX: List[str] = ['derivative1','power2','derivative1_power2']

        regressor_labels = ['_'.join(list(i)) for i in itertools.product(MOTION_BASE, MOTION_DIR)]
        if self.mc_params == 6:
            return regressor_labels
        elif self.mc_params == 24:
            return regressor_labels + ['_'.join(list(i)) for i in itertools.product(MOTION_BASE, MOTION_DIR, MOTION_SUFFIX)]
        else:
            raise ValueError("`self.mc_params` must be set to 6 or 24")
    
class DesignMatrix:
    def __init__(
        self,
        time_window: Tuple[float, float],
        search_frequencies: List[float],
        bold_path: Union[str, Path, None] = None,
        TR: Optional[float] = None,
        confounds_tsv: Union[str, Path, pd.DataFrame, None] = None,
        high_pass_threshold: float = 0.01,
    ):
        self.bold_path = bold_path
        self.confounds_tsv = confounds_tsv
        self.TR = self._get_TR(bold_path, TR)
        self.search_frequencies = search_frequencies
        self.high_pass_threshold = high_pass_threshold
        self.time_points = self._get_time_points(self.TR, time_window)
        self.n_tps = len(self.time_points)
        assert np.all(
            self.time_points[:-1] <= self.time_points[1:]
        ), "`self.time_points` is not sorted in ascending order"

        # design matrix variables
        self.design_matrix = np.zeros((self.n_tps, 0))
        self.column_names = []

    def add_regressor(self, regressors: DesignMatrixRegressors) -> None:
        
        if regressors.confounds_required:
            confounds_df = self._get_confounds(self.bold_path, self.confounds_tsv)
            regressors_array, regressor_labels = regressors.get(confounds_df)
        else:
            regressors_array, regressor_labels = regressors.get()

        # Update design_matrix and column_names
        self.design_matrix = np.hstack((self.design_matrix, regressors_array))
        self.column_names += regressor_labels

    def build_design_matrix(self) -> pd.DataFrame:
        
        assert self.design_matrix.shape[-1] > 0, f"No regressors have been added to the design matrix yet"

        return pd.DataFrame(
            self.design_matrix, index=self.time_points, columns=self.column_names
        )

    def plot_design_matrix(self, figsize=(4, 8)):

        from nilearn.plotting import plot_design_matrix

        fig, ax = plt.subplots(figsize=figsize)

        ax = plot_design_matrix(self.build_design_matrix(), ax=ax)
        ax.set_ylabel("scan volume")

    def get_time_indices(self, time_window: Tuple[float, float]) -> Tuple[int, int]:
        import math

        idx_1 = int(math.floor(time_window[0] / self.TR))
        idx_2 = int(math.floor(time_window[1] / self.TR))

        return (idx_1, idx_2)

    def _get_TR(
        self, bold_path: Union[str, Path, None] = None, TR: Optional[float] = None
    ) -> float:
        if bold_path is not None:
            bold_path = Path(bold_path).resolve()
            if not bold_path.exists():
                raise FileNotFoundError(
                    f"Error: file path does not exist [{bold_path}]"
                )
            return nib.load(bold_path).header.get_zooms()[-1]

        elif TR is not None:
            return TR

        else:
            raise ValueError("Error: either `bold_path` or `TR` must be provided.")

    def _get_time_points(
        self, TR: float, time_window: Tuple[float, float]
    ) -> np.ndarray:
        import math

        tp_1 = math.floor(time_window[0] / self.TR) * self.TR
        tp_2 = math.floor(time_window[1] / self.TR) * self.TR
        time_points = np.arange(tp_1, tp_2 + (self.TR / 10), self.TR)
        time_points -= time_points[0]

        return time_points

    def _get_confounds(
        self, 
        bold_path: Union[str, Path, None] = None,
        confounds_tsv: Union[str, Path, pd.DataFrame, None] = None,
        ) -> pd.DataFrame:

        if isinstance(confounds_tsv, pd.DataFrame):
            return confounds_tsv
        
        elif confounds_tsv is None:
            # Check if a tsv can be found based on oscprep heuristics
            assert bold_path is not None, f"Confounds file cannot be found without a `bold_path` input"
            confounds_tsv = self._get_confounds_path(bold_path, 'tsv')
            return pd.read_csv(confounds_tsv, sep='\t')
        
        else:
            confounds_tsv = Path(confounds_tsv)
            if not confounds_tsv.exists():
                raise FileNotFoundError(
                    f"Error: file path does not exist [{confounds_tsv}]"
                )
            return pd.read_csv(confounds_tsv, sep='\t')

    def _get_confounds_path(
        self, 
        bold_path: Union[str,Path],
        desc: Literal['tsv', 'metadata']
        ) -> Path:

        preproc_bold_suffix = "space-T1w_desc-preproc_bold.nii.gz"
        confounds_suffix = "desc-confounds_timeseries.tsv"
        confounds_metadata_suffix = "desc-confounds_timeseries.json"

        bold_path = Path(bold_path).resolve()
        confounds_tsv = Path(str(bold_path).replace(preproc_bold_suffix, confounds_suffix))
        confounds_metadata = Path(str(bold_path).replace(preproc_bold_suffix, confounds_metadata_suffix))

        for p in [bold_path, confounds_tsv, confounds_metadata]:
            if not p.exists():
                raise FileNotFoundError(f"Error: file path does not exist [{bold_path}]")
        
        if desc == 'tsv':
            return confounds_tsv
        elif desc == 'metadata':
            return confounds_metadata
        else:
            raise ValueError("`desc` must be set to 'tsv' or 'metadata'")