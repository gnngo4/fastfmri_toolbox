from pathlib import Path
from typing import Optional, Union, List, Tuple
import itertools

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib

class DesignMatrix():
    
    def __init__(
        self,
        time_window: Tuple[float, float], 
        search_frequencies: List[float],
        bold_path: Union[str, Path, None] = None,
        TR: Optional[float] = None,
        high_pass_threshold: float = .01,
    ):
        TR = self._get_TR(bold_path, TR)
        self.run_build_design_matrix = False
        
        self.search_frequencies = search_frequencies
        self.high_pass_threshold = high_pass_threshold
        self.time_points = self._get_time_points(TR, time_window)
        self.n_tps = len(self.time_points)
        assert np.all(self.time_points[:-1] <= self.time_points[1:]), f"`self.time_points` is not sorted in ascending order"
        
        # design matrix variables
        self.design_matrix = np.zeros((self.n_tps,0))
        self.column_names = []
        
    def build_design_matrix(self):
        
        if not self.run_build_design_matrix:
            self.run_build_design_matrix = True
            self._frequency_regressors()
            self._drift_regressors()
        
        return pd.DataFrame(
            self.design_matrix,
            index=self.time_points,
            columns=self.column_names
        )
    
    def plot_design_matrix(self):
        
        from nilearn.plotting import plot_design_matrix
        
        plot_design_matrix(self.build_design_matrix())
        plt.show()

    def get_time_indices(self, TR, time_window):

        import math 

        idx_1 = int(math.floor(time_window[0] / TR))
        idx_2 = int(math.floor(time_window[1] / TR))
        
        return (idx_1, idx_2)

    def _get_TR(
        self, 
        bold_path: Union[str, Path, None] = None,
        TR: Optional[float] = None
    ) -> float:

        if bold_path is not None:
            bold_path = Path(bold_path).resolve()
            if not bold_path.exists():
                raise FileNotFoundError(f"Error: file path does not exist [{bold_path}]")
            return nib.load(bold_path).header.get_zooms()[-1]
        elif TR is not None:
            return TR
        else:
            raise ValueError("Error: either `bold_path` or `TR` must be provided.")

    def _get_time_points(self, TR, time_window):

        import math

        tp_1 = math.floor(time_window[0] / TR) * TR
        tp_2 = math.floor(time_window[1] / TR) * TR
        time_points = np.arange(tp_1, tp_2 + (TR/10), TR)
        time_points -= time_points[0]

        return time_points

    def _frequency_regressors(self):
        
        column_names = []
        basis = ['sine','cosine']
        regressors = np.zeros((self.n_tps,len(basis)*len(self.search_frequencies)))
        for ix, (f, _basis) in enumerate(itertools.product(self.search_frequencies,basis)):
            _tps = 2 * np.pi * f * self.time_points
            if _basis == 'sine':
                regressors[:,ix] = np.sin(_tps)
            else:
                regressors[:,ix] = np.cos(_tps)
            column_names.append(f"basis-{_basis}_f-{f}")
    
        self.design_matrix = np.hstack((self.design_matrix,regressors))
        self.column_names += column_names

    def _drift_regressors(self):

        n_times = np.arange(self.n_tps)
        dt = (self.time_points[-1] - self.time_points[0]) / (self.n_tps - 1)
        if self.high_pass_threshold * dt >= .5:
            warn(f"""
            High-pass filter will span all accessible frequencies
            and saturate the design matrix. You may want to reduce
            the high_pass _threshold value. 
            The provided value is {self.high_pass_threshold} Hz.
            """)
        order = np.minimum(
            self.n_tps - 1,
            int(np.floor(2*self.n_tps*self.high_pass_threshold*dt))
        )
        cosine_drift = np.zeros((self.n_tps,order+1))
        normalizer = np.sqrt(2./self.n_tps)

        for k in range(1,order+1):
            cosine_drift[:,k-1] = normalizer * \
                np.cos((np.pi/self.n_tps)*(n_times+.5)*k)

        cosine_drift[:,-1] = 1
        
        column_names = []
        for ix in range(1,cosine_drift.shape[1]):
            column_names.append(f"drift-{str(ix).zfill(2)}")

        self.design_matrix = np.hstack((self.design_matrix,cosine_drift))
        self.column_names += column_names
        self.column_names += ['constant']
        
    def _confound_regressors(self):
        pass