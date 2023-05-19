from pathlib import Path
from typing import Optional, Union, Tuple, Dict, List, Literal
from warnings import warn

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import nibabel as nib
import itertools


class DesignMatrixRegressors:
    def __init__(self):
        self.confounds_required = False
        self.confounds_metadata_required = False

    def get(
        self, confounds: Optional[pd.DataFrame] = None
    ) -> Tuple[np.ndarray, List[str]]:
        raise NotImplementedError


class FrequencyRegressors(DesignMatrixRegressors):
    def __init__(
        self,
        search_frequencies: List[float],
        time_points: Tuple[float, float],
    ):
        super().__init__()
        self.confounds_required = False
        self.confounds_metadata_required = False
        self.search_frequencies = search_frequencies
        self.time_points = time_points
        self.n_tps = len(self.time_points)

    def get(
        self,
        confounds: Optional[pd.DataFrame] = None,
        confounds_metadata: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
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
        self.confounds_metadata_required = False
        self.time_points = time_points
        self.high_pass_threshold = high_pass_threshold
        self.add_constant = add_constant
        self.n_tps = len(self.time_points)

    def get(
        self,
        confounds: Optional[pd.DataFrame] = None,
        confounds_metadata: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        n_times = np.arange(self.n_tps)
        dt = (self.time_points[-1] - self.time_points[0]) / (self.n_tps - 1)
        if self.high_pass_threshold * dt >= 0.5:
            warn(
                f"""
            High-pass filter will span all accessible frequencies
            and saturate the design matrix. You may want to reduce
            the high_pass _threshold value. 
            The provided value is {self.high_pass_threshold} Hz.
            """
            )
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
            return (cosine_drift[:, :-1], column_names)


class MotionParameters(DesignMatrixRegressors):
    def __init__(
        self,
        time_indices: Tuple[int, int],
        mc_params: Literal[6, 24],
    ):
        super().__init__()
        self.confounds_required = True
        self.confounds_metadata_required = False
        self.time_indices = time_indices
        self.mc_params = mc_params

    def get(
        self,
        confounds: Optional[pd.DataFrame] = None,
        confounds_metadata: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        assert confounds is not None, f"`confounds` must be specified to use this class"
        col_names = self._get_column_names()
        idx_1, idx_2 = self.time_indices[0], self.time_indices[1] + 1
        regressors = confounds[col_names][idx_1:idx_2].values
        return (regressors, col_names)

    def _get_column_names(self) -> List[str]:
        MOTION_BASE: List[str] = ["trans", "rot"]
        MOTION_DIR: List[str] = ["x", "y", "z"]
        MOTION_SUFFIX: List[str] = ["derivative1", "power2", "derivative1_power2"]

        regressor_labels = [
            "_".join(list(i)) for i in itertools.product(MOTION_BASE, MOTION_DIR)
        ]
        if self.mc_params == 6:
            return regressor_labels
        elif self.mc_params == 24:
            return regressor_labels + [
                "_".join(list(i))
                for i in itertools.product(MOTION_BASE, MOTION_DIR, MOTION_SUFFIX)
            ]
        else:
            raise ValueError("`self.mc_params` must be set to 6 or 24")


class MeanSignalRegressors(DesignMatrixRegressors):
    """
    Generate regressors for mean GM, WM, CSF, Global signal
    """

    def __init__(
        self,
        time_indices: Tuple[int, int],
        regressor_type: Literal["Global", "WM", "CSF", "CSF+WM"],
        higher_order_flag: bool = False,
    ):
        super().__init__()
        self.confounds_required = True
        self.confounds_metadata_required = False
        self.time_indices = time_indices
        self.regressor_type = regressor_type
        self.higher_order_flag = higher_order_flag

    def get(
        self,
        confounds: Optional[pd.DataFrame] = None,
        confounds_metadata: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        assert confounds is not None, f"`confounds` must be specified to use this class"
        col_names = self._get_column_names()
        idx_1, idx_2 = self.time_indices[0], self.time_indices[1] + 1
        regressors = confounds[col_names][idx_1:idx_2].values
        return (regressors, col_names)

    def _get_column_names(self):
        if self.regressor_type == "Global":
            REGRESSOR_BASE = ["global_signal"]
        elif self.regressor_type == "WM":
            REGRESSOR_BASE = ["white_matter"]
        elif self.regressor_type == "CSF":
            REGRESSOR_BASE = ["csf"]
        elif self.regressor_type == "CSF+WM":
            REGRESSOR_BASE = ["csf_wm"]
        else:
            raise ValueError(
                "`self.regressor_type` must be in [Global, WM, CSF, or CSF+WM]"
            )

        if self.higher_order_flag and self.regressor_type != "CSF+WM":
            return self._add_derivatives(REGRESSOR_BASE)
        elif self.higher_order_flag and self.regressor_type == "CSF+WM":
            raise ValueError(
                "`self.regressor_type` == 'CSF+WM' does not have higher order regressors"
            )
        else:
            return REGRESSOR_BASE

    def _add_derivatives(self, REGRESSOR_BASE: List[str]):
        REGRESSOR_SUFFIX = ["derivative1", "power2", "derivative1_power2"]
        return REGRESSOR_BASE + [
            "_".join(list(i))
            for i in itertools.product(REGRESSOR_BASE, REGRESSOR_SUFFIX)
        ]


class CompCorRegressors(DesignMatrixRegressors):
    """
    Generate CompCor regressors for certain cases
    (1) X regressors that explain a n% variance
    (2) Select top n regressors
    also, consider the different type of compcor regressors
    that are readily available. i.e., wm, csf, edge compcor
    """

    def __init__(
        self,
        time_indices: Tuple[int, int],
        regressor_type: Literal["WM", "CSF", "CSF+WM", "Temporal"],
        top_n_components: Optional[int] = None,
        variance_explained: Optional[float] = None,
    ):
        super().__init__()
        self.confounds_required = True
        self.confounds_metadata_required = True
        self.time_indices = time_indices
        self.regressor_type = regressor_type
        self.top_n_components = top_n_components
        self.variance_explained = variance_explained

    def get(
        self,
        confounds: Optional[pd.DataFrame] = None,
        confounds_metadata: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        assert confounds is not None, "`confounds` must be specified to use this class"
        assert (
            confounds_metadata is not None
        ), "`confounds_metadata` must be specified to use this class"
        col_names = self._get_column_names(confounds_metadata)
        idx_1, idx_2 = self.time_indices[0], self.time_indices[1] + 1
        regressors = confounds[col_names][idx_1:idx_2].values
        return (regressors, col_names)

    def _get_column_names(self, confounds_metadata: Dict[str, Dict[str, float]]):
        if self.top_n_components is not None and self.variance_explained is None:
            return self._get_top_n_regressors(confounds_metadata)
        elif self.top_n_components is None and self.variance_explained is not None:
            return self._get_regressors_by_variance_explained(confounds_metadata)
        else:
            raise ValueError(
                "Error: Only one of `self.top_n_components` or `self.variance_explained` must be specified."
            )

    def _get_top_n_regressors(self, confounds_metadata: Dict[str, Dict[str, float]]):
        assert self.top_n_components is not None, f"`self.top_n_components` is None"
        col_names = self._filter_regressors(confounds_metadata)
        return col_names[: self.top_n_components]

    def _get_regressors_by_variance_explained(
        self, confounds_metadata: Dict[str, Dict[str, float]]
    ):
        assert self.variance_explained is not None, f"`self.variance_explained` is None"
        col_names = []
        _col_names = self._filter_regressors(confounds_metadata)
        for col in _col_names:
            cum_var = confounds_metadata[col]["CumulativeVarianceExplained"]
            col_names.append(col)
            if cum_var > self.variance_explained:
                break
        return col_names

    def _filter_regressors(self, confounds_metadata: Dict[str, Dict[str, float]]):
        regressor_type_mapping = {
            "WM": "w_comp_cor",
            "CSF": "c_comp_cor",
            "CSF+WM": "a_comp_cor",
            "Temporal": "t_comp_cor",
        }

        cols = [
            col
            for col in confounds_metadata.keys()
            if regressor_type_mapping[self.regressor_type] in col
        ]

        # Check if comp cor regressors are ordered in by CumulativeVarianceExplained
        track_cum_var = 0  # Set tracker
        for col in cols:
            cum_var = confounds_metadata[col]["CumulativeVarianceExplained"]
            assert (
                cum_var > track_cum_var
            ), "Regressor labels are not ordered in ascending values [cumulative variance explained]"
            track_cum_var = cum_var  # Update tracker

        return cols


class ScrubbingRegressors(DesignMatrixRegressors):
    """
    Generate motion scrubbing regressors based on a FD or DVARS
    """

    def __init__(
        self,
        time_indices: Tuple[int, int],
        movement_param: Literal["DVARS", "FD"],
        movement_threshold: float,
    ):
        super().__init__()
        self.confounds_required = True
        self.confounds_metadata_required = False
        self.time_indices = time_indices
        self.movement_param = movement_param
        self.movement_threshold = movement_threshold

    def get(
        self,
        confounds: Optional[pd.DataFrame] = None,
        confounds_metadata: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Tuple[np.ndarray, List[str]]:
        assert confounds is not None, f"`confounds` must be specified to use this class"

        col_names = self._get_column_names()
        idx_1, idx_2 = self.time_indices[0], self.time_indices[1] + 1
        regressors = confounds[col_names][idx_1:idx_2].values
        regressors, col_names = self._convert_to_scrubbing_regressors(regressors)
        return (regressors, col_names)

    def _get_column_names(self):
        if self.movement_param == "DVARS":
            REGRESSOR_BASE = ["dvars"]
        elif self.movement_param == "FD":
            REGRESSOR_BASE = ["framewise_displacement"]
        else:
            raise ValueError("`self.regressor_type` must be in [DVARS, FD]")

        return REGRESSOR_BASE

    def _convert_to_scrubbing_regressors(
        self, regressors: np.ndarray
    ) -> Tuple[np.ndarray, List[str]]:
        regressors = regressors >= self.movement_threshold
        n_regressors = regressors.sum()
        if n_regressors == 0:
            return (np.array([0]), [])
        else:
            # Set-up new regressors and labels
            new_regressors = np.zeros((regressors.shape[0], n_regressors))
            new_col_names = []

            coords = np.where(regressors)[0]
            for idx, coord_idx in enumerate(coords):
                new_regressors[coord_idx, idx] = 1
                new_col_names.append(f"scrubbing_{str(idx).zfill(3)}")

            assert (
                regressors.reshape(
                    regressors.size,
                )
                == new_regressors.sum(1)
            ).sum() == regressors.size

            return (new_regressors, new_col_names)


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
        if regressors.confounds_required and not regressors.confounds_metadata_required:
            confounds_df = self._get_confounds(self.bold_path, self.confounds_tsv)
            regressors_array, regressor_labels = regressors.get(confounds_df)
        elif regressors.confounds_required and regressors.confounds_metadata_required:
            assert self.bold_path is not None, "`self.bold_path` must be specified"
            confounds_df = self._get_confounds(self.bold_path, self.confounds_tsv)
            confounds_metadata = self._get_confounds_metadata(self.bold_path)
            regressors_array, regressor_labels = regressors.get(
                confounds_df, confounds_metadata
            )
        else:
            regressors_array, regressor_labels = regressors.get()

        # Update design_matrix and column_names
        if len(regressor_labels) == 0:
            return
        else:
            self.design_matrix = np.hstack((self.design_matrix, regressors_array))
            self.column_names += regressor_labels

    def build_design_matrix(self) -> pd.DataFrame:
        assert (
            self.design_matrix.shape[-1] > 0
        ), f"No regressors have been added to the design matrix yet"

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
            assert (
                bold_path is not None
            ), f"Confounds file cannot be found without a `bold_path` input"
            confounds_tsv = self._get_confounds_path(bold_path, "tsv")
            return pd.read_csv(confounds_tsv, sep="\t")

        else:
            confounds_tsv = Path(confounds_tsv)
            if not confounds_tsv.exists():
                raise FileNotFoundError(
                    f"Error: file path does not exist [{confounds_tsv}]"
                )
            return pd.read_csv(confounds_tsv, sep="\t")

    def _get_confounds_metadata(
        self, bold_path: Union[str, Path]
    ) -> Dict[str, Dict[str, float]]:
        import json

        json_path = self._get_confounds_path(bold_path, "metadata")
        with open(json_path) as json_file:
            json_data = json.load(json_file)

        return json_data

    def _get_confounds_path(
        self, bold_path: Union[str, Path], desc: Literal["tsv", "metadata"]
    ) -> Path:
        preproc_bold_suffix = "space-T1w_desc-preproc_bold.nii.gz"
        confounds_suffix = "desc-confounds_timeseries.tsv"
        confounds_metadata_suffix = "desc-confounds_timeseries.json"

        bold_path = Path(bold_path).resolve()
        confounds_tsv = Path(
            str(bold_path).replace(preproc_bold_suffix, confounds_suffix)
        )
        confounds_metadata = Path(
            str(bold_path).replace(preproc_bold_suffix, confounds_metadata_suffix)
        )

        for p in [bold_path, confounds_tsv, confounds_metadata]:
            if not p.exists():
                raise FileNotFoundError(
                    f"Error: file path does not exist [{bold_path}]"
                )

        if desc == "tsv":
            return confounds_tsv
        elif desc == "metadata":
            return confounds_metadata
        else:
            raise ValueError("`desc` must be set to 'tsv' or 'metadata'")
