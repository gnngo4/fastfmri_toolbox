from pathlib import Path
from typing import Optional, Union

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import nibabel as nib

class PlotSlices:
    def __init__(
        self,
        base_nifti_path: Union[str, Path],
        n_cols: int = 10,
    ):
        self.num_cols = n_cols
        self.base_data = self._load_nifti_path(base_nifti_path)
        self.base_z_indices = self._get_z_indices(self.base_data)

    def plot_base(self, base_vmax: Union[float, None] = None):
        
        x_max, y_max, z_max = self.base_data.shape
        self.num_slices: int = len(self.base_z_indices)
        self.num_rows: int = (self.num_slices + (self.num_cols-1))  // self.num_cols

        self.fig, self.axes = plt.subplots(
            self.num_rows, 
            self.num_cols, 
            figsize=(self.num_cols * 2.4, self.num_rows * 2.4)
        )
        
        if base_vmax is None:
            base_vmax = self.base_data.max()

        for i, z_slice in enumerate(self.base_z_indices):
            row = i // self.num_cols
            col = i % self.num_cols
            if self.num_rows == 1:
                self.axes[col].imshow(
                    self.base_data[:,:,z_slice],
                    cmap = 'gray',
                    vmax = base_vmax,
                    zorder = 1,
                )
                self.axes[col].text(
                    int(.02*x_max),
                    int(.1*y_max),
                    f"z={z_slice}",
                    c='white'
                )
            else:
                self.axes[row, col].imshow(
                    self.base_data[:,:,z_slice],
                    cmap = 'gray',
                    vmax = base_vmax,
                    zorder = 1,
                )
                self.axes[row, col].text(
                    int(.02*x_max),
                    int(.1*y_max),
                    f"z={z_slice}",
                    c='white'
                )

        # Hide empty subplots if there are fewer than `self.num_cols` images in the last row
        if self.num_slices % self.num_cols != 0:
            for j in range(self.num_slices % self.num_cols, self.num_cols):
                if self.num_rows == 1:
                    self.axes[j].axis('off')
                else:
                    self.axes[self.num_rows - 1, j].axis('off')

        plt.subplots_adjust(wspace=0.05, hspace=0.05) 
    
    def add_contours(
        self, 
        mask_nifti_path: Union[str, Path], 
        c: str = 'cyan',
        lw: float = .5,
        zorder: int = 2,
    ):

        from skimage import measure

        self._assert_base_match(mask_nifti_path)
        mask_data = nib.load(mask_nifti_path).get_fdata()
        mask_data[mask_data != 0] = 1 # Binarize all values not equal to zero
        
        for i, z_slice in enumerate(self.base_z_indices):
            row = i // self.num_cols
            col = i % self.num_cols
            contours = measure.find_contours(mask_data[:,:,z_slice], 0.5)
            if self.num_rows == 1:
                for contour in contours:
                    self.axes[col].plot(contour[:,1],contour[:,0],linewidth=lw, color=c, zorder=zorder)
            else:
                for contour in contours:
                    self.axes[row, col].plot(contour[:,1],contour[:,0],linewidth=lw, color=c, zorder=zorder)

    def _assert_base_match(self, nifti_path: Union[str, Path]):
        nifti_path = Path(nifti_path).resolve()
        assert nifti_path.name.endswith("nii.gz"), f"{nifti_path.name} must end with [nii.gz]"
        nifti_data = nib.load(nifti_path).get_fdata()
        assert nifti_data.shape == self.base_data.shape, f"{nifti_path} [{nifti_data.shape}] != [{self.base_data.shape}]"


    def _load_nifti_path(self, nifti_path: Union[str, Path]):
        
        nifti_path = Path(nifti_path).resolve()
        assert nifti_path.name.endswith("nii.gz"), f"{nifti_path.name} must end with [nii.gz]"
        if not nifti_path.exists():
            raise FileNotFoundError(
                f"Error: file path does not exist [{nifti_path}]"
            )
        
        data = nib.load(nifti_path).get_fdata()
        n_dims = len(data.shape)
        assert n_dims in [3,4], f"{n_dims} must be 3 or 4"

        return data if len(data.shape) == 3 else np.mean(data, axis=-1)
    
    def _get_z_indices(self, nifti_data: np.ndarray):

        num_slices = nifti_data.shape[-1]
        return [i for i in range(num_slices) if not np.all(np.equal(nifti_data[:,:,i],0))]