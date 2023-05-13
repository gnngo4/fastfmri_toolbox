from pathlib import Path
from typing import Optional, Union, Tuple, List

import matplotlib.pyplot as plt
import matplotlib.cm as cm

import numpy as np
import nibabel as nib

class PlotSlices:
    def __init__(
        self,
        base_nifti_path: Union[str, Path, nib.Nifti1Image],
        n_cols: int = 10,
        z_slice_range: Union[None, Tuple[int, int]] = None,
    ):
        self.num_cols = n_cols
        self.base_data = self._load_base_nifti(base_nifti_path)
        self.base_z_indices = self._get_z_indices(self.base_data, z_slice_range)

    def plot_base(self, cmap: str = 'gray', base_vmax: Union[float, None] = None):
        
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
                    cmap = cmap,
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
                    int(.15*y_max),
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
        self._turn_off_axis()
    
    def add_contours(
        self, 
        mask_nifti_path: Union[str, Path], 
        c: str = 'cyan',
        lw: float = .5,
        zorder: int = 2,
    ):

        from skimage import measure

        mask_data = self._validate_nifti(mask_nifti_path)
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

    def add_overlay(
        self,
        overlay_nifti: Union[str, Path, nib.Nifti1Image],
        threshold: float,
        cmap: str = 'magma',
        alpha: float = 1.0,
        vmax: Union[float, None] = None,
        zorder: int = 3,
    ):
        
        data = self._validate_nifti(overlay_nifti)
        if vmax is None:
            vmax = data.max()

        for i, z_slice in enumerate(self.base_z_indices):
            row = i // self.num_cols
            col = i % self.num_cols
            slice_data = data[:,:,z_slice]
            masked_slice_data = np.ma.masked_where(slice_data <= threshold, slice_data)
            if self.num_rows == 1:
                self.axes[col].imshow(masked_slice_data, cmap=cmap, alpha=alpha, interpolation='none', zorder=zorder, vmax=vmax)
            else:
                self.axes[row, col].imshow(masked_slice_data, cmap=cmap, alpha=alpha, interpolation='none', zorder=zorder, vmax=vmax)

    def _turn_off_axis(self):

        for ax in self.axes.flatten():
            ax.set_xticks([])
            ax.set_xticklabels([])
            ax.set_yticks([])
            ax.set_yticklabels([])
    
    def _load_base_nifti(self, nifti: Union[str, Path, nib.Nifti1Image]) -> np.ndarray:
        """
        Convert str, Path, or nib.Nifti1Image into numpy array.
        Numpy array is generated for the base image.
        """
        
        if isinstance(nifti, nib.Nifti1Image):
            data = nifti.get_fdata()
        else:
            nifti = Path(nifti).resolve()
            assert nifti.name.endswith("nii.gz"), f"{nifti.name} must end with [nii.gz]"
            if not nifti.exists():
                raise FileNotFoundError(
                    f"Error: file path does not exist [{nifti}]"
                )
            data = nib.load(nifti).get_fdata()
        
        # Image must be 3D or 4D
        n_dims = len(data.shape)
        assert n_dims in [3,4], f"{n_dims} must be 3 or 4"

        # If 4D, the base image is the temporal mean image
        return data if len(data.shape) == 3 else np.mean(data, axis=-1)
    
    def _validate_nifti(self, nifti: Union[str, Path, nib.Nifti1Image]) -> np.ndarray:
        """
        Convert str, Path, or nib.Nifti1Image into numpy array.
        Ensure dimensions matches the base image.
        """
        if isinstance(nifti, nib.Nifti1Image):
            data = nifti.get_fdata()
        else:
            data = nib.load(nifti).get_fdata()

        assert data.shape == self.base_data.shape, f"Dimensions does not match with base image [{self.base_data.shape}]"

        return data
    
    def _get_z_indices(self, nifti_data: np.ndarray, z_slice_range: Union[None, Tuple[int, int]]) -> List[int]:
        """
        Return list of indices corresponding to z-slices that are not empty
        """
        num_slices = nifti_data.shape[-1]
        z_slices = [i for i in range(num_slices) if not np.all(np.equal(nifti_data[:,:,i],0))]
        if z_slice_range is None:
            return z_slices
        else:
            assert z_slice_range[1] > z_slice_range[0]
            return z_slices[z_slice_range[0]:z_slice_range[1]]