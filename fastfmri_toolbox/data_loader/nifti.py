import nibabel as nib
import numpy as np

class NiftiHandler():

    def __init__(self,nii_path,mask_path):
        self.nii_path = nii_path
        self.mask_path = mask_path
        self.nii_img = nib.load(self.nii_path)
        self.mask_img = nib.load(self.mask_path)
        self.coords = None

    def to_array(self,normalization='percent_signal',idx_start=None,idx_end=None,verbose=False):

        data = self.nii_img.get_fdata()
        assert len(data.shape) == 4, f"{self.nii_path} is not 4-dimensions."
        mask_data = self.mask_img.get_fdata()
        assert len(mask_data.shape) < 4, f"{self.mask_path} is greater than 3-dimensions."

        if idx_start is None:
            idx_start = 0

        if idx_end is None:
            idx_end = data.shape[-1]

        if verbose:        
            print(f"""Truncate: [{idx_start},{idx_end}]
Normalization: {normalization}""")
        
        self.coords = np.where((mask_data>0) & (data[:,:,:,idx_start:idx_end].mean(axis=3)>0))
        truncated_data = data[self.coords][:,idx_start:idx_end]
        if normalization == 'percent_signal':
            return self._percent_signal_change(truncated_data)
        else:
            return truncated_data

    def get_array_idx(self,x,y,z):
        
        if self.coords is None:
            assert False, f"Class attribute [coords] is not initialized.\nMethod [to_array()] must be ran."
        
        (x_coords,y_coords,z_coords) = self.coords
        for ix, (_x,_y,_z) in enumerate(zip(x_coords,y_coords,z_coords)):
            if x==_x and y==_y and z==_z:
                return ix

        print(f"Warning: Coordinates [{x},{y},{z}] not found.\nReturning -1")
        return -1

    def to_nifti(self,data,outfile):
        
        assert (self.coords[0].shape[0] == data.shape[0]) \
            and (len(data.shape) in [1,2]), \
            f"Dimensions of `data` must be 1- or 2-D."

        if self.coords is None:
            assert False, f"Class attribute [coords] is not initialized.\nMethod [to_array()] must be ran."

        if len(data.shape) == 1:

            reshaped_data = self._to_nifti(data)

        reshaped_list = []
        if len(data.shape) == 2:
            for ix in range(data.shape[-1]):
                reshaped_list.append(self._to_nifti(data[:,ix]))

            reshaped_data = np.stack(reshaped_list,axis=3)

        self._save_nifti(reshaped_data,outfile)

    def _to_nifti(self,data):

        reshaped_data = np.zeros(self.mask_img.get_fdata().shape)

        (x_coords,y_coords,z_coords) = self.coords
        for ix, (x,y,z) in enumerate(zip(x_coords,y_coords,z_coords)):
            reshaped_data[x,y,z] = data[ix]

        return reshaped_data

    def _save_nifti(self,data,outfile):

        img = nib.Nifti1Image(
            data,
            self.nii_img.affine,
            self.nii_img.header
        )
        nib.save(img,outfile)

    def _percent_signal_change(self,X):

        assert len(X.shape) == 2, f"Input array must be 2-dimensions."

        return (X / X.mean(1,keepdims=True)) - 1
