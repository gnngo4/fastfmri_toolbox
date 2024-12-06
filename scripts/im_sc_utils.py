import numpy as np
import nibabel as nib
import pickle
import subprocess
from pathlib import Path
from collections import defaultdict

from brainsmash.mapgen.sampled import Sampled
from brainsmash.mapgen.memmap import txt2memmap

N_LH_VERTICES = 29696
N_RH_VERTICES = 29716
N_VERTICES = N_LH_VERTICES + N_RH_VERTICES
N_VERTICES_AFTER_RESAMPLING = 32492
GEODESIC_LH = "/opt/app/notebooks/data/surfaces/S1200.L.midthickness_MSMAll.32k_fs_LR_geodesic.txt"
GEODESIC_RH = "/opt/app/notebooks/data/surfaces/S1200.R.midthickness_MSMAll.32k_fs_LR_geodesic.txt"
DSCALAR_TEMPLATE = "/opt/app/notebooks/data/dscalars/S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii"

def contains_all_strings(input_str, string_list):
    for string in string_list:
        if string not in input_str:
            return False
    return True

def find_activations(experiment_id, mri_id, roi_task_id, roi_f_1, fo, sub_id, data_split_id="train", match_str="activations.dtseries.nii", additional_match_strs=None, additional_match_str=None, corr_type="uncp"):
    import os
    directory = f"/scratch/fastfmri/experiment-{experiment_id}_mri-{mri_id}_smooth-0_truncate-39-219_n-400_batch-merged_desc-IMall_roi-{roi_task_id}-{roi_f_1}_pval-{corr_type}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap/"
    if additional_match_strs is not None:
        match_str = additional_match_strs + [match_str, f"data-{data_split_id}"]
        activations_files = []
        for file in os.listdir(directory):
            if contains_all_strings(file, match_str):
                activations_files.append(file)
    else:
        activations_files = [file for file in os.listdir(directory) if f'data-{data_split_id}' in file and match_str in file]

    return [f"{directory}{i}" for i in activations_files]

def get_binary_paths(experiment_id, mri_id, sub_id, roi_task_id, f1, f2, fim, corr_type):
    f_1 = find_activations(experiment_id, mri_id, roi_task_id, f1, .8, sub_id, match_str="activations.dtseries.nii", corr_type=corr_type)
    f_2 = find_activations(experiment_id, mri_id, roi_task_id, f2, .8, sub_id, match_str="activations.dtseries.nii", corr_type=corr_type)
    mask = find_activations(experiment_id, mri_id, roi_task_id, f1, .8, sub_id, match_str="mask.dtseries.nii", corr_type=corr_type)
    f_im = find_activations(experiment_id, mri_id, roi_task_id, fim, .8, sub_id, match_str="activations.dtseries.nii", corr_type=corr_type)
    for i in [f_1, f_2, mask, f_im]:
        assert len(i) == 1

    f_1 = f_1[0]
    f_2 = f_2[0]
    f_im = f_im[0]
    mask = mask[0]

    return {
        "f_1": f_1,
        "f_2": f_2,
        "f_im": f_im,
        "mask": mask,
    }

def read_map(dscalar, dscalar_template=None, dscalar_placeholder="/tmp/read_map_tmp.dscalar.nii"):
    dscalar = Path(dscalar)
    assert dscalar.exists(), f"{dscalar} does not exist."
    if dscalar_template is not None:
        dscalar_template = Path(dscalar_template)
        assert dscalar_template.exists(), f"{dscalar_template} does not exist."
        tmp_dscalar = Path(dscalar_placeholder)
        command = [
            "wb_command",
            "-cifti-create-dense-from-template",
            str(dscalar_template),
            str(tmp_dscalar),
            "-cifti", 
            str(dscalar),
        ]
        result = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        assert tmp_dscalar.exists(), f"{tmp_dscalar} was not generated."

        cortex_data = np.array(nib.load(tmp_dscalar).get_fdata())

        return cortex_data

def convert_multimap_to_fo_maps(multimap_dict,dscalar_template=None):
    binary_map_dict = {}
    for k, v in multimap_dict.items():
        fo_map = read_map(v, dscalar_template=dscalar_template)
        fo_map = fo_map.mean(axis=0)
        fo_map = np.nan_to_num(fo_map, nan=0)
        binary_map_dict[k] = fo_map

    return binary_map_dict

def check_txt2memmap_outputs(surrogatedir):
    exist_flag = True
    for i in ["distmat.npy", "index.npy"]:
        f = surrogatedir / i
        if not f.exists():
            exist_flag = False
            
    return exist_flag

def generate_surrogates(brainmap, wb_coverage_mask, hemi, parentdir, basedir, n_surrogates):
    """
    brainmap (np.ndarray): 1X(32492*2) array, metric of interest
    wb_coverage_mask (or ROI) (np.ndarray): 1X(32492*2), [0,1] where 1==slab, 0==no_slab
    hemi (str): 'lh' or 'rh'
    .../{parentdir}/{basedir}_{hemi}
    n_surrogates (int): number of surrogates
    GEODESIC_LH = "/opt/app/notebooks/data/surfaces/S1200.L.midthickness_MSMAll.32k_fs_LR_geodesic.txt"
    GEODESIC_RH = "/opt/app/notebooks/data/surfaces/S1200.R.midthickness_MSMAll.32k_fs_LR_geodesic.txt"
    """
    assert hemi in ["lh", "rh"], f"hemi must be `lh` or `rh`."
    if hemi == 'lh':
        hemi_coverage_mask = wb_coverage_mask[:32492] # 1x32492
        hemi_brainmap = brainmap[:32492][hemi_coverage_mask==1] # 1x32492
        geodesic = GEODESIC_LH # textfile
    if hemi == 'rh':
        hemi_coverage_mask = wb_coverage_mask[32492:]
        hemi_brainmap = brainmap[32492:][hemi_coverage_mask==1]
        geodesic = GEODESIC_RH
        
    surrogatedir = parentdir / f"{basedir}_hemi-{hemi}"
    if not surrogatedir.exists():
        surrogatedir.mkdir(parents=True, exist_ok=True)

    txt2memmap_flag = check_txt2memmap_outputs(surrogatedir)

    if txt2memmap_flag:
        output_files = {
            'distmat': str(surrogatedir / "distmat.npy"),
            'index': str(surrogatedir / "index.npy"),
        }
    else:
        cortex_mask = np.ones((N_VERTICES_AFTER_RESAMPLING))
        cortex_mask[hemi_coverage_mask==1] = 0
        output_files = txt2memmap(geodesic, surrogatedir, maskfile=cortex_mask)

    surrogate_map_pkl = surrogatedir / "surrogate_maps.pkl"
    if surrogate_map_pkl.exists():
        with open(surrogate_map_pkl, 'rb') as f:
            surrogate_map_array = pickle.load(f)
    else:
        sampled = Sampled(
            hemi_brainmap,
            np.load(output_files['distmat']),
            np.load(output_files['index']),
        )

        _surrogate_map_array = sampled(n=n_surrogates)
        surrogate_map_array = np.zeros((n_surrogates, N_VERTICES_AFTER_RESAMPLING))
        for i in range(n_surrogates):
            surrogate_map_array[i, hemi_coverage_mask==1] = _surrogate_map_array[i,:]
        with open(surrogate_map_pkl, 'wb') as f:
            pickle.dump(surrogate_map_array, f)

    return surrogate_map_array

def convert_array_to_top_n_values(X, n):
    top_indices = np.argsort(X)[-n:]
    boolean_array = np.full_like(X, False, dtype=bool)
    boolean_array[top_indices] = True

    return boolean_array

def dilate_array_by_mm(
    mask_array, 
    dilation, 
    template_cifti="/opt/app/notebooks/data/surfaces/empty_template.dscalar.nii",
    left_surface="/opt/app/notebooks/data/surfaces/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii",
    right_surface="/opt/app/notebooks/data/surfaces/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii",
    CORTEX_ONLY=False,
):
    for p in [template_cifti, left_surface, right_surface]:
        assert Path(p).exists(), f"{p} not found."
        
    img = nib.load(template_cifti)
    new_data = np.zeros_like(img.get_fdata())
    if CORTEX_ONLY:
        padding_size = new_data.shape[-1] - mask_array.shape[0]
        mask_array = np.pad(mask_array, (0, padding_size), constant_values=False)
    new_data[:,mask_array] = 1
    new_img = nib.Cifti2Image(new_data, header=img.header)
    mask_dscalar = f"/tmp/mask.dil-0.dscalar.nii"
    nib.save(new_img, mask_dscalar)
    dilated_mask_dscalar = Path(f"/tmp/mask.dil-{dilation}.dscalar.nii")
    command = [
        "wb_command",
        "-cifti-dilate",
        mask_dscalar,
        "COLUMN",
        str(dilation),
        str(0),
        str(dilated_mask_dscalar),
        "-left-surface", left_surface, 
        "-right-surface", right_surface, 
    ]
    result = subprocess.run(
        command,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    assert dilated_mask_dscalar.exists(), f"{dilated_mask_dscalar} was not generated."
    if CORTEX_ONLY:
        dilated_mask_array = nib.load(dilated_mask_dscalar).get_fdata()[0,:N_VERTICES_AFTER_RESAMPLING*2]
    else:
        dilated_mask_array = nib.load(dilated_mask_dscalar).get_fdata()[0,:]

    return dilated_mask_array

def contains_only_0_1(arr):
    # Check if any element in the array is not 0 or 1
    return not np.any((arr != 0) & (arr != 1))
    
def process_maps(fo_maps, fo):
    predicted_im_roi = (fo_maps["f_im"] >= fo).astype(int) # IM (predicted)
    f1_roi = (fo_maps["f_1"] >= fo).astype(int) # all f1
    f2_roi = (fo_maps["f_2"] >= fo).astype(int) # all f2
    f1_f2_roi = ((f1_roi + f2_roi) == 2) # f1 & f2 intersection
    f1_only_roi = f1_roi - f1_f2_roi # only f1
    f2_only_roi = f2_roi - f1_f2_roi # only f2
    f1_f2_dil_2mm_roi = (dilate_array_by_mm(f1_f2_roi, 2, CORTEX_ONLY=True) == 1).astype(int)
    f1_f2_roi = f1_f2_roi.astype(int)

    all_rois = {
        "IM": predicted_im_roi,
        "f1_only": f1_only_roi,
        "f2_only": f2_only_roi,
        "f1": f1_roi,
        "f2": f2_roi,
        "f1&f2": f1_f2_roi,
        "f1&f2_dil2mm": f1_f2_dil_2mm_roi,
    }

    return all_rois

def compute_metrics(X,Y):
    X = X.flatten()
    Y = Y.flatten()

    intersection = np.logical_and(X,Y)
    union = np.logical_or(X,Y)
    iou = np.sum(intersection) / np.sum(union) # iou
    dice = 2*np.sum(intersection) / (np.sum(X) + np.sum(Y)) # dice
    jaccard = np.sum(intersection) / np.sum(np.logical_or(X,Y)) # jaccard

    metrics = {
        "dice": dice,
        "jaccard": jaccard,
    }

    return metrics

def area_under_curve(X,Y):
    if len(X) != len(Y):
        raise ValueError("Lengths of X, Y must be equal")
    auc = 0.0
    for i in range(1, len(X)):
        width = X[i] - X[i - 1]
        average_height = (Y[i] + Y[i - 1]) / 2.0
        auc += width * average_height

    return -1*auc

def compute_metrics_over_surrogates(surrogate_maps, im_roi, mask_coords):
    metrics = defaultdict(list)
    n_surr = surrogate_maps.shape[0]
    for i in range(n_surr):
        _metrics  = compute_metrics(im_roi[mask_coords], surrogate_maps[i,mask_coords])
        metrics["dice"].append(_metrics["dice"])
        metrics["jaccard"].append(_metrics["jaccard"])

    return metrics

def add_data_to_dict(experiment_id, sub_id, roi_task_id, im_str, metric_dict, surrogate_p_value_dict, data_dict=None):
    if data_dict is None:
        data_dict = defaultdict(list)
    assert isinstance(data_dict, dict), f"{type(data_dict)} must be dictionary."
    for metric_type in metric_dict.keys():
        for roi_str in metric_dict[metric_type].keys():
            data_dict["experiment_id"].append(experiment_id)
            data_dict["sub_id"].append(sub_id)
            data_dict["roi_task_id"].append(roi_task_id)
            data_dict["metric_type"].append(metric_type)
            data_dict["im_map"].append(im_str)
            data_dict["roi_map"].append(roi_str)
            data_dict["metric"].append(metric_dict[metric_type][roi_str])
            data_dict["p_value"].append(surrogate_p_value_dict[metric_type][roi_str])

    return data_dict