from pathlib import Path
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.signal import welch
from collections import defaultdict
from functools import lru_cache

import sys
sys.path.append("/opt/wbplot")
from wbplot import dscalar

"""Global variables
"""
# Working directory
WORKING_DIR = Path("/opt/app/notebooks/ComputeCanada/frequency_tagging")
PICKLE_DIR = WORKING_DIR / "pkl_data"
SCRATCH_DIR = Path("/scratch/fastfmri")
# Parent directories
DSCALAR_DIR = Path("/opt/app/notebooks/data/dscalars")
DLABEL_DIR = Path("/opt/app/notebooks/data/dlabels")
# Figure directories
MAIN = WORKING_DIR / "figures" / "main"
DFM_MAPS = WORKING_DIR / "figures" / "dual_frequency_mapping"
CIFTI_DFM_MAPS = WORKING_DIR / "figures" / "dual_frequency_mapping_cifti"
IM_MAPS = WORKING_DIR / "figures" / "intermodulation_mapping"
CIFTI_IM_MAPS = WORKING_DIR / "figures" / "intermodulation_cifti"
CARPET_PLOTS = WORKING_DIR / "figures" / "dual_frequency_carpet_plots"
DFM_ROI_PSDS = WORKING_DIR / "figures" / "dual_frequency_roi_psds"
# Paths
S1200_MYELINMAP = DSCALAR_DIR / "S1200.MyelinMap_BC_MSMAll.32k_fs_LR.dscalar.nii"
HCP_ATLAS = DLABEL_DIR / "Q1-Q6_RelatedValidation210.CorticalAreas_dil_Final_Final_Areas_Group_Colors.32k_fs_LR.dlabel.nii"
# Cohort info
NORMAL_3T_SUB_IDS = ["000","002","003","004","006","005","007","008","009"] # First 5 and last 4 are stimulated right visual field and left visual field, respectively
NORMAL_3T_CONDITIONS = [("control",.125,.2),("entrain",.125,.2)]
NORMAL_7T_SUB_IDS = ["Pilot001","Pilot009","Pilot010","Pilot011"]
NORMAL_7T_ENTRAIN_CONDITIONS = [("AttendAway",.125,.2)]
VARY_SUB_IDS = ["020","021"]
VARY_020_ENTRAIN_CONDITIONS = [
    ("entrainA",.125,.2),
    ("entrainB",.125,.175),
    ("entrainC",.125,.15),
]
VARY_021_ENTRAIN_CONDITIONS = [
    ("entrainD",.125,.2),
    ("entrainE",.15,.2),
    ("entrainF",.175,.2),
]

# Preprocess info
SMOOTHNESS = str(0)
TRUNCATION_WINDOW = "39-219"
N_BOOTSTRAP_BATCHES = 16
N_BOOTSTRAPS = str(400)


"""General
"""
def set_base_dir(basedir):
    basedir = Path(basedir)
    if not basedir.exists():
        basedir.mkdir(exist_ok=True, parents=True)

    return basedir

def _check_difference(arr, diff=.3):
    differences = np.diff(arr)

    return np.all(np.isclose(differences,diff))

# Stylize
def change_font():
    # Change fontsize
    from matplotlib import font_manager

    font_dirs = ['/opt/app/notebooks/font_library']
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)
    # set font
    plt.rcParams['font.family'] = 'aptos'

def get_experiment_colour_codes():
    """deprecated?
    """
    c_dict = {
        "NORMAL_3T_CONTROL": np.array([i/256 for i in (58,148,42,256)])[np.newaxis,:],
        "NORMAL_3T": np.array([i/256 for i in (191,236,212,256)])[np.newaxis,:],
        "NORMAL_7T": np.array([i/256 for i in (160,210,3,256)])[np.newaxis,:],
        "VARY_3T": np.array([i/256 for i in (242,118,138,256)])[np.newaxis,:],
        "VARY_7T": np.array([i/256 for i in (215,72,48,256)])[np.newaxis,:],
    }

    return c_dict

def get_roi_colour_codes():
    c_dict = {
        "f1": np.array([i/256 for i in (242,51,42,256)])[np.newaxis,:],
        "f2": np.array([i/256 for i in (64,55,255,256)])[np.newaxis,:],
        "f1f2": np.array([i/256 for i in (251,243,64,256)])[np.newaxis,:],
        "f2-f1": np.array([i/256 for i in (53,152,98,256)])[np.newaxis,:],
        "f1+f2": np.array([i/256 for i in (124,19,88,256)])[np.newaxis,:],
        "2f1": np.array([i/256 for i in (132,230,129,256)])[np.newaxis,:],
        "2f2": np.array([i/256 for i in (83,54,146,256)])[np.newaxis,:],
        "2f1-f2": np.array([i/256 for i in (90,236,255,256)])[np.newaxis,:],
        "2f2-f1": np.array([i/256 for i in (244,121,29,256)])[np.newaxis,:],
    }

    return c_dict

def get_frequency_text_codes():
    text_dict = {
        "f1": f"$f_1$",
        "f2": f"$f_2$",
        "f1f2": f"$f_1\cap$$f_2$",
        "f2-f1": f"$f_2-f_1$",
        "f1+f2": f"$f_1+f_2$",
        "2f1": f"2$f_1$",
        "2f2": f"2$f_2$",
        "2f1-f2": f"2$f_1-f_2$",
        "2f2-f1": f"2$f_2-f_1$",
    }

    return text_dict

def convert_to_fractional_overlap(data):

    return data.sum(0) / data.shape[0]

def map_data_to_value(data_list):
    for ix, (k,v) in enumerate(data_list):
        if ix == 0:
            new_data = k.copy() * v
        else:
            new_data += k * v

    return new_data

def get_quadrant_id(mask_path):
    rel_mask_path = mask_path.split("/")[-1]
    idx_start = rel_mask_path.find("Q")
    quadrant_id = rel_mask_path[idx_start:idx_start+2]
    assert quadrant_id in ['Q1', 'Q2'], f"{quadrant_id} not Q1 or Q2"

    return quadrant_id

@lru_cache(maxsize=360)
def read_roi_path(roi_path):
    return nib.load(roi_path).get_fdata()[0,:]

def contains_all_strings(input_str, string_list):
    for string in string_list:
        if string not in input_str:
            return False
    return True

def get_wbplot_colour_palette_details(data_type):
    """Colour settings when plotting with `wbplot`
    """
    if data_type == "dfm":
        # (Palette, red, blue, yellow, black)
        return ("power_surf", -.1, .82, .14, .41)
    elif data_type == "im":
        # (Palette, red, blue, yellow, cyan, black)
        return ("power_surf", -.1, .82, .14, .52, .41)
    else:
        raise ValueError(f"{data_type} not supported.")

def crop_and_save(input_file, output_file, left, top, right, bottom):
    from PIL import Image
    try:
        # Open the input image
        with Image.open(input_file) as img:
            # Crop the image
            cropped_img = img.crop((left, top, right, bottom))
            # Save the cropped image
            cropped_img.save(output_file)
            print("Cropped image saved successfully as", output_file)
    except Exception as e:
        print("An error occurred:", e)

def find_activations(experiment_id, mri_id, roi_task_id, roi_f_1, fo, sub_id, data_split_id="train", match_str="activations.dtseries.nii", additional_match_strs=None, additional_match_str=None, corr_type="uncp", desc_id="basic"):
    # `directory` is generated from merging bootstraps from notebook:step:3
    directory = SCRATCH_DIR / f"experiment-{experiment_id}_mri-{mri_id}_smooth-{SMOOTHNESS}_truncate-{TRUNCATION_WINDOW}_n-{N_BOOTSTRAPS}_batch-merged_desc-{desc_id}_roi-{roi_task_id}-{roi_f_1}_pval-{corr_type}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap"
    if additional_match_strs is not None:
        match_str = additional_match_strs + [match_str, f"data-{data_split_id}"]
        activations_files = []
        for file in os.listdir(directory):
            if contains_all_strings(file, match_str):
                activations_files.append(file)
    else:
        activations_files = [file for file in os.listdir(directory) if f'data-{data_split_id}' in file and match_str in file]

    return [f"{directory}/{i}" for i in activations_files]

def find_task_lock(experiment_id, mri_id, roi_task_id, roi_f_1, fo, sub_id, match_str="tasklock.dtseries.nii",corr_type="uncp", desc_id="basic"):
    # `directory` is generated from merging bootstraps from notebook:step:3
    directory = SCRATCH_DIR / f"experiment-{experiment_id}_mri-{mri_id}_smooth-{SMOOTHNESS}_truncate-{TRUNCATION_WINDOW}_n-{N_BOOTSTRAPS}_batch-merged_desc-{desc_id}_roi-{roi_task_id}-{roi_f_1}_pval-{corr_type}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap/"
    task_lock_file = [file for file in os.listdir(directory) if match_str in file and roi_task_id in file]

    return [f"{directory}/{i}" for i in task_lock_file]

def load_mean_dtseries(dtseries):
    mean_power = nib.load(dtseries).get_fdata().mean(0)

    return mean_power

def extract_im_products(f1,f2):

    assert f2 > f1, f"{f2} <= {f1}"
    
    im_frequencies = {
        "f1": f1,
        "f2": f2,
        "f2-f1": round(f2-f1,10),
        "f1+f2": round(f1+f2, 10),
        "2f1": round(f1*2, 10),
        "2f2": round(f2*2, 10),
        "2f1-f2": round(2*f1-f2, 10),
        "2f2-f1": round(2*f2-f1, 10),
    }

    return im_frequencies

"""
generate dual frequency maps
:ref: notebooks/ComputeCanada/frequency_tagging/4_0_dfm_generate_maps.ipynb
"""
def dfm_combine_f1_f2(f1,f2,fo=1.,mask=None, f1_c=.01, f2_c=.82, f1f2_c=.28, mask_c=.01):
    f1_data = convert_to_fractional_overlap(nib.load(f1).get_fdata())
    f2_data = convert_to_fractional_overlap(nib.load(f2).get_fdata())
    f1_data = (f1_data >= fo).astype(int)
    f2_data = (f2_data >= fo).astype(int)
    f1f2_data = ((f1_data + f2_data) == 2).astype(int)
    f1_data -= f1f2_data
    f2_data -= f1f2_data
    if mask:
        mask_data = convert_to_fractional_overlap(nib.load(mask).get_fdata())
        mask_data = (mask_data >= 1.).astype(int)
        mask_data -= (f1f2_data*mask_data)
        mask_data -= (f1_data*mask_data)
        mask_data -= (f2_data*mask_data)
    data_dict = [(f1_data, f1_c), (f2_data, f2_c), (f1f2_data, f1f2_c)]
    if mask:
        data_dict.append((mask_data,mask_c))

    return map_data_to_value(data_dict)

def dfm_merge_and_binarize_mask(data, f1_c, f2_c, f1f2_c, mask_c):
    data_dict = {
        "f1": data.copy(),
        "f2": data.copy(),
        "f1Uf2": data.copy(),
    }
    data_dict["f1"][(data_dict["f1"]==f1_c) | (data_dict["f1"]==f1f2_c)] = 1
    data_dict["f1"][(data_dict["f1"]==f2_c)] = 0
    data_dict["f2"][(data_dict["f2"]==f2_c) | (data_dict["f2"]==f1f2_c)] = 1
    data_dict["f2"][(data_dict["f2"]==f1_c)] = 0
    data_dict["f1Uf2"][(data_dict["f1Uf2"]==f1f2_c)] = 1
    data_dict["f1Uf2"][(data_dict["f1Uf2"]==f1_c) | (data_dict["f1Uf2"]==f2_c)] = 0
    for v in data_dict.values():
        v[v==mask_c] = 0
    # Create mask
    data_dict["mask"] = data.copy()
    data_dict["mask"][(data_dict["mask"]==mask_c) | (data_dict["mask"]==f1_c) | (data_dict["mask"]==f2_c) | (data_dict["mask"]==f1f2_c)] = 1

    for k,v in data_dict.items():
        assert np.all(np.isin(v, [0,1])), f"{k}, {np.unique(v)} {f1_c} {f2_c} {f1f2_c} {mask_c}"

    return data_dict

def dfm_append_data(
    df_data,
    hcp_mapping,
    map_data,
    activation_f1_data,
    activation_f2_data,
    power_f1_data, 
    power_f2_data, 
    psnr_f1_data,
    psnr_f2_data,
    tasklock_data,
    test_pd_f1_data,
    test_pd_f2_data,
    pd_f1_data, 
    pd_f2_data, 
    q_id,
    experiment_label, 
    sub_id, 
    roi_fo,
    roi_task_id,
):
    """Stores vertex level data for each HCP ROI:
    """
    map_data_keys = ["f1","f2","f1Uf2"]
    roi_slab = map_data["mask"]
    for frequency_of_roi in map_data_keys:
        f_data = map_data[frequency_of_roi]
        for roi_label, roi_path in hcp_mapping.items():
            if q_id == "Q1":
                contra = "L_"
            elif q_id == "Q2":
                contra = "R_"
            else:
                raise ValueError(f"{q_id} not supported.")

            if roi_label.startswith(contra):
                roi_label = f"CONTRA_{roi_label[2:-4]}"
            else:
                roi_label = f"IPSI_{roi_label[2:-4]}"

            roi_mask = read_roi_path(roi_path)
            assert roi_mask.shape == f_data.shape

            hcp_and_slab_roi = roi_mask * roi_slab
            if hcp_and_slab_roi.sum() == 0:
                # Skip if HCP ROI not found in the slab
                continue
            hcp_and_f_roi = roi_mask * f_data
            frequency_coordinates = hcp_and_f_roi==1
            slab_coordinates = hcp_and_slab_roi==1
            # Train
            f1_activation = activation_f1_data[slab_coordinates]
            f2_activation = activation_f2_data[slab_coordinates]
            f1_phase_delay = pd_f1_data[slab_coordinates]
            f2_phase_delay = pd_f2_data[slab_coordinates]
            # Test
            f1_BOLD_power = power_f1_data[slab_coordinates]
            f2_BOLD_power = power_f2_data[slab_coordinates]
            f1_BOLD_psnr = psnr_f1_data[slab_coordinates]
            f2_BOLD_psnr = psnr_f2_data[slab_coordinates]
            test_f1_phase_delay = test_pd_f1_data[slab_coordinates]
            test_f2_phase_delay = test_pd_f2_data[slab_coordinates]
            tasklock = tasklock_data[slab_coordinates]

            # Save to dataframe
            df_data["roi_task_id"].append(roi_task_id)
            df_data["roi_fo"].append(roi_fo)
            df_data["experiment_id"].append(experiment_label)
            df_data["sub_id"].append(sub_id)
            df_data["quadrant_id"].append(q_id)
            df_data["hcp_roi"].append(roi_label)
            df_data["frequency_of_roi"].append(frequency_of_roi)
            df_data["frequency_coordinates"].append(frequency_coordinates)
            df_data["slab_coordinates"].append(slab_coordinates)
            # Train
            df_data["train_f1_activation"].append(f1_activation)
            df_data["train_f2_activation"].append(f2_activation)
            df_data["train_f1_phase_delay"].append(f1_phase_delay)
            df_data["train_f2_phase_delay"].append(f2_phase_delay)
            # Test
            df_data["test_f1_BOLD_power"].append(f1_BOLD_power)
            df_data["test_f2_BOLD_power"].append(f2_BOLD_power)
            df_data["test_f1_BOLD_psnr"].append(f1_BOLD_psnr)
            df_data["test_f2_BOLD_psnr"].append(f2_BOLD_psnr)
            df_data["test_f1_phase_delay"].append(test_f1_phase_delay)
            df_data["test_f2_phase_delay"].append(test_f2_phase_delay)
            df_data["tasklock"].append(tasklock)

    return df_data

def dfm_generate_single_subject_maps(
    hcp_mapping, label, experiment_id, mri_id, sub_ids, 
    roi_task_ids, roi_f_1s, roi_f_2s, roi_fo,
    df_data=None,
    corr_type="uncp",
    ROI_FO=.8,
    LEFT=590, TOP=80, RIGHT=1140, BOTTOM=460, VERTEX_TO = 59412,
):
    
    PALETTE, f1_c, f2_c, f1f2_c, mask_c = get_wbplot_colour_palette_details("dfm")

    if df_data is None:
        df_data = defaultdict(list)
    
    # Loop over cohort
    for sub_id, roi_task_id, roi_f_1, roi_f_2 in zip(
        sub_ids,
        roi_task_ids, 
        roi_f_1s,
        roi_f_2s,
    ):
        # Set up output directories
        _base = f"experiment-{experiment_id}_mri-{mri_id}_sub-{sub_id}"
        _ = set_base_dir(f"{str(DFM_MAPS)}/{_base}")
        _ = set_base_dir(f"{str(CIFTI_DFM_MAPS)}/{_base}")
        # Output paths
        png_out = DFM_MAPS / _base / f"label-{label}_mri-{mri_id}_sub-{sub_id}_task-{roi_task_id}_f-{roi_f_1}-{roi_f_2}_corr-{corr_type}_fo-{ROI_FO}.png"
        dscalar_out = CIFTI_DFM_MAPS / _base / f"label-{label}_mri-{mri_id}_sub-{sub_id}_task-{roi_task_id}_f-{roi_f_1}-{roi_f_2}_corr-{corr_type}_fo-{ROI_FO}.dtseries.nii"
        # Load data
        f1 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_1, .8, sub_id, match_str="activations.dtseries.nii", corr_type=corr_type)
        f2 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_2, .8, sub_id, match_str="activations.dtseries.nii", corr_type=corr_type)
        mask = find_activations(experiment_id, mri_id, roi_task_id, roi_f_1, .8, sub_id, match_str="mask.dtseries.nii", corr_type=corr_type)
        pd_f1 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_1, .8, sub_id, data_split_id = "train", match_str="phasedelay.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{roi_f_1}"], corr_type=corr_type)
        pd_f2 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_2, .8, sub_id, data_split_id = "train", match_str="phasedelay.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{roi_f_2}"], corr_type=corr_type)
        test_pd_f1 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_1, .8, sub_id, data_split_id = "test", match_str="phasedelay.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{roi_f_1}"], corr_type=corr_type)
        test_pd_f2 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_2, .8, sub_id, data_split_id = "test", match_str="phasedelay.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{roi_f_2}"], corr_type=corr_type)
        power_f1 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_1, .8, sub_id, data_split_id = "test", match_str="power.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{roi_f_1}"], corr_type=corr_type)
        power_f2 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_2, .8, sub_id, data_split_id = "test", match_str="power.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{roi_f_2}"], corr_type=corr_type)
        psnr_f1 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_1, .8, sub_id, data_split_id = "test", match_str="pSNR.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{roi_f_1}"], corr_type=corr_type)
        psnr_f2 = find_activations(experiment_id, mri_id, roi_task_id, roi_f_2, .8, sub_id, data_split_id = "test", match_str="pSNR.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{roi_f_2}"], corr_type=corr_type)
        tasklock = find_task_lock(experiment_id, mri_id, roi_task_id, roi_f_1, .8, sub_id)
        for f_label, f in zip(["f1","f2","mask","pd_f1","pd_f2","test_pd_f1","test_pd_f2","power_f1","power_f2","psnr_f1","psnr_f2","tasklock"], [f1,f2, mask, pd_f1, pd_f2, test_pd_f1, test_pd_f2, power_f1, power_f2, psnr_f1, psnr_f2, tasklock]):
            if roi_task_id == "control" and experiment_id == "1_frequency_tagging":
                if f_label in ["f1", "f2", "mask"]:
                    assert len(f) == 1, f"{sub_id}, {f_label} - {f}"
            else:
                assert len(f) == 1, f"{sub_id}, {f_label} - {f}, {experiment_id} {roi_task_id} {pd_f1} {pd_f2} {power_f1} {power_f2}"
        f1, f2 = f1[0], f2[0]
        data = dfm_combine_f1_f2(f1, f2, fo=ROI_FO, mask=mask[0], f1_c=f1_c,f2_c=f2_c,f1f2_c=f1f2_c,mask_c=mask_c)
        data = data[:VERTEX_TO]
        map_data = dfm_merge_and_binarize_mask(data,f1_c,f2_c,f1f2_c,mask_c)
        activation_f1_data = load_mean_dtseries(f1)[:VERTEX_TO]
        activation_f2_data = load_mean_dtseries(f2)[:VERTEX_TO]
        pd_f1_data = load_mean_dtseries(pd_f1[0])[:VERTEX_TO]
        pd_f2_data = load_mean_dtseries(pd_f2[0])[:VERTEX_TO]
        power_f1_data = load_mean_dtseries(power_f1[0])[:VERTEX_TO]
        power_f2_data = load_mean_dtseries(power_f2[0])[:VERTEX_TO]
        psnr_f1_data = load_mean_dtseries(psnr_f1[0])[:VERTEX_TO]
        psnr_f2_data = load_mean_dtseries(psnr_f2[0])[:VERTEX_TO]
        tasklock_data = load_mean_dtseries(tasklock[0])[:VERTEX_TO]
        test_pd_f1_data = load_mean_dtseries(test_pd_f1[0])[:VERTEX_TO]
        test_pd_f2_data = load_mean_dtseries(test_pd_f2[0])[:VERTEX_TO]
        q_id = get_quadrant_id(mask[0])
        """Save data into DataFrame
        """
        df_data = dfm_append_data(
            df_data, 
            hcp_mapping, 
            map_data,
            activation_f1_data,
            activation_f2_data,
            power_f1_data, 
            power_f2_data, 
            psnr_f1_data,
            psnr_f2_data,
            tasklock_data,
            test_pd_f1_data,
            test_pd_f2_data,
            pd_f1_data, 
            pd_f2_data, 
            q_id,
            label, 
            sub_id,
            roi_fo,
            roi_task_id,
        )
        """Save binarized f1/f2/f1&f2 activation maps
        """
        palette_params = {
            "disp-zero": False,
            "disp-neg": True,
            "disp-pos": True,
            "pos-user": (0, 1.),
            "neg-user": (-1,0),
            "interpolate": True,
        }
        dscalar(
            png_out, data, 
            orientation="portrait", 
            hemisphere='right',
            palette=PALETTE, 
            palette_params=palette_params,
            transparent=False,
            flatmap=True,
            flatmap_style='plain',
        )
        crop_and_save(png_out, str(png_out).replace("png", "cropped.png"), LEFT, TOP, RIGHT, BOTTOM)
        """Save (.png) maps of each metric
        """
        metric_types = [
            "train_activation_f1","train_activation_f2",
            "train_phase_delay_f1","train_phase_delay_f2",
            "test_phase_delay_f1","test_phase_delay_f2",
            "test_power_f1","test_power_f2",
            "test_psnr_f1","test_psnr_f2",
            "tasklock",
        ]
        metrics = [
            activation_f1_data, activation_f2_data,
            pd_f1_data, pd_f2_data,
            test_pd_f1_data, test_pd_f2_data,
            power_f1_data, power_f2_data,
            psnr_f1_data, psnr_f2_data,
            tasklock_data
        ]
        for metric_type, metric in zip(metric_types, metrics):
            # Set palette info
            if "activation" in metric_type:
                disp_pos = True
                disp_neg = False
                disp_zero = False
                neg_user = (-1,0)
                interpolate = True
                pos_user = (0, 1)
            elif "phase_delay" in metric_type:
                disp_pos = True
                disp_neg = False
                disp_zero = False
                neg_user = (-1,0)
                interpolate = True
                if metric_type.endswith("f1"):
                    pos_user = (0, 1/roi_f_1)
                elif metric_type.endswith("f2"):
                    pos_user = (0, 1/roi_f_2)
                else:
                    raise ValueError(f"{metric_type} not supported.")
            elif metric_type.startswith("test"):
                disp_pos = True
                disp_neg = False
                disp_zero = False
                neg_user = (-1,0)
                interpolate = True
                _metric = metric.flatten()
                _metric = _metric[(_metric!=0) & (~np.isnan(_metric))]
                percentile = np.percentile(_metric, 95)
                pos_user = (0,percentile)
            elif metric_type == "tasklock":
                disp_pos = True
                disp_neg = True
                disp_zero = False
                interpolate = True
                _metric = metric.flatten()
                _metric = _metric[(_metric!=0) & (~np.isnan(_metric))]
                percentile = np.percentile(_metric, 97.5)
                pos_user = (0,percentile)
                neg_user = (-percentile,0)
            else:
                raise ValueError(f"{metric_type} not supported.")
            palette_params = {"disp-zero": disp_zero, "disp-neg": disp_neg, "disp-pos": disp_pos, "pos-user": pos_user, "neg-user": neg_user, "interpolate": interpolate}
            metric_type_out = metric_type.replace("_","-")
            metric_png_out = DFM_MAPS / _base / f"label-{label}_mri-{mri_id}_sub-{sub_id}_task-{roi_task_id}_f-{roi_f_1}-{roi_f_2}_corr-{corr_type}_fo-{ROI_FO}_metric-{metric_type_out}.png"
            dscalar(
                metric_png_out, metric, 
                orientation="portrait", 
                hemisphere='right',
                palette="magma",
                palette_params=palette_params,
                transparent=False,
                flatmap=True,
                flatmap_style='plain',
            )
            crop_and_save(png_out, str(png_out).replace("png", "cropped.png"), LEFT, TOP, RIGHT, BOTTOM)
        """Save f1f2 map as dtseries
        """
        f1_img = nib.load(f1)
        dscalar_to_save_as_cifti = np.zeros((1,f1_img.shape[-1]))
        dscalar_to_save_as_cifti[0,:VERTEX_TO] = data
        f1f2_img = nib.Cifti2Image(dscalar_to_save_as_cifti, header=f1_img.header)
        f1f2_img.header.matrix[0].number_of_series_points = 1
        nib.save(f1f2_img, dscalar_out)
        """Verbose debug"""
        #track = [len(v) for k,v in df_data.items()]
        #print(track)

    return df_data

"""
generate intermodulation maps
:ref: notebooks/ComputeCanada/frequency_tagging/5_0_im_generate_maps.ipynb
"""
def get_im_frequencies(f1,f2):
    assert f2 > f1, f"{f2} <= {f1}"
    im_frequencies = {}
    im_frequencies["first_order"] = [
        ("f1",f1),
        ("f2",f2),
    ]
    f2_sub_f1 = round(f2-f1,10)
    f1_plus_f2 = round(f1+f2,10)
    f1_mul_2 = round(f1*2,10)
    f2_mul_2 = round(f2*2,10)
    f1_mul_2_sub_f2 = round(2*f1-f2,10)
    f2_mul_2_sub_f1 = round(2*f2-f1,10)
    im_frequencies["second_order"] = [
        ("f2-f1",f2_sub_f1),
        ("f1+f2",f1_plus_f2),
        ("2f1",f1_mul_2),
        ("2f2",f2_mul_2),
    ]
    im_frequencies["third_order"] = [
        ("2f1-f2",f1_mul_2_sub_f2),
        ("2f2-f1",f2_mul_2_sub_f1),
    ]

    return im_frequencies

def convert_f_im(f_im,fo=1.,mask=None,f_im_c=.01,mask_c=.01):
    f_im_data = convert_to_fractional_overlap(nib.load(f_im).get_fdata())
    f_im_data = (f_im_data >= fo).astype(int)
    if mask:
        mask_data = convert_to_fractional_overlap(nib.load(mask).get_fdata())
        mask_data = (mask_data >= 1.).astype(int)
        mask_data -= f_im_data
        mask_data[mask_data==-1] = 0
    data_dict = [(f_im_data,f_im_c)]
    if mask:
        data_dict.append((mask_data,mask_c))

    return map_data_to_value(data_dict)

def convert_f_im_with_f1_f2(f_im,f_1,f_2,fo=1.,mask=None,f_1_c=-.1,f_2_c=.82,f1f2_c=.1,f_im_c=.9,mask_c=.4,):
    f_1_data = convert_to_fractional_overlap(nib.load(f_1).get_fdata())
    f_1_data = (f_1_data >= fo).astype(int)
    f_2_data = convert_to_fractional_overlap(nib.load(f_2).get_fdata())
    f_2_data = (f_2_data >= fo).astype(int)
    f1f2_data = ((f_1_data + f_2_data) == 2).astype(int) # Intersection of f1 & f2
    f_1_data -= f1f2_data # f1 only
    f_2_data -= f1f2_data # f2 only
    f_im_data = convert_to_fractional_overlap(nib.load(f_im).get_fdata())
    f_im_data = (f_im_data >= fo).astype(int) # f_im
    
    # Recolour f_im_data with f_1, f_2 and f1f2 (show f1, f2 and f1f2 that appears in fim) 
    if mask:
        mask_data = convert_to_fractional_overlap(nib.load(mask).get_fdata())
        mask_data = (mask_data >= 1.).astype(int)
    data_dict = []
    for f_data,f_c in zip([f_1_data, f_2_data, f1f2_data],[f_1_c, f_2_c,f1f2_c]):
        _f_data = ((f_data+f_im_data)==2).astype(int) # Intersection of f_im and f (f_data)
        f_im_data -= _f_data
        if mask:
            mask_data -= _f_data
        data_dict.append((_f_data,f_c))
    data_dict.append((f_im_data,f_im_c))
    if mask:
        mask_data -= f_im_data
        data_dict.append((mask_data,mask_c))
    X = map_data_to_value(data_dict)
    
    # Recolour f_im_data with f_1, f_2 and f1f2 (show f1, f2 and f1f2 that does not appears in fim) 
    f_im_data = convert_to_fractional_overlap(nib.load(f_im).get_fdata())
    f_im_data = (f_im_data >= fo).astype(int) # f_im
    if mask:
        mask_data = convert_to_fractional_overlap(nib.load(mask).get_fdata())
        mask_data = (mask_data >= 1.).astype(int)
    data_dict = []
    for f_data,f_c in zip([f_1_data, f_2_data, f1f2_data],[f_1_c, f_2_c,f1f2_c]):
        _f_data = ((f_data+f_im_data)==2).astype(int) # Intersection of f_im and f (f_data)
        _f_data_only = f_data.copy() # f-only, and does not include any IM vertices
        _f_data_only -= _f_data
        if mask:
            mask_data -= _f_data_only
        data_dict.append((_f_data_only,f_c))
    data_dict.append((f_im_data,f_im_c))
    if mask:
        mask_data -= f_im_data
        data_dict.append((mask_data,mask_c))
    Y = map_data_to_value(data_dict)

    return X, Y

def im_binarize_mask(data, f_im_c, mask_c, im_key):
    data_dict = {
        im_key: data.copy(),
    }
    data_dict[im_key][(data_dict[im_key]==f_im_c)] = 1
    for v in data_dict.values():
        v[v==mask_c] = 0
    # Create mask
    data_dict["mask"] = data.copy()
    data_dict["mask"][(data_dict["mask"]==mask_c) | (data_dict["mask"]==f_im_c)] = 1

    for k,v in data_dict.items():
        assert np.all(np.isin(v, [0,1])), f"{k}, {np.unique(v)} {f_im_c} {mask_c} {im_key} {np.unique(data)}"

    return data_dict

                
def im_append_data(
    df_data,
    hcp_mapping,
    map_data,
    activation_f_im_data,
    power_im_data,
    psnr_im_data,
    tasklock_data,
    test_pd_im_data,
    pd_im_data,
    q_id,
    experiment_label,
    sub_id,
    roi_fo,
    roi_task_id,
    im_str,
    im_f
):
    """Stores vertex level data for each HCP ROI:
    """
    roi_slab = map_data["mask"]
    for im_code, f_data in map_data.items():
        if im_code == "mask":
            continue
        assert im_code == im_str, f"{im_code} != {im_str}"
        for roi_label, roi_path in hcp_mapping.items():
            if q_id == "Q1":
                contra = "L_"
            elif q_id == "Q2":
                contra = "R_"
            else:
                raise ValueError(f"{q_id} not supported.")

            if roi_label.startswith(contra):
                roi_label = f"CONTRA_{roi_label[2:-4]}"
            else:
                roi_label = f"IPSI_{roi_label[2:-4]}"

            roi_mask = read_roi_path(roi_path)
            assert roi_mask.shape == f_data.shape

            hcp_and_slab_roi = roi_mask * roi_slab
            if hcp_and_slab_roi.sum() == 0:
                # Skip if HCP ROI not found in the slab
                continue

            hcp_and_f_roi = roi_mask * f_data
            frequency_coordinates = hcp_and_f_roi==1
            slab_coordinates = hcp_and_slab_roi==1
            # Train
            f_im_activation = activation_f_im_data[slab_coordinates]
            f_im_phase_delay = pd_im_data[slab_coordinates]
            # Test
            f_im_power = power_im_data[slab_coordinates]
            f_im_psnr = psnr_im_data[slab_coordinates]
            test_f_im_phase_delay = test_pd_im_data[slab_coordinates]
            tasklock = tasklock_data[slab_coordinates]

            # Save to dataframe
            df_data["roi_task_id"].append(roi_task_id)
            df_data["roi_fo"].append(roi_fo)
            df_data["experiment_id"].append(experiment_label)
            df_data["sub_id"].append(sub_id)
            df_data["quadrant_id"].append(q_id)
            df_data["hcp_roi"].append(roi_label)
            df_data["im_code"].append(im_code)
            df_data["frequency_of_roi"].append(im_f)
            df_data["frequency_coordinates"].append(frequency_coordinates)
            df_data["slab_coordinates"].append(slab_coordinates)
            # Train
            df_data["train_f_im_activation"].append(f_im_activation)
            df_data["train_f_im_phase_delay"].append(f_im_phase_delay)
            # Test
            df_data["test_f_im_BOLD_power"].append(f_im_power)
            df_data["test_f_im_BOLD_psnr"].append(f_im_psnr)
            df_data["test_f_im_phase_delay"].append(test_f_im_phase_delay)
            df_data["tasklock"].append(tasklock)

    return df_data

def im_generate_single_subject_maps(
    hcp_mapping, label, experiment_id, mri_id, sub_ids, 
    roi_task_ids, roi_f_1s, roi_f_2s, roi_fo,
    hcp_labels,
    df_data=None,
    proportion_data=None,
    corr_type="uncp", 
    ROI_FO=.8,
    LEFT=590, TOP=80, RIGHT=1140, BOTTOM=460, VERTEX_TO = 59412,
):

    PALETTE, f_1_c, f_2_c, f1f2_c, f_im_c, mask_c = get_wbplot_colour_palette_details("im")
    
    if df_data is None:
        df_data = defaultdict(list)

    if proportion_data is None:
        proportion_data = defaultdict(list)

    # Loop over cohort
    for sub_id, roi_task_id, roi_f_1, roi_f_2 in zip(
        sub_ids,
        roi_task_ids, 
        roi_f_1s,
        roi_f_2s,
    ):
        im_frequencies = get_im_frequencies(roi_f_1, roi_f_2)
        """ 
        im_frequencies = {
            "first_order": [("f1",.125), ...],
            "second_order": [("f2-f1",.075), ...],
            "third_order": [("2f1-f2",.05), ...],
        }
        """
        for im_order, v in im_frequencies.items():
            im_strs = [i[0] for i in v]
            im_fs = [i[1] for i in v]
            for im_str, im_f in zip(im_strs, im_fs):
                # Set up output directories
                _base = f"experiment-{experiment_id}_mri-{mri_id}_sub-{sub_id}"
                _ = set_base_dir(f"{str(IM_MAPS)}/{_base}")
                _ = set_base_dir(f"{str(CIFTI_IM_MAPS)}/{_base}")
                # Output paths
                png_out = IM_MAPS / _base / f"label-{label}_mri-{mri_id}_sub-{sub_id}_task-{roi_task_id}_f-{im_order}-{im_str}-{im_f}_corr-{corr_type}_fo-{ROI_FO}.png"
                dscalar_out = CIFTI_IM_MAPS / _base / f"label-{label}_mri-{mri_id}_sub-{sub_id}_task-{roi_task_id}_f-{im_order}-{im_str}-{im_f}_corr-{corr_type}_fo-{ROI_FO}.dtseries.nii"
                # Get frequency tagging frequencies
                f_1_str = im_frequencies["first_order"][0][0]
                f_2_str = im_frequencies["first_order"][1][0]
                f_1 = im_frequencies["first_order"][0][1]
                f_2 = im_frequencies["first_order"][1][1]
                assert f_2 > f_1, f"{f_2} <= {f_1}"
                # Load data
                f_1 = find_activations(experiment_id, mri_id, roi_task_id, f_1, .8, sub_id, match_str="activations.dtseries.nii", corr_type=corr_type, desc_id="IMall")
                f_2 = find_activations(experiment_id, mri_id, roi_task_id, f_2, .8, sub_id, match_str="activations.dtseries.nii", corr_type=corr_type, desc_id="IMall")
                f_im = find_activations(experiment_id, mri_id, roi_task_id, im_f, .8, sub_id, match_str="activations.dtseries.nii", corr_type=corr_type, desc_id="IMall")
                mask = find_activations(experiment_id, mri_id, roi_task_id, im_f, .8, sub_id, match_str="mask.dtseries.nii", corr_type=corr_type, desc_id="IMall")
                pd_im = find_activations(experiment_id, mri_id, roi_task_id, im_f, .8, sub_id, data_split_id="train", match_str="phasedelay.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{im_f}"], corr_type=corr_type, desc_id="IMall")
                test_pd_im = find_activations(experiment_id, mri_id, roi_task_id, im_f, .8, sub_id, data_split_id="test", match_str="phasedelay.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{im_f}"], corr_type=corr_type, desc_id="IMall")
                power_im = find_activations(experiment_id, mri_id, roi_task_id, im_f, .8, sub_id, data_split_id="test", match_str="power.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{im_f}"], corr_type=corr_type, desc_id="IMall")
                psnr_im = find_activations(experiment_id, mri_id, roi_task_id, im_f, .8, sub_id, data_split_id="test", match_str="pSNR.dtseries.nii", additional_match_strs=[roi_task_id,f"f-{im_f}"], corr_type=corr_type, desc_id="IMall")
                tasklock = find_task_lock(experiment_id, mri_id, roi_task_id, im_f, .8, sub_id, desc_id="IMall")
                for f_label, f in zip([f_1_str,f_2_str,im_str,"mask",f"pd_{im_str}",f"power_{im_str}",f"test_pd_{im_str}",f"psnr_{im_str}","tasklock"], [f_1,f_2,f_im,mask,pd_im,power_im,test_pd_im,psnr_im,tasklock]):
                    if roi_task_id == "control" and experiment_id == "1_frequency_tagging":
                        if f_label in [im_str,"mask"]:
                            assert len(f) == 1, f"{sub_id}, {f_label} - {f}"
                    else: # AssertionError: 2f2/0.4, 002, power_2f2 - [], 1_frequency_tagging entrain
                        if not f_label.startswith("power"):
                            assert len(f) == 1, f"{im_str}/{im_f}, {sub_id}, {f_label} - {f}, {experiment_id} {roi_task_id}"
                f_1, f_2, f_im = f_1[0], f_2[0], f_im[0]
                # Create image for im only
                data = convert_f_im(f_im, fo=ROI_FO, mask=mask[0], f_im_c=f_im_c, mask_c=mask_c)
                data = data[:VERTEX_TO]
                # Create image for im, contextualized by f1 and f2
                data_contextualized_1, data_contextualized_2 = convert_f_im_with_f1_f2(f_im, f_1, f_2, fo=ROI_FO, mask=mask[0], f_1_c=f_1_c, f_2_c=f_2_c, f1f2_c=f1f2_c, f_im_c=f_im_c, mask_c=mask_c)
                data_contextualized_1 = data_contextualized_1[:VERTEX_TO]
                data_contextualized_2 = data_contextualized_2[:VERTEX_TO]
                """
                Save proportion data into DataFrame
                """
                # Get total vertex count
                c_per_label = [f_1_c,f_2_c,f1f2_c,f_im_c]
                for hcp_label in hcp_labels:
                    hcp_roi_mask = read_roi_path(f"/tmp/{hcp_label}.dscalar.nii")
                    vertex_in_hcp_roi = hcp_roi_mask.sum()
                    _data_contextualized_1 = data_contextualized_1 * hcp_roi_mask
                    total_vertex = 0
                    for _f_c in c_per_label:
                        total_vertex += (_data_contextualized_1 == _f_c).sum()
                    total_vertex_with_slab = total_vertex + (_data_contextualized_1 == mask_c).sum()
                    # Get proportion per label
                    if total_vertex > 0:
                        vertex_label_dict = {}
                        f_labels = ["f1","f2","f1&f2","fim"]
                        for _f_label, _f_c in zip(f_labels, c_per_label):
                            vertex_label_dict[_f_label] = (_data_contextualized_1 == _f_c).sum() / total_vertex
                        vertex_label_dict["hcp_label"] = hcp_label
                        vertex_label_dict["activated_vertex_count"] = total_vertex
                        vertex_label_dict["slab_vertex_count"] = total_vertex_with_slab
                        vertex_label_dict["hcp_vertex_count"] = vertex_in_hcp_roi
                        vertex_label_dict["sub_id"] = sub_id
                        vertex_label_dict["im_code"] = im_str
                        vertex_label_dict["roi_task_id"] = roi_task_id
                        vertex_label_dict["experiment"] = label
                        for k,v in vertex_label_dict.items():
                            proportion_data[k].append(v)
                # Load more data
                map_data = im_binarize_mask(data,f_im_c,mask_c,im_str)
                activation_f_im_data = load_mean_dtseries(f_im)[:VERTEX_TO]
                pd_im_data = load_mean_dtseries(pd_im[0])[:VERTEX_TO]
                power_im_data = load_mean_dtseries(power_im[0])[:VERTEX_TO]
                psnr_im_data = load_mean_dtseries(psnr_im[0])[:VERTEX_TO]
                tasklock_data = load_mean_dtseries(tasklock[0])[:VERTEX_TO]
                test_pd_im_data = load_mean_dtseries(test_pd_im[0])[:VERTEX_TO]
                q_id = get_quadrant_id(mask[0])
                """Save data into DataFrame"""
                df_data = im_append_data(
                    df_data,
                    hcp_mapping,
                    map_data,
                    activation_f_im_data,
                    power_im_data,
                    psnr_im_data,
                    tasklock_data,
                    test_pd_im_data,
                    pd_im_data,
                    q_id,
                    label, 
                    sub_id,
                    roi_fo,
                    roi_task_id,
                    im_str,
                    im_f,
                )
                """Save intermodulation maps
                """
                palette_params = {
                    "disp-zero": False,
                    "disp-neg": True,
                    "disp-pos": True,
                    "pos-user": (0, 1.),
                    "neg-user": (-1,0),
                    "interpolate": True,
                }
                # Save raw intermodulation map 
                dscalar(
                    png_out, data, 
                    orientation="portrait", 
                    hemisphere='right',
                    palette=PALETTE, 
                    palette_params=palette_params,
                    transparent=False,
                    flatmap=True,
                    flatmap_style='plain',
                )
                crop_and_save(png_out, str(png_out).replace("png", "cropped.png"), LEFT, TOP, RIGHT, BOTTOM)
                # Save f1/f2 contextualized intermodulation maps
                png_out = str(png_out).replace(".png", "_contextualized.png")
                dscalar(
                    png_out, data_contextualized_1, 
                    orientation="portrait", 
                    hemisphere='right',
                    palette=PALETTE, 
                    palette_params=palette_params,
                    transparent=False,
                    flatmap=True,
                    flatmap_style='plain',
                )
                crop_and_save(png_out, str(png_out).replace(".png", "_contextualized.cropped.png"), LEFT, TOP, RIGHT, BOTTOM)
                png_out = str(png_out).replace("_contextualized.png", "_contextualized_include_missing.png")
                dscalar(
                    png_out, data_contextualized_2,
                    orientation="portrait", 
                    hemisphere='right',
                    palette=PALETTE, 
                    palette_params=palette_params,
                    transparent=False,
                    flatmap=True,
                    flatmap_style='plain',
                )
                """Save (.png) maps of each metric
                """
                metric_types = [
                    "train_activation_f_im",
                    "train_phase_delay_f_im",
                    "test_phase_delay_f_im",
                    "test_power_f_im",
                    "test_psnr_f_im",
                    "tasklock",
                ]
                metrics = [
                    activation_f_im_data,
                    pd_im_data,
                    test_pd_im_data,
                    power_im_data,
                    psnr_im_data,
                    tasklock_data
                ]
                for metric_type, metric in zip(metric_types, metrics):
                    # Set palette info
                    if "activation" in metric_type:
                        disp_pos = True
                        disp_neg = False
                        disp_zero = False
                        neg_user = (-1,0)
                        interpolate = True
                        pos_user = (0, 1)
                    elif "phase_delay" in metric_type:
                        disp_pos = True
                        disp_neg = False
                        disp_zero = False
                        neg_user = (-1,0)
                        interpolate = True
                        pos_user = (0, 1/im_f)
                    elif metric_type.startswith("test"):
                        disp_pos = True
                        disp_neg = False
                        disp_zero = False
                        neg_user = (-1,0)
                        interpolate = True
                        _metric = metric.flatten()
                        _metric = _metric[(_metric!=0) & (~np.isnan(_metric))]
                        percentile = np.percentile(_metric, 95)
                        pos_user = (0,percentile)
                    elif metric_type == "tasklock":
                        disp_pos = True
                        disp_neg = True
                        disp_zero = False
                        interpolate = True
                        _metric = metric.flatten()
                        _metric = _metric[(_metric!=0) & (~np.isnan(_metric))]
                        percentile = np.percentile(_metric, 97.5)
                        pos_user = (0,percentile)
                        neg_user = (-percentile,0)
                    else:
                        raise ValueError(f"{metric_type} not supported.")
                    palette_params = {"disp-zero": disp_zero, "disp-neg": disp_neg, "disp-pos": disp_pos, "pos-user": pos_user, "neg-user": neg_user, "interpolate": interpolate}
                    metric_type_out = metric_type.replace("_","-")
                    metric_png_out = IM_MAPS / _base / f"label-{label}_mri-{mri_id}_sub-{sub_id}_task-{roi_task_id}_f-{im_order}-{im_str}-{im_f}_corr-{corr_type}_fo-{ROI_FO}_metric-{metric_type_out}.png"
                    dscalar(
                        metric_png_out, metric, 
                        orientation="portrait", 
                        hemisphere='right',
                        palette="magma",
                        palette_params=palette_params,
                        transparent=False,
                        flatmap=True,
                        flatmap_style='plain',
                    )
                    #crop_and_save(png_out, str(png_out).replace("png", "cropped.png"), LEFT, TOP, RIGHT, BOTTOM)
                """Save intermodulation map as dtseries
                """
                f_im_img = nib.load(f_im)
                dscalar_to_save_as_cifti = np.zeros((1,f_im_img.shape[-1]))
                dscalar_to_save_as_cifti[0,:VERTEX_TO] = data
                f_im_img = nib.Cifti2Image(dscalar_to_save_as_cifti, header=f_im_img.header)
                f_im_img.header.matrix[0].number_of_series_points = 1
                nib.save(f_im_img, dscalar_out)
                """Verbose debug"""
                #track = [len(v) for k,v in df_data.items()]
                #print(track)

    return df_data, proportion_data

"""Carpet plots
:ref: 4_1_dfm_bootstrap_carpet_plots.ipynb
"""
def read_pkl(datadir, n_bootstraps, sub_id, roi_task_id, roi_frequency, task_id, experiment_id="1_frequency_tagging", mri_id="7T", fo=.8, pval="uncp"):
    """No longer supporting `control_roi_size`, and `roi_frequency_1` inputs
    These pkls were generated using the following (now, unused) notebooks
        - 3_merge_bootstrap_controlroisize_basic.ipynb
        - 3_merge_bootstrap_intersection_basic.ipynb
    """

    import pickle

    bootstrap_pkl: Path = datadir / f"experiment-{experiment_id}_mri-{mri_id}_smooth-0_truncate-39-219_n-{n_bootstraps}_batch-merged_desc-basic_roi-{roi_task_id}-{roi_frequency}_pval-{pval}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap/task-{task_id}_bootstrapped_data.pkl"
    if not bootstrap_pkl.exists():
        print(f"Warning: {bootstrap_pkl} does not exist.\nReturn None")
        return None

    print(f"Reading: {bootstrap_pkl}")
    with open(bootstrap_pkl, 'rb') as f:
        data = pickle.load(f)

    return data

def find_raw_bold(i):
    
    import os
    
    experiment_id = str(i.parent).split("experiment-")[1].split('_mri-')[0]
    mri_id = str(i.parent).split("mri-")[1].split('_')[0]
    sub_id = i.stem.split('sub-')[1].split('_')[0]
    ses_id = i.stem.split('ses-')[1].split('_')[0]
    task_id = i.stem.split('task-')[1].split('_')[0]
    run_id = i.stem.split('run-')[1].split('_')[0]

    directory = f"/data/{experiment_id}/{mri_id}/bids/derivatives/oscprep_grayords_fmapless/bold_preproc/sub-{sub_id}/ses-{ses_id}/func"
    raw_bold = [f"{directory}/{file}" for file in os.listdir(directory) if f"run-{run_id}" in file and f"task-{task_id}" in file and file.endswith("bold.dtseries.nii")]
    assert len(raw_bold) == 1, f"Multiple raw bolds found: {raw_bold}"

    return Path(raw_bold[0])

def average_bold(bold_list):
    for bold_ix, bold in enumerate(bold_list):
        _bold_data = nib.load(bold).get_fdata()
        if bold_ix == 0:
            y_all = _bold_data.copy() 
        else:
            y_all += _bold_data.copy()
        
    y_all /= len(bold_list)
    y_all = y_all.T
    #y_all = (( y_all - y_all.mean(0)) / y_all.std(0) ).T

    return y_all

def read_bootstrap_txt(bootstrap_txt, bootstrap_idx):
    with open(bootstrap_txt, "r") as f:
        lines = f.readlines()

    fs = lines[bootstrap_idx]
    raw_bolds = []
    raw_windowed_bolds = []
    processed_bolds = []
    for i in fs.split(','):
        i = Path(i.strip())
        raw_windowed_bold = Path(str(i).replace("desc-denoised_bold.dtseries.nii","desc-windowed_bold.dtseries.nii"))
        raw_bold = find_raw_bold(Path(i))
        assert i.exists(), f"{i} not found."
        assert raw_windowed_bold.exists(), f"{raw_windowed_bold} not found"
        assert raw_bold.exists(), f"{raw_bold} not found."
        raw_bolds.append(raw_bold)
        raw_windowed_bolds.append(raw_windowed_bold)
        processed_bolds.append(i)

    raw_avg = average_bold(raw_bolds)
    raw_windowed_avg = average_bold(raw_windowed_bolds)
    processed_avg = average_bold(processed_bolds)

    return raw_avg, raw_windowed_avg, processed_avg

def extract_carpet_data(data, task_id, task_quadrant, bootstrap_id, phased_flag):

    if phased_flag:
        data_tps = data[f'data-test_task-{task_id}{task_quadrant}_roi_phaseadjusted_timepoints']
        data_bold = data[f'data-test_task-{task_id}{task_quadrant}_roi_phaseadjusted_bold'][:,:,bootstrap_id]
        updated_tps, mean_bold = [], []
        n_voxels = data_bold.shape[1]
        for single_tp in np.unique(data_tps):
            coords = (data_tps == single_tp)
            tp_all = np.all((coords).sum(0)) == 1
            if tp_all:
                updated_tps.append(single_tp)
                mean_bold.append(data_bold[coords])
        updated_tps = np.array(updated_tps)
        assert _check_difference(updated_tps)
        
        return updated_tps, np.vstack(mean_bold)

    else:
        # Select timeseries (timepoints x voxels x bootstraps)
        return data[f'data-test_task-{task_id}{task_quadrant}_roi_timepoints'], data[f'data-test_task-{task_id}{task_quadrant}_roi_bold'][:,:,bootstrap_id]

def find_quadrant_id_from_keys(_dict, task_id):
    for i in _dict.keys():
        if task_id in i:
            q_idx = i.find("Q")
            q_id = i[q_idx:q_idx+2]
            assert q_id in ['Q1', 'Q2']
            return q_id
    raise ValueError("No quadrant id found.")

def decorate_fig_carpetplot(y,fig, ax, im, f1, f2, n_f1, n_f2, n_f1f2, FONTSIZE=8, TR=.3,n_pos=160):

    cbar = plt.colorbar(im, ax=ax, shrink=.5, drawedges=False)
    cbar.ax.set_title(r'$\sigma$', fontsize=FONTSIZE, pad=0)
    cbar.ax.set_yticks([-1,0,1])
    cbar.ax.set_yticklabels([-1,0,1], fontdict={'fontsize':FONTSIZE, 'horizontalalignment': 'right'})
    cbar.ax.tick_params(axis="both", length=2,pad=6, width=.5)
    cbar.outline.set_edgecolor('none')

    #ax.set_title(f"{sub_id}, roi-{task_id}, roi-frequency-{f}", fontsize=FONTSIZE)
    ax.title.set_position([.75,1.05])

    ax.set_ylabel("Vertex", fontsize=FONTSIZE,labelpad=0)
    ax.set_yticks([])

    ax.set_xlabel("Time (s)", fontsize=FONTSIZE,labelpad=0)

    xticks = [i*60 for i in range(int((y.shape[-1] * TR) / 40))]
    ax.set_xticks([i/TR for i in xticks])
    ax.set_xticklabels(xticks, fontsize=FONTSIZE)
    ax.tick_params(axis="both", length=2, width=.5, direction="in")

    period_f1 = 1/f1
    period_f2 = 1/f2

    total_vertices = n_f1 + n_f2 + n_f1f2
    top = -total_vertices * .22
    bottom = -total_vertices * .1
    f2_top = -total_vertices * .12
    f2_bottom = -total_vertices * .05
    f1_top = -total_vertices * .22
    f1_bottom = -total_vertices * .15
    ax.plot([0,period_f1/TR], [bottom,bottom], c='white', zorder=1)
    ax.plot([0,period_f2/TR], [top,top], c='white', zorder=2)
    timescale_f1 = plt.Polygon([(0, f1_bottom), (period_f1/TR, f1_bottom), (period_f1/TR, f1_top), (0, f1_top)], closed=True, color='red', linewidth=0., zorder=10)
    timescale_f2 = plt.Polygon([(0, f2_bottom), (period_f2/TR, f2_bottom), (period_f2/TR, f2_top), (0, f2_top)], closed=True, color='blue', linewidth=0., zorder=10)
    
    square_f1 = plt.Polygon([(-2, 0), (-15, 0), (-15, n_f1), (-2, n_f1)], closed=True, color='red', linewidth=0.)
    square_f1f2 = plt.Polygon([(-2, n_f1), (-15, n_f1), (-15, n_f1+n_f1f2), (-2, n_f1+n_f1f2)], closed=True, color='gold', linewidth=0.)
    square_f2 = plt.Polygon([(-2, n_f1+n_f1f2), (-15, n_f1+n_f1f2), (-15, n_f1+n_f1f2+n_f2), (-2, n_f1+n_f1f2+n_f2)], closed=True, color='blue', linewidth=0.)

    for square in [square_f1, square_f1f2, square_f2, timescale_f1, timescale_f2]:
        ax.add_patch(square)

    for i in ("top", "right", "bottom", "left"):
        ax.spines[i].set_visible(False)

    ax.tick_params(axis="both",pad=0)
    ax.text(n_pos/TR,f1_top/3,f"n={total_vertices}", fontsize=FONTSIZE-2,fontdict={"horizontalalignment":"left"})

    return fig, ax

"""
power spectrum analyses
:ref: 4_2_dfm_psd.ipynb
"""
class f1_f2_data:

    def __init__(self, datadir, n_bootstraps, sub_id, roi_task_id, roi_frequency_f1, roi_frequency_f2, task_id, experiment_id="1_frequency_tagging", mri_id="3T", fo=.8, pval="uncp"):
        if roi_frequency_f1 >= roi_frequency_f2:
            raise ValueError(f"$f_1$ must be less than $f_2$.")
        self.f_data = self._read_f1_f2_pkls(
            datadir, 
            n_bootstraps, 
            sub_id, 
            roi_task_id, 
            roi_frequency_f1, 
            roi_frequency_f2, 
            task_id, 
            experiment_id=experiment_id, 
            mri_id=mri_id, 
            fo=fo, 
            pval=pval,
        )
        self.task_id = self._read_task_id()
        self.roi_task_id = roi_task_id
        self.roi_f1 = roi_frequency_f1
        self.roi_f2 = roi_frequency_f2
        self.pval = pval
        self.n_bootstraps = n_bootstraps


    def extract_bootstrapped_mean_from_data(self, bootstrap_id, f_type, rephase=False, rephase_with=None, pure_f=True):
        f1_coords = self.f_data['f1']['roi_coords'].astype(int)
        f2_coords = self.f_data['f2']['roi_coords'].astype(int)
        intersection_roi_coords = f1_coords * f2_coords
        intersection_roi_coords_from_f1 = intersection_roi_coords[f1_coords==1]
        intersection_roi_coords_from_f2 = intersection_roi_coords[f2_coords==1] # redundant

        if rephase:
            f1_bold = self.f_data['f1'][f'data-test_{self.task_id}_roi_phaseadjusted_bold'][:,:,bootstrap_id]
            f1_tps = self.f_data['f1'][f'data-test_{self.task_id}_roi_phaseadjusted_timepoints']
            f2_bold = self.f_data['f2'][f'data-test_{self.task_id}_roi_phaseadjusted_bold'][:,:,bootstrap_id]
            f2_tps = self.f_data['f2'][f'data-test_{self.task_id}_roi_phaseadjusted_timepoints']
            intersection_f1f2_bold_from_f1 = f1_bold[:, intersection_roi_coords_from_f1==1]
            intersection_f1f2_tps_from_f1 = f1_tps[:, intersection_roi_coords_from_f1==1]
            intersection_f1f2_bold_from_f2 = f2_bold[:, intersection_roi_coords_from_f2==1]
            intersection_f1f2_tps_from_f2 = f2_tps[:, intersection_roi_coords_from_f2==1]
            if pure_f:
                pure_f1_coords = f1_coords - intersection_roi_coords
                pure_f1_coords = pure_f1_coords[f1_coords==1]
                f1_bold = f1_bold[:,pure_f1_coords==1]
                f1_tps = f1_tps[:,pure_f1_coords==1]
                pure_f2_coords = f2_coords - intersection_roi_coords
                pure_f2_coords = pure_f2_coords[f2_coords==1]
                f2_bold = f2_bold[:,pure_f2_coords==1]
                f2_tps = f2_tps[:,pure_f2_coords==1]
            if f_type == "f1":
                tps, bold = self._aggregate_timeseries(f1_tps, f1_bold, "mean", rephase=rephase)
            elif f_type == "f2":
                tps, bold = self._aggregate_timeseries(f2_tps, f2_bold, "mean", rephase=rephase)
            elif f_type == "f1f2":
                if rephase_with == "f1":
                    tps, bold = self._aggregate_timeseries(
                        intersection_f1f2_tps_from_f1, 
                        intersection_f1f2_bold_from_f1,
                        "mean",
                        rephase=rephase 
                    )
                elif rephase_with == "f2":
                    tps, bold = self._aggregate_timeseries(
                        intersection_f1f2_tps_from_f2, 
                        intersection_f1f2_bold_from_f2,
                        "mean",
                        rephase=rephase 
                    )
                else:
                    raise ValueError(f"Rephasing of timeseries is turned on, must set `rephase_with` to either f1 or f2.")
                
            return tps, bold

        else:
            f1_bold = self.f_data['f1'][f'data-test_{self.task_id}_roi_bold'][:,:,bootstrap_id]
            f1_tps = self.f_data['f1'][f'data-test_{self.task_id}_roi_timepoints']
            f2_bold = self.f_data['f2'][f'data-test_{self.task_id}_roi_bold'][:,:,bootstrap_id]
            f2_tps = self.f_data['f2'][f'data-test_{self.task_id}_roi_timepoints']
            intersection_f1f2_bold_from_f1 = f1_bold[:, intersection_roi_coords_from_f1==1]
            intersection_f1f2_tps_from_f1 = f1_tps[:, intersection_roi_coords_from_f1==1]
            intersection_f1f2_bold_from_f2 = f2_bold[:, intersection_roi_coords_from_f2==1]
            intersection_f1f2_tps_from_f2 = f2_tps[:, intersection_roi_coords_from_f2==1]
            if pure_f:
                pure_f1_coords = f1_coords - intersection_roi_coords
                pure_f1_coords = pure_f1_coords[f1_coords==1]
                f1_bold = f1_bold[:,pure_f1_coords==1]
                f1_tps = f1_tps[:,pure_f1_coords==1]
                pure_f2_coords = f2_coords - intersection_roi_coords
                pure_f2_coords = pure_f2_coords[f2_coords==1]
                f2_bold = f2_bold[:,pure_f2_coords==1]
                f2_tps = f2_tps[:,pure_f2_coords==1]
            if f_type == "f1":
                tps, bold = self._aggregate_timeseries(f1_tps, f1_bold, "mean", rephase=rephase)
            elif f_type == "f2":
                tps, bold = self._aggregate_timeseries(f2_tps, f2_bold, "mean", rephase=rephase)
            elif f_type == "f1f2":
                tps, bold = self._aggregate_timeseries(intersection_f1f2_tps_from_f1, intersection_f1f2_bold_from_f1, "mean", rephase=rephase)
            else:
                raise ValueError(f"{f_type} not supported")
            
            return tps, bold
        
    def _aggregate_timeseries(self, tps, bold, aggr_type, rephase=False):
        
        assert aggr_type == "mean"

        if rephase:
            updated_tps, mean_bold = [], []
            for single_tp in np.unique(tps):
                coords = (tps == single_tp)
                tp_all = np.all((coords).sum(0)) == 1
                if tp_all:
                    updated_tps.append(single_tp)
                    mean_bold.append(bold[coords].mean())

            updated_tps = np.array(updated_tps)
            mean_bold = np.array(mean_bold)
            assert _check_difference(updated_tps)
            assert mean_bold.shape == updated_tps.shape

            return updated_tps, mean_bold

        else:
            return tps.mean(1), bold.mean(1)

    def _read_f1_f2_pkls(self, datadir, n_bootstraps, sub_id, roi_task_id, roi_frequency_f1, roi_frequency_f2, task_id, experiment_id="1_frequency_tagging", mri_id="3T", fo=.8, pval="uncp"):

        import pickle
        
        f1_pkl: Path = datadir / f"experiment-{experiment_id}_mri-{mri_id}_smooth-0_truncate-39-219_n-{n_bootstraps}_batch-merged_desc-basic_roi-{roi_task_id}-{roi_frequency_f1}_pval-{pval}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap/task-{task_id}_bootstrapped_data.pkl"
        f2_pkl: Path = datadir / f"experiment-{experiment_id}_mri-{mri_id}_smooth-0_truncate-39-219_n-{n_bootstraps}_batch-merged_desc-basic_roi-{roi_task_id}-{roi_frequency_f2}_pval-{pval}_fo-{fo}_bootstrap/sub-{sub_id}/bootstrap/task-{task_id}_bootstrapped_data.pkl"

        assert f1_pkl.exists()
        assert f2_pkl.exists()
        f_data = {}
        for f_type, f_pkl in zip(["f1","f2"], [f1_pkl, f2_pkl]):
            with open(f_pkl, 'rb') as f:
                f_data[f_type] = pickle.load(f)

        return f_data
    
    def _read_task_id(self):
        for i in self.f_data['f1'].keys():
            if i.startswith("task-"):
                return i.split("_")[0]
            
        raise ValueError("No task_id found.")

# Run statistics on a timeseries
class TimeSeries:
    def __init__(self, ts, TR, n_permutations=5_000, nperseg=600):
        self.timeseries = ts
        self.fs = 1/TR
        self.nperseg = nperseg
        self.n_permutations = n_permutations
        self.frequencies = None
                
    def process(self, search_frequencies):
        p_values, observed_statistics = {}, {}
        for f in search_frequencies:
            observed_statistic, observed_power_spectrum = self.calculate_observed_statistic(f)
            observed_statistics[f] = observed_statistic
            null_statistics, null_power_spectrums = self.calculate_null_statistics(f)
            p_values[f] = (np.sum(null_statistics >= observed_statistic) + 1) / (self.n_permutations+1)

        return p_values, observed_statistics, observed_power_spectrum, null_power_spectrums

    def calculate_observed_statistic(self, f):
        if self.frequencies is None:
            self.frequencies, power_spectrum = self._estimate_power_spectrum(self.timeseries)
        else:
            _, power_spectrum = self._estimate_power_spectrum(self.timeseries)
        power = self._estimate_power(self.timeseries, f)

        return power, power_spectrum

    def calculate_null_statistics(self, f):
        null_power_spectrums = []
        null_statistics = []
        for i in range(self.n_permutations):
            y_shuffle = np.random.permutation(self.timeseries.copy())
            null_power_spectrums.append(self._estimate_power_spectrum(y_shuffle)[1])
            null_statistics.append(self._estimate_power(y_shuffle, f))

        return null_statistics, null_power_spectrums

    def _estimate_power_spectrum(self, ts):
        frequencies, power_spectrum = welch(ts, self.fs, nperseg=self.nperseg)

        return (frequencies, power_spectrum)

    def _estimate_power(self, ts, f):
        frequencies, power_spectrum = self._estimate_power_spectrum(ts)
        return np.interp(f, frequencies, power_spectrum)

def psd_analyze_rois(pkl_handler, f_type, test_frequencies, n_bootstraps, TR, n_permutations=1000, nperseg=580, rephase=False, frequency_grid=None, rephase_with=None, get_bootstrapped_psds=False,pure_f=True):
    """
    `f_type` is the type of roi [f1,f2,f1&f2]
    """

    from collections import defaultdict

    # Perform statistics on all bootstrapped-roi-level timeseries
    bootstrapped_means = []
    bootstrapped_psds = []
    bootstrapped_statistics = defaultdict(list)
    for bootstrap_id in range(n_bootstraps):
        tps, mean_bold = pkl_handler.extract_bootstrapped_mean_from_data(bootstrap_id, f_type, rephase=rephase, rephase_with=rephase_with, pure_f=pure_f)
        bootstrapped_means.append(mean_bold) # track
        ts = TimeSeries(mean_bold, TR, n_permutations=n_permutations, nperseg=nperseg)
        p_values, observed_statistics, observed_power_spectrum, null_power_spectrums = ts.process(test_frequencies)
        bootstrapped_psds.append(observed_power_spectrum)
        for test_f in test_frequencies:
            bootstrapped_statistics[f"test-{test_f}"].append((p_values[test_f], observed_statistics[test_f]))

    # Compute mean timeseries across all bootstrapped-roi-level timeseries
    bootstrapped_means = np.vstack(bootstrapped_means)
    y_bootstrapped_mean = bootstrapped_means.mean(0)
    # Compute statistics
    ts = TimeSeries(y_bootstrapped_mean, TR, n_permutations=n_permutations, nperseg=nperseg)
    p_values, observed_statistics, observed_power_spectrum, null_power_spectrums = ts.process(test_frequencies)

    if frequency_grid is None:
        frequency_grid = ts.frequencies
    else:
        assert np.allclose(frequency_grid, ts.frequencies, rtol=1e-05, atol=1e-08)

    if get_bootstrapped_psds:
        return (
            frequency_grid,
            np.vstack(bootstrapped_psds),
            observed_power_spectrum,
        )
    else:
        return (
            frequency_grid, 
            observed_statistics, 
            observed_power_spectrum, 
            null_power_spectrums, 
            p_values, 
            bootstrapped_statistics,
        )

def psd_store_data_in_dict(
    experiment_id,
    sub_id,
    roi_task_id,
    roi_pval,
    roi_fractional_overlap,
    roi_f_type,
    rephase,
    rephase_with,
    im_test_frequencies_map,
    data_dict,
    n_bootstraps,
    loaded_data_dict = None
):
    if loaded_data_dict is None:
        group_data_dict = defaultdict(list)
    else:
        group_data_dict = loaded_data_dict
    group_data_dict["experiment_id"].append(experiment_id)
    group_data_dict["sub_id"].append(sub_id)
    group_data_dict["roi_task_id"].append(roi_task_id)
    group_data_dict["roi_pval"].append(roi_pval)
    group_data_dict["roi_fractional_overlap"].append(roi_fractional_overlap)
    group_data_dict["roi_f_type"].append(roi_f_type)
    group_data_dict["rephase"].append(rephase)
    group_data_dict["rephase_with"].append(rephase_with)

    for metric_f_type, metric_f_value in im_test_frequencies_map.items():
        group_data_dict[metric_f_type].append(metric_f_value)
        group_data_dict[f"power_{metric_f_type}"].append(data_dict['observed_statistics'][metric_f_value])
        group_data_dict[f"pval_{metric_f_type}"].append(-np.log10(data_dict['p_values'][metric_f_value]))
        group_data_dict[f"bootstrap_power_{metric_f_type}"].append(np.median([i[1] for i in data_dict['bootstrapped_statistics'][f'test-{metric_f_value}']]))
        group_data_dict[f"bootstrap_pval_{metric_f_type}"].append(np.sum([-np.log10(i[0]) > -np.log10(.05) for i in data_dict['bootstrapped_statistics'][f'test-{metric_f_value}']]) / n_bootstraps)

    group_data_dict["power_spectrum"].append(data_dict["observed_power_spectrum"])

    return group_data_dict

def plot_power_spectrum(frequency_grid, observed_power_spectrum, null_power_spectrums, n_permutations, frequencies, p_values, observed_statistics, add_im=False, sub_id=None, roi_frequency=None, close_figure=False, png_out=None):
    fig, ax = plt.subplots(figsize=(2,1), dpi=400)
    ax.plot(frequency_grid, observed_power_spectrum, c='k', zorder=2, lw=.5)
    null_power_spectrums = np.vstack(null_power_spectrums)
    null_power_spectrum = np.mean(null_power_spectrums, axis=0)
    std_dev_values = np.std(null_power_spectrum, axis=0)
    confidence_interval = 1.96 * std_dev_values / np.sqrt(n_permutations)
    ax.fill_between(
        frequency_grid, 
        null_power_spectrum - confidence_interval, null_power_spectrum + confidence_interval,
        color='r', 
        alpha=.8,
    )
    fig, ax = _decorate_fig_power_spectrum(fig, ax, frequencies, p_values, observed_statistics, add_im=add_im, sub_id=sub_id, roi_frequency=roi_frequency)

    fig.tight_layout()

    if png_out:
        fig.savefig(png_out,dpi='figure')

    if close_figure:
        plt.close()

def _decorate_fig_power_spectrum(fig, ax, frequencies, p_values, observed_statistics, add_im=False, sub_id=None, roi_frequency=None, fontsize=4):
    for f in frequencies:
        ax.text(f+.005, observed_statistics[f], f"p={-np.log10(p_values[f]):.3f}", fontsize=fontsize)
    if add_im:
        _frequencies = frequencies.copy()
        second_order_frequencies = [
            np.abs(frequencies[0]-frequencies[1]), 
            np.abs(frequencies[1]+frequencies[0]),
            frequencies[0]*2,
            frequencies[1]*2,
        ]
        third_order_frequencies = [
            np.abs(2*frequencies[0] - frequencies[1]),
            np.abs(2*frequencies[1] - frequencies[0]),
        ]
        _frequencies += second_order_frequencies
        _frequencies += third_order_frequencies
    else:
        _frequencies = frequencies
    for f in _frequencies:
        if f not in second_order_frequencies and f not in third_order_frequencies:
            c = 'b'
        elif f in second_order_frequencies:
            c = 'cyan'
        elif f in third_order_frequencies:
            c = 'g'
        else: 
            raise ValueError(f"{f} not identified as a harmonic.")
        ax.axvline(x=f, c=c, linestyle=':', zorder=1, lw=.75)
    ax.set_xlim((0,.5))
    ax.set_ylabel("Power", fontsize=fontsize)
    ax.set_xlabel("Frequency", fontsize=fontsize)
    ax.tick_params(axis="both", length=0, labelsize=fontsize)
    for i in ("top", "right", "bottom", "left"):
        ax.spines[i].set_visible(False)
    ax.set_title(f"{sub_id}, roi-{roi_frequency}, {frequencies}", fontsize=fontsize)

    return fig, ax