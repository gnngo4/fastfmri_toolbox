import typer
import time

import numpy as np
import pickle
from collections import defaultdict

from sklearn.metrics import roc_curve
from pathlib import Path
from im_sc_utils import (
    get_binary_paths,
    convert_multimap_to_fo_maps,
    process_maps,
    compute_metrics,
    generate_surrogates,
    convert_array_to_top_n_values,
    compute_metrics_over_surrogates,
    contains_only_0_1,
    area_under_curve,
    N_VERTICES_AFTER_RESAMPLING,
    GEODESIC_LH,
    GEODESIC_RH,
    DSCALAR_TEMPLATE,
)

app = typer.Typer()

@app.command()
def im_sc(
    experiment_id: str,
    mri_id: str,
    sub_id: str,
    roi_task_id: str,
    im_str: str,
    f1: float,
    f2: float,
    fim: float,
    corr_type: str,
    constant_fo: float = .8,
    fo_spacing: float = .05,
    N_SURROGATES: int = 1000
):

    mosaic = ["IM","f1_only","f2_only","f1","f2","f1&f2","f1&f2_dil2mm"]

    # Directories
    scratch_dir = Path("/scratch/fastfmri")
    surrogatedir = scratch_dir / "surrogate_maps"
    if not surrogatedir.exists():
        surrogatedir.mkdir(exist_ok=True, parents=True)
    pickledir = surrogatedir / "pickles"
    if not pickledir.exists():
        pickledir.mkdir(exist_ok=True, parents=True)
    null_auc_dir = surrogatedir / "null_auc"
    if not null_auc_dir.exists():
        null_auc_dir.mkdir(parents=True, exist_ok=True)

    for i in [GEODESIC_LH, GEODESIC_RH, DSCALAR_TEMPLATE]:
        assert Path(i).exists(), f"{i} not found."

    # Start timer
    start_time = time.time()

    # Get all paths for analysis
    path_dict = get_binary_paths(experiment_id, mri_id, sub_id, roi_task_id, f1, f2, fim, corr_type)

    # Get FO maps for f1, f2, f_im, an whole brain mask
    fo_maps = convert_multimap_to_fo_maps(path_dict, dscalar_template=DSCALAR_TEMPLATE)

    # Process whole brain mask
    wb_coverage_mask = (fo_maps['mask'] == 1).astype(int)
    mask_coords = wb_coverage_mask == 1 # Coordinates

    # Generate f_im map (ground truth)
    im_roi = (fo_maps["f_im"] >= constant_fo).astype(int)
    # Generate predicted maps under the same conditions as the ground truth ROI, then compute dice and jaccard metrics
    dice_metrics = {}
    jaccard_metrics = {}
    constant_all_rois = process_maps(fo_maps, constant_fo)
    for k, v in constant_all_rois.items():
        metrics = compute_metrics(im_roi[mask_coords], v[mask_coords])
        dice_metrics[k] = metrics["dice"]
        jaccard_metrics[k] = metrics["jaccard"]
    # Get n vertices for task IM per hemi
    N_LH = im_roi[:N_VERTICES_AFTER_RESAMPLING].sum()
    N_RH = im_roi[N_VERTICES_AFTER_RESAMPLING:].sum()
    if N_LH + N_RH == 0:
        raise ValueError("No vertices found in the ROI.")

    # Generate SA-preserved surrogate maps of f_im fractional overlap maps
    basename = f"experiment-{experiment_id}_mri-{mri_id}_sub-{sub_id}_roi-task-{roi_task_id}_p-{corr_type}_im-{im_str}_n-{N_SURROGATES}"
    lh_surrogate_maps = generate_surrogates(fo_maps["f_im"], wb_coverage_mask, "lh", surrogatedir, basename, n_surrogates=N_SURROGATES)
    rh_surrogate_maps = generate_surrogates(fo_maps["f_im"], wb_coverage_mask, "rh", surrogatedir, basename, n_surrogates=N_SURROGATES)
    # Binarize surrogate maps
    for i in range(N_SURROGATES):
        lh_surrogate_map = convert_array_to_top_n_values(lh_surrogate_maps[i,:], N_LH)
        rh_surrogate_map = convert_array_to_top_n_values(rh_surrogate_maps[i,:], N_RH)
        if i == 0:
            surrogate_maps = np.concatenate((lh_surrogate_map, rh_surrogate_map)).astype(int)
        else:
            surrogate_maps = np.vstack((surrogate_maps, np.concatenate((lh_surrogate_map, rh_surrogate_map)).astype(int)))

    # Generate iid binary maps of f_im fractional overlap maps
    iid_maps = np.zeros_like(surrogate_maps)
    for i in range(N_SURROGATES):
        for hemi, n_vertices, hemi_mask_coords in zip(
            ["lh","rh"],
            [N_LH, N_RH],
            [mask_coords[:N_VERTICES_AFTER_RESAMPLING], mask_coords[N_VERTICES_AFTER_RESAMPLING:]]
        ):  
            indices = np.random.choice(hemi_mask_coords.sum(), n_vertices, replace=False)
            if hemi == "lh":
                sub_iid_maps = iid_maps[i,:N_VERTICES_AFTER_RESAMPLING][hemi_mask_coords]
                sub_iid_maps[indices] = 1
                iid_maps[i,:N_VERTICES_AFTER_RESAMPLING][hemi_mask_coords] = sub_iid_maps
            elif hemi == "rh":
                sub_iid_maps = iid_maps[i,N_VERTICES_AFTER_RESAMPLING:][hemi_mask_coords]
                sub_iid_maps[indices] = 1
                iid_maps[i,N_VERTICES_AFTER_RESAMPLING:][hemi_mask_coords] = sub_iid_maps
            else:
                raise ValueError(f"{hemi} not supported.")

    # Instantiate tpr and fpr dictionaries
    tpr_dict = defaultdict(list)
    fpr_dict = defaultdict(list)

    # Loop over `fo_range` to obtain ROC-curves
    fo_range = np.arange(0,1+fo_spacing,fo_spacing)
    for fo in fo_range:

        # Generate all maps used to assess spatial overlap with f_im
        all_rois = process_maps(fo_maps,fo)
        predicted_im_roi = all_rois["IM"]
        f1_roi = all_rois["f1"]
        f2_roi = all_rois["f2"]
        f1_f2_roi = all_rois["f1&f2"]
        f1_only_roi = all_rois["f1_only"]
        f2_only_roi = all_rois["f2_only"]
        f1_f2_dil_2mm_roi = all_rois["f1&f2_dil2mm"]

        # Check & verbose
        for roi_str, _roi in zip(
            ["Surrogates",f"IM {im_str}", "f1", "f2", "f1 & f2", "f1 only", "f2 only", "f1 & f2 (dilate 2mm)"],
            [surrogate_maps, predicted_im_roi, f1_roi, f2_roi, f1_f2_roi, f1_only_roi, f2_only_roi, f1_f2_dil_2mm_roi]
        ):
            roi_shape_idx = 0
            if len(_roi.shape) == 2:
                roi_shape_idx = 1
            assert contains_only_0_1(_roi), f"Unique values in roi: {np.unique(_roi)}"
            assert _roi.shape[roi_shape_idx] == N_VERTICES_AFTER_RESAMPLING * 2, f"Expected vertex count: {N_VERTICES_AFTER_RESAMPLING * 2} [{_roi.shape[roi_shape_idx]}]"

        all_rois = [predicted_im_roi,f1_only_roi,f2_only_roi,f1_roi,f2_roi,f1_f2_roi,f1_f2_dil_2mm_roi]
        # Fill TPR and FPR list for all ROIs
        for roi_str, _roi in zip(mosaic, all_rois):
            im_labels = im_roi[mask_coords] # Ground truth
            predicted_labels = _roi[mask_coords] # Predicted
            fpr, tpr, _ = roc_curve(im_labels, predicted_labels)
            tpr_dict[roi_str].append(tpr[1])
            fpr_dict[roi_str].append(fpr[1])

    # Compute metrics over surrogates
    # Instantiate surrogate metrics
    iid_dice_metrics = {}
    surrogate_dice_metrics = {}
    iid_jaccard_metrics = {}
    surrogate_jaccard_metrics = {}
    iid_auc_metrics = defaultdict(list)
    surrogate_auc_metrics = defaultdict(list)
    # Compute dice and jaccard
    for k, v in constant_all_rois.items():
        iid_metrics = compute_metrics_over_surrogates(iid_maps, v, mask_coords)
        surrogate_metrics = compute_metrics_over_surrogates(surrogate_maps, v, mask_coords)
        iid_dice_metrics[k] = iid_metrics["dice"]
        iid_jaccard_metrics[k] = iid_metrics["jaccard"]
        surrogate_dice_metrics[k] = surrogate_metrics["dice"]
        surrogate_jaccard_metrics[k] = surrogate_metrics["jaccard"]
    # Compute auc for each surrogate map 
    # Loop over `fo_range` to obtain ROC-curves
    null_auc_pkl = null_auc_dir / f"{basename}_null_auc.pkl"
    if null_auc_pkl.exists():
        with open(null_auc_pkl, 'rb') as f:
            null_auc_arr = pickle.load(f)
        iid_auc_metrics = null_auc_arr["iid"]
        surrogate_auc_metrics = null_auc_arr["surrogate"]
    else:
        for null_ix in range(N_SURROGATES):
            if null_ix % 10 == 0:
                print(f"[NULL] Computing AUC for {str(null_ix).zfill(4)}/{str(N_SURROGATES).zfill(4)}")
            iid_roi = iid_maps[null_ix,:]
            surrogate_roi = surrogate_maps[null_ix,:]
            iid_tpr_dict = defaultdict(list)
            iid_fpr_dict = defaultdict(list)
            surrogate_tpr_dict = defaultdict(list)
            surrogate_fpr_dict = defaultdict(list)
            for fo in fo_range:
                # Generate all maps used to assess spatial overlap with f_im
                all_rois = process_maps(fo_maps,fo)
                predicted_im_roi = all_rois["IM"]
                f1_roi = all_rois["f1"]
                f2_roi = all_rois["f2"]
                f1_f2_roi = all_rois["f1&f2"]
                f1_only_roi = all_rois["f1_only"]
                f2_only_roi = all_rois["f2_only"]
                f1_f2_dil_2mm_roi = all_rois["f1&f2_dil2mm"]
                all_rois = [predicted_im_roi,f1_only_roi,f2_only_roi,f1_roi,f2_roi,f1_f2_roi,f1_f2_dil_2mm_roi]
                # Fill TPR and FPR list for all ROIs
                for roi_str, _roi in zip(mosaic, all_rois):
                    predicted_labels = _roi[mask_coords] # Predicted
                    iid_labels = iid_roi[mask_coords] # Null distribution (iid) ground truth
                    iid_fpr, iid_tpr, _ = roc_curve(iid_labels, predicted_labels)
                    iid_tpr_dict[roi_str].append(iid_tpr[1])
                    iid_fpr_dict[roi_str].append(iid_fpr[1])
                    surrogate_labels = surrogate_roi[mask_coords] # Null distribution (surrogate) ground truth
                    surrogate_fpr, surrogate_tpr, _ = roc_curve(surrogate_labels, predicted_labels)
                    surrogate_tpr_dict[roi_str].append(surrogate_tpr[1])
                    surrogate_fpr_dict[roi_str].append(surrogate_fpr[1])
            for roi_str in mosaic:
                iid_auc_metrics[roi_str].append(area_under_curve(iid_fpr_dict[roi_str], iid_tpr_dict[roi_str]))
                surrogate_auc_metrics[roi_str].append(area_under_curve(surrogate_fpr_dict[roi_str], surrogate_tpr_dict[roi_str]))
        save_auc_metrics = {"iid":iid_auc_metrics, "surrogate":surrogate_auc_metrics}
        with open(null_auc_pkl, 'wb') as f:
            pickle.dump(save_auc_metrics, f)

    # End timer
    end_time = time.time()
    elapsed_time = (end_time - start_time) / 60
    print(f"Elapsed Time: {elapsed_time:.2f} mins")

if __name__ == "__main__":
    app()
