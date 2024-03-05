import glob
from pathlib import Path
import itertools
import nibabel as nib
import numpy as np
import pandas as pd
import subprocess
import pickle
from brainsmash.mapgen.sampled import Sampled
from brainsmash.mapgen.memmap import txt2memmap

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

N_LH_VERTICES = 29696
N_RH_VERTICES = 29716
N_VERTICES = N_LH_VERTICES + N_RH_VERTICES
N_VERTICES_AFTER_RESAMPLING = 32492
GEODESIC_LH = "/opt/app/notebooks/data/surfaces/S1200.L.midthickness_MSMAll.32k_fs_LR_geodesic.txt"
GEODESIC_RH = "/opt/app/notebooks/data/surfaces/S1200.R.midthickness_MSMAll.32k_fs_LR_geodesic.txt"
TEMPLATE_DSCALAR = "/opt/app/notebooks/data/dscalars/template.dscalar.nii"
for p in [GEODESIC_LH, GEODESIC_RH]:
    assert Path(p).exists(), f"{p} not found."

def filter_run_ids(run_ids):
    KEEP = ['01', '02', '03', 'IMTest', 'IMRetest']
    KEEP = [f"run-{i}" for i in KEEP]
    filtered_run_ids = []
    for run_id in run_ids:
        if run_id.split('/')[-1] in KEEP or "X" in run_id:
            filtered_run_ids.append(run_id)

    return filtered_run_ids

def search(base_dir, wildcard, error=True):
    search_path = Path(base_dir) / wildcard
    files = glob.glob(str(search_path))

    if not files:
        if error:
            raise FileNotFoundError(f"No files were found in: {search_path}")
        else:
            return []

    return files

def convert_array_to_top_n_values(X, n):

    top_indices = np.argsort(X)[-n:]
    boolean_array = np.full_like(X, False, dtype=bool)
    boolean_array[top_indices] = True

    return boolean_array

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
        
    X = np.array(nib.load(dscalar).get_fdata())
    cortex_data = X[0, :N_VERTICES][np.newaxis,:]
    subcortex_data = X[0, N_VERTICES:][np.newaxis,:]
    if "z_score" in str(dscalar):
        Z_THR = 3.0
        print(f"N-grayordinates, Z>{Z_THR}\nCortex: {(cortex_data>Z_THR).sum()}\nSubcortex: {(subcortex_data>Z_THR).sum()}")

    return cortex_data


def get_maps_for_a_run_id(mri_id, smooth_id, truncate_id, scratch_dir, sub_id, task_suffix, run_id, f_1, f_2, f_im, metric_type="z_score", task_base="entrain", experiment_id="1_frequency_tagging"):
    # Get data paths
    bootstrap_str = f"experiment-{experiment_id}_mri-{mri_id}_smooth-{smooth_id}_truncate-{truncate_id}_n-100_batch-00_desc-IMsubtraction_bootstrap"
    sub_dir = (
        scratch_dir / 
        bootstrap_str / 
        "first_level_analysis" / 
        sub_id / 
        "ses-main" / 
        f"task-{task_base}{task_suffix}" /
        run_id /
        "GLM"
    )
    z_score_maps = {
        "f1": sub_dir / f"{sub_id}_ses-main_task-{task_base}{task_suffix}_{run_id}_frequency-{f_1}_{metric_type}.dscalar.nii",
        "f2": sub_dir / f"{sub_id}_ses-main_task-{task_base}{task_suffix}_{run_id}_frequency-{f_2}_{metric_type}.dscalar.nii",
        "IM": sub_dir / f"{sub_id}_ses-main_task-{task_base}{task_suffix}_{run_id}_frequency-{f_im}_{metric_type}.dscalar.nii",
    }
    for v in z_score_maps.values():
        assert v.exists(), f"{v} does not exist."

    return z_score_maps

def load_maps(run_ids, mri_id, smooth_id, truncate_id, scratch_dir, sub_id, task_suffix, f_1, f_2, f_im, metric_type="z_score", task_base="entrain", experiment_id="1_frequency_tagging",dscalar_template=None):
    f1_data, f2_data, im_data = {}, {}, {}
    for run_id in run_ids:
        z_score_maps = get_maps_for_a_run_id(mri_id, smooth_id, truncate_id, scratch_dir, sub_id, task_suffix, run_id, f_1, f_2, f_im, metric_type=metric_type, task_base=task_base, experiment_id=experiment_id)
        f1_data[run_id] = read_map(z_score_maps['f1'], dscalar_template=dscalar_template).flatten()
        f2_data[run_id] = read_map(z_score_maps['f2'], dscalar_template=dscalar_template).flatten()
        im_data[run_id] = read_map(z_score_maps['IM'], dscalar_template=dscalar_template).flatten()

    return f1_data, f2_data, im_data

def load_bootstrapped_stdev_zscore_maps(mri_id, smooth_id, truncate_id, scratch_dir, sub_id, task_suffix, f_1, f_2, f_im, n_batches, task_base="entrain", experiment_id="1_frequency_tagging", dscalar_template=None):
    data = {}
    for batch_id in range(n_batches):
        bootstrap_str = f"experiment-{experiment_id}_mri-{mri_id}_smooth-{smooth_id}_truncate-{truncate_id}_n-100_batch-{str(batch_id).zfill(2)}_desc-IMsubtraction_bootstrap"
        batch_dir = (
            scratch_dir / 
            bootstrap_str /
            sub_id / 
            "bootstrap"
        )
        for f in [f_1, f_2, f_im]:
            bootstrapped_z_scores = batch_dir / f"{sub_id}_ses-main_task-{task_base}{task_suffix}_f-{f}_data-train_n-100_z_score.dtseries.nii"
            assert bootstrapped_z_scores.exists(), f"{bootstrapped_z_scores} not found."
            _data = nib.load(bootstrapped_z_scores).get_fdata()
            if batch_id == 0:
                data[f] = _data
            else:
                data[f] = np.concatenate((data[f], _data), axis=0)

    data = {
        f_1: data[f_1].std(0),
        f_2: data[f_2].std(0),
        f_im: data[f_im].std(0),
    }
    for f, _data in data.items():
        img = nib.load(TEMPLATE_DSCALAR)
        new_data = np.zeros_like(img.get_fdata())
        new_data[0,:] = _data
        new_img = nib.Cifti2Image(new_data, header=img.header)
        tmp_dscalar = f"/tmp/bootstrapped_stdev.dscalar.nii"
        nib.save(new_img, tmp_dscalar)
        data[f] = read_map(tmp_dscalar, dscalar_template=dscalar_template, dscalar_placeholder="/tmp/read_map_tmp.dscalar.nii").flatten()
    
    return data[f_1], data[f_2], data[f_im]

def load_multitype_maps(run_ids, mri_id, smooth_id, truncate_id, scratch_dir, sub_id, task_suffix, f_1, f_2, f_im, metric_types, task_base='entrain',experiment_id='1_frequency_tagging', dscalar_template=None):
    f1_data, f2_data, im_data = {}, {}, {}
    for metric_type in metric_types:
        f1_data[metric_type], f2_data[metric_type], im_data[metric_type] = load_maps(
            run_ids, 
            mri_id, 
            smooth_id, 
            truncate_id, 
            scratch_dir, 
            sub_id, 
            task_suffix, 
            f_1, f_2, f_im,
            metric_type=metric_type,
            task_base=task_base,
            experiment_id=experiment_id,
            dscalar_template=dscalar_template,
        )
    
    return f1_data, f2_data, im_data

def process_mask(data_dict, correction_type = 'fwe'):
    stat_dict = data_dict["stat"]
    z_score_dict = data_dict["z_score"]
    p_value_dict = data_dict["p_value"]
    updated_mask = {}
    for k in stat_dict.keys():
        stat = stat_dict[k]
        zscore = z_score_dict[k]
        pvalue = p_value_dict[k]
        wholebrain_mask = stat > 0
        n_vertices = wholebrain_mask.sum()
        if correction_type == 'fdr':
            import statsmodels.stats.multitest as sm
            corrected_pvalue = np.zeros_like(wholebrain_mask).astype(float)
            qvalues = sm.multipletests(
                pvalue[wholebrain_mask], method='fdr_bh'
            )[1]
            corrected_pvalue[wholebrain_mask] = qvalues
        else:
            corrected_pvalue = n_vertices * pvalue
        corrected_pvalue_mask = corrected_pvalue < .05
        updated_mask[k] = corrected_pvalue_mask * wholebrain_mask
        if updated_mask[k].sum() == 0:
            print(f"Warning: 0 vertices detected in {k}")

    return updated_mask

def custom_cmap(cmap):
    
    import matplotlib.colors as mcolors
    # Define the custom colormap
    cmap = plt.get_cmap(cmap)
    new_colors = cmap(np.linspace(0, 1, 256))  # Create a copy of the 'magma' colormap
    # Set the color for zero values to white
    new_colors[0, :] = [1, 1, 1, 1]  # [R, G, B, A]
    # Create a new colormap using the modified colors
    new_cmap = mcolors.ListedColormap(new_colors)

    return new_cmap

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

def extract_data_from_f1_f2_vertices(run_id, f1_data, f2_data, im_data, z_thr, dilation=0, verbose=False, add_intersection=False):
    
    mask = (f1_data[run_id] >= z_thr).astype(int) + (f2_data[run_id] >= z_thr).astype(int)
    union_mask = mask >= 1
    intersection_mask = mask == 2
    if dilation > 0:
        dilated_intersection_mask = dilate_array_by_mm(intersection_mask, dilation)
        union_mask = (dilated_intersection_mask + mask) >= 1
    
    if verbose:
        print(f"{run_id}: Vertex count in union of f1/f2: {(union_mask).sum()}")
        print(f"{run_id}: Vertex count in intersection of f1/f2: {(intersection_mask).sum()}")
        if dilation > 0:
            print(f"{run_id}: Vertex count in dilated intersection of f1/f2: {(dilated_intersection_mask).sum()}")
    
    data_dict = {
        "f1": f1_data[run_id][union_mask],
        "f2": f2_data[run_id][union_mask],
        "IM": (im_data[run_id][union_mask] > z_thr).astype(int),
    }
    
    if add_intersection:
        data_dict["f1_f2_intersection"] = ((mask == 2)[union_mask]).astype(int)
    
    df = pd.DataFrame(data_dict)
    
    return union_mask, df

def predict_im_with_logistic_regression(f1_data, f2_data, im_data, train_run_ids, test_run_id, z_thr=3, random_state=42, shuffle_training_labels=False, verbose=False, add_intersection=False, dilation=0):
    
    mask = {}
    mask[test_run_id], test_df = extract_data_from_f1_f2_vertices(test_run_id, f1_data, f2_data, im_data, z_thr, verbose=verbose, add_intersection=add_intersection, dilation=dilation)
    
    for train_run_ix, train_run_id in enumerate(train_run_ids):
        if train_run_ix == 0:
            mask[train_run_id], train_df = extract_data_from_f1_f2_vertices(train_run_id, f1_data, f2_data, im_data, z_thr, add_intersection=add_intersection, dilation=dilation)
        else:
            mask[train_run_id], _train_df = extract_data_from_f1_f2_vertices(train_run_id, f1_data, f2_data, im_data, z_thr, add_intersection=add_intersection, dilation=dilation)
            train_df = pd.concat([train_df, _train_df], axis=0)
            train_df = train_df.reset_index(drop=True)

    X_train = train_df.drop("IM", axis=1)
    y_train = train_df["IM"]
    if shuffle_training_labels:
        print("Shuffling training labels...")
        y_train = pd.DataFrame(y_train, columns=["IM"]).sample(frac=1, random_state=random_state)["IM"]
    X_test = test_df.drop("IM", axis=1)
    y_test = test_df["IM"]

    model = LogisticRegression()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {}
    metrics['classification_report'] = classification_report(y_test, y_pred, labels=np.arange(2), output_dict=True)
    metrics['accuracy'] = accuracy_score(y_test, y_pred)

    y_pred_remapped = np.zeros((91282,))
    y_pred_remapped[mask[test_run_id]] = y_pred + 1

    y_ground_truth_remapped = np.zeros((91282,))
    y_ground_truth_remapped[mask[test_run_id]] = y_test + 1
    
    return metrics, y_pred_remapped, y_ground_truth_remapped

def predict_im_with_f1_f2_intersection(f1_data, f2_data, im_data, test_run_id, z_thr=3, verbose=False, dilation=0):

    mask = {}
    mask[test_run_id], test_df = extract_data_from_f1_f2_vertices(test_run_id, f1_data, f2_data, im_data, z_thr, dilation=dilation)
    if dilation > 0:
        y_pred = (f1_data[test_run_id] > z_thr).astype(int) + (f2_data[test_run_id] > z_thr).astype(int)
        y_pred = dilate_array_by_mm(y_pred==2, dilation)[mask[test_run_id]]
    else:
        y_pred = ((f1_data[test_run_id] > z_thr).astype(int) + (f2_data[test_run_id] > z_thr).astype(int))[mask[test_run_id]]
        y_pred = (y_pred==2).astype(int)

    y_test = test_df["IM"]

    if verbose:
        mask_count = (y_pred == 1).sum()
        no_mask_count = (y_pred == 0).sum()
        print(f"[{test_run_id}] Intersection mask with dilation [{dilation}] has a vertex count of [{mask_count}/{mask_count+no_mask_count}]")
        print(f"[{test_run_id}] Intermodulation mask with dilation [{dilation}] has a vertex count of [{y_test.sum()}]")

    metrics = {}
    metrics['classification_report'] = classification_report(y_test, y_pred, labels=np.arange(2), output_dict=True, zero_division=0)
    metrics['accuracy'] = accuracy_score(y_test, y_pred)

    y_pred_remapped = np.zeros((91282,))
    y_pred_remapped[mask[test_run_id]] = y_pred + 1

    y_ground_truth_remapped = np.zeros((91282,))
    y_ground_truth_remapped[mask[test_run_id]] = y_test + 1
    
    return metrics,  y_pred_remapped, y_ground_truth_remapped

def read_metrics(sub_id, run_id, task_suffix, experiment_id, data_dict, metrics):
    
    for label in ['0', '1']:
        for metric in ['precision', 'recall', 'f1-score']:
            data_dict['sub_id'].append(sub_id)
            data_dict['run_id'].append(run_id)
            data_dict['experiment_id'].append(experiment_id)
            data_dict['label'].append(label)
            data_dict['metric_id'].append(metric)
            data_dict['metric'].append(
                metrics['classification_report'][label][metric]
            )
            data_dict['task_suffix'].append(task_suffix)
    
    return data_dict

def check_txt2memmap_outputs(surrogatedir):
    exist_flag = True
    for i in ["distmat.npy", "index.npy"]:
        f = surrogatedir / i
        if not f.exists():
            exist_flag = False
            
    return exist_flag

def generate_surrogates(brainmap, wb_coverage_mask, hemi, parentdir, basedir, n_surrogates):
    
    assert hemi in ["lh", "rh"], f"hemi must be `lh` or `rh`."
    if hemi == 'lh':
        hemi_coverage_mask = wb_coverage_mask[:32492]
        hemi_brainmap = brainmap[:32492][hemi_coverage_mask==1]
        geodesic = GEODESIC_LH
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

def classification_report_for_surrogate_maps(y_true, surrogate_maps, n_surrogates, wb_coverage_mask, condense=False):

    report = {
        '0': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
        '1': {'precision': [], 'recall': [], 'f1-score': [], 'support': []},
    }
    for i in range(n_surrogates):
        y_pred = surrogate_maps[i,:][wb_coverage_mask==1]
        _report = classification_report(
            y_true, 
            y_pred, 
            labels=np.arange(2), 
            output_dict=True, 
            zero_division=0
        )
        for l, m in itertools.product(['0', '1'], ['precision', 'recall', 'f1-score', 'support']):
            report[l][m].append(_report[l][m])

    if condense:
        for l, m in itertools.product(['0', '1'], ['precision', 'recall', 'f1-score', 'support']):
            report[l][m] = np.mean(report[l][m])
    
    return report

def get_classification_reports(
    df, 
    surrogate_maps, 
    n_surrogates, 
    wb_coverage_mask, 
    ignore_control, 
    true_key = "f1_f2_intersection",
):
    # Entrain
    y_true = df[true_key]
    y_pred = df["im"]
    positive_report = classification_report(
        y_true, 
        y_pred, 
        labels=np.arange(2), 
        output_dict=True, 
        zero_division=0
    )
    # Control: Using control task paradigm
    if ignore_control:
        control_report = None
    else:
        y_pred = df["control_im"]
        control_report = classification_report(
            y_true, 
            y_pred, 
            labels=np.arange(2), 
            output_dict=True, 
            zero_division=0,
        )
    # Control: SA-preserving surrogate maps
    surrogate_map_report = classification_report_for_surrogate_maps(
        y_true, 
        surrogate_maps,
        n_surrogates,
        wb_coverage_mask,
        condense=True,
    )

    return positive_report, control_report, surrogate_map_report