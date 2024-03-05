from pathlib import Path
from typing import List

import nibabel as nib
import json

def get_id(idtype: str, nifti_path: Path) -> str:
    assert f"{idtype}-" in nifti_path.stem, f"{idtype}- not in {nifti_path.stem}"
    return nifti_path.stem.split(f"{idtype}-")[1].split('_')[0]

def update_json(k: str, _json: Path, updated_tr: float) -> None:
    # Load json
    with open(str(_json), 'r') as json_file:
        data = json.load(json_file)
    # Check key is in json's metadata
    assert k in data.keys(), f"{k} not in {_json}"
    # Update json
    data[k] = updated_tr
    with open(str(_json), 'w') as json_file:
        json.dump(data, json_file, indent=4)

def reorganize_vaso_niftis(
    basedir: Path, 
    sub_id: str, 
    ses_id: str, 
    task_id: str, 
    vaso_run_id: str, 
    bold_run_id: str,
    TR: float,
) -> None:
    vaso_mag_json = basedir / f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_dir-RL_run-{vaso_run_id}_part-mag_vaso.json"
    vaso_mag_nii = basedir / f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_dir-RL_run-{vaso_run_id}_part-mag_vaso.nii.gz"
    vaso_phase_json = basedir / f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_dir-RL_run-{vaso_run_id}_part-phase_vaso.json"
    vaso_phase_nii = basedir / f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_dir-RL_run-{vaso_run_id}_part-phase_vaso.nii.gz"
    bold_mag_json = basedir / f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_dir-RL_run-{bold_run_id}_part-mag_vaso.json"
    bold_mag_nii = basedir / f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_dir-RL_run-{bold_run_id}_part-mag_vaso.nii.gz"
    bold_phase_json = basedir / f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_dir-RL_run-{bold_run_id}_part-phase_vaso.json"
    bold_phase_nii = basedir / f"sub-{sub_id}_ses-{ses_id}_task-{task_id}_dir-RL_run-{bold_run_id}_part-phase_vaso.nii.gz"
    # Check if all fils exist.
    for f in [
        vaso_mag_json, vaso_mag_nii, vaso_phase_json, vaso_phase_nii,
        bold_mag_json, bold_mag_nii, bold_phase_json, bold_phase_nii
    ]:
        assert f.exists(), f"{f} does not exist."
    # Check vaso and bold nifti using a intensity-based heuristic
    vaso_mean = nib.load(vaso_mag_nii).get_fdata().mean()
    bold_mean = nib.load(bold_mag_nii).get_fdata().mean()
    assert vaso_mean < bold_mean, f"VASO intensity is expected to be lower than BOLD intensity."
    # Update json metadata: RepetitionTime
    for f in [vaso_mag_json, vaso_phase_json, bold_mag_json, bold_phase_json]:
        update_json("RepetitionTime", vaso_mag_json, TR)
    # Rename
    for f in [bold_mag_json, bold_mag_nii, bold_phase_json, bold_phase_nii]:
        new_path = f.with_name(f.name.replace("vaso.", "bold.").replace(f"run-{bold_run_id}", f"run-{vaso_run_id}"))
        f.rename(new_path)

def sort_vaso_data(list_paths: List[Path], TR: float):
    # Read all run ids
    run_info = []
    for _path in list_paths:
        sub_id = get_id("sub", _path)
        ses_id = get_id("ses", _path)
        run_id = get_id("run", _path)
        task_id = get_id("task", _path)
        run_info.append((sub_id, ses_id, run_id, task_id))
    # Remove repeat run_ids
    run_info = list(set(run_info))
    run_info = sorted(run_info, key=lambda x: x[2])
    run_info = [(run_info[i], run_info[i+1]) for i in range(0, len(run_info), 2)] # group runs into pairs 
    # Check if all files exist
    for vaso_info, bold_info in run_info:
        parent_dir = Path(list_paths[0]).parent
        vaso_sub_id, vaso_ses_id, vaso_run_id, vaso_task_id = vaso_info
        bold_sub_id, bold_ses_id, bold_run_id, bold_task_id = bold_info
        assert vaso_sub_id == bold_sub_id
        assert vaso_ses_id == bold_ses_id
        assert vaso_task_id == bold_task_id
        reorganize_vaso_niftis(
            parent_dir, 
            vaso_sub_id, 
            vaso_ses_id, 
            vaso_task_id, 
            vaso_run_id, 
            bold_run_id,
            TR,
        )