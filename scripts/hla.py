from typing import Optional
from typing import List, Tuple
import typer
import time
import os

from pathlib import Path
from hla_utils import (
    search,
    cifti_separate,
    merge_niftis,
    merge_metrics,
    get_surface_vertex_areas,
    get_LR_gii_pairs
)

import numpy as np

app = typer.Typer()

@app.command()
def session_level():
    pass

@app.command()
def session_level_entrain_gt_control(
    experiment_id: str,
    time_window_id: str,
    design_matrix_id: str,
    mri_id: str,
    fla_dir: str,
    out_dir: str,
    sub_id: str,
    ses_id: str,
    n_permutations: int,
    search_frequencies: List[float],
):

    # Logging
    print(
        f" Experiment ID: {experiment_id}\n",
        f"Time Window ID: {time_window_id}\n"
        f"Design Matrix ID: {design_matrix_id}\n"
        f"MRI ID: {mri_id}\n",
        f"first-level analysis directory: {fla_dir}\n",
        f"output directory: {out_dir}\n",
        f"subject ID: {sub_id}\n",
        f"session ID: {ses_id}\n",
        f"number of permutations: {n_permutations}\n"
        f"search frequencies: {search_frequencies} Hz\n",
    )

    # Start the timer
    start_time = time.time()
    
    # Loop over search frequencies
    for ix, _frequency in enumerate(search_frequencies):
        
        # Grab all ciftis for one-sample t-test
        ciftis, n_runs = {}, {}
        temp_files = []
        for task_id in ['entrain', 'control']:
            _ciftis = search(
                fla_dir,
                f"{time_window_id}/{design_matrix_id}/sub-{sub_id}/ses-{ses_id}/task-{task_id}*/run-*/GLM/sub-{sub_id}_ses-{ses_id}_task-{task_id}*_run-*_frequency*{_frequency}_z_score.dscalar.nii"
            )
            _ciftis.sort()
            ciftis[task_id] = _ciftis
            n_runs[task_id] = len(ciftis[task_id])
        
        # Set-up directory
        _out_dir=Path(f"{out_dir}/{time_window_id}/{design_matrix_id}/sub-{sub_id}/ses-{ses_id}")
        if not _out_dir.exists():
            _out_dir.mkdir(parents=True)

        # Skip if processed already
        """
        processed_flag = Path(f"{_out_dir}/sub-{sub_id}/ses-{ses_id}/task-{task_id}/run-{run_id}/GLM").exists()
        if processed_flag:
            print("Skipping.")
            continue
        """

        # Create design matrix (.csv)
        design_matrix = np.zeros((sum(n_runs.values()), 2))
        idx1, idx2 = 0, 0
        for ix, _n_runs in enumerate(n_runs.values()):
            if ix > 0:
                idx1 += _n_runs
            idx2 += _n_runs
            design_matrix[idx1:idx2, ix] = 1        
        dm_csv = f"{_out_dir}/frequency-{_frequency}_designmatrix.csv"
        with open(dm_csv, 'w') as f:
            for ix, i in enumerate(range(design_matrix.shape[0])):
                f.write(','.join([str(j) for j in design_matrix[i,:]]))
                if ix == design_matrix.shape[0] - 1:
                    continue
                f.write('\n')

        # Create contrast (.csv)
        contrast_matrix = np.zeros((3,2))
        # Two sample t-test [entrain>control]
        contrast_matrix[0,0] = 1
        contrast_matrix[0,1] = -1
        contrast_matrix[1,0] = 1 # one sample t-test [entrain]
        contrast_matrix[2,1] = 1 # one sample t-test [control]
        contrast_csv = f"{_out_dir}/frequency-{_frequency}_contrast.csv"
        with open(contrast_csv, 'w') as f:
            for ix,i in enumerate(range(contrast_matrix.shape[0])):
                f.write(','.join([str(j) for j in contrast_matrix[i,:]]))
                if ix == contrast_matrix.shape[0] - 1:
                    continue
                f.write('\n')

        # Concatenate all niftis across task conditions into a single list
        # Note: they must be ordered according to the design matrix
        listify_ciftis = ciftis['entrain'] + ciftis['control']

        # Separate ciftis
        listify_sep_ciftis = {}
        for ix, dscalar in enumerate(listify_ciftis):
            separated_dscalar = cifti_separate(dscalar)
            for k, v in separated_dscalar.items():
                if ix == 0:
                    listify_sep_ciftis[k] = [v]
                else:
                    listify_sep_ciftis[k].append(v)
                temp_files.append(v)

        # Merge separated ciftis
        merged_sep_cifti = {}
        merged_sep_cifti['volume'] = merge_niftis(listify_sep_ciftis['volume'])
        merged_sep_cifti['cortex_left'] = merge_metrics(listify_sep_ciftis['cortex_left'])
        merged_sep_cifti['cortex_right'] = merge_metrics(listify_sep_ciftis['cortex_right'])
        for k, v in merged_sep_cifti.items():
            temp_files.append(v)

        # Get surface and corresponding surface-vertex-area files
        surfaces = {
            'cortex_left': "/scratch/surfaces/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii", 
            'cortex_right': "/scratch/surfaces/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"
        }
        surface_areas = get_surface_vertex_areas(
            surf_left = surfaces['cortex_left'],
            surf_right = surfaces['cortex_right']
        )
        for k, v in surface_areas.items():
            temp_files.append(v)

        # Verbose
        print(f'RUNNING: Frequency - {_frequency}')
        for _cifti in listify_ciftis:
            print(f"   - {_cifti}")

        # Run palm thrice, (1) volume, (2) CORTEX_LEFT, (3) CORTEX_RIGHT
        # Volume
        """
        #os.system(f"""#/opt/PALM/run_palm.sh /opt/matlab \
        #-i {merged_sep_cifti['volume']} \
        #-d {dm_csv} \
        #-t {contrast_csv} \
        #-o {_out_dir}/frequency-{str(_frequency).replace('.','')}_subcortical \
        #-n {n_permutations} \
        #-T -logp""")
        # Cortex Left
        os.system(f"""/opt/PALM/run_palm.sh /opt/matlab \
-i {merged_sep_cifti['cortex_left']} \
-s {surfaces['cortex_left']} {surface_areas['cortex_left']} \
-d {dm_csv} \
-t {contrast_csv} \
-o {_out_dir}/sub-{sub_id}_ses-{ses_id}_frequency-{str(_frequency).replace('.','')}_CortexLeft \
-n {n_permutations} \
-T -tfce2D -logp""")
        # Cortex Right
        os.system(f"""/opt/PALM/run_palm.sh /opt/matlab \
-i {merged_sep_cifti['cortex_right']} \
-s {surfaces['cortex_right']} {surface_areas['cortex_right']} \
-d {dm_csv} \
-t {contrast_csv} \
-o {_out_dir}/sub-{sub_id}_ses-{ses_id}_frequency-{str(_frequency).replace('.','')}_CortexRight \
-n {n_permutations} \
-T -tfce2D -logp""")
        
        # Remove all contrast files
        for f in temp_files:
            os.remove(f)

    # Clean-up to create CIFTI outputs
    # Find all L/R gifti pairs
    gii_pairs = get_LR_gii_pairs(
        search(_out_dir, "*.gii")
    )
    for p in gii_pairs:
        # Create .dscalar.nii
        L_gii, R_gii = p[0], p[1]
        combined_dscalar = L_gii.replace("_CortexLeft","").replace(".gii",".dscalar.nii")
        cmd = f"""wb_command -cifti-create-dense-from-template \
{dscalar} {combined_dscalar} \
-metric CORTEX_LEFT {L_gii} \
-metric CORTEX_RIGHT {R_gii}
"""
        # Run
        os.system(cmd)
        # Cleanup
        os.remove(L_gii)
        os.remove(R_gii)

    # Calculate the elapsed time in seconds
    elapsed_time = (time.time() - start_time) / 60
    print(f"Elapsed time: {elapsed_time:.2f} mins")

if __name__ == "__main__":
    app()