from typing import Optional
from typing import List, Tuple
import typer
import time

from pathlib import Path
from fla_utils import (
    search,
    PathLoader,
    REGRESSOR_COMBINATIONS,
    build_design_matrix,
    run_glm,
    extract_glm_metrics,
)

app = typer.Typer()

@app.command()
def run_level(
    experiment_id: str,
    mri_id: str,
    oscprep_dir: str,
    out_dir: str,
    sub_id: str,
    ses_id: str,
    task_id: str,
    run_id: str,
    search_frequencies: List[float],
    time_window: Tuple[float, float]
):

    # Logging
    print(
        f" Experiment ID: {experiment_id}\n",
        f"MRI ID: {mri_id}\n",
        f"oscprep directory: {oscprep_dir}\n",
        f"output directory: {out_dir}\n",
        f"subject ID: {sub_id}\n",
        f"session ID: {ses_id}\n",
        f"task ID: {task_id}\n",
        f"run ID: {run_id}\n",
        f"search frequencies: {search_frequencies} Hz\n",
        f"time window: {time_window} secs"
    )

    # Loop over various design matrix schemes
    for experiment_ix, dm_type in enumerate(REGRESSOR_COMBINATIONS.keys()):
        
        print(f"\n[{str(experiment_ix).zfill(2)}/{str(len(REGRESSOR_COMBINATIONS)).zfill(2)}] Design matrix: {dm_type}")

        # Start the timer
        start_time = time.time()
        
        # Set-up directory
        _out_dir=f"{out_dir}/{str(experiment_ix).zfill(2)}_experiment-{dm_type}"
        
        # Create figures directory
        figures_dir = Path(f"{_out_dir}/figures")
        if not figures_dir.exists():
            figures_dir.mkdir(parents=True)
        
        # Skip if processed already
        processed_flag = Path(f"{_out_dir}/sub-{sub_id}/ses-{ses_id}/task-{task_id}/run-{run_id}/GLM").exists()
        if processed_flag:
            print("Skipping.")
            continue

        # Instantiate path object
        path_loader = PathLoader(
            oscprep_dir,
            sub_id, 
            ses_id,
            task_id,
            run_id
        )

        # Build and save design matrix
        design_matrix, dm_fig = build_design_matrix(
            path_loader,
            time_window,
            search_frequencies,
            dm_type,
            show_flag = False
        )
        dm_fig.savefig(f"{figures_dir}/sub-{path_loader.sub_id}_ses-{path_loader.ses_id}_task-{path_loader.task_id}_run-{path_loader.run_id}_DM.png")

        # Fit and run frequency-based GLM on NIFTI and CIFTI data
        image_type = 'CIFTI'
        run_glm(
            path_loader,
            time_window, 
            search_frequencies, 
            image_type, 
            design_matrix, 
            _out_dir
        )

        # Calculate the elapsed time in seconds
        elapsed_time = (time.time() - start_time) / 60
        print(f"Elapsed time: {elapsed_time:.2f} mins [Design matrix shape: {design_matrix.shape}]")

if __name__ == "__main__":
    app()