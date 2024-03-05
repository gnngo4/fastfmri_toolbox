from typing import Optional
from typing import List, Tuple, Union, Literal
import typer
import time
import fcntl

from pathlib import Path
from fla_utils import (
    search,
    PathLoader,
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
    time_window: Tuple[float, float],
    smooth_mm: int = 0,
    denoise_only: bool = False,
    nordic_dir: str = 'None',
    image_type: str = 'CIFTI',
):

    assert image_type in ["NIFTI", "CIFTI"]

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
        f"time window: {time_window} secs\n",
        f"smooth (mm): {smooth_mm}\n",
        f"denoise_only: {denoise_only}\n",
        f"nordic_dir: {nordic_dir}\n",
        f"image_type: {image_type}\n",
    )

    if denoise_only:
        search_frequencies = [] # Force empty
        from fla_utils import REGRESSOR_COMBINATIONS_AGGR as REGRESSOR_COMBINATIONS
    else:
        from fla_utils import REGRESSOR_COMBINATIONS_SOFT as REGRESSOR_COMBINATIONS

    # Loop over various design matrix schemes
    for experiment_ix, dm_type in enumerate(REGRESSOR_COMBINATIONS.keys()):
        
        print(f"\n[{str(experiment_ix).zfill(2)}/{str(len(REGRESSOR_COMBINATIONS)).zfill(2)}] Design matrix: {dm_type}")

        # Start the timer
        start_time = time.time()
        
        # Set-up directory
        _out_dir=f"{out_dir}/{str(experiment_ix).zfill(2)}_experiment-{dm_type}"
        
        # Create figures directory
        lock_file = Path(f"/scratch/{experiment_id}_{mri_id}_truncate-{time_window[0]}-{time_window[1]}_experiment-{dm_type}.lock") # Define the lock file path
        # try to acquire lock
        with open(lock_file, "w") as lock:
            while True:
                try:
                    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    break
                except BlockingIOError:
                    time.sleep(.1)
            figures_dir = Path(f"{_out_dir}/figures") # Create directory
            if not figures_dir.exists():
                try:
                    figures_dir.mkdir(parents=True)
                except FileExistsError:
                    pass
                except Exception as e:
                    print(f"Error creating figures directory: {e}")
            fcntl.flock(lock, fcntl.LOCK_UN)

        if not figures_dir.exists():
            try:
                lock_file = figures_dir / ".lock"
                with open(lock_file, "w") as f:
                    fcntl.flock(f, fcntl.LOCK_EX)  # Acquire an exclusive lock
                    if not figures_dir.exists():
                        figures_dir.mkdir(parents=True)
                    fcntl.flock(f, fcntl.LOCK_UN)  # Release the lock
            except Exception as e:
                print(f"Error creating figures directory: {e}")

        
        # Skip if processed already
        GLM_DIR = Path(f"{_out_dir}/sub-{sub_id}/ses-{ses_id}/task-{task_id}/run-{run_id}/GLM")
        processed_flag = GLM_DIR.exists()
        if processed_flag:
            print(f"Skipping.\n{GLM_DIR} exists.")
            continue

        # Instantiate path object
        path_loader = PathLoader(
            oscprep_dir,
            sub_id, 
            ses_id,
            task_id,
            run_id,
            nordic_dir = nordic_dir
        )

        # Build and save design matrix
        design_matrix, dm_fig = build_design_matrix(
            path_loader,
            time_window,
            search_frequencies,
            dm_type,
            show_flag = False,
            denoise_only = denoise_only,
            nordic_dir = nordic_dir
        )
        print(f"DEBUGGING: {design_matrix.shape}")
        dm_fig.savefig(f"{figures_dir}/sub-{path_loader.sub_id}_ses-{path_loader.ses_id}_task-{path_loader.task_id}_run-{path_loader.run_id}_DM.png")

        # Fit and run frequency-based GLM on NIFTI and CIFTI data
        #image_type = 'CIFTI'
        run_glm(
            path_loader,
            time_window, 
            search_frequencies, 
            image_type, 
            design_matrix, 
            _out_dir,
            smooth_mm,
        )

        # Calculate the elapsed time in seconds
        elapsed_time = (time.time() - start_time) / 60
        print(f"Elapsed time: {elapsed_time:.2f} mins [Design matrix shape: {design_matrix.shape}]")

if __name__ == "__main__":
    app()