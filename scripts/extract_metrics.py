from typing import Optional
from typing import List, Tuple
import typer

from pathlib import Path
from extract_metrics_utils import (
    filter_parcel_name,
    FLAExtractZScore,
    HCP_ROIS
)

import pandas as pd


app = typer.Typer()

@app.command()
def test():
    pass

@app.command()
def extract_metrics(
    out_pkl: Path,
    top_percentile: float,
    z_threshold: float,
    data_list: List[str],
    hcp_roi_dir: str = "/scratch/tmp"
):

    print(f"Number of Z maps: {len(data_list)}")

    # Initialize dataframe
    df_columns = [
        'experiment_id',
        'mri_id',
        'truncate_id',
        'dm_id',
        'smooth_id',
        'sub_id',
        'ses_id',
        'task_id',
        'run_id',
        'frequency_tag',
        'roi',
        'n_dpvs',
        'median',
    ]
    df = pd.DataFrame(columns = df_columns)

    # Set all HCP ROI dscalar paths
    parcel_dscalars = {}
    for parcel_name in HCP_ROIS:
        parcel_dscalars[parcel_name] = f"{hcp_roi_dir}/{parcel_name}.dscalar.nii"
        assert Path(parcel_dscalars[parcel_name]).exists()


    # Loop over all data (all data are z_score maps)
    for ix, z_score_path in enumerate(data_list):

        if ix % 5 == 0:
            print(f"[Progress] {str(ix).zfill(4)}/{str(len(data_list)).zfill(4)}")

        data_extractor = FLAExtractZScore(z_score_path)
        
        for parcel_name, parcel_path in parcel_dscalars.items():
            if not filter_parcel_name(parcel_name):
                continue
            # Extract metrics
            n_dpvs = data_extractor.get_threshold_count(z_threshold,parcel_path)
            median_val = data_extractor.get_median(top_percentile,parcel_path)
            add_df = pd.DataFrame(
                [[
                    data_extractor.experiment_id,
                    data_extractor.mri_id,
                    data_extractor.truncate_id,
                    data_extractor.dm_id,
                    data_extractor.smooth_id,
                    data_extractor.sub_id,
                    data_extractor.ses_id,
                    data_extractor.task_id,
                    data_extractor.run_id,
                    data_extractor.frequency_tag,
                    parcel_name,
                    n_dpvs,
                    median_val
                ]],
                columns = df_columns,
            )
            df = pd.concat([df, add_df], ignore_index=True)

    print(f"Saving:\n {out_pkl.stem}")
    df.to_pickle(out_pkl)

if __name__ == "__main__":
    app()