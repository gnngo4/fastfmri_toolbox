OSCPREP_DIR = "oscprep_grayords_fmapless"
SMOOTH_MM = 0
STIM_START_TIME = 14
STIM_RAMPUP_TIME = 25
CONTAINER = '/project/def-mmur/gngo4/containers/neuroimaging-notebook-v2.simg'
IMAGE_TYPE = 'CIFTI'

attention_settings = {
    "experiment_ids": ["1_attention"],
    "mri_ids": ["7T"], 
    "oscprep_dir": OSCPREP_DIR, 
    "smooth_mm": SMOOTH_MM,
    "relevant_task_ids": ("AttendAway", "AttendInF2", "AttendInF1", "AttendInF1F2"),
    "experiment_search_frequencies": {
        "AttendAway_7T": [.125, .2],
        "AttendInF2_7T": [.125, .2],
        "AttendInF1_7T": [.125, .2],
        "AttendInF1F2_7T": [.125, .2],
    },
    "TR": {
        "AttendAway_7T": .25,
        "AttendInF2_7T": .25,
        "AttendInF1_7T": .25,
        "AttendInF1F2_7T": .25,
    },
    "stim_start_time": STIM_START_TIME,
    "stim_rampup_time": STIM_RAMPUP_TIME,
    "experiment_stim_end": {
        "AttendAway_7T": STIM_START_TIME+205,
        "AttendInF2_7T": STIM_START_TIME+205,
        "AttendInF1_7T": STIM_START_TIME+205,
        "AttendInF1F2_7T": STIM_START_TIME+205,
    },
    "CONTAINER": CONTAINER,
    "image_type": IMAGE_TYPE,
}

frequency_tagging_settings = {
    "experiment_ids": ["1_frequency_tagging"],
    "mri_ids": ["7T", "3T"], 
    "oscprep_dir": OSCPREP_DIR, 
    "smooth_mm": SMOOTH_MM,
    "relevant_task_ids": ("localizer", "entrain", "control"),
    "experiment_search_frequencies": {
        "localizer_7T": [.2],
        "entrain_7T": [.2, .5],
        "control_7T": [.2, .5],
        "localizer_3T": [.125],
        "entrain_3T": [.125, .2],
        "control_3T": [.125, .2],
    },
    "TR": {
        "localizer_7T": .225,
        "entrain_7T": .225,
        "control_7T": .225,
        "localizer_3T": .3,
        "entrain_3T": .3,
        "control_3T": .3,
    },
    "stim_start_time": STIM_START_TIME,
    "stim_rampup_time": STIM_RAMPUP_TIME,
    "experiment_stim_end": {
        "localizer_7T": STIM_START_TIME+185,
        "entrain_7T": STIM_START_TIME+185,
        "control_7T": STIM_START_TIME+185,
        "localizer_3T": STIM_START_TIME+205,
        "entrain_3T": STIM_START_TIME+205,
        "control_3T": STIM_START_TIME+205,
    },
    "CONTAINER": CONTAINER,
    "image_type": IMAGE_TYPE,
}

def setting_exceptions(
    experiment_id, 
    mri_id, 
    sub_id, 
    settings,
):
    
    if experiment_id == "1_frequency_tagging" and mri_id == "7T" and sub_id == "sub-007":
        settings["relevant_task_ids"] = ("entrain", "control")

    if experiment_id == "1_frequency_tagging" and mri_id == "7T" and sub_id == "sub-005":
        settings["experiment_search_frequencies"] = {
            f"entrain_{mri_id}": [.2, .5],
            f"control_{mri_id}": [.2, .5],
            f"localizer_{mri_id}": [.125],
        }
        settings["TR"] = {
            f"entrain_{mri_id}": .3,
            f"control_{mri_id}": .3,
            f"localizer_{mri_id}": .3,
        }

    if experiment_id == "1_frequency_tagging" and mri_id == "7T" and sub_id in ["sub-020"]:
        settings["relevant_task_ids"] = ("entrainA", "entrainB", "entrainC")
        settings["experiment_search_frequencies"] = {
            f"entrainA_{mri_id}": [.125, .2],
            f"entrainB_{mri_id}": [.125, .175],
            f"entrainC_{mri_id}": [.125, .15],
        }
        settings["TR"] = {
            f"entrainA_{mri_id}": .3,
            f"entrainB_{mri_id}": .3,
            f"entrainC_{mri_id}": .3,
        }
        settings["experiment_stim_end"] = {
            f"entrainA_{mri_id}": 14+205,
            f"entrainB_{mri_id}": 14+205,
            f"entrainC_{mri_id}": 14+205,
        }
    
    if experiment_id == "1_frequency_tagging" and mri_id == "7T" and sub_id in ["sub-021"]:
        settings["relevant_task_ids"] = ("entrainD", "entrainE", "entrainF")
        settings["experiment_search_frequencies"] = {
            f"entrainD_{mri_id}": [.125, .2],
            f"entrainE_{mri_id}": [.150, .2],
            f"entrainF_{mri_id}": [.175, .2],
        }
        settings["TR"] = {
            f"entrainD_{mri_id}": .3,
            f"entrainE_{mri_id}": .3,
            f"entrainF_{mri_id}": .3,
        }
        settings["experiment_stim_end"] = {
            f"entrainD_{mri_id}": 14+205,
            f"entrainE_{mri_id}": 14+205,
            f"entrainF_{mri_id}": 14+205,
        }
    
    if experiment_id == "1_frequency_tagging" and mri_id == "3T" and sub_id in ["sub-020"]:
        settings["relevant_task_ids"] = ("entrainA", "entrainB", "entrainC")
        settings["experiment_search_frequencies"] = {
            f"entrainA_{mri_id}": [.125, .2],
            f"entrainB_{mri_id}": [.125, .175],
            f"entrainC_{mri_id}": [.125, .15],
        }
        settings["TR"] = {
            f"entrainA_{mri_id}": .3,
            f"entrainB_{mri_id}": .3,
            f"entrainC_{mri_id}": .3,
        }
        settings["experiment_stim_end"] = {
            f"entrainA_{mri_id}": 14+205,
            f"entrainB_{mri_id}": 14+205,
            f"entrainC_{mri_id}": 14+205,
        }
    
    if experiment_id == "1_frequency_tagging" and mri_id == "3T" and sub_id in ["sub-021"]:
        settings["relevant_task_ids"] = ("entrainD", "entrainE", "entrainF")
        settings["experiment_search_frequencies"] = {
            f"entrainD_{mri_id}": [.125, .2],
            f"entrainE_{mri_id}": [.150, .2],
            f"entrainF_{mri_id}": [.175, .2],
        }
        settings["TR"] = {
            f"entrainD_{mri_id}": .3,
            f"entrainE_{mri_id}": .3,
            f"entrainF_{mri_id}": .3,
        }
        settings["experiment_stim_end"] = {
            f"entrainD_{mri_id}": 14+205,
            f"entrainE_{mri_id}": 14+205,
            f"entrainF_{mri_id}": 14+205,
        }
    
    if experiment_id == "1_attention" and mri_id == "7T" and sub_id.startswith("sub-Pilot"):
        settings["TR"] = {
            "AttendAway_7T": .3,
            "AttendInF2_7T": .3,
            "AttendInF1_7T": .3,
            "AttendInF1F2_7T": .3,
        }

    return settings