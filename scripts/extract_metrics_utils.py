from pathlib import Path
import nibabel as nib
import numpy as np

HCP_ROIS = [
    'R_V1_ROI',
    'R_MST_ROI',
    'R_V6_ROI',
    'R_V2_ROI',
    'R_V3_ROI',
    'R_V4_ROI',
    'R_V8_ROI',
    'R_4_ROI',
    'R_3b_ROI',
    'R_FEF_ROI',
    'R_PEF_ROI',
    'R_55b_ROI',
    'R_V3A_ROI',
    'R_RSC_ROI',
    'R_POS2_ROI',
    'R_V7_ROI',
    'R_IPS1_ROI',
    'R_FFC_ROI',
    'R_V3B_ROI',
    'R_LO1_ROI',
    'R_LO2_ROI',
    'R_PIT_ROI',
    'R_MT_ROI',
    'R_A1_ROI',
    'R_PSL_ROI',
    'R_SFL_ROI',
    'R_PCV_ROI',
    'R_STV_ROI',
    'R_7Pm_ROI',
    'R_7m_ROI',
    'R_POS1_ROI',
    'R_23d_ROI',
    'R_v23ab_ROI',
    'R_d23ab_ROI',
    'R_31pv_ROI',
    'R_5m_ROI',
    'R_5mv_ROI',
    'R_23c_ROI',
    'R_5L_ROI',
    'R_24dd_ROI',
    'R_24dv_ROI',
    'R_7AL_ROI',
    'R_SCEF_ROI',
    'R_6ma_ROI',
    'R_7Am_ROI',
    'R_7PL_ROI',
    'R_7PC_ROI',
    'R_LIPv_ROI',
    'R_VIP_ROI',
    'R_MIP_ROI',
    'R_1_ROI',
    'R_2_ROI',
    'R_3a_ROI',
    'R_6d_ROI',
    'R_6mp_ROI',
    'R_6v_ROI',
    'R_p24pr_ROI',
    'R_33pr_ROI',
    'R_a24pr_ROI',
    'R_p32pr_ROI',
    'R_a24_ROI',
    'R_d32_ROI',
    'R_8BM_ROI',
    'R_p32_ROI',
    'R_10r_ROI',
    'R_47m_ROI',
    'R_8Av_ROI',
    'R_8Ad_ROI',
    'R_9m_ROI',
    'R_8BL_ROI',
    'R_9p_ROI',
    'R_10d_ROI',
    'R_8C_ROI',
    'R_44_ROI',
    'R_45_ROI',
    'R_47l_ROI',
    'R_a47r_ROI',
    'R_6r_ROI',
    'R_IFJa_ROI',
    'R_IFJp_ROI',
    'R_IFSp_ROI',
    'R_IFSa_ROI',
    'R_p9-46v_ROI',
    'R_46_ROI',
    'R_a9-46v_ROI',
    'R_9-46d_ROI',
    'R_9a_ROI',
    'R_10v_ROI',
    'R_a10p_ROI',
    'R_10pp_ROI',
    'R_11l_ROI',
    'R_13l_ROI',
    'R_OFC_ROI',
    'R_47s_ROI',
    'R_LIPd_ROI',
    'R_6a_ROI',
    'R_i6-8_ROI',
    'R_s6-8_ROI',
    'R_43_ROI',
    'R_OP4_ROI',
    'R_OP1_ROI',
    'R_OP2-3_ROI',
    'R_52_ROI',
    'R_RI_ROI',
    'R_PFcm_ROI',
    'R_PoI2_ROI',
    'R_TA2_ROI',
    'R_FOP4_ROI',
    'R_MI_ROI',
    'R_Pir_ROI',
    'R_AVI_ROI',
    'R_AAIC_ROI',
    'R_FOP1_ROI',
    'R_FOP3_ROI',
    'R_FOP2_ROI',
    'R_PFt_ROI',
    'R_AIP_ROI',
    'R_EC_ROI',
    'R_PreS_ROI',
    'R_H_ROI',
    'R_ProS_ROI',
    'R_PeEc_ROI',
    'R_STGa_ROI',
    'R_PBelt_ROI',
    'R_A5_ROI',
    'R_PHA1_ROI',
    'R_PHA3_ROI',
    'R_STSda_ROI',
    'R_STSdp_ROI',
    'R_STSvp_ROI',
    'R_TGd_ROI',
    'R_TE1a_ROI',
    'R_TE1p_ROI',
    'R_TE2a_ROI',
    'R_TF_ROI',
    'R_TE2p_ROI',
    'R_PHT_ROI',
    'R_PH_ROI',
    'R_TPOJ1_ROI',
    'R_TPOJ2_ROI',
    'R_TPOJ3_ROI',
    'R_DVT_ROI',
    'R_PGp_ROI',
    'R_IP2_ROI',
    'R_IP1_ROI',
    'R_IP0_ROI',
    'R_PFop_ROI',
    'R_PF_ROI',
    'R_PFm_ROI',
    'R_PGi_ROI',
    'R_PGs_ROI',
    'R_V6A_ROI',
    'R_VMV1_ROI',
    'R_VMV3_ROI',
    'R_PHA2_ROI',
    'R_V4t_ROI',
    'R_FST_ROI',
    'R_V3CD_ROI',
    'R_LO3_ROI',
    'R_VMV2_ROI',
    'R_31pd_ROI',
    'R_31a_ROI',
    'R_VVC_ROI',
    'R_25_ROI',
    'R_s32_ROI',
    'R_pOFC_ROI',
    'R_PoI1_ROI',
    'R_Ig_ROI',
    'R_FOP5_ROI',
    'R_p10p_ROI',
    'R_p47r_ROI',
    'R_TGv_ROI',
    'R_MBelt_ROI',
    'R_LBelt_ROI',
    'R_A4_ROI',
    'R_STSva_ROI',
    'R_TE1m_ROI',
    'R_PI_ROI',
    'R_a32pr_ROI',
    'R_p24_ROI',
    'L_V1_ROI',
    'L_MST_ROI',
    'L_V6_ROI',
    'L_V2_ROI',
    'L_V3_ROI',
    'L_V4_ROI',
    'L_V8_ROI',
    'L_4_ROI',
    'L_3b_ROI',
    'L_FEF_ROI',
    'L_PEF_ROI',
    'L_55b_ROI',
    'L_V3A_ROI',
    'L_RSC_ROI',
    'L_POS2_ROI',
    'L_V7_ROI',
    'L_IPS1_ROI',
    'L_FFC_ROI',
    'L_V3B_ROI',
    'L_LO1_ROI',
    'L_LO2_ROI',
    'L_PIT_ROI',
    'L_MT_ROI',
    'L_A1_ROI',
    'L_PSL_ROI',
    'L_SFL_ROI',
    'L_PCV_ROI',
    'L_STV_ROI',
    'L_7Pm_ROI',
    'L_7m_ROI',
    'L_POS1_ROI',
    'L_23d_ROI',
    'L_v23ab_ROI',
    'L_d23ab_ROI',
    'L_31pv_ROI',
    'L_5m_ROI',
    'L_5mv_ROI',
    'L_23c_ROI',
    'L_5L_ROI',
    'L_24dd_ROI',
    'L_24dv_ROI',
    'L_7AL_ROI',
    'L_SCEF_ROI',
    'L_6ma_ROI',
    'L_7Am_ROI',
    'L_7PL_ROI',
    'L_7PC_ROI',
    'L_LIPv_ROI',
    'L_VIP_ROI',
    'L_MIP_ROI',
    'L_1_ROI',
    'L_2_ROI',
    'L_3a_ROI',
    'L_6d_ROI',
    'L_6mp_ROI',
    'L_6v_ROI',
    'L_p24pr_ROI',
    'L_33pr_ROI',
    'L_a24pr_ROI',
    'L_p32pr_ROI',
    'L_a24_ROI',
    'L_d32_ROI',
    'L_8BM_ROI',
    'L_p32_ROI',
    'L_10r_ROI',
    'L_47m_ROI',
    'L_8Av_ROI',
    'L_8Ad_ROI',
    'L_9m_ROI',
    'L_8BL_ROI',
    'L_9p_ROI',
    'L_10d_ROI',
    'L_8C_ROI',
    'L_44_ROI',
    'L_45_ROI',
    'L_47l_ROI',
    'L_a47r_ROI',
    'L_6r_ROI',
    'L_IFJa_ROI',
    'L_IFJp_ROI',
    'L_IFSp_ROI',
    'L_IFSa_ROI',
    'L_p9-46v_ROI',
    'L_46_ROI',
    'L_a9-46v_ROI',
    'L_9-46d_ROI',
    'L_9a_ROI',
    'L_10v_ROI',
    'L_a10p_ROI',
    'L_10pp_ROI',
    'L_11l_ROI',
    'L_13l_ROI',
    'L_OFC_ROI',
    'L_47s_ROI',
    'L_LIPd_ROI',
    'L_6a_ROI',
    'L_i6-8_ROI',
    'L_s6-8_ROI',
    'L_43_ROI',
    'L_OP4_ROI',
    'L_OP1_ROI',
    'L_OP2-3_ROI',
    'L_52_ROI',
    'L_RI_ROI',
    'L_PFcm_ROI',
    'L_PoI2_ROI',
    'L_TA2_ROI',
    'L_FOP4_ROI',
    'L_MI_ROI',
    'L_Pir_ROI',
    'L_AVI_ROI',
    'L_AAIC_ROI',
    'L_FOP1_ROI',
    'L_FOP3_ROI',
    'L_FOP2_ROI',
    'L_PFt_ROI',
    'L_AIP_ROI',
    'L_EC_ROI',
    'L_PreS_ROI',
    'L_H_ROI',
    'L_ProS_ROI',
    'L_PeEc_ROI',
    'L_STGa_ROI',
    'L_PBelt_ROI',
    'L_A5_ROI',
    'L_PHA1_ROI',
    'L_PHA3_ROI',
    'L_STSda_ROI',
    'L_STSdp_ROI',
    'L_STSvp_ROI',
    'L_TGd_ROI',
    'L_TE1a_ROI',
    'L_TE1p_ROI',
    'L_TE2a_ROI',
    'L_TF_ROI',
    'L_TE2p_ROI',
    'L_PHT_ROI',
    'L_PH_ROI',
    'L_TPOJ1_ROI',
    'L_TPOJ2_ROI',
    'L_TPOJ3_ROI',
    'L_DVT_ROI',
    'L_PGp_ROI',
    'L_IP2_ROI',
    'L_IP1_ROI',
    'L_IP0_ROI',
    'L_PFop_ROI',
    'L_PF_ROI',
    'L_PFm_ROI',
    'L_PGi_ROI',
    'L_PGs_ROI',
    'L_V6A_ROI',
    'L_VMV1_ROI',
    'L_VMV3_ROI',
    'L_PHA2_ROI',
    'L_V4t_ROI',
    'L_FST_ROI',
    'L_V3CD_ROI',
    'L_LO3_ROI',
    'L_VMV2_ROI',
    'L_31pd_ROI',
    'L_31a_ROI',
    'L_VVC_ROI',
    'L_25_ROI',
    'L_s32_ROI',
    'L_pOFC_ROI',
    'L_PoI1_ROI',
    'L_Ig_ROI',
    'L_FOP5_ROI',
    'L_p10p_ROI',
    'L_p47r_ROI',
    'L_TGv_ROI',
    'L_MBelt_ROI',
    'L_LBelt_ROI',
    'L_A4_ROI',
    'L_STSva_ROI',
    'L_TE1m_ROI',
    'L_PI_ROI',
    'L_a32pr_ROI',
    'L_p24_ROI',
]

def filter_parcel_name(parcel_name):

    INCLUDE_ROIS = [
        'V1',
        'V2',
        'V3',
        'V4',
        'PGs',
        'PGi',
    ]
    for _roi in INCLUDE_ROIS:
        if f"_{_roi}_" in parcel_name:
            return True
    return False
    
class FLAExtractZScore:

    def __init__(self, z_score_path):
        self.z_score = Path(z_score_path)
        self._validate_z_score_path()
        self._get_info()
        self.data = nib.load(self.z_score).get_fdata()
        
    def get_threshold_count(self, z_thr, roi_path):
        roi_data = self._extract_roi_data(roi_path)
        return (roi_data > 2).sum()

    def get_median(self, top_percentile, roi_path):
        roi_data = self._extract_roi_data(roi_path)
        roi_data.sort()
        percentile_index = int(len(roi_data) * (1-top_percentile))
        top_pct = roi_data[percentile_index:]
        return np.median(top_pct)
    
    def _validate_z_score_path(self):
        assert self.z_score.exists(), f"{self.z_score} does not exist."
        assert str(self.z_score).endswith('z_score.dscalar.nii'), f"{self.z_score} does not end with [z_score.dscalar.nii]."
        assert 'frequency-' in str(self.z_score), f"{self.z_score} must contain [frequency-]."

    def _get_info(self):
        self.frequency_tag = float(self._get_value_from_stem('frequency'))
        self.sub_id = self._get_value_from_stem('sub')
        self.ses_id = self._get_value_from_stem('ses')
        self.task_id = self._get_value_from_stem('task')
        self.run_id = self._get_value_from_stem('run')
        # Experiment id
        self.experiment_id = [i for i in str(self.z_score).split('/') if i.startswith('1_')]
        assert len(self.experiment_id) == 1, f"Found {self.experiment_id}."
        self.experiment_id = self.experiment_id[0]
        # MRI id
        self.mri_id = [i for i in str(self.z_score).split('/') if (i.endswith('T')) and len(i)==2]
        assert len(self.mri_id) == 1, f"Found {self.mri_id}."
        self.mri_id = self.mri_id[0]
        # Truncate id
        self.truncate_id = [i for i in str(self.z_score).split('/') if i.startswith("truncate-")]
        assert len(self.truncate_id) == 1, f"Found {self.truncate_id}."
        self.truncate_id = self.truncate_id[0]
        # Designmatrix id
        self.dm_id = [i for i in str(self.z_score).split('/') if 'experiment-' in i]
        assert len(self.dm_id) == 1, f"Found {self.dm_id}."
        self.dm_id = self.dm_id[0]
        # Smooth id
        self.smooth_id = [i.split('_')[-1] for i in str(self.z_score).split('/') if i.startswith("run_level")]
        assert len(self.smooth_id) == 1, f"Found {self.smooth_id}."
        self.smooth_id = self.smooth_id[0]
        

    def _get_value_from_stem(self, tag):
        assert f"{tag}-" in self.z_score.stem, f"{tag} not found in [{self.z_score.stem}]."
        return self.z_score.stem.split(f"{tag}-")[-1].split('_')[0]

    def _extract_roi_data(self, roi_path):
        roi_coords = np.where( nib.load(roi_path).get_fdata() == 1 )
        return self.data[roi_coords]