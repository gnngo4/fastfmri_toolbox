import os
import glob
import tempfile
from pathlib import Path

TMP_DIR = '/scratch/tmp'

def search(base_dir, wildcard):
    search_path = Path(base_dir) / wildcard
    files = glob.glob(str(search_path))

    if not files:
        raise FileNotFoundError(f"No files were found in: {search_path}")

    return files

def create_temp_file(suffix, temp_dir=TMP_DIR):
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False, dir=temp_dir) as temp_file:
        return temp_file.name

def cifti_separate(dscalar):
    data = {}
    data['volume'] = create_temp_file('.nii')
    data['cortex_left'] = create_temp_file('.func.gii')
    data['cortex_right'] = create_temp_file('.func.gii')
    # Separate dsclar
    cmd = f"wb_command -cifti-separate {dscalar} COLUMN -volume-all {data['volume']} -metric CORTEX_LEFT {data['cortex_left']} -metric CORTEX_RIGHT {data['cortex_right']}"
    os.system(cmd)
    for i in data.values():
        assert Path(i).exists()
    # Uncompress format
    for func in [data['cortex_left'], data['cortex_right']]:
        cmd = f"wb_command -gifti-convert BASE64_BINARY {func} {func}"
        os.system(cmd)
    
    return data

def get_surface_vertex_areas(
    surf_left="/scratch/surfaces/S1200.L.midthickness_MSMAll.32k_fs_LR.surf.gii", 
    surf_right="/scratch/surfaces/S1200.R.midthickness_MSMAll.32k_fs_LR.surf.gii"
):
    for s in [surf_left, surf_right]:
        assert Path(s).exists()
    data = {}
    for hemi, surf_path in zip(
        ['cortex_left','cortex_right'],
        [surf_left,surf_right]
    ):
        data[hemi] = create_temp_file('.func.gii')
        cmd = f"wb_command -surface-vertex-areas {surf_path} {data[hemi]}"
        os.system(cmd)
        assert Path(data[hemi]).exists()
        
    return data

def merge_niftis(niftis):
    merged_nifti = create_temp_file('.nii')
    cmd = f"fslmerge -t {merged_nifti} {' '.join(niftis)}"
    os.system(cmd)
    assert Path(merged_nifti).exists()

    return merged_nifti

def merge_metrics(func_giis):
    merged_func_gii = create_temp_file('.func.gii')
    cmd = f"wb_command -metric-merge {merged_func_gii} -metric {' -metric '.join(func_giis)}"
    os.system(cmd)
    assert Path(merged_func_gii).exists()

    return merged_func_gii

def get_LR_gii_pairs(_list):
    pairs = []
    for i in _list:
        for j in _list:
            
            if not 'CortexLeft' in i:
                continue
                
            if i == j.replace('CortexRight', 'CortexLeft') and 'CortexRight' in j:
                pairs.append((i,j))

    assert len(pairs)%2 == 0, f"Number of pairs {len(pairs)} is not even."
    return pairs