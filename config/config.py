"""
Configuration file of the project
"""
import pathlib

# Define root path
root_path = pathlib.Path(__file__).resolve().parents[1]

# Define raw data path and individual raw data files
data_raw_path = root_path.joinpath('data', 'raw')
# data_raw_ssqdx = data_raw_path.joinpath('data for paper.xlsx')
# data_raw_ssq = data_raw_path.joinpath('ssi_validation_older_added_hla.xlsx')
data_raw_anic = data_raw_path.joinpath('anic_labat_ssi_mergedEM2.xlsx')
data_raw_okun = data_raw_path.joinpath('okun_data_paper.xlsx')

# Define pre-processed data path and individual pre-processed data files
data_pre_proc_path = root_path.joinpath('data', 'pproc')
data_pp_ssq_merged_ssqdx = data_pre_proc_path.joinpath('SSQDX_pp.csv')
data_pp_ssq_merged_ssqdx_imputed = data_pre_proc_path.joinpath('SSQDX_imputed_pp.csv')
data_pp_anic_okun = data_pre_proc_path.joinpath('pp_anic_okun.csv')


# Define results path
results_path = root_path.joinpath('results')
# Configuration dictionary
config = {
    'root_path': root_path,
    'data_raw_path': data_raw_path,
    'data_pre_proc_path': data_pre_proc_path,
    # 'results_path': results_path,
    'data_raw_files': {
        # 'ssqdx': data_raw_ssqdx,
        # 'ssq': data_raw_ssq,
        'okun': data_raw_okun,  # 475 rows, mainly cases
        'anic': data_raw_anic,  # 984 rows, mainly controls
    },
    'data_pre_proc_files': {
        # 'ssq_ssqdx': data_pp_ssq_merged_ssqdx,
        # 'ssq_ssqdx_imputed':data_pp_ssq_merged_ssqdx_imputed,
        'anic_okun': data_pp_anic_okun,
    },

    'results_path': {
        'results': results_path,
        'ess_model': results_path.joinpath('ess_model'),
        'main_model': results_path.joinpath('main_model')

    },

}
