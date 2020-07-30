# raw dataset paths
root_dir_path = '/Users/kevin/Desktop/stress'
raw_dataset_dir_path = f'{root_dir_path}/data/initial dataset/raw sensing dataset'
test_dataset_dir_path = f'{root_dir_path}/data/initial dataset/test dataset'
ground_truths_file_path = f'{root_dir_path}/data/initial dataset/all_ESMs.csv'

# processed dataset paths
not_filtered_dataset_dir_path = f'{root_dir_path}/data/processed dataset/1. not filtered dataset'
ppg_filtered_dataset_dir_path = f'{root_dir_path}/data/processed dataset/2. ppg filtered dataset'
acc_filtered_dataset_dir_path = f'{root_dir_path}/data/processed dataset/3. acc filtered dataset'
combined_filtered_dataset_dir_path = f'{root_dir_path}/data/processed dataset/4. combined filtered dataset'
behavioral_features_dataset_dir_path = f'{root_dir_path}/data/processed dataset/5. behavioral features dataset'
behavioral_features_combinations_dir_path = f'{root_dir_path}/data/processed dataset/6. behavioral features combinations'

# result paths
no_filter_test_results_dir_path = f'{root_dir_path}/data/test results/1. not filtered'
ppg_filter_test_results_dir_path = f'{root_dir_path}/data/test results/2. ppg filtered'
acc_filter_test_results_dir_path = f'{root_dir_path}/data/test results/3. acc filtered'
combined_filter_test_results_dir_path = f'{root_dir_path}/data/test results/4. combined filtered'
behavioral_data_test_results_dir_path = f'{root_dir_path}/data/test results/5. behavioral data'

# participants / emails
participants = [
    'aliceblackwood123@gmail.com',
    'azizsambo58@gmail.com',
    'jskim@nsl.inha.ac.kr',
    'jumabek4044@gmail.com',
    'laurentkalpers3@gmail.com',
    'mr.khikmatillo@gmail.com',
    'nazarov7mu@gmail.com',
    'nnarziev@gmail.com',
    'nslabinha@gmail.com',
    'salman@nsl.inha.ac.kr',
]

# all & selected features
all_features = [
    # time domain features
    'mean_nni', 'sdnn', 'sdsd', 'rmssd', 'median_nni', 'nni_50', 'pnni_50', 'nni_20', 'pnni_20', 'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr',
    # frequency domain features
    'total_power', 'vlf', 'lf', 'hf', 'lf_hf_ratio', 'lfnu', 'hfnu',
    # cardiac sympathetic / vagal index features
    'csi', 'cvi', 'Modified_csi',
    # geometrical features
    'triangular_index', 'tinn',
    # pointcare plot features
    'sd1', 'sd2', 'ratio_sd2_sd1',
    # sample entropy
    'sampen'
]
basic_selected_features = [
    # time domain features
    'mean_nni', 'sdnn', 'rmssd', 'nni_50',
    # frequency domain features
    'lf', 'hf', 'lf_hf_ratio',
    # pointcare plot features
    'ratio_sd2_sd1', 'sd2',
    # sample entropy
    'sampen'
]
behavior_enhanced_selected_features = [
    # time domain features
    'mean_nni', 'sdnn', 'rmssd', 'nni_50',
    # frequency domain features
    'lf', 'hf', 'lf_hf_ratio',
    # pointcare plot features
    'ratio_sd2_sd1', 'sd2',
    # sample entropy
    'sampen',
    # behavioral features
    '1st_moment', '2nd_moment', '3rd_moment', '4th_moment'
]

# header used for created datasets
filter_datasets_header = f"timestamp,{','.join(all_features)},gt_self_report,gt_timestamp,gt_pss_control,gt_pss_difficult,gt_pss_confident,gt_pss_yourway,gt_likert_stresslevel,gt_score,gt_label\n"
behavioral_dataset_header = f"timestamp,{','.join(all_features)},1st_moment,2nd_moment,3rd_moment,4th_moment,gt_self_report,gt_timestamp,gt_pss_control,gt_pss_difficult,gt_pss_confident,gt_pss_yourway,gt_likert_stresslevel,gt_score,gt_label\n"

# data sources (id <-> name map)
data_source_id_name_map = {
    '31': 'LOCATION',
    '32': 'LIGHT_INTENSITY',
    '33': 'HAD_ACTIVITY',
    '34': 'HR',
    '35': 'RR_INTERVAL',
    "36": 'HAD_SLEEP_PATTERN',
    '37': 'ACCELEROMETER',
    '38': 'AMBIENT_LIGHT',
}

# window sizes
dataset_augmentation_window_size = 3600000  # 1 hour
feature_aggregation_window_size = 60000  # 1 minute
activity_window_size = 10000  # 10 seconds

# thresholds
ppg_thresholds = {
    'aliceblackwood123@gmail.com': 70000,  # 60000,
    'azizsambo58@gmail.com': 80000,  # 110000,
    'jskim@nsl.inha.ac.kr': 120000,  # 140000,
    'jumabek4044@gmail.com': 80000,  # 100000,
    'laurentkalpers3@gmail.com': 60000,  # 90000,
    'mr.khikmatillo@gmail.com': 70000,  # 70000,
    'nazarov7mu@gmail.com': 60000,  # 50000,
    'nnarziev@gmail.com': 120000,  # 70000,
    'nslabinha@gmail.com': 80000,  # 120000,
    'salman@nsl.inha.ac.kr': 110000,  # 80000,
}
acc_thresholds = {
    'aliceblackwood123@gmail.com': 1.1,  # 1.10,
    'azizsambo58@gmail.com': 0.5,  # 0.65,
    'jskim@nsl.inha.ac.kr': 1.85,  # 0.50,
    'jumabek4044@gmail.com': 0.65,  # 0.50,
    'laurentkalpers3@gmail.com': 1.25,  # 0.65,
    'mr.khikmatillo@gmail.com': 2.0,  # 1.10,
    'nazarov7mu@gmail.com': 1.1,  # 0.65,
    'nnarziev@gmail.com': 1.1,  # 0.80,
    'nslabinha@gmail.com': 0.8,  # 1.40,
    'salman@nsl.inha.ac.kr': 1.55,  # 2.00
}
combined_thresholds = {
    'PPG': {
        'aliceblackwood123@gmail.com': 70000,  # 70000, #50000,
        'azizsambo58@gmail.com': 100000,  # 70000, #100000,
        'jskim@nsl.inha.ac.kr': 150000,  # 100000, #50000,
        'jumabek4044@gmail.com': 110000,  # 60000, #80000,
        'laurentkalpers3@gmail.com': 50000,  # 60000, #130000,
        'mr.khikmatillo@gmail.com': 60000,  # 60000, #70000,
        'nazarov7mu@gmail.com': 50000,  # 80000, #50000,
        'nnarziev@gmail.com': 60000,  # 50000, #60000,
        'nslabinha@gmail.com': 80000,  # 130000, #70000,
        'salman@nsl.inha.ac.kr': 150000,  # 130000, #60000,
    },
    'ACC': {
        'aliceblackwood123@gmail.com': 1.1,  # 1.7, #1.10,
        'azizsambo58@gmail.com': 1.7,  # 0.8, #1.10,
        'jskim@nsl.inha.ac.kr': 1.25,  # 0.5, #0.65,
        'jumabek4044@gmail.com': 0.5,  # 1.7, #0.65,
        'laurentkalpers3@gmail.com': 1.85,  # 1.7, #0.65,
        'mr.khikmatillo@gmail.com': 0.65,  # 1.25, #1.70,
        'nazarov7mu@gmail.com': 1.25,  # 0.65, #0.65,
        'nnarziev@gmail.com': 0.5,  # 1.25, #0.50,
        'nslabinha@gmail.com': 1.7,  # 1.7, #1.10,
        'salman@nsl.inha.ac.kr': 1.25,  # 1.55, #1.55,
    }
}
