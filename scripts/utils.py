from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values
from hrvanalysis import get_time_domain_features, get_frequency_domain_features
from hrvanalysis import get_csi_cvi_features, get_geometrical_features
from hrvanalysis import get_poincare_plot_features, get_sampen
from sklearn.metrics import confusion_matrix, recall_score
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
from scipy.stats import moment
import statistics
import datetime
from scripts import settings
import xgboost
import pandas
import numpy
import math
import csv
import os


# converts string time into timestamp (i.e., 2020-04-23T11:00+0900 --> 1587607200000)
def string_to_timestamp(_str):
    if _str == '':
        return None
    elif _str[-3] == ':':
        _str = _str[:-3] + _str[-2:]
    return int(datetime.datetime.strptime(_str, '%Y-%m-%dT%H:%M:%S.%f%z').timestamp()) * 1000


# converts a numeric string into a number
def string_to_number(_str):
    if _str == '':
        return None
    else:
        try:
            return int(_str)
        except ValueError:
            return None


# converts a numeric string into a fraction number
def string_to_float(_str):
    if _str == '':
        return None
    else:
        try:
            return float(_str)
        except ValueError:
            return None


# loads ESM responses, calculates scores, and adds a label (i.e., "stressed" / "not-stressed")
def load_ground_truths(participant):
    res = []
    with open(settings.ground_truths_file_path, 'r') as r:
        csv_reader = csv.reader(r, delimiter=',', quotechar='"')
        for csv_row in csv_reader:
            if csv_row[0] != participant:
                continue
            rt = string_to_timestamp(csv_row[11])
            st = string_to_timestamp(csv_row[12])
            control = string_to_number(csv_row[16])
            difficulty = string_to_number(csv_row[17])
            confident = string_to_number(csv_row[18])
            yourway = string_to_number(csv_row[19])
            row = (
                st is None,  # is self report
                rt if st is None else st,  # timestamp
                control,  # (-)PSS:Control
                difficulty,  # (-)PSS:Difficult
                confident,  # (+)PSS:Confident
                yourway,  # (+)PSS:YourWay
                string_to_number(csv_row[20]),  # LikeRt:StressLevel,
            )
            if None in row:
                continue
            score = (control + difficulty + (6 - confident) + (6 - yourway)) / 4
            res += [(row) + (score,)]
        res.sort(key=lambda e: e[1])
        mean = sum(row[7] for row in res) / len(res)
        for i in range(len(res)):
            res[i] += (res[i][7] > mean,)
    return res


# loads participant's IBI readings
def load_rr_data(participant):
    res = []
    file_path = os.path.join(settings.raw_dataset_dir_path, participant, 'RR_INTERVAL.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r:
            ts, rr = line[:-1].split(',')
            try:
                ts, rr = int(ts), int(rr)
            except ValueError:
                continue
            res += [(ts, rr)]
        res.sort(key=lambda e: e[0])
    return res


# loads participant's ACC readings
def load_acc_data(participant):
    res = []
    file_path = os.path.join(settings.raw_dataset_dir_path, participant, 'ACCELEROMETER.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r:
            cells = line[:-1].split(',')
            if len(cells) == 2:
                continue
            try:
                ts, x, y, z = int(cells[0]), float(cells[1]), float(cells[2]), float(cells[3])
            except ValueError:
                continue
            res += [(ts, x, y, z)]
        res.sort(key=lambda e: e[0])
    return res


# loads participant's PPG light intensity readings
def load_ppg_data(participant):
    res = []
    file_path = os.path.join(settings.raw_dataset_dir_path, participant, 'LIGHT_INTENSITY.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r:
            ts, rr = line[:-1].split(',')
            try:
                ts, rr = int(ts), int(rr)
            except ValueError:
                continue
            res += [(ts, rr)]
        res.sort(key=lambda e: e[0])
    return res


# calculate STD of ppg light intensities
def get_ppg_stdev(ppg_values):
    return statistics.stdev(ppg_values)


# searches for element in dataset with binary search
def bin_find(dataset, low, high, ts):
    if high > low:
        mid = (high + low) // 2
        if dataset[mid][0] == ts:
            return mid
        elif dataset[mid][0] > ts:
            return bin_find(dataset, low, mid, ts)
        else:
            return bin_find(dataset, mid + 1, high, ts)
    else:
        return min(high, low)


# selects data within the specified timespan
def select_data(dataset, from_ts, till_ts, with_timestamp=False):
    res = []
    index = bin_find(dataset, 0, len(dataset) - 1, from_ts)
    end_index = len(dataset)
    while index < end_index:
        if from_ts <= dataset[index][0] < till_ts:
            if with_timestamp:
                res += [tuple(dataset[index])]
            elif len(dataset[index][1:]) == 1:
                res += tuple(dataset[index][1:])
            else:
                res += [tuple(dataset[index][1:])]
        elif dataset[index][0] >= till_ts:
            break
        index += 1
    return res if len(res) >= 60 else None


# selects the closest data point (w/ timestamp)
def find_closest(start_ts, ts, dataset):
    while ts not in dataset and ts != start_ts:
        ts -= 1
    if ts == start_ts:
        # print('error occurred, reached the start_ts!')
        # exit(1)
        return None
    else:
        return dataset[ts]


# downsamples data
def select_downsample_data(dataset, from_ts, till_ts):
    selected_data = {}
    for row in dataset:
        if from_ts <= row[0] < till_ts:
            selected_data[row[0]] = row[1]
        elif row[0] >= till_ts:
            break
    timestamps = [ts for ts in range(from_ts, till_ts, 1000)]
    res = []
    for ts in timestamps:
        closest = find_closest(start_ts=from_ts, ts=ts, dataset=selected_data)
        if closest is not None:
            res += [closest]
    return res if len(res) >= 20 else None


# checks if window of accelerations is considered as active
def is_acc_window_active(acc_values, activity_threshold):
    activeness_scores = []
    magnitudes = []
    end = acc_values[0][0] + settings.activity_window_size
    last_ts = acc_values[-1][0]
    for ts, x, y, z in acc_values:
        if ts >= end or ts == last_ts:
            activeness_scores += [numpy.mean(magnitudes)]
            end += settings.activity_window_size
            magnitudes = []
        magnitudes += [math.sqrt(x ** 2 + y ** 2 + z ** 2)]
    # low_limit = np.percentile(stdevs, 1)
    # high_limit = np.percentile(stdevs, 99)
    # activeness_range = high_limit - low_limit
    active_count = 0
    for activeness_score in activeness_scores:
        # if activeness_score > (low_limit + activity_threshold * activeness_range):
        if activeness_score > activity_threshold:
            active_count += 1
    if active_count > len(activeness_scores) / 2:
        return True
    else:
        return False


# calculates stress features from provided IBI readings
def calculate_features(rr_intervals):
    # process the RR-intervals
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rr_intervals, low_rri=300, high_rri=2000, verbose=False)
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers, interpolation_method='linear')
    nn_intervals = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method='malik')
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals)

    # extract the features
    time_domain_features = get_time_domain_features(nn_intervals=interpolated_nn_intervals)
    frequency_domain_features = get_frequency_domain_features(nn_intervals=interpolated_nn_intervals)
    csi_cvi_features = get_csi_cvi_features(nn_intervals=interpolated_nn_intervals)
    geometrical_features = get_geometrical_features(nn_intervals=interpolated_nn_intervals)
    poincare_plot_features = get_poincare_plot_features(nn_intervals=interpolated_nn_intervals)
    sample_entropy = get_sampen(nn_intervals=interpolated_nn_intervals)

    return [
        time_domain_features['mean_nni'],  # The mean of RR-intervals
        time_domain_features['sdnn'],  # The standard deviation of the time interval between successive normal heart beats (i.e. the RR-intervals)
        time_domain_features['sdsd'],  # The standard deviation of differences between adjacent RR-intervals
        time_domain_features['rmssd'],  # The square root of the mean of the sum of the squares of differences between adjacent NN-intervals. Reflects high frequency (fast or parasympathetic) influences on hrV (i.e., those influencing larger changes from one beat to the next)
        time_domain_features['median_nni'],  # Median Absolute values of the successive differences between the RR-intervals
        time_domain_features['nni_50'],  # Number of interval differences of successive RR-intervals greater than 50 ms
        time_domain_features['pnni_50'],  # The proportion derived by dividing nni_50 (The number of interval differences of successive RR-intervals greater than 50 ms) by the total number of RR-intervals
        time_domain_features['nni_20'],  # Number of interval differences of successive RR-intervals greater than 20 ms
        time_domain_features['pnni_20'],  # The proportion derived by dividing nni_20 (The number of interval differences of successive RR-intervals greater than 20 ms) by the total number of RR-intervals
        time_domain_features['range_nni'],  # Difference between the maximum and minimum nn_interval
        time_domain_features['cvsd'],  # Coefficient of variation of successive differences equal to the rmssd divided by mean_nni
        time_domain_features['cvnni'],  # Coefficient of variation equal to the ratio of sdnn divided by mean_nni
        time_domain_features['mean_hr'],  # Mean heart rate value
        time_domain_features['max_hr'],  # Maximum heart rate value
        time_domain_features['min_hr'],  # Minimum heart rate value
        time_domain_features['std_hr'],  # Standard deviation of heart rate values

        frequency_domain_features['total_power'],  # Total power density spectral
        frequency_domain_features['vlf'],  # variance (=power) in HRV in the Very low Frequency (.003 to .04 Hz by default). Reflect an intrinsic rhythm produced by the heart which is modulated primarily by sympathetic activity
        frequency_domain_features['lf'],  # variance (=power) in HRV in the low Frequency (.04 to .15 Hz). Reflects a mixture of sympathetic and parasympathetic activity, but in long-term recordings, it reflects sympathetic activity and can be reduced by the beta-adrenergic antagonist propanolol
        frequency_domain_features['hf'],
        # variance (=power) in HRV in the High Frequency (.15 to .40 Hz by default). Reflects fast changes in beat-to-beat variability due to parasympathetic (vagal) activity. Sometimes called the respiratory band because it corresponds to HRV changes related to the respiratory cycle and can be increased by slow, deep breathing (about 6 or 7 breaths per minute) and decreased by anticholinergic drugs or vagal blockade
        frequency_domain_features['lf_hf_ratio'],  # lf/hf ratio is sometimes used by some investigators as a quantitative mirror of the sympatho/vagal balance
        frequency_domain_features['lfnu'],  # normalized lf power
        frequency_domain_features['hfnu'],  # normalized hf power

        csi_cvi_features['csi'],  # Cardiac Sympathetic Index
        csi_cvi_features['cvi'],  # Cardiac Vagal Index
        csi_cvi_features['Modified_csi'],  # Modified CSI is an alternative measure in research of seizure detection

        geometrical_features['triangular_index'],  # The HRV triangular index measurement is the integral of the density distribution (= the number of all NN-intervals) divided by the maximum of the density distribution
        geometrical_features['tinn'],  # The triangular interpolation of NN-interval histogram (TINN) is the baseline width of the distribution measured as a base of a triangle, approximating the NN-interval distribution

        poincare_plot_features['sd1'],  # The standard deviation of projection of the Poincaré plot on the line perpendicular to the line of identity
        poincare_plot_features['sd2'],  # SD2 is defined as the standard deviation of the projection of the Poincaré plot on the line of identity (y=x)
        poincare_plot_features['ratio_sd2_sd1'],  # Ratio between SD2 and SD1

        sample_entropy['sampen'],  # The sample entropy of the Normal to Normal Intervals
    ]


# calculate behavioral features from provided acceleration readings
def calculate_behavioral_features(acc_values, moments=None):
    acc_magnitudes = [math.sqrt(x ** 2 + y ** 2 + z ** 2) for x, y, z in acc_values]
    if moment is None:
        return [moment(acc_magnitudes, moment=_moment) for _moment in range(1, 5)]
    else:
        return [moment(acc_magnitudes, moment=_moment) for _moment in moments]


# makes suitable behavioral dataset header for the given moments
def get_behavioral_dataset_header(moments):
    moments_map = {1: '1st_moment', 2: '2nd_moment', 3: '3rd_moment', 4: '4th_moment'}
    return f"timestamp,{','.join(settings.all_features)},{','.join([moments_map[moment] for moment in moments])},gt_self_report,gt_timestamp,gt_pss_control,gt_pss_difficult,gt_pss_confident,gt_pss_yourway,gt_likert_stresslevel,gt_score,gt_label\n"


# calculates combinations of the elements in array
def calculate_combinations(array):
    def recursive_combinations(array):
        if len(array) == 1:
            return [[], [array[0]]]
        else:
            res = []
            for i, val in enumerate(array):
                sub_array = array[:i] + array[i + 1:]
                for combination in recursive_combinations(array=sub_array):
                    combination.sort()
                    if combination not in res:
                        res += [combination]
                    combination = [val] + combination
                    combination.sort()
                    if combination not in res:
                        res += [combination]
            return res

    combinations = recursive_combinations(array=array)
    combinations.remove([])
    return combinations


# loads dataset, excluding some samples if needed
def load_dataset(directory, participant, selected_column_names, screen_out_timestamps=None):
    _dataset = pandas.read_csv('{dir}/{participant}.csv'.format(dir=directory, participant=participant)).replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis=0)
    # .drop_duplicates(subset='timestamp')
    if screen_out_timestamps is not None:
        _dataset = _dataset[~_dataset.timestamp.isin(screen_out_timestamps)]
    _features = _dataset[selected_column_names]
    _label = _dataset.gt_label.astype(int)
    return list(_dataset.timestamp), _features, _label


# trains and tests on a test dataset
def participant_train_test_xgboost(participant, train_dir, selected_features, tuning=True):
    # train & test dataset
    ts, test_features, test_labels = load_dataset(directory=settings.test_dataset_dir_path, participant=participant, selected_column_names=selected_features)
    _, train_features, train_labels = load_dataset(directory=train_dir, participant=participant, selected_column_names=selected_features, screen_out_timestamps=ts)

    # configure test dataset
    scaler = MinMaxScaler()
    scaler.fit(test_features)
    test_features_scaled = scaler.transform(test_features)
    test_features = pandas.DataFrame(test_features_scaled, index=test_features.index, columns=test_features.columns)

    k_folds = []
    splitter = StratifiedKFold(n_splits=5, shuffle=True)
    for idx, (train_indices, test_indices) in enumerate(splitter.split(train_features, train_labels)):
        try:
            x_train = train_features.iloc[train_indices]
            y_train = train_labels.iloc[train_indices]
            x_test = train_features.iloc[test_indices]
            y_test = train_labels.iloc[test_indices]
            k_folds.append((x_train, y_train, x_test, y_test))
        except ValueError:
            continue

    # print('# Features : rows({rows}) cols({cols})'.format(rows=train_features.shape[0], cols=train_features.shape[1]))
    # print(train_features.head(), '\n')
    # print('# Labels : stressed({stressed}) not-stressed({not_stressed})'.format(stressed=numpy.count_nonzero(train_labels == 1), not_stressed=numpy.count_nonzero(train_labels == 0)))
    # print(train_labels.head(), '\n')

    k_folds_sampled = []
    for idx, (x_train, y_train, x_test, y_test) in enumerate(k_folds):
        try:
            sampler = SMOTE()
            x_sample, y_sample = sampler.fit_resample(x_train, y_train)
            x_sample = pandas.DataFrame(x_sample, columns=x_train.columns)
            y_sample = pandas.Series(y_sample)
            k_folds_sampled.append((x_sample, y_sample, x_test, y_test))
        except ValueError:
            continue

    k_folds_scaled = []
    for x_train, y_train, x_test, y_test in k_folds_sampled:
        scaler = MinMaxScaler()
        scaler.fit(x_train)
        x_train_scale = scaler.transform(x_train)
        x_test_scale = scaler.transform(x_test)
        x_train = pandas.DataFrame(x_train_scale, index=x_train.index, columns=x_train.columns)
        x_test = pandas.DataFrame(x_test_scale, index=x_test.index, columns=x_test.columns)
        k_folds_scaled.append((x_train, y_train, x_test, y_test))

    conf_mtx = numpy.zeros((2, 2))  # 2 X 2 confusion matrix
    train_data = xgboost.DMatrix(data=train_features, label=train_labels.to_numpy())

    # Parameter tuning / grid search
    best_params = {
        'max_depth': 6,
        'min_child_weight': 1,
        'eta': .3,
        'subsample': 1,
        'colsample_bytree': 1,
        'objective': 'binary:logistic',
        'booster': 'gbtree',
        'verbosity': 0,
        'eval_metric': "auc"
    }
    if tuning:
        print('tuning parameters...')
        tmp_params = {
            'max_depth': 6,
            'min_child_weight': 1,
            'eta': .3,
            'subsample': 1,
            'colsample_bytree': 1,
            'objective': 'binary:logistic',
            'booster': 'gbtree',
            'verbosity': 0,
            'eval_metric': "auc"
        }
        grid_search_params = [(max_depth, min_child_weight) for max_depth in range(0, 12) for min_child_weight in range(0, 8)]
        current_test_auc = -float("Inf")
        for max_depth, min_child_weight in grid_search_params:
            try:
                # Update our parameters
                tmp_params['max_depth'] = max_depth
                tmp_params['min_child_weight'] = min_child_weight
                # Run CV
                cv_results = xgboost.cv(tmp_params, train_data, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                # Update best MAE
                mean_mae = cv_results['test-auc-mean'].max()
                if mean_mae > current_test_auc:
                    current_test_auc = mean_mae
                    best_params['max_depth'] = max_depth
                    best_params['min_child_weight'] = min_child_weight
            except xgboost.core.XGBoostError:
                continue

        grid_search_params = [(subsample, colsample) for subsample in [i / 10. for i in range(7, 11)] for colsample in [i / 10. for i in range(7, 11)]]
        current_test_auc = -float("Inf")
        tmp_params = {'subsample': None, 'colsample_bytree': None}
        # We start by the largest values and go down to the smallest
        for sub_sample, col_sample in reversed(grid_search_params):
            try:
                # We update our parameters
                best_params['subsample'] = sub_sample
                best_params['colsample_bytree'] = col_sample
                # Run CV
                cv_results = xgboost.cv(best_params, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                mean_mae = cv_results['test-auc-mean'].max()
                if mean_mae > current_test_auc:
                    current_test_auc = mean_mae
                    tmp_params = {'subsample': sub_sample, 'colsample_bytree': col_sample}
            except xgboost.core.XGBoostError:
                continue
        best_params['subsample'] = tmp_params['subsample']
        best_params['colsample_bytree'] = tmp_params['colsample_bytree']

        current_test_auc = -float("Inf")
        tmp_params = None
        for eta in [.3, .2, .1, .05, .01, .005]:
            try:
                # We update our parameters
                best_params['eta'] = eta
                # Run and time CV
                cv_results = xgboost.cv(best_params, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                # Update best score
                mean_mae = cv_results['test-auc-mean'].max()
                if mean_mae > current_test_auc:
                    current_test_auc = mean_mae
                    tmp_params = eta
            except xgboost.core.XGBoostError:
                continue
        best_params['eta'] = tmp_params

        current_test_auc = -float("Inf")
        tmp_params = None
        gamma_range = [i / 10.0 for i in range(0, 25)]
        for gamma in gamma_range:
            try:
                # We update our parameters
                best_params['gamma'] = gamma
                # Run and time CV
                cv_results = xgboost.cv(best_params, train_data, num_boost_round=1000, nfold=5, metrics=['auc'], early_stopping_rounds=25)
                # Update best score
                mean_mae = cv_results['test-auc-mean'].max()
                if mean_mae > current_test_auc:
                    current_test_auc = mean_mae
                    tmp_params = gamma
            except xgboost.core.XGBoostError:
                continue
        best_params['gamma'] = tmp_params

    print('training and testing...')
    xgb_models = []  # This is used to store models for each fold.
    folds_scores_tmp = {'Accuracy (balanced)': [], 'F1 score': [], 'ROC AUC score': [], 'True Positive rate': [], 'True Negative rate': []}
    for x_train, y_train, x_test, y_test in k_folds_scaled:
        train_data = xgboost.DMatrix(data=x_train, label=y_train.to_numpy())
        evaluation_data = xgboost.DMatrix(data=x_test, label=y_test.to_numpy())
        test_data = xgboost.DMatrix(data=test_features, label=test_labels.to_numpy())

        # docs : https://xgboost.readthedocs.io/en/latest/parameter.html
        results = {}
        booster = xgboost.train(best_params, dtrain=train_data, num_boost_round=1000, early_stopping_rounds=25, evals=[(evaluation_data, 'test')], verbose_eval=False, evals_result=results)
        # print('Fold evaluation results : ', results)
        predicted_probabilities = booster.predict(data=test_data, ntree_limit=booster.best_ntree_limit)
        predicted_labels = numpy.where(predicted_probabilities > 0.5, 1, 0)

        acc = balanced_accuracy_score(test_labels, predicted_labels)
        f1 = f1_score(test_labels, predicted_labels, average='macro')
        roc_auc = roc_auc_score(test_labels, predicted_probabilities)
        tpr = recall_score(test_labels, predicted_labels)
        tnr = recall_score(test_labels, predicted_labels, pos_label=0)

        folds_scores_tmp['Accuracy (balanced)'].append(acc)
        folds_scores_tmp['F1 score'].append(f1)
        folds_scores_tmp['ROC AUC score'].append(roc_auc)
        folds_scores_tmp['True Positive rate'].append(tpr)
        folds_scores_tmp['True Negative rate'].append(tnr)

        conf_mtx += confusion_matrix(test_labels, predicted_labels)
        xgb_models.append(booster)

    folds_scores = {}
    for k, v in folds_scores_tmp.items():
        folds_scores[k] = numpy.mean(v)
    return best_params, folds_scores
