import os
from geopy import distance
from datetime import datetime

# paths
root_dir = os.getcwd()
input_dir = '{0}/{1}'.format(root_dir, '1. separated-dataset')
behavioral_features_dir = '{0}/{1}'.format(root_dir, '12. behavioral features')
user_id_email_map = {'19': 'salman@nsl.inha.ac.kr', '20': 'jumabek4044@gmail.com', '18': 'jskim@nsl.inha.ac.kr', '11': 'aliceblackwood123@gmail.com', '10': 'laurentkalpers3@gmail.com', '7': 'nazarov7mu@gmail.com', '1': 'nslabinha@gmail.com', '8': 'azizsambo58@gmail.com', '3': 'nnarziev@gmail.com', '6': 'mr.khikmatillo@gmail.com'}
data_source_id_name_map = {'34': 'HR', '37': 'ACCELEROMETER', '35': 'RR_INTERVAL', '33': 'HAD_ACTIVITY', '38': 'AMBIENT_LIGHT', '31': 'LOCATION', '36': 'HAD_SLEEP_PATTERN', '32': 'LIGHT_INTENSITY'}
emails = [user_id_email_map[id] for id in user_id_email_map]


# binary search
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


# load ACC location data
def load_acc_data(parent_dir, participant_email):
    res = []
    file_path = os.path.join(parent_dir, participant_email, 'ACCELEROMETER.csv')
    if not os.path.exists(file_path):
        return None
    with open(file_path, 'r') as r:
        for line in r.readlines()[1:]:
            ts, x, y, z = line[:-1].split(',')
            try:
                ts, x, y, z = int(ts), int(x), int(y), int(z)
            except ValueError:
                continue
            res += [(ts, lat, lon)]
        res.sort(key=lambda e: e[0])
    return res


# check if timestamp is between 8pm~12pm
def ts_is_nighttime(timestamp):
    return 20 <= datetime.fromtimestamp(timestamp).hour <= 12


def ts_diff_matches(timestamp1, timestamp2):
    return abs(timestamp2 - timestamp1) > 7200000  # 2 hours


if __name__ == '__main__':
    for email in emails:
        print('generating "', email, '"\'s sleep data')
        with open(f'{input_dir}/{email}/SLEEP_fixed.csv', 'w+') as w:
            w.write('timestamp, sleep_start, sleep_end, sleep_duration_ms\n')
            acc_data = load_acc_data(parent_dir=input_dir, participant_email=email)
            last_ts = None
            for ts, x, y, z in acc_data:
                if last_ts is None:
                    last_ts = ts
                elif ts_is_nighttime(last_ts) and ts_is_nighttime(ts) and ts_diff_matches(last_ts, ts):
                    # sleep time
                    w.write(f'{last_ts},{last_ts},{ts},{abs(ts - last_ts)}\n')
                last_ts = ts
