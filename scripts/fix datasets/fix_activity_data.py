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

activity_values = {'STATIONARY': 0, 'WALKING': 1}
for email in emails:
    # print('generating "', email, '"\'s sleep data')
    with open(f'{input_dir}/{email}/HAD_ACTIVITY.csv', 'r') as r, open(f'{input_dir}/{email}/HAD_ACTIVITY_fixed.csv', 'w+') as w:
        w.write('timestamp,had_activity\n')
        for line in r.readlines()[1:]:
            ts, activity = line[:-1].split(',')
            w.write(f'{ts},{activity_values[activity]}\n')
