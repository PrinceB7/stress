import os
from geopy import distance

# paths
root_dir = os.getcwd()
input_dir = '{0}/{1}'.format(root_dir, '1. separated-dataset')
behavioral_features_dir = '{0}/{1}'.format(root_dir, '12. behavioral features')
user_id_email_map = {'19': 'salman@nsl.inha.ac.kr', '20': 'jumabek4044@gmail.com', '18': 'jskim@nsl.inha.ac.kr', '11': 'aliceblackwood123@gmail.com', '10': 'laurentkalpers3@gmail.com', '7': 'nazarov7mu@gmail.com', '1': 'nslabinha@gmail.com', '8': 'azizsambo58@gmail.com', '3': 'nnarziev@gmail.com', '6': 'mr.khikmatillo@gmail.com'}
data_source_id_name_map = {'34': 'HR', '37': 'ACCELEROMETER', '35': 'RR_INTERVAL', '33': 'HAD_ACTIVITY', '38': 'AMBIENT_LIGHT', '31': 'LOCATION', '36': 'HAD_SLEEP_PATTERN', '32': 'LIGHT_INTENSITY'}
emails = [user_id_email_map[id] for id in user_id_email_map]

for email in emails:
    print('fixing "', email, '"\'s location timestamps')
    with open(f'{input_dir}/{email}/LOCATION.csv', 'r') as r, open(f'{input_dir}/{email}/LOCATION_fixed.csv', 'w+') as w:
        w.write('timestamp,lat,lon,distance_traveled\n')
        gap_ts, count, last_loc = 1100, None, None
        for line in r.readlines()[1:]:
            ts, lat, lon = line[:-1].split(',')
            ts, lat, lon = int(ts), float(lat), float(lon)
            if lat == 200.0 and lon == 200.0:
                continue
            if last_loc is None or last_loc[0] != ts:
                w.write(f'{ts},{lat},{lon},0\n')
                count = 0
            else:
                dist = distance.distance((last_loc[1], last_loc[2]), (lat, lon)).meters
                fixed_ts = ts + count * gap_ts
                w.write(f'{fixed_ts},{lat},{lon},{dist}\n')
            last_loc = (ts, lat, lon)
            count += 1
