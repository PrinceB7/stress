from scripts import utils, settings
import os

combinations = [el[:-11] for el in os.listdir(settings.behavioral_data_combinations_test_results_dir_path) if el.endswith('_scores.csv')]
f1s = {}

for moments in combinations:
    f1s[moments] = []
    with open(f'{settings.behavioral_data_combinations_test_results_dir_path}/{moments}_scores.csv', 'r') as r:
        for line in r.readlines()[1:]:
            participant, acc, f1, roc_auc, tnr, tpr = line[:-1].split(',')
            f1s[moments] += [float(f1)]
    f1s[moments] = sum(f1s[moments]) / len(f1s[moments])

print(f1s)
