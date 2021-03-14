import argparse
from mmdet.utils.openimage_categories import get_group_ids

parser = argparse.ArgumentParser()

parser.add_argument('--file')

args = parser.parse_args()

group_names = get_group_ids(return_names=True)
group_accs = [[] for _ in range(len(group_names))]
name_id_map = dict()
for i, names in enumerate(group_names):
    for name in names:
        name_id_map[name] = i

with open(args.file, 'r') as f:
    mAP_line = f.readline().strip()
    mAP = mAP_line.split(',')[-1]
    while True:
        line = f.readline()
        if not line:
            break
        name, acc = line.strip().replace('OpenImagesDetectionChallenge_PerformanceByCategory/AP@0.5IOU/b', '').split(',')
        if acc == 'nan':
            print(f"detect nan!!: {name}")
        else:
            name = name.replace("'", "")
            group_id = name_id_map[name]
            group_accs[group_id].append(float(acc))

print(f"mAP {mAP}")
for i, accs in enumerate(group_accs):
    AP = sum(accs) / len(accs)
    print(f"mAP{i}: {AP}")
