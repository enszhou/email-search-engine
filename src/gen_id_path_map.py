import os
import csv


def gen_id_path_map(dataset_path, id_path_map):
    id_path_dict = {}
    num_files = 0
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for f in files:
            id_path_dict[num_files] = os.path.join(root.replace(dataset_path, ""), f)
            num_files += 1
    with open(id_path_map, "w+", newline="") as fp:
        w = csv.writer(fp)
        w.writerows(id_path_dict.items())
    return num_files


def get_id_path_map(id_path_map):
    with open(id_path_map, newline="") as fp:
        r = csv.reader(fp)
        id_path_dict = dict((rows[0], rows[1]) for rows in r)
        return id_path_dict


if __name__ == "__main__":
    dataset_path = os.path.join("..", "dataset", "")
    id_path_map = os.path.join("..", "output", "id_path_map.csv")
    gen_id_path_map(dataset_path, id_path_map)
