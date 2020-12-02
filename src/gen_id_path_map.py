import json
import os


def gen_id_path_map(dataset_path, id_path_map):
    id_path_dict = {}
    num_files = 0
    for root, dirs, files in os.walk(dataset_path, topdown=False):
        for f in files:
            id_path_dict[num_files] = os.path.join(root.replace(dataset_path, ""), f)
            num_files += 1
    with open(id_path_map, "w+") as fp:
        json.dump(id_path_dict, fp)
    return num_files


if __name__ == "__main__":
    dataset_path = os.path.join("..", "dataset", "")
    id_path_map = os.path.join("..", "output", "id_path_map.json")
    gen_id_path_map(dataset_path, id_path_map)
