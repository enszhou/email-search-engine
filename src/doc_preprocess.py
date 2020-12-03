# generate total terms frequency of top 1000
import os
from collections import Counter
import csv
import time
import util


def write_tokens(path, tokens):
    with open(path, "w+", newline="") as fp:
        token_cnt = Counter(tokens)
        token_cnt_ordered = token_cnt.most_common()
        w = csv.writer(fp)
        w.writerows(token_cnt_ordered)


if __name__ == "__main__":

    dataset_path = os.path.join("..", "dataset", "")
    id_path_map = os.path.join("..", "output", "id_path_map.csv")
    id_path_dict = util.gen_id_path_map(dataset_path, id_path_map)

    all_tokens = []
    total_tf_table = {}

    cost_time = [0, 0, 0, 0, 0, 0]
    temp_time = [0, 0, 0, 0, 0, 0]

    for iter in range(util.low_id, util.high_id):
        if iter % 1000 == 0:
            print(iter)
        doc_id = iter
        doc_path = id_path_dict[doc_id]
        # read docs
        with open(os.path.join("..", "dataset", doc_path)) as doc_fp:
            temp_time[0] = time.time()
            doc_str = util.doc2str(doc_fp)
            temp_time[1] = time.time()
            tokens = util.tokenize(doc_str)
            temp_time[2] = time.time()
            tokens = map(util.stem, tokens)
            temp_time[3] = time.time()
            tokens = filter(util.del_stop, tokens)
            temp_time[4] = time.time()
            path = os.path.join("..", "output", "doc_tokens", "%d.csv" % doc_id)
            write_tokens(path, tokens)
            temp_time[5] = time.time()
            for i in range(5):
                cost_time[i] += temp_time[i + 1] - temp_time[i]
            cost_time[5] = sum(cost_time[:-1])

    print(cost_time)
