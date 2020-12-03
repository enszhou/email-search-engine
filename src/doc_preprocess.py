# generate total terms frequency of top 1000
import os
from collections import Counter
import csv
import time
import util
from multiprocessing import Pool


def write_tokens(path, tokens):
    with open(path, "w+", newline="") as fp:
        token_cnt = Counter(tokens)
        token_cnt_ordered = token_cnt.most_common()
        w = csv.writer(fp)
        w.writerows(token_cnt_ordered)


def preprocess(core_id, low_id, high_id, id_path_dict):
    print("Task%d starts..." % core_id)
    for iter in range(low_id, high_id):
        if iter % 1000 == 0:
            print(iter)
        doc_id = iter
        doc_path = id_path_dict[doc_id]
        with open(os.path.join("..", "dataset", doc_path)) as doc_fp:
            doc_str = util.doc2str(doc_fp)
            tokens = util.tokenize(doc_str)
            tokens = map(util.stem, tokens)
            tokens = filter(util.del_stop, tokens)
            path = os.path.join("..", "output", "doc_tokens", "%d.csv" % doc_id)
            write_tokens(path, tokens)
    print("Task%d ends..." % core_id)


if __name__ == "__main__":

    dataset_path = os.path.join("..", "dataset", "")
    id_path_map = os.path.join("..", "output", "id_path_map.csv")
    id_path_dict = util.gen_id_path_map(dataset_path, id_path_map)

    cores = 8
    core_payload = 12000
    print("Start preprocessing")
    start_time = time.time()
    p = Pool(cores)
    for i in range(cores):
        p.apply_async(
            preprocess, args=(i, i * core_payload, (i + 1) * core_payload, id_path_dict)
        )
    p.close()
    p.join()
    end_time = time.time()
    print("Total time: %fs" % (end_time - start_time))
