# generate total terms frequency of top 1000
import os
from collections import Counter
import csv
import time
import util
from multiprocessing import Pool
import math


def wordvec(core_id, low_id, high_id):
    print("Task%d starts..." % core_id)

    with open(os.path.join("..", "output", "df_1000.csv")) as fp:
        r = csv.reader(fp)
        target_tokens_df = dict((rows[0], int(rows[1])) for rows in r)
        target_tokens_list = sorted(list(map(lambda x: x[0], target_tokens_df)))

    for iter in range(low_id, high_id):
        if iter >= util.docs_num:
            break
        if iter % 1000 == 0:
            print(iter)
        doc_id = iter
        doc_tokens_path = os.path.join("..", "output", "doc_tokens", "%d.csv" % doc_id)
        with open(doc_tokens_path) as doc_fp:
            r = csv.reader(doc_fp)
            doc_tokens_tf = dict((rows[0], int(rows[1])) for rows in r)

            w_td = [0] * 1000
            j = 0
            for token in target_tokens_list:
                if token in doc_tokens_tf:
                    w_td[j] = (1 + math.log(doc_tokens_tf[token], 10)) * math.log(
                        util.docs_num / target_tokens_df[token], 10
                    )
                j += 1

            path = os.path.join("..", "output", "doc_wordvec", "%d.csv" % doc_id)
            with open(path, "w+", newline="") as fp:
                w = csv.writer(fp)
                w.writerow(w_td)
    print("Task%d ends..." % core_id)


if __name__ == "__main__":
    cores = 8
    core_payload = int(util.docs_num / cores) + 1
    # core_payload = 1000
    print("Start generating")
    start_time = time.time()
    p = Pool(cores)
    for i in range(cores):
        p.apply_async(wordvec, args=(i, i * core_payload, (i + 1) * core_payload))
    p.close()
    p.join()
    end_time = time.time()
    print("Total time: %fs" % (end_time - start_time))
