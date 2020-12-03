# generate total terms frequency of top 1000
import os
from collections import Counter
import csv
import time
import util
from multiprocessing import Pool
import math
import numpy as np


def get_top(dic, top=10):
    return sorted(dic.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[:top]


def gen_query_vec(query_str):
    with open(os.path.join("..", "output", "df_1000.csv")) as fp:
        r = csv.reader(fp)
        target_tokens_df = dict((rows[0], int(rows[1])) for rows in r)
        target_tokens_list = sorted(list(map(lambda x: x[0], target_tokens_df)))

    query_tokens = util.tokenize(query_str)
    query_tokens = map(util.stem, query_tokens)
    query_tokens = filter(util.del_stop, query_tokens)

    str_tokens_tf = Counter(query_tokens)

    query_vec = [0] * 1000
    j = 0
    for token in target_tokens_list:
        if token in query_tokens:
            query_vec[j] = (1 + math.log(str_tokens_tf[token], 10)) * math.log(
                util.docs_num / target_tokens_df[token], 10
            )
        j += 1
    return query_vec


def query(core_id, query_vec, low_id, high_id):
    print("Task%d starts..." % core_id)
    docs_cos = {}
    for iter in range(low_id, high_id):
        if iter >= util.docs_num:
            break
        if iter % 1000 == 0:
            print(iter)
        doc_id = iter
        doc_wordvec_path = os.path.join(
            "..", "output", "doc_wordvec", "%d.csv" % doc_id
        )
        with open(doc_wordvec_path) as fp:
            r = csv.reader(fp)
            doc_vec = list(map(lambda x: float(x), list(r)[0]))
        v1 = np.asarray(query_vec)
        v2 = np.asarray(doc_vec)
        cos = np.inner(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
        docs_cos[doc_id] = cos
    print("Task%d ends..." % core_id)
    return get_top(docs_cos, 10)


if __name__ == "__main__":

    id_path_map = os.path.join("..", "output", "id_path_map.csv")
    id_path_dict = util.get_id_path_map(id_path_map)

    cores = 8
    core_payload = int(util.docs_num / cores) + 1
    # core_payload = 1000

    while True:
        query_str = input("Query sentence: ")
        query_vec = gen_query_vec(query_str)

        print("Start searching")
        start_time = time.time()
        results = []
        p = Pool(cores)
        for i in range(cores):
            results.append(
                p.apply_async(
                    query, args=(i, query_vec, i * core_payload, (i + 1) * core_payload)
                )
            )
        p.close()
        p.join()
        docs_cos_top_80 = []
        for res in results:
            docs_cos_top_80.extend(res.get())
        docs_cos = get_top(dict(docs_cos_top_80), 10)
        query_results = map(lambda x: (x[0], x[1], id_path_dict[x[0]]), docs_cos)
        end_time = time.time()
        print("Total time: %fs" % (end_time - start_time))
        print(query_results)
