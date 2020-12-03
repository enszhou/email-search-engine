# generate total terms frequency of top 1000
import os
from collections import Counter
import csv
import time
import util


def append_ttf_df(tokens_tf, tokens_ttf, tokens_df):
    for token, tf in tokens_tf.items():
        # ttf (Total Token Frequency)
        if token in tokens_ttf:
            tokens_ttf[token] += tf
        else:
            tokens_ttf[token] = tf
        if token in tokens_df:
            tokens_df[token] += 1
        else:
            tokens_df[token] = 1


if __name__ == "__main__":

    tokens_ttf = {}
    tokens_df = {}
    cost_time = [0, 0, 0]
    temp_time = [0, 0, 0]

    for iter in range(util.low_id, util.high_id):
        if iter % 1000 == 0:
            print(iter)
        doc_id = iter
        doc_tokens_path = os.path.join("..", "output", "doc_tokens", "%d.csv" % doc_id)
        with open(doc_tokens_path) as doc_fp:
            temp_time[0] = time.time()
            r = csv.reader(doc_fp)
            tokens_tf = dict((rows[0], int(rows[1])) for rows in r)
            temp_time[1] = time.time()
            append_ttf_df(tokens_tf, tokens_ttf, tokens_df)
            temp_time[2] = time.time()
            for i in range(2):
                cost_time[i] += temp_time[i + 1] - temp_time[i]
            cost_time[2] = sum(cost_time[:-1])

    print(cost_time)

    ttf_1000 = sorted(tokens_ttf.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)[
        :1000
    ]
    with open(os.path.join("..", "output", "ttf_1000.csv"), "w+", newline="") as fp:
        w = csv.writer(fp)
        w.writerows(ttf_1000)

    tokens_1000 = list(map(lambda x: x[0], ttf_1000))
    df_1000 = list(map(lambda x: (x, tokens_df[x]), tokens_1000))
    with open(os.path.join("..", "output", "df_1000.csv"), "w+", newline="") as fp:
        w = csv.writer(fp)
        w.writerows(df_1000)
