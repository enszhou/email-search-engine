import os
import time
import csv
import util


def token_filter(token):
    return token in target_tokens


def append_tokens(tokens, doc_id, inverted_indices):
    for token in tokens:
        if token in inverted_indices:
            if doc_id != inverted_indices[token][-1]:
                inverted_indices[token].append(doc_id)
        else:
            inverted_indices[token] = [doc_id]


if __name__ == "__main__":
    # get top 1000 tf tokens
    with open(os.path.join("..", "output", "ttf_1000.csv")) as fp:
        r = csv.reader(fp)
        total_tf_1000 = list(r)
    target_tokens = set(map(lambda x: x[0], total_tf_1000))

    inverted_indices = {}
    cost_time = [0, 0, 0, 0]
    temp_time = [0, 0, 0, 0]

    for iter in range(util.low_id, util.high_id):
        if iter % 1000 == 0:
            print(iter)
        doc_id = iter
        doc_tokens_path = os.path.join("..", "output", "doc_tokens", "%d.csv" % doc_id)
        with open(doc_tokens_path) as doc_fp:
            temp_time[0] = time.time()
            r = csv.reader(doc_fp)
            tokens = set(map(lambda x: x[0], list(r)))
            temp_time[1] = time.time()
            tokens = filter(token_filter, tokens)
            temp_time[2] = time.time()
            # add tokens of a certain doc into inverted index table
            append_tokens(tokens, doc_id, inverted_indices)
            temp_time[3] = time.time()
            for i in range(3):
                cost_time[i] += temp_time[i + 1] - temp_time[i]
            cost_time[3] = sum(cost_time[:-1])

    print(cost_time)
    i = 0
    for key, value in inverted_indices.items():
        i += 1
        print(i, key)
        path = os.path.join("..", "output", "inverted_index_table", key + ".csv")
        with open(path, "w+", newline="") as fp:
            w = csv.writer(fp)
            for j in value:
                w.writerow([j])
