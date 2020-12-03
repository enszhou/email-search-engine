# generate total terms frequency of top 1000
import os
import nltk
import email
import string
import csv

del_letters = string.punctuation + string.digits
del_tran_table = str.maketrans(del_letters, " " * len(del_letters))
stopwords = set(nltk.corpus.stopwords.words("english"))
stemmer = nltk.stem.SnowballStemmer("english")

docs_num = 517401
low_id = 0
high_id = docs_num
universe_set = set(range(low_id, high_id))


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
    return id_path_dict


def get_id_path_map(id_path_map):
    with open(id_path_map, newline="") as fp:
        r = csv.reader(fp)
        id_path_dict = dict((int(rows[0]), rows[1]) for rows in r)
        return id_path_dict


def doc2str(doc_fp):
    try:
        mail = email.parser.Parser().parse(doc_fp)
        mail_subject = mail.get("Subject")
        mail_body = mail.get_payload()
        return mail_subject + " " + mail_body
    except Exception as e:
        return "the"


def tokenize(doc_str):
    doc_str = doc_str.translate(del_tran_table)
    tokens = nltk.tokenize.word_tokenize(doc_str)
    return tokens


def stem(token):
    token = token.lower()
    return stemmer.stem(token)


def del_stop(token):
    return token not in stopwords


def op_and(op1, op2):
    return op1 & op2


def op_or(op1, op2):
    return op1 | op2


def op_not(op):
    return universe_set - op


operator_func = {"&": op_and, "|": op_or, "!": op_not}
operators_level = {"$": -1, ")": 0, "|": 1, "&": 2, "!": 3, "(": 4}


def get_indices(word):
    token = stemmer.stem(word)
    path = os.path.join("..", "output", "inverted_index_table", token + ".csv")
    if os.path.exists(path):
        with open(path) as fp:
            r = csv.reader(fp)
            return set(map(lambda x: int(x[0]), r))
    else:
        return set()
