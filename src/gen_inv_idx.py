import os
import nltk
from nltk.corpus import wordnet
import email
import time
import string
import csv
import gen_id_path_map

id_path_map = os.path.join("..", "output", "id_path_map.csv")
id_path_dict = gen_id_path_map.get_id_path_map(id_path_map)

# get top 1000 tf tokens
with open(os.path.join("..", "output", "ttf_1000.csv")) as fp:
    r = csv.reader(fp)
    total_tf_1000 = list(r)
target_tokens = set(map(lambda x: x[0], total_tf_1000))

del_letters = string.punctuation + string.digits
del_tran_table = str.maketrans(del_letters, " " * len(del_letters))
stopwords = set(nltk.corpus.stopwords.words("english"))
pos_tran = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

lemmer = nltk.stem.WordNetLemmatizer()
porter_stemmer = nltk.stem.porter.PorterStemmer()
lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
snowball_stemmer = nltk.stem.SnowballStemmer("english")
stemmer = snowball_stemmer


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


def lem(tokens):
    token_tags = nltk.pos_tag(tokens)
    stem_tokens = []
    for token, tag in token_tags:
        token = token.lower()
        if tag[0] in pos_tran:
            stem_tokens.append(lemmer.lemmatize(token, pos=pos_tran[tag[0]]))
        else:
            stem_tokens.append(token)
    return stem_tokens


def stem(token):
    token = token.lower()
    return stemmer.stem(token)


def del_duplicates(tokens):
    # remove duplicates in tokens
    tokens = list(set(tokens))
    return tokens


def token_filter(token):
    return token in target_tokens


def del_stop(token):
    return token not in stopwords


def append_tokens(tokens, doc_id, inverted_indices):
    for token in tokens:
        if token in inverted_indices:
            if doc_id != inverted_indices[token][-1]:
                inverted_indices[token].append(doc_id)
        else:
            inverted_indices[token] = [doc_id]


max_iters = 10000
inverted_indices = {}
cost_time = [0, 0, 0, 0, 0, 0]
temp_time = [0, 0, 0, 0, 0, 0]
iter = 0

for doc_id, doc_path in id_path_dict.items():
    if iter % 1000 == 0:
        print(iter)
    iter += 1
    if iter > max_iters:
        break
    doc_id = int(doc_id)
    # read docs
    with open(os.path.join("..", "dataset", doc_path)) as doc_fp:
        temp_time[0] = time.time()
        doc_str = doc2str(doc_fp)
        temp_time[1] = time.time()
        tokens = tokenize(doc_str)
        temp_time[2] = time.time()
        tokens = map(stem, tokens)
        temp_time[3] = time.time()
        tokens = filter(token_filter, tokens)
        temp_time[4] = time.time()
        # add tokens of a certain doc into inverted index table
        append_tokens(tokens, doc_id, inverted_indices)
        temp_time[5] = time.time()
        for i in range(5):
            cost_time[i] += temp_time[i + 1] - temp_time[i]
        cost_time[5] = sum(cost_time[:-1])


i = 0
for key, value in inverted_indices.items():
    i += 1
    print(i, key)
    with open("../output/inverted_index_table/" + key + ".csv", "w+", newline="") as fp:
        w = csv.writer(fp)
        for j in value:
            w.writerow((j, id_path_dict[str(j)]))
