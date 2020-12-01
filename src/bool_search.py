# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import os
import nltk
from nltk.corpus import wordnet
import email
import time
import string
from multiprocessing import Pool


punct_tran_table = str.maketrans(string.punctuation, " " * len(string.punctuation))
stopwords = set(nltk.corpus.stopwords.words("english"))
pos_tran = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

lemmer = nltk.stem.WordNetLemmatizer()
porter_stemmer = nltk.stem.porter.PorterStemmer()
lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
snowball_stemmer = nltk.stem.SnowballStemmer("english")
stemmer = snowball_stemmer


# %%
def gen_id_path_map(dataset_path, id_path_map):
    with open(id_path_map, "w+") as f_id_path_map:
        num_files = 0
        for root, dirs, files in os.walk(dataset_path, topdown=False):
            for f in files:
                f_id_path_map.write(
                    str(num_files)
                    + " "
                    + os.path.join(root.replace(dataset_path, ""), f)
                    + "\n"
                )
                num_files += 1
    return num_files


def doc2str(doc_fp):
    try:
        mail = email.parser.Parser().parse(doc_fp)
        mail_subject = mail.get("Subject")
        mail_body = mail.get_payload()
        return mail_subject + " " + mail_body
    except Exception as e:
        return "the"


def tokenize(doc_str):
    doc_str = doc_str.translate(punct_tran_table)
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


def stem_single(token):
    token = token.lower()
    return stemmer.stem(token)


def stem(tokens):
    stem_tokens = []
    for token in tokens:
        token = token.lower()
        stem_tokens.append(stemmer.stem(token))
    return stem_tokens


def del_duplicates(tokens):
    # remove duplicates in tokens
    tokens = list(set(tokens))
    return tokens


def del_stops(tokens):
    filtered_tokens = [token for token in tokens if token not in stopwords]
    return filtered_tokens


def append_tokens(tokens, doc_id, inverted_indices):
    for token in tokens:
        if token in inverted_indices:
            if doc_id != inverted_indices[token][-1]:
                inverted_indices[token].append(doc_id)
        else:
            inverted_indices[token] = [doc_id]


# %%
if __name__ == "__main__":
    # run only once to generate map file between doc id and doc path
    dataset_path = os.path.join("..", "dataset", "")
    id_path_map = os.path.join("..", "output", "id_path_map.txt")
    # gen_id_path_map(dataset_path, id_path_map)

    max_iters = 1000
    inverted_indices = {}
    cost_time = [0, 0, 0, 0, 0, 0]
    temp_time = [0, 0, 0, 0, 0, 0]
    with open(id_path_map, "r") as f_id_path_map:
        iter = 0
        while True:
            print(iter)
            iter += 1
            line = f_id_path_map.readline().strip("\n")
            if not line or iter > max_iters:
                break
            doc_id, doc_path = line.split(" ")
            doc_id = int(doc_id)
            # read docs
            with open(os.path.join("..", "dataset", doc_path), "r") as doc_fp:
                temp_time[0] = time.time()
                doc_str = doc2str(doc_fp)
                temp_time[1] = time.time()
                tokens = tokenize(doc_str)
                temp_time[2] = time.time()
                # tokens = stem(tokens)
                with Pool(4) as p:
                    tokens = p.map(stem_single, tokens)
                temp_time[3] = time.time()
                tokens = del_stops(tokens)
                temp_time[4] = time.time()
                # add tokens of a certain doc into inverted index table
                append_tokens(tokens, doc_id, inverted_indices)
                temp_time[5] = time.time()
                for i in range(5):
                    cost_time[i] += temp_time[i + 1] - temp_time[i]
                cost_time[5] = sum(cost_time[:-2])
    print(cost_time)