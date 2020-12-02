# generate total terms frequency of top 1000
import os
import nltk
from nltk.corpus import wordnet
import email
import string
from collections import Counter
import json


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


def stem(token):
    token = token.lower()
    return stemmer.stem(token)


def del_stop(token):
    return token not in stopwords


dataset_path = os.path.join("..", "dataset", "")
id_path_map = os.path.join("..", "output", "id_path_map.json")

with open(id_path_map) as fp:
    id_path_dict = json.load(fp)

max_iters = 10000
all_tokens = []
total_tf_table = {}
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
        doc_str = doc2str(doc_fp)
        tokens = tokenize(doc_str)
        tokens = map(stem, tokens)
        tokens = filter(del_stop, tokens)
        all_tokens.extend(tokens)


total_tf_table = Counter(all_tokens)
total_tf_table_1000 = total_tf_table.most_common(1000)
with open("../output/ttf_1000.json", "w+") as fp:
    json.dump(total_tf_table_1000, fp)
