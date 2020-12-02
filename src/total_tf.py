# generate total terms frequency of top 1000
import os
import nltk
from nltk.corpus import wordnet
import email
import time
import string
from collections import Counter
import yaml


punct_tran_table = str.maketrans(string.punctuation, " " * len(string.punctuation))
stopwords = set(nltk.corpus.stopwords.words("english"))
pos_tran = {"N": wordnet.NOUN, "V": wordnet.VERB, "J": wordnet.ADJ, "R": wordnet.ADV}

lemmer = nltk.stem.WordNetLemmatizer()
porter_stemmer = nltk.stem.porter.PorterStemmer()
lancaster_stemmer = nltk.stem.lancaster.LancasterStemmer()
snowball_stemmer = nltk.stem.SnowballStemmer("english")
stemmer = snowball_stemmer


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


def stem(token):
    token = token.lower()
    return stemmer.stem(token)



def del_stop(token):
    return token not in stopwords



# run only once to generate map file between doc id and doc path
dataset_path = os.path.join("..", "dataset", "")
id_path_map = os.path.join("..", "output", "id_path_map.txt")
# gen_id_path_map(dataset_path, id_path_map)

max_iters = 1000000
all_tokens = []
total_tf_table = {}
cost_time = [0, 0, 0, 0, 0, 0]
temp_time = [0, 0, 0, 0, 0, 0]
with open(id_path_map, "r") as f_id_path_map:
    iter = 0
    while True:
        if iter % 1000 == 0:
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
            tokens = map(stem, tokens)
            temp_time[3] = time.time()
            tokens = filter(del_stop, tokens)
            temp_time[4] = time.time()
            all_tokens.extend(tokens)
            temp_time[5] = time.time()
            for i in range(5):
                cost_time[i] += temp_time[i + 1] - temp_time[i]
            cost_time[5] = sum(cost_time[:-1])



total_tf_table = Counter(all_tokens)
with open("../output/total_tf.yaml","w+") as fp:
    yaml.dump(total_tf_table, fp)


tf_table_ordered = total_tf_table.most_common()
with open("../output/total_tf_ordered.yaml","w+") as fp:
    yaml.dump(tf_table_ordered, fp)


tf_table_1000 = total_tf_table.most_common(1000)
with open("../output/total_tf_1000.yaml","w+") as fp:
    yaml.dump(tf_table_1000, fp)

