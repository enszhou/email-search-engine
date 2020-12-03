import json
import os
import nltk

universe_set = set(range(517401))
stemmer = nltk.stem.SnowballStemmer("english")


def op_and(op1, op2):
    return op1 & op2


def op_or(op1, op2):
    return op1 | op2


def op_not(op):
    return universe_set - op


def get_indices(word):
    token = stemmer.stem(word)
    path = os.path.join("..", "output", "inverted_index_table", token)
    if os.path.exists(path):
        with open(path) as fp:
            l = json.load(fp)
            return set(l)
    else:
        return set()


operators_level = {"$": -1, ")": 0, "|": 1, "&": 2, "!": 3, "(": 4}
operator_func = {"&": op_and, "|": op_or, "!": op_not}


def bool_query(origin_query_str):
    query_str = (
        origin_query_str.lower()
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace("and", "&")
        .replace("or", "|")
        .replace("not", "!")
    )
    query_exp = query_str.split()
    query_exp.append("$")

    operand_stack = list()
    operator_stack = list("$")
    i = 0
    while True:
        element = query_exp[i]
        if element in operators_level:
            operator = operator_stack.pop()
            if operators_level[operator] < operators_level[element]:
                operator_stack.append(operator)
                operator_stack.append(element)
                i += 1
            else:
                if operator == "$":
                    break
                elif operator == "(":
                    if element != ")":
                        operator_stack.append(operator)
                        operator_stack.append(element)
                    i += 1
                elif operator == "!":
                    operand = operand_stack.pop()
                    result = operator_func[operator](operand)
                    operand_stack.append(result)
                else:
                    operand1 = operand_stack.pop()
                    operand2 = operand_stack.pop()
                    result = operator_func[operator](operand1, operand2)
                    operand_stack.append(result)
        else:
            operand = get_indices(element)  # set
            operand_stack.append(operand)
            i += 1
    return operand_stack.pop()


while True:
    query_str = input("bool search expression: ")
    print("doc ids are: ", bool_query(query_str))

