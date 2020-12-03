import os
import util


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
        if element in util.operators_level:
            operator = operator_stack.pop()
            if util.operators_level[operator] < util.operators_level[element]:
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
                    result = util.operator_func[operator](operand)
                    operand_stack.append(result)
                else:
                    operand1 = operand_stack.pop()
                    operand2 = operand_stack.pop()
                    result = util.operator_func[operator](operand1, operand2)
                    operand_stack.append(result)
        else:
            operand = util.get_indices(element)  # set
            operand_stack.append(operand)
            i += 1
    return operand_stack.pop()


if __name__ == "__main__":
    id_path_map = os.path.join("..", "output", "id_path_map.csv")
    id_path_dict = util.get_id_path_map(id_path_map)

    while True:
        query_str = input("bool search expression: ")
        result = list(bool_query(query_str))
        result.sort()
        result_2d = list(map(lambda x: (x, id_path_dict[str(x)]), result))
        print("doc ids are: ", result_2d)

