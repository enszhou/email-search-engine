# Web Lab1

## 实验环境

+ windows10
+ python3.8

## 运行方式

将预生成的output放置好，执行`*_search.py`脚本，进行对应的检索。

或者按照后面报告所述方式，一步一步生成倒排表词向量等，再进行检索。

## 目录结构

+ output：
  + doc_tokens：文档词条化的中间结果
  + doc_wordvec：文档的词向量
  + inverted_index_table：top1000词的倒排索引
  + df_1000.csv：top1000词条的文档频率
  + ttf_1000.csv：top1000词条的总词项频率
+ src：
  + bool_search.py：布尔检索入口脚本
  + doc_preprocess.py：文档词条化预处理
  + gen_inv_idx.py：生成倒排索引
  + gen_ttf_df_1000.py：生成top1000词条以及其文档频率
  + gen_wordvec.py：生成文档的词向量表示
  + semantic_search.py：语义检索
  + util.py：一些全局的工具类函数

## 关键函数说明

见`实验报告.pdf`