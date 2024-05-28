# NL2PQL datasets

This folder contains data and code for building and evaluating systems that convert natural language to polystore query language.

For each dataset, we provide:

- A json file contains NL-PQL pairs
- A database
- A database schema



## Original NL2SQL datasets

Fork from repository [text2sql-data](https://github.com/jkkummerfeld/text2sql-data) and [WikiSQL](https://github.com/salesforce/WikiSQL).



## Tips

`.db`、`.sqlite`：sqlite数据库文件

`*_old.json`：自原数据集转换后得到的文件

`*_original.json`：包含alias的PQL

`*.json`：删去了alias和无法执行的PQL

`*_preprocessed.json`：预处理后输入SESD的文件

