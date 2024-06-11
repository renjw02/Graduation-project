# NL2PQL: A Task of Natural Language Processing
## Overview
This repository is a graduation research about how to convert natural language to polystore query language, which contains NL2PQL datasets, experimental code and a simple demonstration system.

## Motivation
In the era of big data, polystore systems are receiving increasing attention and wider adoption. Complex polystore query languages pose challenges for both ordinary users and professional developers in querying polystore systems. To facilitate users of polystore systems in quickly constructing query statements, this paper proposes a natural language processing task, natural language to polystore query language(NL2PQL), aiming to convert natural language into polystore query language to support polystore systems. The paper constructs a benchmark dataset and baseline methods for the natural language to polystore query language task. Additionally, it introduces an innovative approach focused on the scope and skeleton information of polystore query statements. This involves designing a neural network model based on an Encoder-Decoder architecture, injecting model scope information related to the query statements into the Encoder part, and utilizing skeleton features of the query statements in the Decoder part to enhance model performance. Experimental results on multiple datasets demonstrate that the new method significantly outperforms baseline methods, thereby confirming its effectiveness. The paper also suggests several future directions for the NL2PQL task, including multi-modal support, cross-language support, real-time interaction, and feedback mechanisms. This work is of significant importance for query generation in polystore systems.

## Folder structure

| Folder | Role | 
| ---- | ---- | 
| datasets | Datasets modified based on current NL2SQL datasets. | 
| experimental-code | Including baselines and our method code. | 
| demo-system | A demo system built for presentation. | 
