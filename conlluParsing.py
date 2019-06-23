from io import open
import matplotlib.pyplot as plt
import pandas as pd
import sys
import numpy as np
import csv
import os

"""
conlluParsing.py : 
    Building the network from the raw text files downloaded from Universal Dependencies (https://universaldependencies.org). 
    The language file (in .txt format) is read line-by-line and the sentences are formed by considering an empty line. 
    Each sentence is, then, parsed to convert into usable edge csv and node csv format. These are appended to corresponding
    separate node and edge data for the language. 
"""


def parseSentence(df, nodeind):
    """
    It parses a sentence given in a dataframe format. The sentence in CONLL-U format has been converted to
    a dataframe with rows as each word (or punctuation) and columns as the CONLL-U fields (along with a concatenated "Lemma:UPOS") field.
    The dataframe is then converted to a dataframe suitable for further network analysis with each row corresponding to an edge and 
    the columns as : 
        1. SourceNode : node from which the edge originates (node identified by "<Lemma>:<UPOS>")
        2. TargetNode : node at which the edge goes to (lands on) (node identified by "<Lemma>:<UPOS>")
        3. InteractionType : type of edge (directed or undirected)
        4. DEPREL : dependency relation of this edge (annotated in the CONLL-U)
        5. SententialDistance : no. of constituents (words) between the two nodes of the edge
        6. SourceNodeIndex : node index in the original node csv (which has all CONLL-U fields) to get further information about the source node
        7. TargetNodeIndex : node index in the original node csv (which has all CONLL-U fields) to get further information about the target node
    """
    df["TargetNode"] = df["LemmaPos"]
    # nodeind means the starting node index of this sentence in the original node csv (in order to relate node data with edge data)
    df["TargetNodeIndex"] = list(map(lambda x: x + nodeind, df.index))
    df["tni"] = df.index
    df["SententialDistance"] = df["HEAD"] - df["INDEX"]
    df["SourceNodeIndex"] = df["TargetNodeIndex"] + df["SententialDistance"]
    df["sni"] = df["tni"] + df["SententialDistance"]
    # "pp" means directed in Cytoscape
    df["InteractionType"] = ["pp"] * len(df.index)
    for i in df.index:
        try:
            df.at[i, "SourceNode"] = df.at[df.at[i, "sni"], "LemmaPos"]
        except:
            pass
    # no edge is counted for the head dependency
    df = df.loc[df["HEAD"] != 0]
    # removing punctuations from the edge data
    df = df.loc[df["UPOS"] != "PUNCT"]
    df = df[["SourceNode", "InteractionType",
             "TargetNode", "DEPREL", "SententialDistance", "SourceNodeIndex", "TargetNodeIndex"]]
    return df

# All languages can be listed from here and then this code can be run for each of them.
# langs = list(map(lambda lang: lang[:-4], os.listdir("./data_in_txt")))


max_sentences = 30000
# Language taken as the first argument
language = sys.argv[1]

cols = ['LemmaPos', 'INDEX', 'FORM', 'LEMMA', 'UPOS',
        'XPOS', 'FEATS', 'HEAD', 'DEPREL', 'DEPS', 'MISC']

edgeCols = ["SourceNode", "InteractionType", "TargetNode", "DEPREL",
            "SententialDistance", "SourceNodeIndex", "TargetNodeIndex"]

dict1 = {'LemmaPos': [], 'INDEX': [], 'FORM': [], 'LEMMA': [], 'UPOS': [],
         'XPOS': [], 'FEATS': [], 'HEAD': [], 'DEPREL': [], 'DEPS': [], 'MISC': []}

node_index = 0
prev_node_index = 0

with open('data_in_txt/' + language + '.txt', 'r', encoding="utf-8") as f, open("./NetworkData/NodeCsv/"+language+".csv", "w+", encoding="utf8", newline='') as dataF, open("./NetworkData/EdgeCsv/" + language + ".csv", "w+", encoding="utf8", newline='') as dataEdgeF:
    node_writer = csv.DictWriter(dataF, fieldnames=cols)
    node_writer.writeheader()
    edge_writer = csv.DictWriter(dataEdgeF, fieldnames=edgeCols)
    edge_writer.writeheader()
    count_sentences = 0
    for line in f.readlines():
        if (line[0] == '#'):
            # commented lines
            pass
        else:
            if (count_sentences == max_sentences):
                # no. of sentences for this language reach the max. bound
                break
            elif (line == '\n'):
                NodeDF = pd.DataFrame(dict1, columns=cols)
                dict1 = {'LemmaPos': [], 'INDEX': [], 'FORM': [], 'LEMMA': [], 'UPOS': [],
                         'XPOS': [], 'FEATS': [], 'HEAD': [], 'DEPREL': [], 'DEPS': [], 'MISC': []}
                if (len(NodeDF.index) >= 5):    # sentence length at least 5
                    try:
                        for i in range(len(NodeDF.index)):
                            node_writer.writerow(dict(NodeDF.iloc[i]))
                        count_sentences += 1
                        EdgeDF = parseSentence(NodeDF, prev_node_index)
                        for i in range(len(EdgeDF.index)):
                            edge_writer.writerow(dict(EdgeDF.iloc[i]))
                        prev_node_index = node_index
                        if ((count_sentences % 1000) == 0):
                            # For logging how many sentences done
                            print(count_sentences + " done")
                    except:
                        node_index = prev_node_index
                        pass
                else:
                    node_index = prev_node_index
            # elif ((len(line) > 4) and ((line[1] == '-') or (line[2] == '-') or (line[3] == '-'))):
            #     # Check or think whether morphemes should be taken
            #     # print('left a line')
            #     pass
            # do nothing : dont't consider this line as this contain multiple morphenes which are seperated in next lines
            else:
                try:
                    description = line.split()
                    index = int(description[0])
                    if ((index >= 1) and (description[2] != "_") and (len(description) == 10)):
                        dict1['LemmaPos'].append(
                            description[2]+":"+description[3])
                        dict1['INDEX'].append(int(description[0]))
                        dict1['FORM'].append(description[1])
                        dict1['LEMMA'].append(description[2])
                        dict1['UPOS'].append(description[3])
                        dict1['XPOS'].append(description[4])
                        dict1['FEATS'].append(description[5])
                        dict1['HEAD'].append(int(description[6]))
                        dict1['DEPREL'].append(description[7])
                        dict1['DEPS'].append(description[8])
                        dict1['MISC'].append(description[9])
                        node_index += 1
                except:
                    pass

print("DONE " + language + "!!!")
