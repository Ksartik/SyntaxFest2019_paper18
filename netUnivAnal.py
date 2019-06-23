import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import silhouette_score

"""
netUnivAnal.py:
    For analyzing the results obtained from the analysis (using Cytoscape) of the layered network 
    derived from the layering.py
    It plots the graphs and prints the silhouette score. 
    It is suggested to use it on the terminal or some other file import functions from here. 
"""


def silhouette(df, feat, par):
    n_samples = len(df.index)
    X = np.zeros(shape=(n_samples, 1))
    X[:, 0] = list(df[par])
    labels = list(df[feat])
    return silhouette_score(X, labels, metric='euclidean')


def plotFeat(df, feat, par1, par2):
    """
    Plots the par: par1(par2) for different languages and styling (giving color and shape) based on
    the "feat" (language feature from WALS)
    par1 is one of the node properties derived from the analysis of the layered network and par2 is
    the node (one of the Layer 2) whose properties are seen. 
    """
    temp_df = df[["Languages", feat]]
    temp_df = temp_df.rename(columns={feat: feat[4:]})
    feat = feat[4:]
    par = par1 + "(" + par2 + ")"
    temp_df[par] = df[par1][par2]
    temp_df = temp_df.dropna(axis=0, how='any').reset_index()
    temp_df = temp_df.sort_values(by=feat).reset_index()
    sns.scatterplot(x=par, y="Languages", hue=feat, style=feat, data=temp_df)
    plt.xticks(rotation='vertical')
    plt.tight_layout()


# Analyzed network in csv format. For more info, look into the csv file
langnet = pd.read_csv("layeredNetAnalysis.csv", header=[0, 1], nrows=33)
langs = langnet.ix[:, 0]
wos = ["SOV", "SVO", "VOS", "VSO", "OSV", "OVS"]
# WALS data for the considered languages
langs_data = pd.read_csv("LanguageWALS.csv", nrows=33)
features = list(langs_data.columns)
categories = dict(
    list(map(lambda feat: [feat, set(langs_data[feat])], features)))
for feat in features:
    langnet[feat] = langs_data[feat]

langs = list(langnet["Languages"])
worders = list(langnet["Outdegree"].columns)

# "Outperc" is found which is just outdegree of the node divided by all the nodes.
for wo in wos:
    langnet["Outperc", wo] = list(map(
        lambda x, y: x/y, langnet["Outdegree"][wo],  langnet["Outdegree"][wos].sum(axis=1)))


layer2nodes = ["SV", "VS", "OV", "VO", "VI", "IV",
               "SOV", "SVO", "VOS", "VSO", "OSV", "OVS", "ISV", "IVS", "VSI",
               "VIS", "SIV", "SVI", "OVI", "OIV", "IVO", "IOV", "VIO", "VOI",
               "SVOI", "SVIO", "SIVO", "SIOV", "SOIV", "SOVI", "OVSI", "OVIS",
               "OIVS", "OISV", "OSIV", "OSVI", "IVSO", "IVOS", "ISVO", "ISOV",
               "IOSV", "IOVS", "VSOI", "VSIO", "VISO", "VIOS", "VOSI", "VOIS"]


def plotAnalysis(langnet, feat, par2):
    """
    Returns the node parameter (and the corresponding score) whose distribution for the node "par2" 
    is best clustered by "feat" given by WALS.
    It also plots the distribution and the corresponding cluster (by "feat") for all the node 
    parameters (and optionally printing the silhouette scores) in order to decide the clusters by 
    visual examination (against the corresponding silhoette scores).
    """
    networkPars = ['AverageShortestPathLength', 'BetweennessCentrality',
                   'ClosenessCentrality', 'ClusteringCoefficient', 'Eccentricity',
                   'EdgeCount', 'Indegree', 'NeighborhoodConnectivity', 'Outdegree',
                   'PartnerOfMultiEdgedNodePairs', 'SelfLoops', 'Stress']
    if (par2 in ["SOV", "SVO", "VSO", "VOS", "OSV", "OVS"]):
        networkPars.append('Outperc')
    silhouettes = {}
    i = 0
    for par in networkPars:
        temp_df = pd.DataFrame(langnet[par][par2])
        temp_df[feat] = langnet[feat]
        temp_df = temp_df.dropna(
            axis=0, how='any').reset_index().rename(columns={par2: par})
        try:
            silhouettes[par] = silhouette(temp_df, feat, par)
        except:
            pass
        # print(par, " : ", silhouette(temp_df, feat, par))
        plt.figure(i)
        plotFeat(langnet, feat, par, par2)
        i += 1
    maxkey = max(silhouettes, key=silhouettes.get)
    # maxkey = sorted(silhouettes, key=silhouettes.get)
    return (maxkey, silhouettes[maxkey])
    # return (silhouettes)


def meanDiffCluster(langnet, feat, par2):
    """
    An alternate way to calculate cluster scores. The details are not important since we do not 
    report it - it's just to assist "visual examination" described above. 
    """
    networkPars = ['AverageShortestPathLength', 'BetweennessCentrality',
                   'ClosenessCentrality', 'ClusteringCoefficient', 'Eccentricity',
                   'EdgeCount', 'Indegree', 'NeighborhoodConnectivity', 'Outdegree',
                   'PartnerOfMultiEdgedNodePairs', 'SelfLoops', 'Stress']
    if (par2 in ["SOV", "SVO", "VSO", "VOS", "OSV", "OVS"]):
        networkPars.append('Outperc')
    meanDiffs = {}
    i = 0
    for par in networkPars:
        temp_df = pd.DataFrame(langnet[par][par2])
        temp_df[feat] = langnet[feat]
        temp_df = temp_df.dropna(
            axis=0, how='any').reset_index().rename(columns={par2: par})
        dfgrp = temp_df.groupby(by=feat)
        dfgmean = dfgrp.mean()[par]
        dfgstd = dfgrp.std()[par]
        largestDist = max(temp_df[par]) - min(temp_df[par])
        # dfgmax = dfgrp.max()[par]
        # dfgmin = dfgrp.min()[par]
        s = 0
        for g1 in dfgrp.groups:
            for g2 in dfgrp.groups:
                if (g1 != g2):
                    s += abs(dfgmean[g1] - dfgmean[g2]) - \
                        (0 if (np.isnan(dfgstd[g1])) else dfgstd[g1]) - \
                        (0 if (np.isnan(dfgstd[g2])) else dfgstd[g2])
        meanDiffs[par] = np.nan if (np.isnan(largestDist) or (largestDist == 0)
                                    ) else (s/(2*largestDist))
        print(par, meanDiffs[par])
        plt.figure(i)
        plotFeat(langnet, feat, par, par2)
        i += 1
    maxkey = max(meanDiffs, key=meanDiffs.get)
    return(maxkey, meanDiffs[maxkey])


def univAnal(langnet, feat, univi):
    """
    Plots the distribution of each node parameter for each node in the Layer 2 of the proposed network.
    It also saves a csv storing the parameter and the node which clusters the best and the corresponding
    cluster scores (silhouette and mean difference).
    """
    global layer2nodes
    df = {"Node": [], "NetworkParameter1": [], "ClusterScore1": [],
          "NetworkParameter2": [], "ClusterScore2": []}
    i = 0
    for n in layer2nodes:
        try:
            netpar1, score1 = plotAnalysis(langnet, feat, n)
            netpar2, score2 = meanDiffCluster(langnet, feat, n)
            df["Node"].append(n)
            df["NetworkParameter1"].append(netpar1)
            df["ClusterScore1"].append(score1)
            df["NetworkParameter2"].append(netpar2)
            df["ClusterScore2"].append(score2)
            plt.figure(i)
            plotFeat(langnet, feat, netpar1, n)
            i += 1
            plt.figure(i)
            plotFeat(langnet, feat, netpar2, n)
            i += 1
            # print(n, netpar, score)
        except:
            pass
    # return (pd.DataFrame(df))
    pd.DataFrame(df).to_csv("UniversalAnalysis/univ" +
                            str(univi) + ".csv", index=False, columns=["Node", "NetworkParameter1", "ClusterScore1", "NetworkParameter2", "ClusterScore2"])

#
#
#
# Experiment 1 :
#
#
#


"""
Universal 1
"In declarative sentences with nominal subject and object,
the dominant order is almost always one in which the subject precedes the object."
"""

nnodes = langnet["Outdegree"]
maxwos = []
for i in nnodes.index:
    maxperc = 0
    s = nnodes.ix[i, :].sum()
    for wo in wos:
        woperc = nnodes.at[i, wo]/s
        if (woperc > maxperc):
            maxperc = woperc
            maxwo = wo
    maxwos.append(maxwo)

langnet["WordOrder"] = maxwos
langnet["WordOrder"].value_counts().plot(kind='bar')
plt.ylabel("No. of languages")
plt.xlabel("Word-order")
plt.savefig("UniversalAnalysis/univ1.png")


"""
Universal 3
"Languages with dominant VSO order are always prepositional."
"""
# plotFeat(langnet, "85A Order of Adposition and Noun Phrase", "Outperc", "VSO")
plotAnalysis(langnet, "85A Order of Adposition and Noun Phrase", "VSO")
# -- gave opposite results

"""
Universal 4
"With overwhelmingly greater than chance frequency, languages with normal SOV order
 are post-positional."
"""
# plotFeat(langnet, "85A Order of Adposition and Noun Phrase", "Outperc", "VSO")
plotAnalysis(langnet, "85A Order of Adposition and Noun Phrase", "SOV")

"""
Universal 5
"If a language has dominant SOV order and the genitive follows the governing noun,
then the adjective likewise follows the noun."
"""
genitive_feat = "86A Order of Genitive and Noun"
adj_feat = "87A Order of Adjective and Noun"
df = langnet.loc[langnet[genitive_feat] == "2 Noun-Genitive"]
netpar1, score1 = plotAnalysis(df, adj_feat, "SOV")
netpar2, score2 = meanDiffCluster(df, adj_feat, "SOV")
# plt.figure(1)
# plotFeat(df, adj_feat, netpar1, "SOV")
# plt.figure(2)
# plotFeat(df, adj_feat, netpar2, "SOV")
# plt.show()


"""
Universal 6
"All languages with dominant VSO order
have SVO as an alternative or as the only alternative basic order."
Correlation between Outperc of SOV and VSO outpercs
"""
df = pd.DataFrame(langnet["Languages"])
df["SVO"] = langnet["Outperc"]["SVO"]
df["VSO"] = langnet["Outperc"]["VSO"]
df = df.dropna(axis=0, how='any').reset_index()
plt.plot(df["Languages"], list(df["SVO"]), label="SVO")
plt.plot(df["Languages"], list(df["VSO"]), label="VSO", linestyle='--')
plt.legend(loc='best')
plt.xlabel("Languages")
plt.ylabel("Outperc")
plt.xticks(rotation="vertical")
plt.show()
# >>> from scipy import stats
# >>> stats.pearsonr(df["SVO"], df["VSO"])
# (0.069297330662097104, 0.73125673386347478)


"""
Universal 12
"If a language has dominant order VSO in declarative sentences,
it always puts interrogative words or phrases first in interrogative word questions;
if it has dominant order SOV in declarative sentences,
there is never such an invariant rule."
"""
interr_feat = "93A Position of Interrogative Phrases in Content Questions"
plotFeat(langnet, interr_feat, "Outperc", "VSO")
plotFeat(langnet, interr_feat, "Outperc", "SOV")

#
#
#
#
# Experiment 2 :
#
#
#

"""
Order of subject, verb and object
"""
wos = ["SOV", "SVO", "VSO", "VOS", "OVS", "OSV"]
dfwo = {"WordOrder": [], "NetworkParameter": [], "Score": []}
for wo in wos:
    netpar, score = plotAnalysis(
        langnet, "81A Order of Subject, Object and Verb", wo)
    dfwo["WordOrder"].append(wo)
    dfwo["NetworkParameter"].append(netpar)
    dfwo["Score"].append(score)
    # print(meanDiffCluster(langnet, "81A Order of Subject, Object and Verb", wo))

pd.DataFrame(dfwo).to_csv("UniversalAnalysis/word_order.csv")


"""
Order of Adposition and Noun Phrase
"""
adpos_feat = "85A Order of Adposition and Noun Phrase"
univAnal(langnet, adpos_feat, 3)


"""
Order of Genitive and Noun
Order of Adjective and Noun
"""
genitive_feat = "86A Order of Genitive and Noun"
adj_feat = "87A Order of Adjective and Noun"
univAnal(langnet, genitive_feat, 51)
univAnal(langnet, adj_feat, 52)


"""
Position of Interrogative Phrase in Content Questions
"""
univAnal(langnet, interr_feat, 12)
