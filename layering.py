# -*- coding: utf-8 -*-
import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as plt
import os
import math
from functools import reduce

"""
layering.py:
    The main script to convert a language parsed data from conlluParsing.py into the layers described in the paper : 
    "Can Greenbergian universals be induced from the language networks?" by Sharma et al.
"""

# Helper functions for probability calculations :
def sphere_project(p):
    # converting the vector/point p to its unit vector
    m = 1/((p[0]**2 + p[1]**2 + p[2]**2)**0.5)
    return (p[0]*m, p[1]*m, p[2]*m)


def spherical_distance(p1, p2):
    # great-circle distance between two points lying on a unit sphere
    C = ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2 + (p1[2] - p2[2])**2)**0.5
    return (2*math.asin(C/2))


def spherical_norm_pdf(mean_dist, var, p, mean_point):
    """
    Assuming that the points p are normally distributed (according to great-circle distance) 
    around the mean points (described below), it finds the pdf of such a normal distribution
    "mean_dist" is the distance from the closest mean point.
    "var" is a given variance
    """
    d = spherical_distance(p, mean_point)
    return math.exp(- (d - mean_dist)**2/(2*var))/((2*math.pi*var)**2)


def gaussian_prob(mean_point, var, points, point):
    """
    Returns the list of probabilities for the "point" to be in each class of the "points" given 
    "mean_point" is the closest point from the "points" (thus closest class),
    "var" is a given variance of each normal distributions around the points.
    """
    mean_dist = spherical_distance(point, mean_point)
    probs = []
    for p in points:
        probs.append(spherical_norm_pdf(mean_dist, var, point, p))
    s = sum(probs)
    return [x/s for x in probs]


# Probability calculation :

"""
means of clusters set to the "pure" word-order types -
    (1,0,0) : SV/VS
    (0,1,0) : VO/OV
    (0,0,1) : IV/VI
    (1/2,1/2,0) : SVO/SOV/OVS/OSV/VSO/VOS
    (1/2,0,1/2) : SVI/SIV/IVS/ISV/VSI/VIS
    (0,1/2,1/2) : IVO/IOV/OVI/OIV/VIO/VOI
    (1/3,1/3,1/3) : SVOI/SOVI/OVSI/OSVI/VSOI/VOSI/SVIO/SOIV/OVIS/OSIV/VSIO/VOIS/
                    SIVO/SIOV/OIVS/OISV/VISO/VIOS/ISVO/ISOV/IOVS/IOSV/IVSO/IVOS

These are then converted to their spherical projection or in other words, converted to the 
corresponding unit vectors.

"""
kmeans = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1/2, 1/2, 0),
          (1/2, 0, 1/2), (0, 1/2, 1/2), (1/3, 1/3, 1/3)]

kmeans_spherical = [sphere_project(p) for p in kmeans]

# An arbitrary variance set to the spherical distance between a "pure SIVO/SVOI/..." verb and a "pure SOV/SVO/..."
variance = spherical_distance(sphere_project(
    (1/3, 1/3, 1/3)), sphere_project((1/2, 1/2, 0)))/8


def probs(p, variance):
    """
    Returns the probabilities of the point "p" being in each class given by "kmeans/kmeans_spherical"
    with a given "variance" of the corresponding normal distribution.
    """
    global kmeans
    global kmeans_spherical
    if (p == (0, 0, 0)):
        return ([1/9] * 7)
    else:
        # Min Euclidean distance in order to find mean_point
        mind = np.inf
        cluster = -1
        for k in range(len(kmeans)):
            mpoint = kmeans[k]
            d = ((p[0] - mpoint[0])**2 + (p[1] - mpoint[1])
                 ** 2 + (p[2] - mpoint[2])**2)**0.5
            if (d < mind):
                mind = d
                cluster = k
        return gaussian_prob(kmeans_spherical[cluster], variance, kmeans_spherical,
                             sphere_project(p))


def word_order(sdist, odist, iodist, defaultwo=(0, 1, 2, 3)):
    def strwo(n):
        # Helper function for word_order
        if (n == 0):
            return "V"
        elif (n == 1):
            return "S"
        elif (n == 2):
            return "O"
        elif (n == 3):
            return "I"
    # Possible :
    # SVOI, SVIO, SIVO, SIOV, SOIV, SOVI
    # OVSI, OVIS, OIVS, OISV, OSIV, OSVI
    # IVSO, IVOS, ISVO, ISOV, IOSV, IOVS
    # VSOI, VSIO, VISO, VIOS, VOSI, VOIS
    dists = (0.0, sdist, odist, iodist)
    wo = defaultwo   # 0 -> Verb, 1 -> Subject, 2 -> Object, 3 -> Indirect Object
    """
    Based on the dependency distance betweeen a verb and the "S", "O" and "I" (with "V" taken as 0),
    the word order is found by sorting these distances (which can be negative as well) and then replacing
    the key with the corresponding labels ("S" or "V" or "I" or "O")
    """
    return reduce(lambda x, y: x + strwo(y), sorted(wo, key=lambda x: dists[x])[::-1], "")


def clusterVerb(sfreq, ofreq, iofreq, sdist, odist, iodist, variance=variance):
    """
    Returns the dictionary with verb_class-probability pair. Verb classes are, in general, 7 as defined above -
    "pure subject only", "pure object only", "pure indirect object only", "pure subject-object" and so on.
    These are then classified further using the word order property derived from "sdist", "odist", "iodist" (and the default).
    For each of the 7 classes thus formed, are then assigned probability such that the verb encapsulated by 
    (sfreq, ofreq, iofreq, sdist, odist, iodist) belongs to a certain class with the corresponding probability.
    """
    def non_args(i):
        # 0 -> (1,0,0), 1 -> (0,1,0), 2 -> (0,0,1),
        # 3 -> (0.5,0.5,0), 4 -> (0.5,0,0.5), 5 -> (0,0.5,0.5)
        # 6 -> (1/3, 1/3, 1/3)
        s = []
        if (i == 1 or i == 2 or i == 5):
            s.append("S")
        if (i == 0 or i == 2 or i == 4):
            s.append("O")
        if (i == 0 or i == 1 or i == 3):
            s.append("I")
        return s

    wo = word_order(sdist, odist, iodist, defaultwo=(
        1, 0, 2, 3))  # defaultwo = SVOI
    probabs = probs((sfreq, ofreq, iofreq), variance)
    clusterDict = {}
    for i in range(len(probabs)):
        nags = non_args(i)
        wo_cat = wo
        for ag in nags:
            wo_cat = wo_cat.replace(ag, "")
        # if (probabs[i])
        clusterDict[wo_cat] = probabs[i]
    return clusterDict
    # return list(map(lambda x, y, z: min_dist_mean((x, y, z)), sfreq, ofreq, iofreq))


def clusterColumn(df):
    """
    Returns a dataframe with column "VerbClass" storing a dictionary returned by "clusterVerb" function
    defined above, for each verb in the row of "df".
    """
    clusterDF = pd.DataFrame(index=df.index, columns=["VerbClass"])
    for i in df.index:
        clusterDF.at[i, "VerbClass"] = clusterVerb(df.at[i, "subj_freq"], df.at[i, "obj_freq"],
                                                   df.at[i, "iobj_freq"], df.at[i,
                                                                                "subj_avdist"],
                                                   df.at[i, "obj_avdist"], df.at[i, "iobj_avdist"])
    return clusterDF


def finite_verbs(verb_df):
    """
    Returns the instances (dependencies/edges) of the verb which are "finite" in nature. 
    Here, the verb is considered if it has an auxiliary even if its "VerbForm" field is not "Fin".
    """
    def haveDirectAux(vedges):
        # checks if the edges (dependencies) with this verb as its head, has any direct auxiliary.
        for i in vedges.index:
            if (vedges.at[i, "TargetNode"].endswith("AUX")):
                return True
        return False
    global node_df
    vdf_out = pd.DataFrame([], columns=verb_df.columns)
    verb_dfg = verb_df.groupby(by="SourceNodeIndex")
    for i in verb_dfg.groups.keys():
        vi_edges = verb_dfg.get_group(i)
        vi_node = node_df.iloc[i]
        try:
            vform = next(filter(lambda x: x.startswith(
                'VerbForm'), vi_node["FEATS"].split('|')))[9:]
            if (vform == 'Fin'):
                vdf_out = vdf_out.append(vi_edges)
            elif (haveDirectAux(vi_edges)):
                vdf_out = vdf_out.append(vi_edges)
        except:
            pass
    return vdf_out


"""

Raw network : 

"""

lang = sys.argv[1]
# langs = list(map(lambda x : x[:-4], os.listdir("NetworkData/EdgeCsv/")))
# for lang in langs :

edge_df = pd.read_csv("NetworkData/EdgeCsv/" + lang + ".csv")
node_df = pd.read_csv("NetworkData/NodeCsv/" + lang + ".csv")


"""

Layer 1 formation : 

Taking the verbs out with some extra information from the raw network (averaged distances and frequencies)

"""

core_args = ["nsubj", "csubj", "nsubj:pass",
             "obj", "ccomp", "iobj", "xcomp"]
core_args_dict = {"subj": ["nsubj", "csubj", "nsubj:pass"], "obj": [
    "obj", "ccomp"], "iobj": ["iobj", "xcomp"]}

layer1 = {"verb": [], "prev_layer": [], "subj_freq": [], "obj_freq": [], "iobj_freq": [],
          "subj_avdist": [], "obj_avdist": [], "iobj_avdist": []}

# Grouping the edge data by source nodes
edge_dfg = edge_df.groupby(by="SourceNode")
# node_dfg = node_df.groupby(by="LemmaPos")

# Listing the verbs out of all the (distinct) nodes of the network
verbs = list(filter(lambda x: x.endswith(":VERB"), edge_dfg.groups.keys()))

for v in verbs:
    layer1["verb"].append(v)
    verb_df = edge_dfg.get_group(v)
    layer1["prev_layer"].append(verb_df)
    # Only taking the finite instances of verb v.
    verb_df = finite_verbs(verb_df)
    verb_df["SententialDistance"] = verb_df["SententialDistance"].astype(float)
    verb_df["SourceNodeIndex"] = verb_df["SourceNodeIndex"].astype(int)
    verb_df["TargetNodeIndex"] = verb_df["TargetNodeIndex"].astype(int)
    """
    Finding mean sentential distance and frequency for each core argument type in all the (free) instances 
    of the verb v. 
    The edge's dependency relation (DEPREL) is checked for the core_args (core arguments) and then the 
    corresponding type (subject, object, indirect object) is changed. 
    For example, subj_avdist will store the average dependency distance (signed) of the "subj" type argument relations.
    subj_freq stores the fraction of instances there was a "subj" type argument relation with this verb v. 
    """
    if (len(verb_df.index) == 0):
        for ag in core_args_dict.keys():
            layer1[ag + "_freq"].append(0.0)
            layer1[ag + "_avdist"].append(0.0)
    else:
        core_verb_df = verb_df.loc[verb_df["DEPREL"].isin(core_args)]
        coregrps = core_verb_df.groupby("DEPREL")
        coregrps_mean = coregrps.mean()
        nsents = len(core_verb_df)
        for ag in core_args_dict.keys():
            freql = 0
            distl = 0
            i = 0
            for arg in core_args_dict[ag]:
                try:
                    freql += len(coregrps.get_group(arg))/nsents
                    distl += coregrps_mean.at[arg, "SententialDistance"]
                    i += 1
                except Exception:
                    continue
            layer1[ag + "_freq"].append(freql)
            layer1[ag + "_avdist"].append(distl/i if (i != 0) else distl)

layer1cols = layer1.keys()
layer1df = pd.DataFrame(layer1, columns=layer1cols)
layer1df = layer1df.loc[~((layer1df["subj_freq"] == 0) & (
    layer1df["obj_freq"] == 0) & (layer1df["iobj_freq"] == 0))]
layer1df = layer1df.reset_index(drop=True)
# VerbClass is a dictionary for each verb (as described in clusterColumn) for use in the formation of the next layer.
layer1df["VerbClass"] = clusterColumn(layer1df)


"""

Layer 2 Formation : 

Contains the verb classes defined as below and each verb from Layer 1 is being assigned a probability of being in each of them. 

"""

layer2df = pd.DataFrame(columns=["prev_layer"],
                        index=["SV", "VS", "OV", "VO", "VI", "IV",
                               "SOV", "SVO", "VOS", "VSO", "OSV", "OVS", "ISV", "IVS", "VSI",
                               "VIS", "SIV", "SVI", "OVI", "OIV", "IVO", "IOV", "VIO", "VOI",
                               "SVOI", "SVIO", "SIVO", "SIOV", "SOIV", "SOVI", "OVSI", "OVIS",
                               "OIVS", "OISV", "OSIV", "OSVI", "IVSO", "IVOS", "ISVO", "ISOV",
                               "IOSV", "IOVS", "VSOI", "VSIO", "VISO", "VIOS", "VOSI", "VOIS"])

layer2prevcols = ["probability", "verb", "prev_layer", "subj_freq", "obj_freq",
                  "iobj_freq", "subj_avdist", "obj_avdist", "iobj_avdist"]

for i in layer2df.index:
    layer2df.at[i, "prev_layer"] = pd.DataFrame(
        columns=layer2prevcols)

# Connecting each node of this layer with the corresponding previous ones.
for k in layer1df.index:
    classes = layer1df.at[k, "VerbClass"]
    verb_df = layer1df.iloc[k][layer1cols]
    for c in classes.keys():
        prevdict = {"probability": [classes[c]]}
        for col in layer2prevcols[1:]:
            prevdict[col] = [verb_df[col]]
        layer2df.at[c, "prev_layer"] = layer2df.at[c, "prev_layer"].append(
            pd.DataFrame(prevdict)).reset_index(drop=True)


"""
Getting an idea of the verbs for each class and their probabilities : 
"""
try:
    os.mkdir("layer_analysis/" + lang)
except:
    pass

wo_lang = {"Word_order": [], "Count": []}
for wo in layer2df.index:
    wo_lang["Word_order"].append(wo)
    highp = layer2df.at[wo, "prev_layer"].loc[layer2df.at[wo,
                                                          "prev_layer"]["probability"] > 0.1]
    wo_lang["Count"].append(len(highp))
    highp.drop(["prev_layer"], axis=1).to_csv("layer_analysis/" + lang + "/" + lang +
                                              "_" + wo + ".csv", index=False, columns=layer2prevcols, encoding="utf-8-sig")
    # layer2df.at[wo, "prev_layer"].loc[layer2df.at[wo, "prev_layer"]["probability"] > 0.1].drop(["prev_layer"], axis=1).to_csv(
    #     "layer_analysis/" + lang + "/" + lang + "_" + wo + ".csv", index=False, columns=["probability", "verb", "subj_freq", "obj_freq",
    #                                                                                      "iobj_freq", "subj_avdist", "obj_avdist", "iobj_avdist"], encoding="utf-8-sig")


"""
Selecting only those connections between layer 1 and layer 2 that have probability greater than 0.2
"""
for wo in layer2df.index:
    layer2df.at[wo, "prev_layer"] = layer2df.at[wo,
                                                "prev_layer"].loc[layer2df.at[wo, "prev_layer"]["probability"] > 0.2]

"""
Appending the connections between Layer 1 and Layer 2 connections, to the edge data of the language. 
This complete network data will be then analyzed by Cytoscape or some network analysis tool. 
"""

layer2edges = {}
wo_ind = 0

for col in edge_df.columns:
    layer2edges[col] = []

for wo in layer2df.index:
    prev = layer2df.at[wo, "prev_layer"]
    for i in prev.index:
        layer2edges["SourceNode"].append(wo)
        layer2edges["InteractionType"].append("pp")
        layer2edges["TargetNode"].append(prev.at[i, "verb"])
        layer2edges["DEPREL"].append("VerbClass")
        layer2edges["SententialDistance"].append(
            (1.0/prev.at[i, "probability"] if (prev.at[i, "probability"] != 0) else np.inf))
        layer2edges["SourceNodeIndex"].append(wo_ind)
        layer2edges["TargetNodeIndex"].append(i)
    wo_ind += 1

edge_df.append(pd.DataFrame(layer2edges, columns=edge_df.columns)).to_csv(
    "layer_edges/" + lang + ".csv", encoding="utf-8-sig", index=False)

print("Done for " + lang + "!!")
# wo_lang = pd.DataFrame(wo_lang)
# wo_lang.to_csv("layer_analysis/" + lang + "/" + lang + ".csv",
#                index=False, columns=["Word_order", "Count"])

# plt.pie(wo_lang["Count"], labels=wo_lang["Word_order"], autopct='%1.1f%%')
# plt.savefig("layer_analysis/" + lang + "/" + lang + ".png")
