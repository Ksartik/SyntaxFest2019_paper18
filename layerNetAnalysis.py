import pandas as pd
import sys
import numpy as np
import os

"""
layerNetAnalysis.py : 
    Summarizes the analyzed results from the Cytoscape of the layered network into one file named
    "layeredNetAnalysis.csv" which contains the node properties of all the nodes of layer 2 of all the languages. 
"""

layer2nodes = ["SV", "VS", "OV", "VO", "VI", "IV",
               "SOV", "SVO", "VOS", "VSO", "OSV", "OVS", "ISV", "IVS", "VSI",
               "VIS", "SIV", "SVI", "OVI", "OIV", "IVO", "IOV", "VIO", "VOI",
               "SVOI", "SVIO", "SIVO", "SIOV", "SOIV", "SOVI", "OVSI", "OVIS",
               "OIVS", "OISV", "OSIV", "OSVI", "IVSO", "IVOS", "ISVO", "ISOV",
               "IOSV", "IOVS", "VSOI", "VSIO", "VISO", "VIOS", "VOSI", "VOIS"]

# lang = sys.argv[1]
langs = os.listdir("layer_net_analysis")
network_pars = ["AverageShortestPathLength", "BetweennessCentrality", "ClosenessCentrality", "ClusteringCoefficient", "Eccentricity",
                "EdgeCount", "Indegree", "IsSingleNode", "NeighborhoodConnectivity", "Outdegree", "PartnerOfMultiEdgedNodePairs", "SelfLoops", "Stress"]
k = len(layer2nodes)
col1 = np.array([network_pars[0]]*k)
for par in network_pars[1:]:
    col1 = np.append(col1, np.array([par]*k))

# col2 = np.array([langs]*n)
n = len(network_pars)
col2 = np.array(layer2nodes*n)

# df_dict = {}
data = []
for lang in langs:
    df = pd.read_csv("layer_net_analysis/" + lang + "/" + lang + "_node.csv",
                     dtype={"SUID": int, "AverageShortestPathLength": float, "BetweennessCentrality": float,
                            "ClosenessCentrality": float, "ClusteringCoefficient": float, "Degree": int, "Eccentricity": float,
                            "IsSingleNode": bool, "name": str, "NeighborhoodConnectivity": float, "NumberOfDirectedEdges": int,
                            "NumberOfUndirectedEdges": int, "PartnerOfMultiEdgedNodePairs": int, "Radiality": float, "selected": bool,
                            "SelfLoops": int, "shared name": str, "Stress": float, "TopologicalCoefficient": float})
    df = df.loc[df["name"].isin(layer2nodes)].reset_index()
    # df_dict[lang] = df
    row = []
    for par in network_pars:
        for wo in layer2nodes:
            try:
                row.append(list(df.loc[df["name"] == wo][par])[0])
            except:
                row.append(np.nan)
    data.append(row)

ddf = pd.DataFrame(data=data, columns=pd.MultiIndex.from_tuples(
    list(zip(col1, col2))), index=langs)
ddf.to_csv("layeredNetAnalysis.csv")
# for i in range(len(layer2nodes)) :
#     row = []
#     for par in network_pars :
#         for lang in langs:
#             if (i < df_dict)
#                 row.append(df_dict[lang].at[i, par])
#     data.append(row)
