# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 11:52:01 2024

@author: lzl
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy import sparse
from random import sample
import os
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import matplotlib.cm as cm
import anndata as ad
import scvelo as scv
import sys
from sklearn.preprocessing import StandardScaler
import seaborn as sns  
from matplotlib.colors import ListedColormap  


def scDataCluster(path, cls = 'h5', min_cells = None, percent = None, highlyVarGene = None):
    if cls == 'h5':
        adata = sc.read_h5ad(path)
    elif cls == 'csv':
        adata = sc.read_csv(path)
    elif cls == 'loom':
        adata = sc.read_loom(path)
    else:
        adata = sc.read_text(path,delimiter = ',')
        
        
    if min_cells:
        sc.pp.filter_genes(adata, min_cells= min_cells)
    elif percent:
        sc.pp.filter_genes(adata, min_cells= adata.X.shape[0] * percent)
    
    sc.pp.log1p(adata)
    if highlyVarGene: 
        sc.pp.highly_variable_genes(adata, n_top_genes = highlyVarGene)
        adata = adata[:,adata.var['highly_variable']]
    
    return adata

workDir = sys.path[0]   
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['figure.dpi'] = 300

GSEname = 'Pro_inflammatory'   

Data_path = 'C:/aging_human_skin/data/'
file_path = Data_path + 'aging_human_skin.h5ad'
adata = sc.read_h5ad(file_path)

label = adata.obs['age']    
dataName = GSEname + '2000'   
csnDataDir =  'D:/aging_skin/Cell_type/Pro_inflammatory/csnData/'  
savePath = workDir + '/data/Pro_inflammatory/' + 'result'   

mtx = adata.X #(cell,gene)   

adata_Pro_inflammatory= adata[adata.obs['Celltype'] == 'Pro-inflammatory', :]
adata = adata_Pro_inflammatory

sc.pp.filter_genes(adata, min_cells=5)
scv.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)

# hvg annotation
sc.pp.highly_variable_genes(adata)
print(f"This detected {adata.var['highly_variable'].sum()} highly variable genes. ")

sc.pp.highly_variable_genes(adata,n_top_genes=2001,subset=True)

GirM_adata = pd.read_csv('./data/Pro_inflammatory2000/result/GIRMatrix.csv', header=None)

GirM_adata = ad.AnnData(GirM_adata)

GirM_adata.obs = adata.obs   
GirM_adata.var = adata.var   

# adata = GirM_adata
# adata
GirM = GirM_adata.X
GirM_adata
GirM = GirM_adata.X

min_val = np.min(GirM)
max_val = np.max(GirM)
range_val = max_val - min_val
scaled_matrix = (GirM - min_val) / range_val

GirM_adata.rename_categories('age', np.unique(GirM_adata.obs['age']))

sc.tl.rank_genes_groups(GirM_adata, 'age', method='t-test')
sc.pl.rank_genes_groups(GirM_adata, n_genes=10, sharey=False)

marker_genes_num = 10
marker_genes = []
for i in range(marker_genes_num):  
    marker_genes += list(GirM_adata.uns['rank_genes_groups']['names'][i])



#%%
path = './data/Fig_2/networks2000_10node_2/'
df = pd.read_csv(path + 'select_gene_edge_matrix_OLD.csv', header=None,)
df = np.array(df)


base_folder_path = './data/Fig_2/networks2000_10node_2/networks_all_edgewight/'
# Ensure the directory exists
if not os.path.exists(base_folder_path):
    os.makedirs(base_folder_path)

G = nx.Graph()
for idx, gene in enumerate(marker_genes):
    G.add_node(idx, label=gene)

for i in range(len(df)):
    for j in range(len(df)):
        if df[i][j] > 0:
            normalized_weight = df[i][j] / np.max(df)  
            G.add_edge(i, j, weight=normalized_weight)


pos = nx.circular_layout(G)
node_degrees = dict(G.degree())
max_degree = max(node_degrees.values())
normalized_degrees = {node: degree / max_degree for node, degree in node_degrees.items()}
node_sizes = [normalized_degrees[node] * 3000 for node in G.nodes()]
node_colors = [normalized_degrees[node] for node in G.nodes()]
node_cmap = cm.get_cmap('viridis')
edge_cmap = cm.get_cmap('coolwarm')
plt.figure(figsize=(8, 8))
edges = nx.draw_networkx_edges(G, pos, width=[d['weight']*6 for i, j, d in G.edges(data=True)], 
                               edge_color=[edge_cmap(d['weight']) for i, j, d in G.edges(data=True)], alpha=0.65)
nodes = nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, cmap=node_cmap, alpha=0.8)
node_labels = nx.get_node_attributes(G, 'label')
nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=36, font_family='Times New Roman')   
plt.axis('off')   
norm = plt.Normalize(0, 1)
node_sm = cm.ScalarMappable(cmap=node_cmap, norm=norm)
node_sm.set_array([])
edge_sm = cm.ScalarMappable(cmap=edge_cmap, norm=norm)
edge_sm.set_array([])
plt.savefig(base_folder_path+ 'MarkerGeneNetwork_OLD' +'.jpg',format = 'jpg', dpi=300)
plt.savefig(base_folder_path+ 'MarkerGeneNetwork_OLD' +'.pdf',format = 'pdf', dpi=300)
plt.show()

#%%
path = './data/Fig_2/networks2000_10node_2/'
df = pd.read_csv(path + 'all_gene_edget_matrix_OLD.csv', header=None,)
df = np.array(df)

adata_X = df  
X = adata_X.T  
cellsMat = np.where(df > 0.0, 1, 0)
G = nx.Graph(cellsMat)
num_to_remove = int(len(G) / 2)
nodes = sample(list(G.nodes), num_to_remove)
G.remove_nodes_from(nodes)
components = nx.connected_components(G)
largest_component = max(components, key=len)
H = G.subgraph(largest_component)
centrality = nx.degree_centrality(G)  
lpc = nx.community.label_propagation_communities(H)
community_index = {n: i for i, com in enumerate(lpc) for n in com}
fig, ax = plt.subplots(figsize=(20, 15))
pos = nx.spring_layout(H, k=0.01, seed=4572321)  

node_color = [community_index[n] for n in H]
max_color = max(node_color)
node_color = [max_color - c for c in node_color]

normalized_degree_centrality = {node: centrality / (G.number_of_nodes() - 1) 
                                   for node, centrality in centrality.items()}

node_size = [v * 500 for v in centrality.values()]

nx.draw_networkx(
    H,
    pos=pos,
    with_labels=False,
    node_color=node_color,
    node_size=node_size,
    edge_color="gainsboro",
    alpha=0.4,
)

font = {"color": "k", "fontweight": "bold", "fontsize": 20}
ax.set_title(f"Gene functional association network", font)
ax.margins(0.1, 0.05)
fig.tight_layout()
plt.axis("off")
plt.savefig(base_folder_path+ 'ALLGeneNetwork_OLD' +'.jpg',format = 'jpg', dpi=300)
plt.savefig(base_folder_path+ 'ALLGeneNetwork_OLD' +'.pdf',format = 'pdf', dpi=300)
plt.show()







