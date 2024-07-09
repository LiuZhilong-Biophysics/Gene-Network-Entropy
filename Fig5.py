# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:43:15 2024

@author: lzl
"""

import os
import sys
import time
import stat
import numpy as np
import pandas as pd
import scanpy as sc
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.colors import Normalize
import anndata as ad
import scvelo as scv
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

Data_path = 'C:/Users/lzl/Desktop/work5/林海/aging_human_skin/data/'
file_path = Data_path + 'aging_human_skin.h5ad'

adata = sc.read_h5ad(file_path)
adata

label = adata.obs['age']    

dataName = GSEname + '2000'   
csnDataDir =  'D:/lzl/Work/work5/Data_Set/aging_skin/Cell_type/Pro_inflammatory/csnData/'  
savePath = workDir + '/data/Pro_inflammatory/' + 'result'     

mtx = adata.X #(cell,gene)   
adata.obs['Celltype'].value_counts()

adata_Pro_inflammatory= adata[adata.obs['Celltype'] == 'Pro-inflammatory', :]
adata = adata_Pro_inflammatory

# filter, normalize total counts and log-transform
sc.pp.filter_genes(adata, min_cells=10)
scv.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)


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

min_val = np.min(GirM)
max_val = np.max(GirM)
range_val = max_val - min_val

scaled_matrix = (GirM - min_val) / range_val

GirM_adata.rename_categories('age', np.unique(GirM_adata.obs['age']))
sc.tl.rank_genes_groups(GirM_adata, 'age', method='t-test')
sc.pl.rank_genes_groups(GirM_adata, n_genes=50, sharey=False)

marker_genes_num = 5
marker_genes = []
for i in range(marker_genes_num):  
    marker_genes += list(GirM_adata.uns['rank_genes_groups']['names'][i])

adata = GirM_adata   

cellN = 1
netName = adata.obs['age'][cellN]   

adata.obs['age'][cellN]

mean_G = {}       
for i in np.unique(adata.obs['age']):
    mean_G[i] =[]
mean_G

for i in range(len(adata.obs['age'])):
    mean_G[adata.obs['age'][i]].append(i)     

mean_GM = []
for cell in np.unique(adata.obs['age']):
    c_l = []
    for i in range(mtx.shape[1]):
        g = []
        for c in mean_G[cell]:
            g.append(mtx[c,i])
        c_l.append(np.mean(np.array(g)))
    mean_GM.append(c_l)                              


def calculate_entropy(G):
    # Calculate the degree of each node
    degrees = np.array([d for _, d in G.degree()])
#     print(f"度：{degrees}")
    return degrees


import csv

# Ensure the directory exists
folder_path = './data/Fig_2/networks2000_10node_2/'
if not os.path.exists(folder_path):
    os.makedirs(folder_path)

# Define lists to store entropy and degree results
Entropies_save = []
Degrees_save = []
G_matrix_degrees_list = []

all_gene_edge_matrix = np.zeros((2001, 2001))
select_gene_edge_matrix = np.zeros((marker_genes_num*2, marker_genes_num*2))


for c in ['YOUNG']:  

    cell_counts = adata.obs['age'].value_counts()  
    print(cell_counts)
    YOUNG_counts = cell_counts.get('YOUNG',0)
    OLD_counts = cell_counts.get('OLD',0)
  
    cell_list = range(1, YOUNG_counts)
    
    
    
    for idx, i in enumerate(cell_list):
    # for i in cell_list:      
        netName = c      
       
        print(c)            
        print(mean_G[c][i]) 
        
        cellN = mean_G[c][i]
        k = cellN           
        
        adata_X = adata.X  
        X = adata_X.T      
        
        
        ## =========================================
        from scipy import sparse    
        path = csnDataDir   
#         path = './data/Secretory_reticular/csnData/'
        cellsSP = sparse.load_npz(path + 'csn' + str(k) + '.npz') 
        cellsMat = cellsSP.toarray()   
        # np.savetxt('csn0.csv', cellsMat, delimiter=",") 
        all_gene_edge_matrix += cellsMat
#         print(all_gene_edge_matrix)
        print()
  
        
        ##====================================
        geneExp = X[:,k]  
        M = np.multiply(cellsMat, geneExp)   
        sumM = np.sum(M, axis = 1) + 0.0001  
        broSumM = np.ones(M.shape)*sumM  
        edgeW = M / broSumM.T  
        l1,l2 = np.nonzero(edgeW)  
        
        #   ==================================
        DG = nx.from_numpy_array(cellsMat, create_using=nx.DiGraph)  
        for i in range(l1.shape[0]):  
            DG.add_weighted_edges_from([(l1[i], l2[i], edgeW[l1[i]][l2[i]])])  
        

        gi = [] 
        for g in marker_genes: 
            gi.append(adata.var_names.get_loc(g))  ##

        

        colormap = cm.coolwarm  

        nodes_to_draw = gi  
        node_name = marker_genes  
        node_label = {}   
        for i in range(len(node_name)):
            node_label[nodes_to_draw[i]] = node_name[i]   

        G = DG.subgraph(nodes_to_draw)  
           
    
        nx.set_node_attributes(G, node_label, "label")  
        pos = nx.layout.circular_layout(G)    
             
        adjacency_matrix = nx.to_numpy_array(G)  
        binary_adjacency_matrix = (adjacency_matrix != 0).astype(int)   
        # print(binary_adjacency_matrix) 

        select_gene_edge_matrix  += binary_adjacency_matrix        
        
        G_matrix = nx.from_numpy_array(adjacency_matrix)
        degrees = calculate_entropy(G_matrix)
        
        print("node degrees:", degrees)
                
        G_matrix_degre=G_matrix.degree()
        G_matrix_degrees_list.append(G_matrix_degre)  # Append degree sequence to the list

        #######
    
# Save degree sequences into a CSV file
degrees_csv_file_path = folder_path + 'MarkerGene_Node_degrees_YOUNG.csv'
with open(degrees_csv_file_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(['Cell'] + marker_genes)  # Write header with gene labels
    for idx, degrees in enumerate(G_matrix_degrees_list):
        writer.writerow([f'Cell {idx}'] + [degree for node, degree in degrees])

print(f"All G_matrix_degre results have been saved to {degrees_csv_file_path}")


select_gene_edge_matrix = select_gene_edge_matrix/OLD_counts
all_gene_edge_matrix = all_gene_edge_matrix/OLD_counts

np.savetxt(folder_path + 'select_gene_edge_matrix_YOUNG.csv', select_gene_edge_matrix, delimiter=',')
np.savetxt(folder_path + 'all_gene_edget_matrix_YOUNG.csv', all_gene_edge_matrix, delimiter=',')


#%%  
import math
import pandas as pd
from collections import Counter

def calculate_network_entropy(degree_list):
    degree_counts = Counter(degree_list)
    total_nodes = len(degree_list)
    degree_probabilities = {k: v / total_nodes for k, v in degree_counts.items() if v > 0}
    network_entropy = -sum(p * math.log2(p) for p in degree_probabilities.values())
    return network_entropy

data_Young = pd.read_csv('MarkerGene_Node_degrees_YOUNG.csv') 
data_Young = data_Young.iloc[:, 1:]  
entropies_YOUNG = [calculate_network_entropy(row.dropna().astype(int).tolist()) for index, row in data_Young.iterrows()]
data_Old = pd.read_csv('MarkerGene_Node_degrees_OLD.csv')
data_Old = data_Old.iloc[:, 1:]  
entropies_OLD = [calculate_network_entropy(row.dropna().astype(int).tolist()) for index, row in data_Old.iterrows()]

import pandas as pd
import numpy as np


entropies_YOUNG = np.array(entropies_YOUNG)
entropies_OLD = np.array(entropies_OLD)

length_YOUNG = len(entropies_YOUNG)
length_OLD = len(entropies_OLD)

max_length = max(length_YOUNG, length_OLD)

if length_YOUNG > length_OLD:
    entropies_OLD = np.pad(entropies_OLD, (0, max_length - length_OLD), mode='constant', constant_values=np.nan)
elif length_OLD > length_YOUNG:
    entropies_YOUNG = np.pad(entropies_YOUNG, (0, max_length - length_YOUNG), mode='constant', constant_values=np.nan)

entropy_df = pd.DataFrame({
    'YOUNG': entropies_YOUNG,
    'OLD': entropies_OLD
})


entropy_df.to_csv('Network_entropies_YOUNG_OLD.csv', index=False)
print("Network entropies of YOUNG and OLD saved with NaN padding")

#%%

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_OLD = pd.read_csv('./MarkerGene_Node_degrees_OLD.csv')
data_YOUNG = pd.read_csv('./MarkerGene_Node_degrees_YOUNG.csv')
Gene_name = data_YOUNG.columns.tolist()[1:]
print(Gene_name)

plt.figure(figsize=(18, 6))
data_YOUNG['Group'] = 'YOUNG'
data_OLD['Group'] = 'OLD'
combined_data = pd.concat([data_YOUNG,data_OLD])
combined_data_melted = pd.melt(combined_data, id_vars=['Group'], value_vars=Gene_name, var_name='Gene', value_name='Degree')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.figure(figsize=(10, 8))
sns.violinplot(x='Gene', y='Degree', hue='Group', data=combined_data_melted, split=True, inner=None)
plt.ylabel('Degree', fontsize=36)
plt.xlabel('Gene', fontsize=36)
plt.xticks(rotation=90)
plt.tick_params(axis='both', which='major', labelsize=24)
plt.legend(fontsize=26)
plt.tight_layout()
plt.savefig('./Degree_Entropy/' + 'MarkerGene_degreeDistribution_YOUNG+OLD' +'.jpg',format = 'jpg', dpi=300)
plt.savefig('./Degree_Entropy/' + 'MarkerGene_degreeDistribution_YOUNG+OLD' +'.pdf',format = 'pdf', dpi=300)
plt.show()


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


data_OLD = pd.read_csv('./MarkerGene_Node_degrees_OLD.csv')
data_YOUNG = pd.read_csv('./MarkerGene_Node_degrees_YOUNG.csv')
data_OLD = data_OLD.iloc[1:, 1:]
data_YOUNG = data_YOUNG.iloc[1:, 1:]
data_OLD_stacked = data_OLD.stack().reset_index(drop=True)
data_YOUNG_stacked = data_YOUNG.stack().reset_index(drop=True)
combined_data = pd.DataFrame({'YOUNG': data_YOUNG_stacked,'OLD': data_OLD_stacked })
combined_data.to_csv('./Marker_gene_degree_combined_data.csv', index=False)
plt.figure(figsize=(10, 6))
plt.hist(data_YOUNG_stacked, bins=20, alpha=0.5, label='YOUNG', color='orange')
plt.hist(data_OLD_stacked, bins=20, alpha=0.5, label='OLD', color='blue')
plt.title('Degree Distribution Comparison', fontsize=26)  
plt.xlabel('Degree', fontsize=18)  
plt.ylabel('Frequency', fontsize=18)  
plt.tick_params(axis='both', which='major', labelsize=14)
plt.legend(fontsize=18)
plt.show()


ENTROPY_ALL = pd.read_csv('Network_entropies_YOUNG_OLD.csv')
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman'] + plt.rcParams['font.serif']
plt.figure(figsize=(10, 8))
sns.violinplot(data=ENTROPY_ALL[['YOUNG', 'OLD']], inner='point', hue_order=['YOUNG', 'OLD'])
plt.title('Entropy Distribution', fontsize=26)  
plt.ylabel('Entropy', fontsize=36)  
plt.xlabel('Age', fontsize=36)  
plt.xticks(ticks=[0, 1], labels=['YOUNG', 'OLD'], fontsize=22)  
plt.tick_params(axis='both', which='major', labelsize=24)
plt.legend(fontsize=26)
plt.tight_layout()
plt.savefig( 'MarkerGene_EntropyDistribution_YOUNG+OLD' +'.jpg',format='jpg', dpi=300)
plt.savefig( 'MarkerGene_EntropyDistribution_YOUNG+OLD' +'.pdf',format='pdf', dpi=300)
plt.show()




