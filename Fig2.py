# -*- coding: utf-8 -*-
"""
Created on Sun Jul  7 22:15:25 2024

@author: lzl
"""

import scanpy as sc
import matplotlib.pyplot as plt
import seaborn as sns

Data_path = 'C:/aging_human_skin/data/'
file_path = Data_path + 'aging_human_skin.h5ad'

adata = sc.read_h5ad(file_path)
adata.obs['age'].unique()
adata.obs['Celltype'].unique()

sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.tl.tsne(adata)
sc.pl.tsne(adata, color='Celltype', palette=plt.cm.Spectral.name)



def hide_old_cells(adata):
    mask = adata.obs['age'] != 'OLD'
    return mask
mask = hide_old_cells(adata)
sc.pl.tsne(adata[mask], color='Celltype', palette=plt.cm.Spectral.name)


def hide_old_cells(adata):
    mask = adata.obs['age'] != 'YOUNG'
    return mask
mask = hide_old_cells(adata)
sc.pl.tsne(adata[mask], color='Celltype', palette=plt.cm.Spectral.name)



#%%

import scanpy as sc
import matplotlib.pyplot as plt
import scvelo as scv
import cellrank as cr

Data_path = 'C:/aging_human_skin/data/'
file_path = Data_path + 'aging_human_skin.h5ad'

adata = sc.read_h5ad(file_path)
adata.obs['Celltype'].value_counts()

adata_Pro_inflammatory= adata[adata.obs['Celltype'] == 'Pro-inflammatory', :]
adata = adata_Pro_inflammatory

sc.pp.filter_genes(adata, min_cells=10)
scv.pp.normalize_per_cell(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata)
sc.pp.highly_variable_genes(adata,n_top_genes=2000,subset=True)

adata.layers["spliced"] = adata.X
adata.layers["unspliced"] = adata.X
scv.pp.moments(adata, n_pcs=30, n_neighbors=30)

del adata.layers['spliced']
del adata.layers['unspliced']
del adata.layers['Mu']

from cellrank.kernels import CytoTRACEKernel
ctk = CytoTRACEKernel(adata).compute_cytotrace()

sc.pl.embedding(
    adata,
    color=["ct_pseudotime"],
    basis="X_pca",
    color_map="gnuplot2",
)
plt.rcParams.update({'font.size':20})

ctk.compute_transition_matrix(threshold_scheme="soft", nu=0.2)
ctk.plot_projection(basis="X_pca", color="ct_pseudotime", legend_loc="right")


