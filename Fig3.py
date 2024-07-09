# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 09:56:26 2024

@author: lzl
"""

import scanpy as sc
import matplotlib.pyplot as plt
import scvelo as scv
import numpy as np
import pandas as pd
import anndata as ad
from sklearn.preprocessing import StandardScaler


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
sc.pp.highly_variable_genes(adata,n_top_genes=2001,subset=True)

IM_data = pd.read_csv('./Pro_inflammatory_GIRMatrix.csv', header=None)
IM_adata = ad.AnnData(IM_data)
IM_adata

IM_adata.obs = adata.obs   
IM_adata.var = adata.var  

adata = IM_adata
adata.X = StandardScaler().fit_transform(adata.X)


sc.tl.pca(adata)
adata.obsm['X_pca'] = adata.obsm['X_pca'][::-1]
adata.obs['age'] = adata.obs['age'][::-1]


fig, ax = plt.subplots()
sc.pl.pca(adata, color=['age'], show=False, ax=ax)
plt.savefig('GIM_Pro_inflammatory_reversed_pca_plot.pdf', dpi=300)

sc.tl.pca(adata,  svd_solver='arpack')
sc.pl.pca(adata, color='age',components=['1,2','1,3','1,4','2,3','2,4','3,4'])

plt.rcParams.update({'font.size':20})

adata.layers["spliced"] = adata.X
adata.layers["unspliced"] = adata.X
scv.pp.moments(adata, n_pcs=40, n_neighbors=40)

del adata.layers['spliced']
del adata.layers['unspliced']
del adata.layers['Mu']

from cellrank.kernels import CytoTRACEKernel
ctk = CytoTRACEKernel(adata).compute_cytotrace()
plt.rcParams.update({'font.size':20})


cmap = plt.get_cmap('gnuplot2')
axes = sc.pl.embedding(adata, basis='X_pca', color=['ct_pseudotime'], 
                       components=['1,2','1,3','1,4','2,3','2,4','3,4'], cmap=cmap.name, palette=cmap.name, show=False, size=100)

sc.pl.pca(adata, color='age', components=['1,2','1,3','1,4','2,3','2,4','3,4'], palette={'YOUNG': '#1F77B4','OLD': '#FF7F0E'})

cmap1 = plt.get_cmap('viridis')
axes1 = sc.pl.embedding(adata, basis='X_pca', color=[ 'ct_pseudotime'],
                        components=['1,2','1,3','1,4','2,3','2,4','3,4'], cmap=cmap1.name, show=False, size=300,alpha=0.7)


fig, ax = plt.subplots()
sc.pl.pca(adata, color=['age'], components=['3,4'], palette={'YOUNG': '#1F77B4','OLD': '#FF7F0E'}, show=False, size=200, ax=ax)

plt.savefig('GIM_Pro_inflammatory_pca_plot.pdf', dpi=300)


IM_adata.rename_categories('age', np.unique(adata.obs['age']))
sc.tl.rank_genes_groups(adata, 'age', method='t-test')

plt.rcParams['font.family'] = 'Times New Roman'
plt.figure(figsize=(4, 10))
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False, fontsize=14)

plt.savefig('rank_genes_groups_1.pdf', dpi=300, bbox_inches='tight')
sc.pl.rank_genes_groups(adata, n_genes=10, sharey=False, save='rank_genes_groups.pdf', show=False, dpi=300, figsize=(4, 10))


