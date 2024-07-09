# -*- coding: utf-8 -*-
"""
Created on Tue Jul  9 10:34:42 2024

@author: lzl
"""

import scanpy as sc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import umap
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering,KMedoids
from sklearn import metrics
import os
import stat
import scipy.stats as st
from scipy import sparse
from tqdm import tqdm

import networkx as nx
from tqdm import tqdm
from scipy import sparse

def clusterKD(matrix, label, cls = 'KMeans', n_clusters = 4, n_components = 8, decompose = 'PCA', target = 'ARI'):
    switch = {
        'KMeans' : KMeans(n_clusters= n_clusters).fit,
        'Hierarchical' : AgglomerativeClustering(n_clusters= n_clusters).fit,
        'KMedoids' : KMedoids(n_clusters= n_clusters).fit,   
        'Spectral' : SpectralClustering(n_clusters= n_clusters).fit,
    }
    if decompose == 'TSNE':
        newx = TSNE(n_components = n_components).fit_transform(matrix)
    elif decompose == 'PCA': 
        newx = PCA(n_components = n_components).fit_transform(matrix)
    else: 
        newx = matrix
    y_pred = switch[cls](newx)
    
    if target == 'ARI':
        score = metrics.adjusted_rand_score(y_pred.labels_,label)
    elif target == 'FM':
        score = metrics.fowlkes_mallows_score(y_pred.labels_,label)
    elif target == 'NMI':
        score = metrics.normalized_mutual_info_score(y_pred.labels_,label)
    return score

def save2Csv(
    matrix, 
    name,
    dir = './',
    header=None,
    index= None
):
    """
        save matrix to csv file
    Args:
        matrix
            numpy array or Dataframe. 
        dir
            directory of file, Default = './'
        header
            Write out the column names. If a list of strings is given it is
            assumed to be aliases for the column names.
        index
            Write row names.
    """
    if isinstance(matrix, pd.DataFrame):
        matrix.to_csv(dir+name+'.csv',header= header, index= index)
    elif isinstance(matrix, np.ndarray):
        if len(matrix.shape) == 2:
            np.savetxt(dir+name+'.csv', matrix, delimiter=",")
        elif len(matrix.shape) == 3:
            with open(dir+name+'.csv', 'w') as f:
                for slice_2d in matrix:
                    np.savetxt(f, slice_2d, fmt = '%f', delimiter = ',')
    else: raise NameError('Ensure that the data types are numpy of dataframe')

def draw(
    x,
    y,
    xlabel = 'x',
    ylabel = 'y',
    title = 'title',
    xfontsize = 12,
    yfontsize = 12,
    tfontsize = 15,
    cls = 'plot',
    color = 'red',
    legend = 'best',
    scatter_s = 10,
):
    plt.title(title, fontsize = tfontsize)
    plt.xlabel(xlabel, fontsize = xfontsize)
    plt.ylabel(ylabel, fontsize = yfontsize)
    if cls == 'plot':
        plt.plot(x,y,c = color)
    elif cls == 'scatter':
        plt.scatter(x,y,c = color, s=scatter_s, label = legend)
    plt.legend(loc = legend)
    plt.show()


def drawHist(
    x,
    xlabel = 'p',
    ylabel = 'Density',
    title = 'title',
    figsize = [12, 8],
    dpi = 80,
    xfontsize = 12,
    yfontsize = 12,
    tfontsize = 15,
    color = 'red',
    range = None,
    save = False,
    fit = True,
    dir = './hist.png',
    width = 100,
):
    plt.figure(figsize=figsize, dpi = dpi)
    plt.title(title, fontsize = tfontsize)
    plt.xlabel(xlabel, fontsize = xfontsize)
    plt.ylabel(ylabel, fontsize = yfontsize)
    n, bins, patches = plt.hist(x, bins=width, range = range)
    if fit:
        X = bins[0:width] + (bins[1] - bins[0])/2.0
        Y = n
        plt.plot(X,Y, color = 'green')
        p1 = np.polyfit(X, Y, 7)
        Y1 = np.polyval(p1, X)
        plt.plot(X, Y1, color = 'red')
    if save: plt.savefig(dir)
    plt.show()

def drawUmap(
    mtx,
    name,
    cls = 'KMeans',
    embedding_way = 'UMAP',
    savePath = None,
    size = [5, 5],
    n_clusters = 4,
    component = 8,
    n_neighbors = 6,
    min_dist = 0.1,
    u_components = 2,
    c = 'cet_rainbow4',
    random_state = 99,
):
    switch = {
        'KMeans' : KMeans(n_clusters= n_clusters).fit,
        'Hierarchical' : AgglomerativeClustering(n_clusters= n_clusters).fit,
        'KMedoids' : KMedoids(n_clusters= n_clusters).fit,   
        'Spectral' : SpectralClustering(n_clusters= n_clusters).fit,
        
    }
    fig = plt.figure(figsize=(size[0],size[1]))
    pca = PCA(n_components= component)
    dmtx = pca.fit_transform(mtx)
    y_pred = switch[cls](dmtx)
    if embedding_way == 'UMAP':
        embedding = umap.UMAP(n_components= u_components).fit_transform(mtx)
    elif embedding_way == 'TSNE':
        
        embedding = TSNE(n_components = u_components).fit_transform(dmtx)
    else:
        embedding = PCA(n_components = u_components).fit_transform(mtx)
        
    unique_labels = np.unique(y_pred.labels_)
    colors = plt.get_cmap(c)(np.linspace(0, 1, len(unique_labels)))
    colors = ['#83639f','#c22f2f', '#3490de','#449945', '#1f70a9', '#ea7827', '#F8766D','#abedd8','#f6416c','#ffd460']#
    
    for i, label in enumerate(unique_labels):
        plt.scatter(embedding[y_pred.labels_ == label, 0], embedding[y_pred.labels_ == label, 1], c=colors[i], label=f'Cluster {label}', s = 0.6)
    plt.title(name, size=12)
    plt.xticks([])
    plt.yticks([])
    if embedding_way == 'PCA':
        plt.xlabel('PC1',size = 8)
        plt.ylabel('PC2',size = 8)
    else:
        plt.xlabel('t-SNE1',size = 8)
        plt.ylabel('t-SNE2',size = 8)
    plt.tight_layout()
    plt.rcParams['figure.dpi'] = 300
    plt.savefig(savePath+name)
    plt.show()
    plt.close(fig)


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

def neighborhoodDef(
    rows, #n1
    cols, #n2
    boxsize,
    mat,
)->np.array:
    """
    Define the neighborhood of each plot
    ----------
    Args:
        rows
            Gene expression
        cols 
            Cells
        boxsize
            Size of neighborhood.
        mat 
            Gene expression matrix, rows = genes, columns = cells.
    Returns:
        gene_k boundary
    """
    upper = np.zeros((rows, cols)) #box boundary
    lower = np.zeros((rows, cols))
    for i in range(rows): 
        geneIdx = np.argsort(mat[i,:])#s2
        geneX = np.sort(mat[i,:]) #s1
        epsCells = np.sum(np.sign(geneX))
        unepsCells = cols - epsCells #n3
        h = boxsize/2 * epsCells #side of box, epscells * boxsize = n_x or n_y = 0.1n
        if h == 0.5:
            h = 1
        else:
            h = np.round(h)
        k = 0
        while k < cols:
            s = 0
            while k+s+1 < cols and geneX[k+s+1] == geneX[k]: 
                s = s + 1

            if s >= h:
                for j in range(s+1):
                    upper[i, geneIdx[k+j]] = mat[i, geneIdx[k]]
                    lower[i, geneIdx[k+j]] = mat[i, geneIdx[k]]
            else:
                for j in range(s+1):
                    upper[i, geneIdx[k+j]] = mat[i, geneIdx[int(min(cols-1, k+s+h))]]
                    lower[i, geneIdx[k+j]] = mat[i, geneIdx[int(max(unepsCells*(unepsCells>h), k-h))]]
                    
            k = k + s + 1
    
    return upper, lower


def csnConstruct(
    mat,
    path = './',
    cells = None,
    alpha = 0.01,
    boxsize = 0.1,
    weighted = 0,
    p = None,
) -> np.array :
    """
    Input gene x cell matrix
    Construct matrix of cell-specific networks.
    ----------
    Args:
        mat
            Gene expression matrix, rows = genes, columns = cells.
        cells
            Construct the CSNs for all cells, set c = [] (Default);
            Construct the CSN for cell k, set  c = k
        alpha
            Significant level (eg. 0.001, 0.01, 0.05 ...).
        boxsize
            Size of neighborhood, Default = 0.1.
        weighted
            1  edge is weighted
            0  edge is not weighted (Default)
        p
            Level of significance
    Returns:
        Cell-specific network 
    """
    if not os.path.exists(path):
        os.mkdir(path)
    os.chmod(path, stat.S_IWRITE)
    mat.astype(np.float32)
    if not cells :
        cells = []
        for i in range(np.size(mat, 1)):
            cells.append(i)
    rows =  mat.shape[0] # genes
    cols = mat.shape[1] # cells
    upper, lower = neighborhoodDef(rows, cols, boxsize, mat)
    csn = []
    prox = []
    B = np.zeros((rows, cols))
    
    if not p:
        p = -st.norm.ppf(alpha)
        
    
    for k in tqdm(cells):  
        for j in range(cols):
            for g in range(rows):
                if mat[g,j] <= upper[g,k] and mat[g,j] >= lower[g,k]:
                    B[g, j] = 1
                else: B[g, j] = 0
        colSumMat = np.sum(B, axis=1)
        colSumMat = colSumMat.reshape((np.shape(colSumMat)[0],1))
        eps = np.finfo(float).eps

        distence = (np.dot(B,B.T)*cols-np.dot(colSumMat,colSumMat.T))/np.sqrt(np.dot(colSumMat,colSumMat.T)*(np.dot((cols-colSumMat),(cols-colSumMat).T)/(cols-1)+eps))
        distence = distence - distence*np.eye(distence.shape[0])
        
        retM = np.zeros(distence.shape)
        retM[distence>=p] = 1

        retMSp=sparse.csr_matrix(retM)
        sparse.save_npz(path + 'csn' + str(k) + '.npz',retMSp)


def degreeCentrality(G):
    if len(G) <= 1:
        return np.array([n for n in G])
    s = 1.0/(len(G) - 1.0)
    centrality = np.array([d * s for n, d in G.degree()])
    return centrality

def eigenvectorCentrality(G):
    dict = nx.eigenvector_centrality(G)
    ret = []
    for _, v in dict.items():
        ret.append(v)
    return np.array(ret)

def betweennessCentrality(G):
    dict = nx.betweenness_centrality(G)
    ret = []
    for _, v in dict.items():
        ret.append(v)
    return np.array(ret)

def pageRankCentrality(G):
    dict = nx.pagerank(G)
    ret = []
    for _, v in dict.items():
        ret.append(v)
    return np.array(ret)

def closenessCentrality(G):
    dict = nx.closeness_centrality(G)
    ret = []
    for _, v in dict.items():
        ret.append(v)
    return np.array(ret)

def geneCentralityRank(X, path, savePath, centralityDir, cells, cls = 'PageRank'):
    switch = {
        'Degree' : degreeCentrality,
        'Eigenvector' : eigenvectorCentrality,
        'Betweenness' : betweennessCentrality,
        'PageRank' : pageRankCentrality,
        'Closeness' : closenessCentrality,
    }
    for k in tqdm(range(cells)):
        cellsSP = sparse.load_npz(path + 'csn' + str(k) + '.npz') # 读取
        cellsMat = cellsSP.toarray()
        geneExp = X[:,k]
        M = np.multiply(cellsMat, geneExp)
        sumM = np.sum(M, axis = 1) + 0.0001
        broSumM = np.ones(M.shape)*sumM
        edgeW = M / broSumM.T
        l1,l2 = np.nonzero(edgeW)
        DG = nx.from_numpy_array(cellsMat, create_using=nx.DiGraph)
        for i in range(l1.shape[0]):
            DG.add_weighted_edges_from([(l1[i], l2[i], edgeW[l1[i]][l2[i]])])
        n = switch[cls](DG).reshape(1,-1)
        with open(centralityDir, 'a') as f:
            np.savetxt(f, n, fmt = '%f', delimiter = ',')

def martrixCentrality(path, savePath, centralityDir, cells, cls = 'Degree'):
    switch = {
        'Degree' : degreeCentrality,
        'Eigenvector' : eigenvectorCentrality,
        'Betweenness' : betweennessCentrality,
        'PageRank' : pageRankCentrality,
        'Closeness' : closenessCentrality,
    }
    for k in tqdm(range(cells)):
        cellsSP = sparse.load_npz(path + 'csn' + str(k) + '.npz') # 读取
        cellsMat = cellsSP.toarray()
        DG = nx.from_numpy_array(cellsMat, create_using=nx.DiGraph)
        n = switch[cls](DG).reshape(1,-1)
        with open(centralityDir, 'a') as f:
            np.savetxt(f, n, fmt = '%f', delimiter = ',')

def scGIR(dataDir, name, min_cells, highlyVarGene):       
    
    adata = scDataCluster(dataDir, cls = 'h5', min_cells = min_cells, highlyVarGene = highlyVarGene)
    mtx = adata.X   
    dataName = name + str(highlyVarGene)  
    print(dataName)
    X = np.array(mtx).T  
    print(X)
    csnDataDir = dataDir  ####
    # savePath =  './data/Buettner'  ####   
    print(csnDataDir)

    savePath =  './data/' + name + '/'+ 'result/'  
    if not os.path.exists(savePath):
        os.makedirs(savePath)
    os.chmod(savePath, stat.S_IWRITE)
    
    
    csnConstruct_savepath = './data/' + name +  '/' +  'csnData/'  
    print(csnConstruct_savepath)
    if not os.path.exists(csnConstruct_savepath):
        os.makedirs(csnConstruct_savepath)
    os.chmod(csnConstruct_savepath, stat.S_IWRITE)
    
    
    adata.write(savePath + 'dataName_'+ 'highlyVarGene.h5ad')    
    
    
    csnConstruct(X, path=csnConstruct_savepath)   
      
    cls = 'GIR'   # cell_type.csv    
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    geneCentralityRank(X, path = csnConstruct_savepath, savePath =  savePath, centralityDir = centralityDir, cells = X.shape[1])
    
    
    cls = 'PageRank'
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    martrixCentrality(path = csnConstruct_savepath, savePath = savePath, centralityDir = centralityDir,  cells = X.shape[1], cls = cls)
        
    cls = 'Degree'    
    centralityDir = savePath + '/' +cls +'Matrix.csv'
    martrixCentrality(path = csnConstruct_savepath, savePath = savePath, centralityDir = centralityDir,  cells = X.shape[1], cls = cls)
  
    
scGIR('./Pro-inflammatory.h5ad', 'Pro-inflammatory', 10, 2000)


























