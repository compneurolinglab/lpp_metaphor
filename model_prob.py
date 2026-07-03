import os, sys
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
from scipy.stats import zscore, pearsonr

DIR = "/scratch/ResearchGroups/lt_jixingli/lpp_metaphor/"
os.chdir(DIR)

sem_embs = np.load('Analysis/embs/lppCN_sem_last.npy')[:,1:,:]
prag_embs = np.load('Analysis/embs/lppCN_prag_last.npy')[:,1:,:]
rand_embs = np.array([np.random.permutation(i) for i in sem_embs])

n_layers = sem_embs.shape[1]
layers = range(n_layers)

n_components = 100
alpha = 1.0
n_splits = 5
kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

def probe_one_layer(X_raw, Y_raw):
    fold_scores = []
    for train_idx, test_idx in kf.split(X_raw):
        X_tr, X_te = X_raw[train_idx], X_raw[test_idx]
        Y_tr, Y_te = Y_raw[train_idx], Y_raw[test_idx]

        pca_X = PCA(n_components=n_components, random_state=42)
        X_tr = pca_X.fit_transform(X_tr)
        X_te = pca_X.transform(X_te)

        pca_Y = PCA(n_components=n_components, random_state=42)
        Y_tr = pca_Y.fit_transform(Y_tr)
        Y_te = pca_Y.transform(Y_te)

        sc_X = StandardScaler()
        X_tr = sc_X.fit_transform(X_tr)
        X_te = sc_X.transform(X_te)

        sc_Y = StandardScaler()
        Y_tr = sc_Y.fit_transform(Y_tr)
        Y_te = sc_Y.transform(Y_te)

        model = Ridge(alpha=alpha)
        model.fit(X_tr, Y_tr)
        Y_scores = model.score(X_te, Y_te)

        # cos_sim = np.diag(cosine_similarity(Y_pred, Y_te))
        # cos_sim = [pearsonr(Y_pred[i], Y_te[i])[0] for i in range(len(Y_te))]
        fold_scores.append(np.mean(Y_scores))
    return np.mean(fold_scores)

layer_scores = []
ctrl_scores = []

for layer in layers:
    print(f"Layer {layer}/{n_layers-1}")
    X_raw = sem_embs[:, layer, :]
    X_raw = np.nan_to_num(zscore(X_raw,nan_policy='omit'))
    prag_layer = prag_embs[:, layer, :]
    prag_layer = np.nan_to_num(zscore(prag_layer,nan_policy='omit'))
    rand_layer = rand_embs[:, layer, :]
    rand_layer = np.nan_to_num(zscore(rand_layer,nan_policy='omit'))

    layer_scores.append(probe_one_layer(X_raw, prag_layer))
    ctrl_scores.append(probe_one_layer(rand_layer, prag_layer))

layer_scores = np.array(layer_scores)
ctrl_scores = np.array(ctrl_scores)

print("\n===== Results =====")
for l, true, ctrl in zip(layers, layer_scores, ctrl_scores):
    print(f"Layer {l:2d}: true={true:.4f}  ctrl={ctrl:.4f}  Δ={true-ctrl:.4f}")

np.save('Results/model_probe/probe_scores_new.npy', layer_scores)
np.save('Results/model_probe/ctrl_scores_new.npy', ctrl_scores)