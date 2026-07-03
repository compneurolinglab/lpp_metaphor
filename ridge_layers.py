import os
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)
import numpy as np
from pathlib import Path
import pandas as pd
from scipy.stats import zscore
from joblib import Parallel, delayed
from himalaya.backend import set_backend
from himalaya.kernel_ridge import MultipleKernelRidgeCV, Kernelizer, ColumnKernelizer
from himalaya.scoring import correlation_score_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.decomposition import PCA
import nibabel as nib
import sys
import time
# Use CPU backend
backend = set_backend("numpy")

DIR = "/scratch/ResearchGroups/lt_jixingli/lpp_metaphor/"
os.chdir(DIR)
subj_id = int(sys.argv[1])
subj = 'sub-CN00%s' %subj_id if subj_id < 10 else 'sub-CN0%s' %subj_id

sem_embs = np.load('Analysis/embs/hrf/lppCN_sem_hrf.npy')
prag_embs = np.load('Analysis/embs/hrf/lppCN_prag_hrf.npy')
df_run = pd.read_csv('Analysis/lppCN_run_info.csv')

train_scans = df_run[df_run['run'] <= 8]['n_scans'].sum()  # 2576
test_scans = df_run[df_run['run'] == 9]['n_scans'].iloc[0]  # 401
train_idx = np.arange(train_scans)
test_idx = np.arange(train_scans, train_scans + test_scans)
print(f"Train scans: {len(train_idx)}, Test scans: {len(test_idx)}")  # 2576 + 401 = 2977

# HRF shape: (n_layers, n_scans, n_dims)
n_layers, n_scans_total, n_dims = sem_embs.shape
n_components = 100
n_train, n_test = len(train_idx), len(test_idx)
X_sem_train = np.zeros((n_layers, n_train, n_components))
X_sem_test = np.zeros((n_layers, n_test, n_components))
X_prag_train = np.zeros((n_layers, n_train, n_components))
X_prag_test = np.zeros((n_layers, n_test, n_components))

print(f"PCA preprocessing {n_layers} layers...")
for layer in range(n_layers):
    # sem
    sem_layer = sem_embs[layer]  # (n_scans, n_dims)
    sem_z = np.nan_to_num(zscore(sem_layer, axis=0, nan_policy='omit'))
    pca_sem = PCA(n_components=n_components)
    X_sem_train[layer] = pca_sem.fit_transform(sem_z[train_idx])
    X_sem_test[layer] = pca_sem.transform(sem_z[test_idx])
    
    # prag
    prag_layer = prag_embs[layer]
    prag_z = np.nan_to_num(zscore(prag_layer, axis=0, nan_policy='omit'))
    pca_prag = PCA(n_components=n_components)
    X_prag_train[layer] = pca_prag.fit_transform(prag_z[train_idx])
    X_prag_test[layer] = pca_prag.transform(prag_z[test_idx])

# StandardScaler per layer
for layer in range(n_layers):
    scaler_sem = StandardScaler()
    X_sem_train[layer] = scaler_sem.fit_transform(X_sem_train[layer])
    X_sem_test[layer] = scaler_sem.transform(X_sem_test[layer])
    scaler_prag = StandardScaler()
    X_prag_train[layer] = scaler_prag.fit_transform(X_prag_train[layer])
    X_prag_test[layer] = scaler_prag.transform(X_prag_test[layer])

def fit_layer(layer_idx, Y_train_z, Y_test_z):
    X_train = np.hstack([X_sem_train[layer_idx], X_prag_train[layer_idx]])
    X_test = np.hstack([X_sem_test[layer_idx], X_prag_test[layer_idx]])
    slices = [slice(0, n_components), slice(n_components, 2*n_components)]
    kernelizers = [(name, Kernelizer(), sl) for name, sl in zip(['sem', 'prag'], slices)]
    column_kernelizer = ColumnKernelizer(kernelizers)
    # solver_params = dict(alphas=np.logspace(0,20,10), n_iter=50, progress_bar=False)
    model = MultipleKernelRidgeCV(kernels='precomputed',solver_params=None,random_state=42)
    # model = MultipleKernelRidgeCV(kernels='precomputed', solver='random_search', solver_params=solver_params, random_state=42)
    pipe = make_pipeline(column_kernelizer, model)
    pipe.fit(X_train.astype(np.float32), Y_train_z.astype(np.float32))
    y_pred_split = pipe.predict(X_test.astype(np.float32), split=True)
    y_test_backend = backend.asarray(Y_test_z.astype(np.float32))
    corr_split = correlation_score_split(y_true=y_test_backend, y_pred=y_pred_split)
    return backend.to_numpy(corr_split)  # (n_kernels, n_vertices)

# load fmri data
subj_dir = Path('/scratch/ResearchGroups/lt_jixingli/lpp/Data/derivatives')/subj/'func'
print(f"\nProcessing subject {subj}...")
fmri_runs_L, fmri_runs_R = [], []
for hemi in ['L', 'R']:
    gii_files = sorted(subj_dir.glob(f"*hemi-{hemi}_space-fsaverage*.gii"))
    runs = []
    for p in gii_files:
        gii = nib.load(p)
        fmri = np.vstack([d.data for d in gii.darrays])[:-4, :2562]
        runs.append(fmri)
    if hemi == 'L':
        fmri_runs_L = runs
    else:
        fmri_runs_R = runs

fmri_L = np.concatenate(fmri_runs_L, axis=0)
fmri_R = np.concatenate(fmri_runs_R, axis=0)
Y_all = np.hstack([fmri_L, fmri_R]).mean(axis=-1)

Y_train = Y_all[train_idx].reshape(-1, 1)
Y_test = Y_all[test_idx].reshape(-1, 1)
y_scaler = StandardScaler()
Y_train_z = y_scaler.fit_transform(Y_train)
Y_test_z = y_scaler.transform(Y_test)

print(f"Parallel fitting {n_layers} layers...")
start = time.time()
layer_scores = Parallel(n_jobs=-1, verbose=10)(
    delayed(fit_layer)(layer_idx, Y_train_z, Y_test_z) 
    for layer_idx in range(n_layers)
)
print(f"Time: {time.time()-start:.1f}s")

scores = np.stack(layer_scores, axis=0)
print(f"Scores shape: {scores.shape}")  # (32, 2, n_vertices)
print(f"Sem mean corr per layer: {scores[:, 0, :].mean(axis=1)[:5]}...")
print(f"Prag mean corr per layer: {scores[:, 1, :].mean(axis=1)[:5]}...")
np.save('Results/ridge_layers/%s.npy' %subj, scores)
