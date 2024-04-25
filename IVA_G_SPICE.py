import pickle
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from independent_vector_analysis import iva_g
from spice_iva_g import iva_spice
import os
import time

if not os.path.exists('iva_out'):
    os.mkdir('iva_out')

co_oc_file_path = 'combined_co_occurence.pkl'

with open(co_oc_file_path, 'rb') as file:
    loaded_co_oc = pickle.load(file)

cooc_lens = [len(df) for df in loaded_co_oc]
cooc_lens.sort(reverse=True)

mat_dim = 19
iva_type = 'S'

word_limit = cooc_lens[mat_dim -1]

co_oc = [df for df in loaded_co_oc if len(df) >= word_limit]


x_pca = []
#instantiate scaler on training data
scl_txt = StandardScaler(with_mean = True, with_std = False)
pca_txt_mod = PCA(n_components = word_limit, svd_solver='full')
for item in co_oc:
    item_scaled = scl_txt.fit_transform(item.T)
    pca_result = pca_txt_mod.fit_transform(item_scaled)
    x_pca.append(pca_result)

x_data = np.stack(x_pca, axis=0)

X = np.transpose(x_data, (2, 1, 0))

start_time = time.time() 

if iva_type == 'G':
    W, cost, Sigma_n, isi = iva_g(X, whiten=False)
elif iva_type == 'S':
    W, cost, Sigma_n, isi = iva_spice(X, whiten=False)

time_spent = time.time() - start_time

IVA_file_path = f'iva_out/IVA_{iva_type}_{mat_dim}_output.pkl'
with open(IVA_file_path, 'wb') as file:
    pickle.dump({'W': W, 'Cost': cost, 'Sigma_n': Sigma_n, 'ISI': isi, 'Time': time_spent}, file)
