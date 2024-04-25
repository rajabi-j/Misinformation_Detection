import pickle
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV
import numpy as np
from sklearn.svm import SVC
import pandas as pd
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

# Fucntion Definition
def load_data():
    data_file_path = 'Vocab_Train_Test_Data.pkl'
    with open(data_file_path, 'rb') as f:
        loaded_data = pickle.load(f)

    co_oc_file_path = 'combined_co_occurence.pkl'
    with open(co_oc_file_path, 'rb') as file:
        loaded_co_oc = pickle.load(file)

    return loaded_data, loaded_co_oc


# Fucntion Definition
def calculate_pca(word_limit):
    co_oc = [df for df in loaded_co_oc if len(df) >= word_limit]

    x_pca = []
    #instantiate scaler on training data
    scl_txt = StandardScaler(with_mean = True, with_std = False)
    pca_txt_mod = PCA(n_components = word_limit, svd_solver='full')
    for item in co_oc:
        item_scaled = scl_txt.fit_transform(item.T)
        pca_result = pca_txt_mod.fit_transform(item_scaled)
        x_pca.append(pca_result)

    return np.stack(x_pca, axis=0)


# Fucntion Definition
def calculate_A(data, mat_dim, iva_type, word_limit):
    co_oc = [df for df in loaded_co_oc if len(df) >= word_limit]

    iva_file_path = f'iva_out/IVA_{iva_type}_{mat_dim}_output.pkl'
    with open(iva_file_path, 'rb') as file:
        context = pickle.load(file)
    iva_spice = pd.Series(context)
    W = iva_spice["W"]
    # Ws = np.transpose(W, (2, 0, 1))

    F_list = []
    for X, PC in zip(co_oc, data):
        F = np.dot(np.linalg.pinv(X.T), PC)
        F_list.append(F)

    A_list = []
    for i in range(len(F_list)):
        A = np.dot(np.linalg.pinv(F_list[i].T), np.linalg.inv(W[:,:,i])) 
        A_list.append(A)

    return np.vstack(A_list)

# Fucntion Definition
def find_vocabs(word_limit):
    file_path = 'event_vocabs.pkl'
    with open(file_path, 'rb') as f:
        vocabs_t = pickle.load(f)  

    event_vocabs = []
    for t in vocabs_t:
        if len(vocabs_t[t]) >= word_limit:
            event_vocabs.append(vocabs_t[t])

    return np.concatenate(event_vocabs).tolist()



# Fucntion Definition
# create function to recode label field
def recode_annotation(txt_field):
    if txt_field == "fake":
        return 1
    else:
        return 0

# Fucntion Definition
def embedding_feats(list_of_vects, list_of_labels, data_vect, total_event_vocabs, word_limit):
    DIMENSION = word_limit
    feats = []
    labels = []
    for idx, tweet in enumerate(list_of_vects):
        feat_for_this = np.zeros(DIMENSION)
        count_for_this = 0
        for token in tweet:
            if token in total_event_vocabs:
                index = total_event_vocabs.index(token)
                count_for_this += 1
                feat_for_this += data_vect.T[:, index]  # Access the vector for the token

        if count_for_this != 0:
            feat_for_this = feat_for_this / count_for_this  # Average the vectors
            feats.append(feat_for_this)
            labels.append(recode_annotation(list_of_labels[idx]))

    return feats, labels


# Function to calculate metrics
def calculate_metrics(labels, predictions):
    return {
        'Accuracy': accuracy_score(labels, predictions),
        'Precision': precision_score(labels, predictions),
        'Recall': recall_score(labels, predictions),
        'F1 Score': f1_score(labels, predictions)
    }


#Function to train
def find_best_model(train_vectors, train_labels):

    parameters = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf']
    }

    scoring = {
        'accuracy': make_scorer(accuracy_score),
        'precision': make_scorer(precision_score, average='binary'),
        'recall': make_scorer(recall_score, average='binary'),
        'f1': make_scorer(f1_score, average='binary')
    }

    # Cross-Validation setup
    svm = SVC(gamma='scale', random_state=42)
    svm_GS = GridSearchCV(svm, param_grid=parameters, scoring=scoring, refit='f1', return_train_score=True)
    svm_GS.fit(train_vectors, train_labels)

    # The best estimator found by GridSearchCV
    return svm_GS.best_estimator_, svm_GS.best_params_


# Start of the main program
if __name__ == "__main__":

    all_results = {}

    loaded_data, loaded_co_oc = load_data()

    # Accessing the lists from the loaded data
    total_vocab = loaded_data['total_vocab']
    x_train = loaded_data['x_train']
    y_train = loaded_data['y_train']
    x_test = loaded_data['x_test']
    y_test = loaded_data['y_test']

    mat_dim_list = [i for i in range(7, 20)]
    iva_type_list = ['G', 'S']
    output_text_path = f'A_out.txt'

    for mat_dim in mat_dim_list:
        for iva_type in iva_type_list:

            cooc_lens = [len(df) for df in loaded_co_oc]
            cooc_lens.sort(reverse=True)
            word_limit = cooc_lens[mat_dim -1]

            data = calculate_pca(word_limit)

            try:
                data_vect = calculate_A(data, mat_dim, iva_type, word_limit)
            except:
                with open(output_text_path, 'a') as f:
                    output = f"Error occured at {iva_type}_{mat_dim}"
                    print(output, file=f)
                continue

            total_event_vocabs = find_vocabs(word_limit)

            train_vectors, train_labels = embedding_feats(x_train, y_train, data_vect, total_event_vocabs, word_limit)
            test_vectors, test_labels = embedding_feats(x_test, y_test, data_vect, total_event_vocabs, word_limit)

            best_model, best_params = find_best_model(train_vectors, train_labels)
            results = {
                'Train': calculate_metrics(train_labels, best_model.predict(train_vectors)),
                'Test': calculate_metrics(test_labels, best_model.predict(test_vectors)),
                'Best Model Parameters': best_params
            }

            results_key = f"{iva_type}_{mat_dim}"
            all_results[results_key] = results

            with open(output_text_path, 'a') as f:
                output = f"{results_key}\n {results}"
                print(output, file=f)

    results_file = 'model_results_A.pkl'
    with open(results_file, 'wb') as f:
        pickle.dump(all_results, f)

