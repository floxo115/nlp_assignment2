import re
from collections import Counter
from pathlib import Path
from string import punctuation
from typing import Tuple

import dill
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.probability import FreqDist
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.base import clone, BaseEstimator
from sklearn.decomposition import TruncatedSVD
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, accuracy_score, ConfusionMatrixDisplay
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from tqdm import tqdm


def import_datasets(size: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    :param size: "medium"|"small" selects the loaded datasets
    :return: train / test / validation sets as Dataframes
    """

    path = Path(".", "datasets")

    train_path = path.joinpath(Path(f"thedeep.{size}.train.txt"))
    test_path = path.joinpath(Path(f"thedeep.{size}.test.txt"))
    validation_path = path.joinpath(Path(f"thedeep.{size}.validation.txt"))

    names = ["id", "text", "label"]
    index_col = "id"
    train_df: pd.DataFrame = pd.read_csv(train_path, names=names, index_col=index_col)
    test_df: pd.DataFrame = pd.read_csv(test_path, names=names, index_col=index_col)
    validation_df: pd.DataFrame = pd.read_csv(validation_path, names=names, index_col=index_col)

    return train_df, test_df, validation_df


def preprocess_txt(input_df: pd.DataFrame):
    """
    Takes a DataFrame representation of the text datasets and preprocesses it.
    :param input_df: DataFrame representing a dataset
    :return: preprocessed Dataframe
    """
    def preprocess(text: str):
        text = text.lower()
        text = text.translate(str.maketrans('', '', punctuation))
        words_lst = text.split(" ")

        # stem words with PorterStemmer
        stemmer = PorterStemmer()

        for i in range(len(words_lst)):

            if words_lst[i] in stopwords.words("english") or words_lst[i].isnumeric():
                words_lst[i] = ""

            words_lst[i] = stemmer.stem(words_lst[i])

        p_bar.update(1)
        return " ".join([word for word in words_lst if word != ""])

    working_copy = input_df.copy()
    p_bar = tqdm(total=len(working_copy))
    working_copy["text"] = working_copy["text"].apply(preprocess)
    p_bar.close()
    return working_copy


def get_n_most_common_tokens(input_df, n):
    counter = Counter(" ".join(input_df["text"].values).split(" "))
    del counter[""]
    return counter.most_common(n)


def clean_text_corpus(input_df: pd.DataFrame, token_count_dict: pd.DataFrame, threshold: int = None):
    """
    Removes all nontrivial tokens from the text corpus
    :param input_df:
    :param token_count_dict:
    :param threshold:
    :return: dataframe with cleaned text corpuses
    """
    def remove_words(vocabulary: list):
        def inner(text):
            text = " ".join([w for w in text.split(" ") if w in vocabulary])
            p_bar.update(1)
            return text
        return inner

    if threshold is not None and threshold > 0:
        vocabulary = (token_count_dict[token_count_dict["count"] > threshold])["token"].to_list()
    else:
        vocabulary = token_count_dict["token"].to_list()

    working_copy = input_df.copy()
    p_bar = tqdm(len(working_copy))
    working_copy["text"] = working_copy["text"].apply(remove_words(vocabulary))
    p_bar.close()
    return working_copy


def compute_sparsity(input_df: pd.DataFrame, vocabulary) -> int:
    """
    Retruns the degree of sparsity of the given array
    :param input_df:
    :param vocabulary:
    :return:
    """
    features = get_vocab_tokens_from_df(input_df, vocabulary).to_numpy(dtype=float)
    empty = (features == 0).sum()
    return empty / features.size


def get_dummy_baseline(baseline: pd.Series):
    """
    Creates a function acts as a baseline for our task. The given series provides a base for the calculation.
    The resulting function returns a randomly drawn label of those appearing in the baseline series with
    the probability of the label distribution of the baseline series.
    :param baseline:
    :return: function that draws randomly lables from the given baseline
    """
    def inner():
        return np.random.choice(len(probs), 1, p=probs)

    frequencies = baseline.value_counts(normalize=True)
    frequencies = frequencies.sort_index()
    probs = frequencies.to_numpy()

    return inner


def calc_baseline_confusion(lables, baseline_function):
    """
    Returns the confusion matrix and the accuracy of our baseline model
    :param lables: true lables
    :param baseline_function: Random lable generator for our baseline prediction
    :return: confusion matrix and the accuracy of our baseline model
    """
    bl_label = np.zeros(len(lables))

    for i in range(len(lables)):
        bl_label[i] = baseline_function()

    return confusion_matrix(lables, bl_label), accuracy_score(lables, bl_label)


def create_corpus_dict(data: pd.Series, threshold: int = None):
    """
    Returns dictionary of tokens that are
    :param data:
    :param threshold:
    :return:
    """
    token_counter = Counter((" ".join(data.values).split(" ")))
    if "" in token_counter:
        _ = token_counter.pop("")

    corpus_dict = pd.DataFrame({"token": token_counter.keys(), "count": token_counter.values()})

    if threshold is not None and int(threshold) > 1:
        corpus_dict = corpus_dict[corpus_dict["count"] >= threshold]

    sorted_dict = corpus_dict.sort_values("count", ascending=False)
    sorted_dict.reset_index(inplace=True, drop=True)
    return sorted_dict


def plot_token_dict(dict_to_plot: dict):
    """
    Plots a token dictionary
    :param dict_to_plot:
    :return:
    """
    plt.plot(range(1, len(dict_to_plot) + 1), dict_to_plot["count"])
    plt.yscale("log")
    plt.ylabel("occurences")
    plt.xlabel("tokens")
    plt.show()


def vectorize_dataframe(input_df, vocabulary):
    """
    Vectorizes text corpus of the given input dataframe and appends the measurement for the tokens of the given
    vocabulary as separate columns for each token.
    :param input_df:
    :param vocabulary:
    :return:
    """
    vocab_tokens = vocabulary["token"]
    result_df = input_df.copy()
    result_df[vocab_tokens] = None

    print("Create TC Dataset")
    p_bar = tqdm(total=(len(result_df) * len(vocab_tokens)))
    for r_index, r_value in result_df["text"].items():
        tokens = word_tokenize(r_value)
        fd = FreqDist(tokens)
        for t in vocab_tokens:
            result_df.loc[r_index, t] = fd[t]
            p_bar.update(1)
    p_bar.close()

    print("Create TF Dataset")
    p_bar = tqdm(total=len(vocab_tokens))
    result_log_df = result_df.copy()
    for t in vocab_tokens:
        result_log_df[t] = result_log_df[t].apply(lambda x: np.log(x + 1))
        p_bar.update(1)
    p_bar.close()

    return result_df, result_log_df


def latent_semantic_analysis(data_sets: dict, vocabulary, k: int = 50):
    """
    Gets a dictionary with train, test and validation datasets and performs svd on thoose. The k most informative
    features are kept and returned. Therefore the inital datasets are reduced in space.
    :param data_sets:
    :param vocabulary:
    :param k:
    :return:
    """
    vocab_tokens = vocabulary["token"].to_list()
    train_df = data_sets['train'][vocab_tokens].astype(dtype=np.float64)
    val_df = data_sets['val'][vocab_tokens].astype(dtype=np.float64)
    test_df = data_sets['test'][vocab_tokens].astype(dtype=np.float64)

    svd = TruncatedSVD(n_components=k)
    train_svd = svd.fit_transform(train_df)
    val_svd = svd.transform(val_df)
    test_svd = svd.transform(test_df)

    return train_svd, val_svd, test_svd, svd.explained_variance_ratio_


def get_best_lsa_reduced_feature_set(data_sets: dict, vocabulary, cutoff_delta: float = 3e-3):
    """
    The function returns the smallest but most informative representation of the given datasets. The size of
    the given datasets is reduced via latent segmantiv analysis (LSA). To get the smallest yet most informative
    representation, LSA is performed while increasing k. As long as the total explained variation
    between LSA of k and LSA of k-1 is bigger than the cutoff delta search is continued.

    :param data_sets:
    :param vocabulary:
    :param cutoff_delta:
    :return: Smallest and most informative representation of the given datasets
    """
    train_df = data_sets['train']
    max_num_latent_factors = train_df.shape[1]
    last_exp_ratio = 0
    l_tr_svd, l_v_svd, l_te_svd = None, None, None

    for i in range(1, max_num_latent_factors + 1):
        tr_svd, v_svd, te_svd, ratio = latent_semantic_analysis(data_sets, vocabulary, k=i)
        exp_variation = np.sum(ratio)

        if np.abs(exp_variation - last_exp_ratio) < cutoff_delta:
            print(f"Choose {len(ratio)} features that have a explained variation of {exp_variation:.3f}")
            break

        last_exp_ratio = exp_variation
        l_tr_svd = tr_svd
        l_v_svd = v_svd
        l_te_svd = te_svd

    return l_tr_svd, l_v_svd, l_te_svd


def get_vocab_tokens_from_df(df: pd.DataFrame, vocabulary):
    """
    Reduces the given input dataframe to just the token information holding columns
    :param df:
    :param vocabulary:
    :return:
    """
    return df[vocabulary["token"].to_list()]


def get_best_model_featureset(datasets: dict, classifier: dict):
    """
    Returns a dictionary with the results of all experiments done with the given input data.

    Result dictionary has the from:
        {
        'experiments':
            'feature_set_1': {
                'ml_modelclass_1': {
                    'hyperparameter_1': {
                        'value_1': {
                            'prediction_val': XX,
                            'prediction_test': XX,
                            'confusion_matrix_val': XX,
                            'confusion_matrix_test': XX,
                            'accuracy_val': XX,
                            'accuracy_test' XX
                        },
                        'value_2': {
                            ...
                        }
                        ...
                    },
                    'hyperparameter_2': {
                        ...
                    },
                    'hyperparameter_3': {
                        ...
                    }
                    ...
                },
                'ml_modelclass_2': {
                    ...
                }
                ...
            },
            'feature_set_2': {
                ...
            }
            ...
        'best_values': {
            'model': None,
            'features': None,
            'parameter': None,
            'parameter_value': None,
            'prediction': None,
            'accuracy': float('-inf'),
            'confusion_matrix': None
        }
    }
    :param datasets:
    :param classifier:
    :return:
    """
    labels = datasets['lables']

    result_dict = {
        'experiments': dict(),
        'best_values': {
            'model': None,
            'features': None,
            'parameter': None,
            'parameter_value': None,
            'prediction_val': None,
            'prediction_test': None,
            'accuracy_val': float('-inf'),
            'accuracy_test': float('-inf'),
            'confusion_matrix_val': None,
            'confusion_matrix_test': None
        }
    }
    # parameter_test_per_feature_set = 0
    # for cl_name, cl_data in classifier:
    #     for param, param_values in cl_data['parameters'].items():
    #         parameter_test_per_feature_set += len(param_values)

    for ds_name, ds_data in datasets['features'].items():
        for cl_name, cl_data in classifier.items():
            working_classifier: BaseEstimator = cl_data['modelclass']
            working_classifier = clone(working_classifier)
            for param, param_values in cl_data['parameters'].items():
                for pv in param_values:
                    working_classifier.set_params(**{param: pv})
                    working_classifier.fit(ds_data['train'], labels['train'])

                    experiment = dict()
                    experiment['prediction_val'] = working_classifier.predict(ds_data['val'])
                    experiment['confusion_matrix_val'] = confusion_matrix(y_true=labels['val'], y_pred=experiment['prediction_val'])
                    experiment['accuracy_val'] = accuracy_score(y_true=labels['val'], y_pred=experiment['prediction_val'])
                    experiment['prediction_test'] = working_classifier.predict(ds_data['test'])
                    experiment['confusion_matrix_test'] = confusion_matrix(y_true=labels['test'], y_pred=experiment['prediction_test'])
                    experiment['accuracy_test'] = accuracy_score(y_true=labels['test'], y_pred=experiment['prediction_test'])

                    if ds_name not in result_dict['experiments']:
                        result_dict['experiments'][ds_name] = dict()
                    if cl_name not in result_dict['experiments'][ds_name]:
                        result_dict['experiments'][ds_name][cl_name] = dict()
                    if param not in result_dict['experiments'][ds_name][cl_name]:
                        result_dict['experiments'][ds_name][cl_name][param] = dict()
                    if pv not in result_dict['experiments'][ds_name][cl_name][param]:
                        result_dict['experiments'][ds_name][cl_name][param][pv] = experiment
                    else:
                        raise ValueError(f"Multiple key for 'experiments' -> '{ds_name}' -> "
                                         f"'{cl_name}' -> '{param}' -> '{pv}'")

                    if experiment['accuracy_val'] > result_dict['best_values']['accuracy_val']:
                        result_dict['best_values']['model'] = cl_name
                        result_dict['best_values']['features'] = ds_name
                        result_dict['best_values']['parameter'] = param
                        result_dict['best_values']['parameter_value'] = pv
                        result_dict['best_values']['prediction_val'] = experiment['prediction_val']
                        result_dict['best_values']['prediction_test'] = experiment['prediction_test']
                        result_dict['best_values']['accuracy_val'] = experiment['accuracy_val']
                        result_dict['best_values']['accuracy_test'] = experiment['accuracy_test']
                        result_dict['best_values']['confusion_matrix_val'] = experiment['confusion_matrix_val']
                        result_dict['best_values']['confusion_matrix_test'] = experiment['confusion_matrix_test']

    return result_dict


def get_model_statistics(best_models: dict):
    statistics_data = []
    for ds_name, ds_data  in best_models['experiments'].items():
        for cl_name, cl_data in ds_data.items():
            for p_name, p_value_dict in cl_data.items():
                best_pv, best_acc_val, best_acc_test = None, float('-inf'), float('-inf')
                for p_value, exp_param_dict in p_value_dict.items():
                    if exp_param_dict['accuracy_val'] > best_acc_val:
                        best_pv = p_value
                        best_acc_val = exp_param_dict['accuracy_val']
                        best_acc_test = exp_param_dict['accuracy_test']

                statistics_data.append([ds_name, cl_name, p_name, best_pv, best_acc_val, best_acc_test])

    statistics_dataframe = pd.DataFrame(statistics_data,
                                        columns=['feature set',
                                                 'modelclass',
                                                 'parameter',
                                                 'best value',
                                                 'accuracy validation',
                                                 'accuracy test'])
    return statistics_dataframe


def make_scatter_plot(X, y, lables, title):

    # default color map
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
              '#bfef45', '#000000']

    plt.title(title)
    uq_lables = np.unique(lables)
    for l in uq_lables:
        (l_ids) = np.where(lables == l)
        plt.scatter(X[l_ids], y[l_ids], label=f"Class {l}", c=colors[l], alpha=0.4)

    plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1.15))
    plt.show()


def plot_features(feature_sets, best_prediction):
    orig_test_data = feature_sets['features'][best_prediction['features']]['test']
    orig_labels = feature_sets['lables']['test'].to_numpy()
    predicted_labels = best_prediction['prediction_test']

    feature_reduced = TSNE(n_components=2).fit_transform(orig_test_data)

    make_scatter_plot(feature_reduced[:, 0], feature_reduced[:, 1], orig_labels, "Given Test Data")
    make_scatter_plot(feature_reduced[:, 0], feature_reduced[:, 1], predicted_labels, "Predicted Test Data")


if __name__ == '__main__':
    # from pprint import pprint

    # train_df, test_df, val_df = import_datasets("small")
    # train_df = train_df.iloc[range(0,10)]
    # print(get_n_most_common_tokens(train_df, 20))
    # preprocess_txt(train_df)
    # print(get_n_most_common_tokens(train_df, 20))
    #
    # token_counter = Counter((" ".join(train_df["text"].values).split(" ")))
    #
    # corpus_dict = pd.DataFrame({"token": token_counter.keys(), "count": token_counter.values()})
    # corpus_dict = corpus_dict.sample(frac=1).reset_index(drop=True)
    #
    # threshold = 3
    #
    # reduced_corpus_dict = corpus_dict[corpus_dict["count"] >= threshold]
    #
    # pprint(train_df["text"])
    # reduce_words(train_df, corpus_dict, threshold)
    # pprint(train_df["text"])
    #
    # train_tc_vecs = []
    # train_tf_vecs = []
    # for doc_idx in range(len(train_df)):
    #     train_tc_vecs.append(create_tc_vec_from_doc(train_df, corpus_dict, doc_idx))
    #     train_tf_vecs.append(np.log(train_tc_vecs[-1] + 1))
    #
    # train_tc_vecs = np.array(train_tc_vecs).T
    # train_tf_vecs = np.array(train_tf_vecs).T
    #
    # pprint(train_tc_vecs)
    # pprint(train_tf_vecs)
    #
    # print(f"sparsity of training vector: {compute_sparsity(train_tc_vecs)}")
    #
    # k = 5
    # lsa_train_tc = latent_semantic_analysis(train_tc_vecs, k)
    # lsa_train_tf = latent_semantic_analysis(train_tf_vecs, k)
    # nltk.download()
    PREPROC = False
    REDUCE_WORDS = False
    VECTORIZE = False
    DIM_REDUCT = False
    COMPARE_MODELS = False

    if PREPROC:
        train_df, test_df, val_df = import_datasets("small")

        train_df = preprocess_txt(train_df)
        test_df = preprocess_txt(test_df)
        val_df = preprocess_txt(val_df)

        with open("train_df_preprocessed.pkl", "wb") as f:
            f.write(dill.dumps(train_df))
        with open("test_df_preprocessed.pkl", "wb") as f:
            f.write(dill.dumps(test_df))
        with open("val_df_preprocessed.pkl", "wb") as f:
            f.write(dill.dumps(val_df))
    else:
        with open("train_df_preprocessed.pkl", "rb") as f:
            train_df = dill.loads(f.read())
        with open("test_df_preprocessed.pkl", "rb") as f:
            test_df = dill.loads(f.read())
        with open("val_df_preprocessed.pkl", "rb") as f:
            val_df = dill.loads(f.read())

    orig_dict = create_corpus_dict(train_df["text"])
    reduced_dict = create_corpus_dict(train_df["text"], threshold=150)

    if REDUCE_WORDS:
        clean_text_corpus(train_df, reduced_dict)
        clean_text_corpus(test_df, reduced_dict)
        clean_text_corpus(val_df, reduced_dict)

        with open("train_df_reduced.pkl", "wb") as f:
            f.write(dill.dumps(train_df))
        with open("test_df_reduced.pkl", "wb") as f:
            f.write(dill.dumps(test_df))
        with open("val_df_reduced.pkl", "wb") as f:
            f.write(dill.dumps(val_df))
    else:
        with open("train_df_reduced.pkl", "rb") as f:
            train_df = dill.loads(f.read())
        with open("test_df_reduced.pkl", "rb") as f:
            test_df = dill.loads(f.read())
        with open("val_df_reduced.pkl", "rb") as f:
            val_df = dill.loads(f.read())

    if VECTORIZE:
        train_df_tc, train_df_tf = vectorize_dataframe(train_df, reduced_dict)
        test_df_tc, test_df_tf = vectorize_dataframe(test_df, reduced_dict)
        val_df_tc, val_df_tf = vectorize_dataframe(val_df, reduced_dict)

        with open("train_df_vectorize_tc.pkl", "wb") as f:
            f.write(dill.dumps(train_df_tc))
        with open("test_df_vectorize_tc.pkl", "wb") as f:
            f.write(dill.dumps(test_df_tc))
        with open("val_df_vectorize_tc.pkl", "wb") as f:
            f.write(dill.dumps(val_df_tc))
        with open("train_df_vectorize_tf.pkl", "wb") as f:
            f.write(dill.dumps(train_df_tf))
        with open("test_df_vectorize_tf.pkl", "wb") as f:
            f.write(dill.dumps(test_df_tf))
        with open("val_df_vectorize_tf.pkl", "wb") as f:
            f.write(dill.dumps(val_df_tf))
    else:
        with open("train_df_vectorize_tc.pkl", "rb") as f:
            train_df_tc = dill.loads(f.read())
        with open("test_df_vectorize_tc.pkl", "rb") as f:
            test_df_tc = dill.loads(f.read())
        with open("val_df_vectorize_tc.pkl", "rb") as f:
            val_df_tc = dill.loads(f.read())
        with open("train_df_vectorize_tf.pkl", "rb") as f:
            train_df_tf = dill.loads(f.read())
        with open("test_df_vectorize_tf.pkl", "rb") as f:
            test_df_tf = dill.loads(f.read())
        with open("val_df_vectorize_tf.pkl", "rb") as f:
            val_df_tf = dill.loads(f.read())

    print(f"sparsity of training vector: {compute_sparsity(train_df_tc, reduced_dict)}")
    print(f"sparsity of testing vector: {compute_sparsity(test_df_tc, reduced_dict)}")
    print(f"sparsity of validation vector: {compute_sparsity(val_df_tc, reduced_dict)}")

    #create training sets
    dict_feature_sets = {
        'features': {
            'tc': dict(),
            'tf': dict(),
            'tc_svd': dict(),
            'tf_svd': dict()
        },
        'lables': dict()
    }

    dict_feature_sets['features']['tc']['train'] = get_vocab_tokens_from_df(train_df_tc, reduced_dict)
    dict_feature_sets['features']['tc']['val'] = get_vocab_tokens_from_df(val_df_tc, reduced_dict)
    dict_feature_sets['features']['tc']['test'] = get_vocab_tokens_from_df(test_df_tc, reduced_dict)

    dict_feature_sets['features']['tf']['train'] = get_vocab_tokens_from_df(train_df_tf, reduced_dict)
    dict_feature_sets['features']['tf']['val'] = get_vocab_tokens_from_df(val_df_tf, reduced_dict)
    dict_feature_sets['features']['tf']['test'] = get_vocab_tokens_from_df(test_df_tf, reduced_dict)

    dict_feature_sets['lables']['train'] = train_df['label']
    dict_feature_sets['lables']['val'] = val_df['label']
    dict_feature_sets['lables']['test'] = test_df['label']

    del train_df, test_df, val_df, train_df_tf, test_df_tf, val_df_tf, train_df_tc, test_df_tc, val_df_tc

    if DIM_REDUCT:
        dict_feature_sets['features']['tc_svd']['train'], \
        dict_feature_sets['features']['tc_svd']['val'], \
        dict_feature_sets['features']['tc_svd']['test'] = \
            get_best_lsa_reduced_feature_set(dict_feature_sets['features']['tc'], reduced_dict)
        dict_feature_sets['features']['tf_svd']['train'], \
        dict_feature_sets['features']['tf_svd']['val'], \
        dict_feature_sets['features']['tf_svd']['test'] = \
            get_best_lsa_reduced_feature_set(dict_feature_sets['features']['tf'], reduced_dict)

        with open("train_df_vectorize_tc_lsa.pkl", "wb") as f:
            f.write(dill.dumps(dict_feature_sets['features']['tc_svd']['train']))
        with open("test_df_vectorize_tc_lsa.pkl", "wb") as f:
            f.write(dill.dumps(dict_feature_sets['features']['tc_svd']['test']))
        with open("val_df_vectorize_tc_lsa.pkl", "wb") as f:
            f.write(dill.dumps(dict_feature_sets['features']['tc_svd']['val']))
        with open("train_df_vectorize_tf_lsa.pkl", "wb") as f:
            f.write(dill.dumps(dict_feature_sets['features']['tf_svd']['train']))
        with open("test_df_vectorize_tf_lsa.pkl", "wb") as f:
            f.write(dill.dumps(dict_feature_sets['features']['tf_svd']['test']))
        with open("val_df_vectorize_tf_lsa.pkl", "wb") as f:
            f.write(dill.dumps(dict_feature_sets['features']['tf_svd']['val']))
    else:
        with open("train_df_vectorize_tc_lsa.pkl", "rb") as f:
            dict_feature_sets['features']['tc_svd']['train'] = dill.loads(f.read())
        with open("test_df_vectorize_tc_lsa.pkl", "rb") as f:
            dict_feature_sets['features']['tc_svd']['test'] = dill.loads(f.read())
        with open("val_df_vectorize_tc_lsa.pkl", "rb") as f:
            dict_feature_sets['features']['tc_svd']['val'] = dill.loads(f.read())
        with open("train_df_vectorize_tf_lsa.pkl", "rb") as f:
            dict_feature_sets['features']['tf_svd']['train'] = dill.loads(f.read())
        with open("test_df_vectorize_tf_lsa.pkl", "rb") as f:
            dict_feature_sets['features']['tf_svd']['test'] = dill.loads(f.read())
        with open("val_df_vectorize_tf_lsa.pkl", "rb") as f:
            dict_feature_sets['features']['tf_svd']['val'] = dill.loads(f.read())

    # TASK-B - 1
    dummy_baseline_confusion, baseline_acc = calc_baseline_confusion(dict_feature_sets['lables']['test'],
                                                                     get_dummy_baseline(baseline=dict_feature_sets['lables']['train']))

    dict_parameter_sets = {
        'knn': {
            'modelclass': KNeighborsClassifier(),
            'parameters': {
                'n_neighbors': range(40, 55)
            }
        },
        'randomForest': {
            'modelclass': RandomForestClassifier(),
            'parameters': {
                'n_estimators': range(1, 510, 100),
                'max_depth': range(1, 15),
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_leaf_nodes': range(2, 11, 2),
                'criterion': ['gini', 'entropy']
            }
        }
    }

    if COMPARE_MODELS:
        model_comparison = get_best_model_featureset(datasets=dict_feature_sets, classifier=dict_parameter_sets)
        with open("model_comparison.pkl", "wb") as f:
            f.write(dill.dumps(model_comparison))
    else:
        with open("model_comparison.pkl", "rb") as f:
            model_comparison = dill.loads(f.read())

    # For the comparison task we choose as ML-Models the KNearestNeighbors and the RandomForest Classifier.
    # The following parameters and corresponding values were chosen and compared.
    #   KNN:
    #       n_neighbors: 40-60
    #   RandomForest:
    #       n_estimators: {1, 101, 201, 301, 401, 501},
    #       max_depth: 1-15,
    #       max_features: {'auto', 'sqrt', 'log2'},
    #       max_leaf_nodes: {2, 4, 6, 8, 10},
    #       criterion: {'gini', 'entropy'}
    # Each of the 50+ possible model variations was trained and tested against the 4 feature sets.
    #   tc: Counts of each token in the document
    #   tf: log(count +1) of the tokens
    #   tc_svd: SVD performed on the 'tc' dataset with a reduction to the '{k_tc}' most informative components
    #   tf_svd: SVD performed on the 'tf' dataset with a reduction to the '{k_tf}' most informative components


    best_predictions = model_comparison['best_values']
    print(f"The best combination over all 200+ model-dataset combinations was the {best_predictions['model']} model "
          f"on the {best_predictions['features']} feature set, where the {best_predictions['parameter']} was set "
          f"to '{best_predictions['parameter_value']}' and the accuracy gained a value of "
          f"'{best_predictions['accuracy_val']}' on the validation set.")

    statistics = get_model_statistics(model_comparison)
    print(statistics)

    disp = ConfusionMatrixDisplay(confusion_matrix=best_predictions['confusion_matrix_test'])
    disp.plot()
    plt.show()

    plot_features(dict_feature_sets, best_predictions)



