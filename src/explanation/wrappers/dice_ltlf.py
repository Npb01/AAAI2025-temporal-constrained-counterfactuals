import warnings
import os
from datetime import datetime
import dice_ml
import numpy as np
import pandas as pd
import pm4py
from scipy.spatial.distance import _validate_vector
from scipy.spatial.distance import cdist, pdist
from scipy.stats import median_abs_deviation
from pm4py import convert_to_event_log,format_dataframe
from Declare4Py.D4PyEventLog import D4PyEventLog
from Declare4Py.ProcessModels.LTLModel import LTLModel,LTLTemplate
from Declare4Py.ProcessMiningTasks.ConformanceChecking.LTLAnalyzer import LTLAnalyzer
from src.encoding.common import get_encoded_df, EncodingType
import regex as re
from src.predictive_model.common import ClassificationMethods

warnings.filterwarnings("ignore", category=UserWarning)

class IterationCallback:
    def __init__(self):
        self.iteration_count = 0

    def __call__(self, population, scores, best_solution, best_score, model):
        self.iteration_count += 1

# Create an instance of the callback
iteration_callback = IterationCallback()

def dice_ltlf_explain(CONF, predictive_model, encoder, df, query_instances, method, optimization, heuristic,
                 model_path, case_ids=None, random_seed=None, adapted=None,
                 ltlf_model=None,path_results=None, dfa=None, percentage=None):
    """
    This function generation counterfactuals for LTLf models using the DICE framework.
    :param CONF: Configuration dictionary containing the necessary settings.
    :param predictive_model: Predictive model used for generating counterfactuals.
    :param encoder: Data encoder for transforming data as required by the model.
    :param df: Reference population DataFrame for generating the explanations.
    :param query_instances: Test set samples uesd for counterfactual generation
    :param method: Method used for generating counterfactuals.
    :param optimization: Type of optimization strategy used for generating counterfactuals.
    :param heuristic: The strategy used for heuristic search.
    :param model_path: Path to the saved model, if applicable.
    :param case_ids: IDs of the cases to be explained.
    :param random_seed: Seed for random number generation.
    :param adapted: Boolean indicating if the adapted version of the explanation method should be used.
    :param ltlf_model: LTLf model for compliance checking.
    :param path_results: Path where the results should be saved.
    :param dfa: Deterministic Finite Automaton for LTLf formula compliance.
    :param percentage: Coverage percentage value used for counterfactual generation.
    :return:
    """
    features_names = df.columns.values[:-1]
    feature_selection = CONF['feature_selection']
    dataset = CONF['data'].rpartition('/')[0].replace('../datasets/', '')
    path_results = path_results + '/' + dataset + '/' + str(percentage) + '/'
    try:
        if not os.path.exists(path_results):
            os.makedirs(path_results)
    except OSError as error:
        print("Directory '%s' can not be created" % path_results)
    if 'bpic2012' in dataset:
        dataset_created = dataset.replace('-COMPLETE', '').replace('bpic2012', 'BPIC12')
    black_box = predictive_model.model_type
    categorical_features, continuous_features, cat_feature_index, cont_feature_index = split_features(
        df.iloc[:, :-1], encoder)
    ratio_cont = len(continuous_features) / len(categorical_features)
    time_start = datetime.now()
    query_instances_for_cf = query_instances.iloc[:2, :-1]
    d = dice_ml.Data(dataframe=df, continuous_features=continuous_features, outcome_name='label')
    m = dice_model(predictive_model)
    dice_query_instance = dice_ml.Dice(d, m, method)
    time_train = (datetime.now() - time_start).total_seconds()
    index_test_instances = range(len(query_instances_for_cf))

    encoder.decode(df)
    alphabet = pd.unique(df[[col for col in df.columns if 'prefix' in col]].values.ravel())
    result = extract_matching_substrings(ltlf_model.formula, alphabet)
    encoder.encode(df)


    try:
        if not os.path.exists(model_path):
            os.makedirs(model_path)
            print("Directory '%s' created successfully" % model_path)
    except OSError as error:
        print("Directory '%s' can not be created" % model_path)

    for test_id, i in enumerate(index_test_instances):
        print(datetime.now(), dataset, black_box, test_id, len(index_test_instances),
              '%.2f' % (test_id + 1 / len(index_test_instances)))
        cf_list_all = list()
        x_eval_list = list()
        desired_cfs_all = list()
        x = query_instances_for_cf.iloc[[i]]
        predicted_outcome = predictive_model.model.predict(x.values.reshape(1, -1))[0]
        for k in [5, 10, 15, 20]:
            time_start_i = datetime.now()
            if method == 'genetic_ltlf':
                dice_result = dice_query_instance.generate_counterfactuals(x, encoder=encoder, desired_class='opposite',
                                                                           verbose=False,
                                                                           posthoc_sparsity_algorithm='linear',
                                                                           total_CFs=k, dataset=dataset + '_' + str(
                        CONF['prefix_length']),
                                                                           model_path=model_path,
                                                                           optimization=optimization,
                                                                           heuristic=heuristic, random_seed=random_seed,
                                                                           adapted=adapted,ltlf_model=ltlf_model,original_activities=result,
                                                                           dfa= dfa)
            else:
                dice_result = dice_query_instance.generate_counterfactuals(x, encoder=encoder, desired_class='opposite',
                                                                           verbose=False,
                                                                           posthoc_sparsity_algorithm='linear',
                                                                           total_CFs=k, dataset=dataset + '_' + str(
                        CONF['prefix_length']),
                                                                           )
            # function to decode cf from train_df and show it decoded before adding to list
            generated_cfs = dice_result.cf_examples_list[0].final_cfs_df
            cf_list = np.array(generated_cfs).astype('float64')
            y_pred = predictive_model.model.predict(x.values.reshape(1, -1))[0]
            time_test = (datetime.now() - time_start_i).total_seconds()

            x_eval = evaluate_cf_list(cf_list, x.values.reshape(1, -1), cont_feature_index, cat_feature_index,
                                      df=df,
                                      nr_of_cfs=k, y_pred=y_pred, predictive_model=predictive_model,
                                      query_instances=query_instances, continuous_features=continuous_features,
                                      categorical_features=categorical_features, ratio_cont=ratio_cont
                                      )
            x_eval['dataset'] = dataset
            x_eval['idx'] = test_id + 1
            x_eval['model'] = predictive_model.model_type
            x_eval['desired_nr_of_cfs'] = k
            x_eval['time_train'] = time_train
            x_eval['time_test'] = time_test
            x_eval['runtime'] = time_train + time_test
            #  x_eval['generated_cfs'] = x_eval['nbr_cf']
            if adapted == False and method == 'genetic_ltlf':
                x_eval['method'] = 'genetic_ltlf_baseline_operators'
            else:
                x_eval['method'] = method
            x_eval['explainer'] = CONF['explanator']
            x_eval['prefix_length'] = CONF['prefix_length']
            x_eval['heuristic'] = heuristic
            x_eval['optimization'] = optimization
            x_eval_list.append(x_eval)
            if cf_list.size > 4:
                if method == 'random':
                    cf_list = cf_list[:, :-1]
                elif method == 'genetic':
                    cf_list = cf_list[:, :-1]
                elif method == 'genetic_ltlf':
                    cf_list = cf_list[:, :-1]
                df_conf = pd.DataFrame(data=cf_list, columns=features_names)
                sat_score = conformance_score(encoder, df=df_conf,
                                              features_names=features_names,
                                              ltlf_model=ltlf_model, dfa=dfa)
                x_eval['sat_score'] = sat_score
                cf_list_all.extend(cf_list[:5])
                desired_cfs = [float(k) * np.ones_like(cf_list[:5, 0])]

                desired_cfs_all.extend(*desired_cfs)
        filename_results = path_results + 'cfeval_%s_%s_dice_%s.csv' % (dataset, black_box, feature_selection)
        if len(cf_list_all) > 0:
            df_cf = pd.DataFrame(data=cf_list_all, columns=features_names)
            encoder.decode(df_cf)
            df_cf['desired_cfs'] = desired_cfs_all
            if case_ids:
                df_cf['case_id'] = case_ids[i]
            else:
                df_cf['idx'] = test_id + 1 * len(cf_list_all)
            # df_cf['method']= method
            df_cf['test_id'] = np.arange(0, len(cf_list_all))
            df_cf['dataset'] = [dataset] * len(cf_list_all)
            df_cf['black_box'] = [black_box] * len(cf_list_all)
            try:
                if not os.path.exists(path_results):
                    #os.makedirs(path_cf)
                    print("Directory '%s' created successfully" % path_results)
            except OSError as error:
                print("Directory '%s' can not be created" % path_results)
            if optimization != 'baseline':
                filename_cf = path_results + 'cf_%s_%s_dice_%s_%s_%s_%s.csv' % (
                dataset, black_box, feature_selection, method, optimization,
                CONF['prefix_length'])
            else:
                filename_cf = path_results + 'cf_%s_%s_dice_%s_%s_%s.csv' % (dataset, black_box, feature_selection, method,
                                                                        CONF['prefix_length'])
            if not os.path.isfile(filename_cf):
                df_cf.to_csv(filename_cf, index=False)
            else:
                df_cf.to_csv(filename_cf, mode='a', index=False, header=False)
        else:
            x_eval['sat_score'] = 0
        result_dataframe = pd.DataFrame(data=x_eval_list)
        result_dataframe = result_dataframe[columns]
        if not os.path.isfile(filename_results):
            result_dataframe.to_csv(filename_results, index=False)
        else:
            result_dataframe.to_csv(filename_results, mode='a', index=False, header=False)


def dice_model(predictive_model):
    if predictive_model.model_type is ClassificationMethods.RANDOM_FOREST.value:
        m = dice_ml.Model(model=predictive_model.model, backend='sklearn')
    elif predictive_model.model_type is ClassificationMethods.PERCEPTRON.value:
        m = dice_ml.Model(model=predictive_model.model, backend='sklearn')
    elif predictive_model.model_type is ClassificationMethods.MLP.value:
        m = dice_ml.Model(model=predictive_model.model, backend='sklearn')
    elif predictive_model.model_type is ClassificationMethods.XGBOOST.value:
        m = dice_ml.Model(model=predictive_model.model, backend='sklearn')
    else:
        m = dice_ml.Model(model=predictive_model.model, backend='TF2')
    return m


def split_features(df, encoder):
    # This function splits the features of the DataFrame into categorical and continuous features.
    categorical_features = [col for col in df.columns if col in list(encoder._label_dict.keys())]
    cat_feature_index = [df.columns.get_loc(c) for c in categorical_features if c in df]
    continuous_features = [col for col in df.columns if col in list(encoder._numeric_encoder.keys())]
    cont_feature_index = [df.columns.get_loc(c) for c in continuous_features if c in df]
    return categorical_features, continuous_features, cat_feature_index, cont_feature_index


def evaluate_cf_list(cf_list, query_instance, cont_feature_index, cat_feature_index, df, y_pred, nr_of_cfs,
                     predictive_model, query_instances, continuous_features, categorical_features, ratio_cont):
    # This function evaluates the counterfactuals generated by the DICE framework.
    nbr_features = query_instance.shape[1]
    if cf_list.size > 4:
        nbr_cf_ = len(cf_list)
        nbr_features = cf_list.shape[1]
        plausibility_sum = plausibility(query_instance, predictive_model, cf_list, nr_of_cfs, query_instances, y_pred,
                                        cont_feature_index, cat_feature_index, df, ratio_cont
                                        )
        plausibility_max_nbr_cf_ = plausibility_sum / nr_of_cfs
        plausibility_nbr_cf_ = plausibility_sum / nbr_cf_
        distance_l2_ = continuous_distance(query_instance, cf_list, cont_feature_index, metric='euclidean', X=df)
        distance_mad_ = continuous_distance(query_instance, cf_list, cont_feature_index, metric='mad', X=df)
        distance_j_ = categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard')
        distance_h_ = categorical_distance(query_instance, cf_list, cat_feature_index, metric='hamming')
        distance_l2j_ = distance_l2j(query_instance, cf_list, cont_feature_index, cat_feature_index)
        distance_l1j_ = distance_l1j(query_instance, cf_list, cont_feature_index, cat_feature_index)
        distance_mh_ = distance_mh(query_instance, cf_list, cont_feature_index, cat_feature_index, df)

        distance_l2_min_ = continuous_distance(query_instance, cf_list, cont_feature_index, metric='euclidean', X=df,
                                               agg='min')
        distance_mad_min_ = continuous_distance(query_instance, cf_list, cont_feature_index, metric='mad', X=df,
                                                agg='min')
        distance_j_min_ = categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard', agg='min')
        distance_h_min_ = categorical_distance(query_instance, cf_list, cat_feature_index, metric='hamming', agg='min')
        distance_l2j_min_ = distance_l2j(query_instance, cf_list, cont_feature_index, cat_feature_index,
                                         agg='min')
        distance_l1j_min_ = distance_l1j(query_instance, cf_list, cont_feature_index, cat_feature_index,
                                         agg='min')
        distance_mh_min_ = distance_mh(query_instance, cf_list, cont_feature_index, cat_feature_index, df, agg='min')

        distance_l2_max_ = continuous_distance(query_instance, cf_list, cont_feature_index, metric='euclidean', X=df,
                                               agg='max')
        distance_mad_max_ = continuous_distance(query_instance, cf_list, cont_feature_index, metric='mad', X=df,
                                                agg='max')
        distance_j_max_ = categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard', agg='max')
        distance_h_max_ = categorical_distance(query_instance, cf_list, cat_feature_index, metric='hamming', agg='max')
        distance_l2j_max_ = distance_l2j(query_instance, cf_list, cont_feature_index, cat_feature_index, agg='max')
        distance_l1j_max_ = distance_l1j(query_instance, cf_list, cont_feature_index, cat_feature_index, agg='max')

        distance_mh_max_ = distance_mh(query_instance, cf_list, cont_feature_index, cat_feature_index, X=df, agg='max')
        avg_nbr_changes_per_cf_ = avg_nbr_changes_per_cf(query_instance, cf_list, continuous_features)
        avg_nbr_changes_ = avg_nbr_changes(query_instance, cf_list, nbr_features, continuous_features)
        if len(cf_list) > 1:
            diversity_l2_ = continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=df)
            diversity_mad_ = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=df)
            diversity_j_ = categorical_diversity(cf_list, cat_feature_index, metric='jaccard')
            diversity_h_ = categorical_diversity(cf_list, cat_feature_index, metric='hamming')
            diversity_l2j_ = diversity_l2j(cf_list, cont_feature_index, cat_feature_index)
            diversity_mh_ = diversity_mh(cf_list, cont_feature_index, cat_feature_index, df)

            diversity_l2_min_ = continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=df, agg='min')
            diversity_mad_min_ = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=df, agg='min')
            diversity_j_min_ = categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg='min')
            diversity_h_min_ = categorical_diversity(cf_list, cat_feature_index, metric='hamming', agg='min')
            diversity_l2j_min_ = diversity_l2j(cf_list, cont_feature_index, cat_feature_index, agg='min')
            diversity_mh_min_ = diversity_mh(cf_list, cont_feature_index, cat_feature_index, df, agg='min')

            diversity_l2_max_ = continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=None, agg='max')
            diversity_mad_max_ = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=df, agg='max')
            diversity_j_max_ = categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg='max')
            diversity_h_max_ = categorical_diversity(cf_list, cat_feature_index, metric='hamming', agg='max')
            diversity_l2j_max_ = diversity_l2j(cf_list, cont_feature_index, cat_feature_index, agg='max')
            diversity_mh_max_ = diversity_mh(cf_list, cont_feature_index, cat_feature_index, df, agg='max')

        else:
            diversity_l2_ = 0.0
            diversity_mad_ = 0.0
            diversity_j_ = 0.0
            diversity_h_ = 0.0
            diversity_l2j_ = 0.0
            diversity_mh_ = 0.0

            diversity_l2_min_ = 0.0
            diversity_mad_min_ = 0.0
            diversity_j_min_ = 0.0
            diversity_h_min_ = 0.0
            diversity_l2j_min_ = 0.0
            diversity_mh_min_ = 0.0

            diversity_l2_max_ = 0.0
            diversity_mad_max_ = 0.0
            diversity_j_max_ = 0.0
            diversity_h_max_ = 0.0
            diversity_l2j_max_ = 0.0
            diversity_mh_max_ = 0.0

        count_diversity_cont_ = count_diversity(cf_list, cont_feature_index, nbr_features, cont_feature_index)
        count_diversity_cate_ = count_diversity(cf_list, cat_feature_index, nbr_features, cont_feature_index)
        count_diversity_all_ = count_diversity_all(cf_list, nbr_features, cont_feature_index)
        res = {'generated_cfs': nr_of_cfs,
               'implausibility_sum': plausibility_sum,
               'implausibility_max_nbr_cf': plausibility_max_nbr_cf_,
               'implausibility_nbr_cf': plausibility_nbr_cf_,
               'distance_l2': distance_l2_,
               'distance_mad': distance_mad_,
               'distance_j': distance_j_,
               'distance_h': distance_h_,
               'distance_l2j': distance_l2j_,
               'distance_l1j': distance_l1j_,
               'distance_mh': distance_mh_,

               'distance_l2_min': distance_l2_min_,
               'distance_mad_min': distance_mad_min_,
               'distance_j_min': distance_j_min_,
               'distance_h_min': distance_h_min_,
               'distance_l2j_min': distance_l2j_min_,
               'distance_l1j_min': distance_l1j_min_,
               'distance_mh_min': distance_mh_min_,

               'distance_l2_max': distance_l2_max_,
               'distance_mad_max': distance_mad_max_,
               'distance_j_max': distance_j_max_,
               'distance_h_max': distance_h_max_,
               'distance_l2j_max': distance_l2j_max_,
               'distance_l1j_max': distance_l1j_max_,
               'distance_mh_max': distance_mh_max_,

               'diversity_l2': diversity_l2_,
               'diversity_mad': diversity_mad_,
               'diversity_j': diversity_j_,
               'diversity_h': diversity_h_,
               'diversity_l2j': diversity_l2j_,
               'diversity_mh': diversity_mh_,

               'diversity_l2_min': diversity_l2_min_,
               'diversity_mad_min': diversity_mad_min_,
               'diversity_j_min': diversity_j_min_,
               'diversity_h_min': diversity_h_min_,
               'diversity_l2j_min': diversity_l2j_min_,
               'diversity_mh_min': diversity_mh_min_,

               'diversity_l2_max': diversity_l2_max_,
               'diversity_mad_max': diversity_mad_max_,
               'diversity_j_max': diversity_j_max_,
               'diversity_h_max': diversity_h_max_,
               'diversity_l2j_max': diversity_l2j_max_,
               'diversity_mh_max': diversity_mh_max_,

               'count_diversity_cont': count_diversity_cont_,
               'count_diversity_cate': count_diversity_cate_,
               'count_diversity_all': count_diversity_all_,
               'avg_nbr_changes_per_cf': avg_nbr_changes_per_cf_,
               'avg_nbr_changes': avg_nbr_changes_}
    else:
        res = {
            'generated_cfs': 0,
            'distance_l2': np.nan,
            'distance_mad': np.nan,
            'distance_j': np.nan,
            'distance_h': np.nan,
            'distance_l2j': np.nan,
            'distance_l1j': np.nan,
            'distance_mh': np.nan,
            'distance_l2_min': np.nan,
            'distance_mad_min': np.nan,
            'distance_j_min': np.nan,
            'distance_h_min': np.nan,
            'distance_l2j_min': np.nan,
            'distance_l1j_min': np.nan,
            'distance_mh_min': np.nan,
            'distance_l2_max': np.nan,
            'distance_mad_max': np.nan,
            'distance_j_max': np.nan,
            'distance_h_max': np.nan,
            'distance_l2j_max': np.nan,
            'distance_l1j_max': np.nan,
            'distance_mh_max': np.nan,
            'avg_nbr_changes_per_cf': np.nan,
            'avg_nbr_changes': np.nan,
            'diversity_l2': np.nan,
            'diversity_mad': np.nan,
            'diversity_j': np.nan,
            'diversity_h': np.nan,
            'diversity_l2j': np.nan,
            'diversity_mh': np.nan,
            'diversity_l2_min': np.nan,
            'diversity_mad_min': np.nan,
            'diversity_j_min': np.nan,
            'diversity_h_min': np.nan,
            'diversity_l2j_min': np.nan,
            'diversity_mh_min': np.nan,
            'diversity_l2_max': np.nan,
            'diversity_mad_max': np.nan,
            'diversity_j_max': np.nan,
            'diversity_h_max': np.nan,
            'diversity_l2j_max': np.nan,
            'diversity_mh_max': np.nan,
            'count_diversity_cont': np.nan,
            'count_diversity_cate': np.nan,
            'count_diversity_all': np.nan,

            'implausibility_sum': 0.0,
            'implausibility_max_nbr_cf': 0.0,
            'implausibility_nbr_cf': 0.0,
            'sat_score': 0.0
        }
    return res


def continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=None, agg=None):
    if metric == 'mad':
        mad = median_abs_deviation(X.iloc[:, cont_feature_index], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)

        dist = pdist(cf_list[:, cont_feature_index], metric=_mad_cityblock)
    else:
        dist = pdist(cf_list[:, cont_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg=None):
    dist = pdist(cf_list[:, cat_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def diversity_mh(cf_list, cont_feature_index, cat_feature_index, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=X, agg=agg)
    dist_cate = categorical_diversity(cf_list, cat_feature_index, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def count_diversity(cf_list, features, nbr_features, cont_feature_index):
    nbr_cf = cf_list.shape[0]
    nbr_changes = 0
    for i in range(nbr_cf):
        for j in range(i + 1, nbr_cf):
            for k in features:
                if cf_list[i][k] != cf_list[j][k]:
                    nbr_changes += 1 if j in cont_feature_index else 0.5
    return nbr_changes / (nbr_cf * nbr_cf * nbr_features)


# piu alto e' meglio conta variet' tra cf
def count_diversity_all(cf_list, nbr_features, cont_feature_index):
    return count_diversity(cf_list, range(cf_list.shape[1]), nbr_features, cont_feature_index)


def continuous_distance(query_instance, cf_list, cont_feature_index, metric='euclidean', X=None, agg=None):
    if metric == 'mad':
        mad = median_abs_deviation(X.iloc[:, cont_feature_index], axis=0)
        mad = np.array([v if v != 0 else 1.0 for v in mad])

        def _mad_cityblock(u, v):
            return mad_cityblock(u, v, mad)

        dist = cdist(query_instance[:, cont_feature_index], cf_list[:, cont_feature_index], metric=_mad_cityblock)
    else:
        dist = cdist(query_instance[:, cont_feature_index], cf_list[:, cont_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def mad_cityblock(u, v, mad):
    u = _validate_vector(u)
    v = _validate_vector(v)
    l1_diff = abs(u - v)
    l1_diff_mad = l1_diff / mad
    return l1_diff_mad.sum()


def categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard', agg=None):
    try:
        dist = cdist(query_instance.reshape(1, -1)[:, cat_feature_index], cf_list[:, cat_feature_index], metric=metric)
    except:
        print('Problem with categorical distance')
    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def euclidean_jaccard(query_instance, A, cont_feature_index, cat_feature_index, ratio_cont=None):
    nbr_features = A.shape[1]
    dist_cont = cdist(query_instance.reshape(1, -1)[:, cont_feature_index], A[:, cont_feature_index],
                      metric='euclidean')
    dist_cate = cdist(query_instance.reshape(1, -1)[:, cat_feature_index], A[:, cat_feature_index], metric='jaccard')
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def distance_l2j(query_instance, cf_list, cont_feature_index, cat_feature_index, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(query_instance, cf_list, cont_feature_index, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def distance_l1j(query_instance, cf_list, cont_feature_index, cat_feature_index, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(query_instance, cf_list, cont_feature_index, metric='cityblock', X=None, agg=agg)
    dist_cate = categorical_distance(query_instance, cf_list, cat_feature_index, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def distance_mh(query_instance, cf_list, cont_feature_index, cat_feature_index, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_distance(query_instance, cf_list, cont_feature_index, metric='mad', X=X, agg=agg)
    dist_cate = categorical_distance(query_instance, cf_list, cat_feature_index, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg=None):
    dist = pdist(cf_list[:, cat_feature_index], metric=metric)

    if agg is None or agg == 'mean':
        return np.mean(dist)

    if agg == 'max':
        return np.max(dist)

    if agg == 'min':
        return np.min(dist)


def diversity_l2j(cf_list, cont_feature_index, cat_feature_index, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, cont_feature_index, metric='euclidean', X=None, agg=agg)
    dist_cate = categorical_diversity(cf_list, cat_feature_index, metric='jaccard', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def diversity_mh(cf_list, cont_feature_index, cat_feature_index, X, ratio_cont=None, agg=None):
    nbr_features = cf_list.shape[1]
    dist_cont = continuous_diversity(cf_list, cont_feature_index, metric='mad', X=X, agg=agg)
    dist_cate = categorical_diversity(cf_list, cat_feature_index, metric='hamming', agg=agg)
    if ratio_cont is None:
        ratio_continuous = len(cont_feature_index) / nbr_features
        ratio_categorical = len(cat_feature_index) / nbr_features
    else:
        ratio_continuous = ratio_cont
        ratio_categorical = 1.0 - ratio_cont
    dist = ratio_continuous * dist_cont + ratio_categorical * dist_cate
    return dist


def nbr_changes_per_cf(x, cf_list, continuous_features):
    x = x.ravel()
    nbr_features = cf_list.shape[1] - 1  # exclude label
    nbr_changes = np.zeros(len(cf_list))
    for i, cf in enumerate(cf_list):
        cf = cf[:-1]
        for j in range(nbr_features):
            if cf[j] != x[j]:
                nbr_changes[i] += 1 if j in continuous_features else 0.5
    return nbr_changes


def avg_nbr_changes_per_cf(x, cf_list, continuous_features):
    return np.mean(nbr_changes_per_cf(x, cf_list, continuous_features))


def avg_nbr_changes(x, cf_list, nbr_features, continuous_features):
    val = np.sum(nbr_changes_per_cf(x, cf_list, continuous_features))
    nbr_cf, _ = cf_list.shape
    return val / (nbr_cf * nbr_features)


def plausibility(query_instance, predictive_model, cf_list, nr_of_cfs, query_instances,
                 y_pred, continuous_features, categorical_features, df, ratio_cont):
    sum_dist = 0.0
    full_df = pd.concat([query_instances, df], ignore_index=False)
    for cf in cf_list:
        # X_y = full_df[full_df['label'] == y_label]
        X_y = full_df
        # neigh_dist = exp.cdist(x.reshape(1, -1), X_test_y)
        neigh_dist = distance_mh(query_instance.reshape(1, -1), X_y.to_numpy(), continuous_features,
                                 categorical_features, df, ratio_cont)
        idx_neigh = np.argsort(neigh_dist)[0]
        # closest_idx = closest_idx = idx_neigh[0]
        # closest = X_test_y[closest_idx]
        closest = X_y.to_numpy()[idx_neigh]
        d = distance_mh(cf.reshape(1, -1), closest.reshape(1, -1), continuous_features,
                        categorical_features, df, ratio_cont)
        sum_dist += d
    return sum_dist


def extract_matching_substrings(input_string, alphabet):
    # Create a regex pattern by joining substrings with the | character
    model_dict = {}
    for i in range(len(alphabet)):
        # print(alphabet[i
        #      ])
        match = re.findall(alphabet[i].lower().replace(" ", "").replace("\\", ""), input_string)
        if len(match):
            print(alphabet[i], match)
            model_dict[match[0]] = alphabet[i]
    # Find all occurrences of the pattern in the input string
    #matches = re.findall(pattern, input_string)

    return model_dict

def conformance_score(encoder, df, features_names, ltlf_model, dfa):
    # This function calculates the conformance score of the counterfactuals generated by the DICE framework using the LTLf model.
    population_df = pd.DataFrame(df, columns=features_names)
    encoder.decode(population_df)
    population_df.insert(loc=0, column='Case ID', value=np.divmod(np.arange(len(population_df)), 1)[0] + 1)
    population_df.insert(loc=1, column='label', value=1)
    long_data = pd.wide_to_long(population_df, stubnames=['prefix'], i='Case ID',
                                j='order', sep='_', suffix=r'\w+')
    timestamps = pd.date_range('1/1/2011', periods=len(long_data), freq='H')
    long_data_sorted = long_data.sort_values(['Case ID', 'order'], ).reset_index(drop=False)
    long_data_sorted['time:timestamp'] = timestamps
    long_data_sorted['label'].replace({1: 'regular'}, inplace=True)
    long_data_sorted.drop(columns=['order'], inplace=True)
    columns_to_rename = {'Case ID': 'case:concept:name', 'prefix': 'concept:name'}
    long_data_sorted.rename(columns=columns_to_rename, inplace=True)
    long_data_sorted['label'].replace({'regular': 'false', 'deviant': 'true'}, inplace=True)
    long_data_sorted.replace('0', 'other', inplace=True)
    dataframe = format_dataframe(long_data_sorted, case_id='case:concept:name', activity_key='concept:name',
                                       timestamp_key='time:timestamp')
    log = convert_to_event_log(dataframe)
    event_log = D4PyEventLog()
    event_log.load_xes_log(log)
    analyzer = LTLAnalyzer(event_log, ltlf_model)
    conf_check_res_df = analyzer.run(
        #jobs=4,
        dfa=dfa
    )
    conformance_score = conf_check_res_df['accepted'].replace({True: 1, False: 0})
    population_conformance = conf_check_res_df[conf_check_res_df['accepted'] == True].shape[0] / conf_check_res_df.shape[0]
    print('Conformance score:', population_conformance)
    return population_conformance

# Example usage
def extract_names(text):
    # Use regular expression to find text within parentheses
    pattern = r'\((.*?)\)'
    matches = re.findall(pattern, text)

    # Remove spaces and letters from each match
    cleaned_names = [re.sub(r'[^a-zA-Z]', '', match).lstrip('F').lower() for match in matches]

    return cleaned_names

columns = ['dataset', 'heuristic', 'model', 'method', 'optimization', 'prefix_length', 'idx', 'desired_nr_of_cfs',
           'generated_cfs', 'time_train', 'time_test',
           'runtime', 'distance_l2', 'distance_mad', 'distance_j', 'distance_h', 'distance_l1j', 'distance_l2j',
           'distance_mh',
           'distance_l2_min', 'distance_mad_min', 'distance_j_min', 'distance_h_min', 'distance_l1j_min',
           'distance_l2j_min',
           'distance_mh_min', 'distance_l2_max', 'distance_mad_max', 'distance_j_max', 'distance_h_max',
           'distance_l1j_max', 'distance_l2j_max', 'distance_mh_max', 'diversity_l2',
           'diversity_mad', 'diversity_j', 'diversity_h', 'diversity_l2j', 'diversity_mh', 'diversity_l2_min',
           'diversity_mad_min', 'diversity_j_min', 'diversity_h_min', 'diversity_l2j_min', 'diversity_mh_min',
           'diversity_l2_max', 'diversity_mad_max', 'diversity_j_max', 'diversity_h_max', 'diversity_l2j_max',
           'diversity_mh_max', 'count_diversity_cont', 'count_diversity_cate', 'count_diversity_all',
           'avg_nbr_changes_per_cf', 'avg_nbr_changes', 'implausibility_sum',
           'implausibility_max_nbr_cf', 'implausibility_nbr_cf', 'sat_score']
