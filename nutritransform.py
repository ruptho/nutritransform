import itertools
import numbers
import re

import pandas as pd
import numpy as np
import json
from collections import defaultdict
from sentence_transformers.util import cos_sim
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import train_test_split, KFold
import sklearn.metrics as skmetrics

USDA_COLUMNS = ['Energy', 'Water', 'Carbohydrate, by difference', 'Protein', 'Total lipid (fat)',
                'Fiber, total dietary', 'Sugars, total including NLEA', 'Cholesterol', 'Alcohol, ethyl',
                'Caffeine', 'Vitamin C, total ascorbic acid', 'Vitamin D (D2 + D3)']
FOODCOM_COLUMNS = ['kcal_100g', 'carb_100g', 'fat_100g', 'prot_100g', 'sodium_100g', 'chol_100g']
model = None
KEYWORD_POSTFIX = ['with', 'without', 'from', 'stuffed with', 'trimmed', 'made']
RE_POSTFIX = re.compile(rf"^({'|'.join(KEYWORD_POSTFIX)}).*?")

# ref from our columns -> groundtruth foodcom columnsload_food_data
col_dict = {'Energy': 'kcal_100g',
            'Carbohydrate, by difference': 'carb_100g',
            'Total lipid (fat)': 'fat_100g',
            'Protein': 'prot_100g',
            'Cholesterol': 'chol_100g'}
col_dict_cj = {'Energy': 'calories',
               'Carbohydrate, by difference': 'carbohydrates_total_g',
               'Total lipid (fat)': 'fat_total_g',
               'Protein': 'protein_g',
               'Cholesterol': 'cholesterol_mg'}


def load_all_food_submissions():
    df_subs = pd.read_json('../../data/RedditCrawl/202204/wellbeing/submissions/food/food_submissions.txt', lines=True)
    return df_subs[df_subs.subreddit == 'food']


def filter_relevant_data(df_subs_food, date_column='created_utc'):
    # without deleted authors
    reg_str = '\[ *(i ?(ate|eat)|home[ -]?made|pro[/ ]chef) *\]'

    df_subs_posts = df_subs_food.copy()
    print('All posts ever', len(df_subs_posts))

    df_subs_posts['date_utc'] = pd.to_datetime(df_subs_posts[date_column], unit='s')
    df_subs_posts['date_day'] = df_subs_posts['date_utc'].dt.date
    df_subs_posts = df_subs_posts[(df_subs_posts.date_utc >= pd.to_datetime('2017-01-01')) & (
            df_subs_posts.date_utc <= pd.to_datetime('2021-01-01'))]

    print('>2017', len(df_subs_posts))
    df_subs_posts_norem = df_subs_posts[~(
            df_subs_posts.title.isin(['[removed]', '[deleted]', '[deleted by user]']) | pd.isna(
        df_subs_posts.title) | df_subs_posts.author.str.lower().isin(['[automoderator]']))].copy()
    print('Removed and deleted posts', len(df_subs_posts_norem))

    df_subs_posts_norem_nodup = df_subs_posts_norem.drop_duplicates(subset=['author', 'title', 'date_day'], keep='last')
    print('Remove duplicates', len(df_subs_posts_norem_nodup))

    df_sub_regex_posts = df_subs_posts_norem_nodup[
        df_subs_posts_norem_nodup.title.str.lower().str.contains(reg_str, regex=True)].copy()
    df_sub_nonregex_posts = df_subs_posts_norem_nodup[
        ~df_subs_posts_norem_nodup.title.str.lower().str.contains(reg_str, regex=True)].copy()

    print('Matched Regex', len(df_sub_regex_posts))

    df_sub_regex_posts['clean_title'] = df_sub_regex_posts.title.str.lower().str.replace('&amp;', ' and ', regex=False)
    # first remove tags, than remove weird signs at start and end, then remove apostropohes,
    # then in the end clear out all the double whitespaces resulting from that
    df_sub_regex_posts['clean_title'] = \
        df_sub_regex_posts.clean_title.str.replace(r'\[.+?\]', ' ', regex=True) \
            .str.strip(',.|-;:!?()/\\\n\t\'" ') \
            .str.replace('"', ' ', regex=False) \
            .str.replace('( )+', ' ', regex=True).str.strip()

    df_filtered = df_sub_regex_posts[df_sub_regex_posts.clean_title.str.len() >= 3]
    print('Remaining after stripping empty', len(df_filtered))

    df_filtered_nodel = df_filtered[
        ~pd.isna(df_filtered.author) & ~df_filtered.author.isin(['[removed]', '[deleted]', '[AutoModerator]'])]
    print('Remaining after removing deleted users', len(df_filtered_nodel))
    return df_filtered_nodel


def flatten(l):
    return [item for sublist in l for item in sublist]


def load_model(model_name='all-mpnet-base-v2'):
    global model
    model = SentenceTransformer(model_name)


def generate_embedding_dict(titles, model_name='all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    embedded_db = model.encode(titles)
    embedded_db_dict = dict(zip(titles, embedded_db))
    return {title: vector for title, vector in embedded_db_dict.items() if vector.any()}


def extract_food_data(path, food_db):
    data_struct = []
    with open(path, 'r') as myfile:
        data = json.load(myfile)[food_db]

    units = defaultdict(set)
    for food_item in data:
        food_data = {}
        food_data['name'] = food_item['description']

        food_data['parent_name'] = food_item['wweiaFoodCategory'][
            'wweiaFoodCategoryDescription'] if food_db == 'SurveyFoods' else food_data['name']
        for nutrient_info in food_item['foodNutrients']:
            if 'amount' in nutrient_info:
                if nutrient_info['nutrient']['name'] == 'Energy':
                    if nutrient_info['nutrient']['unitName'] == 'kcal':
                        food_data[nutrient_info['nutrient']['name']] = nutrient_info['amount']
                        units[nutrient_info['nutrient']['name']].add(nutrient_info['nutrient']['unitName'])

                else:
                    food_data[nutrient_info['nutrient']['name']] = nutrient_info['amount']
                    units[nutrient_info['nutrient']['name']].add(nutrient_info['nutrient']['unitName'])
            else:
                # print(nutrient_info)
                pass  # No info available
        data_struct.append(food_data)
    return pd.DataFrame(data_struct), units


def load_food_data(filter_uncooked=True):
    foundation_data, foundation_units = extract_food_data(
        'food_data/FoodData_Central_foundation_food_json_2022-10-28.json', food_db='FoundationFoods')
    foundation_data['data_source'] = 'foundation'
    survey_data, survey_units = extract_food_data('food_data/FoodData_Central_survey_food_json_2022-10-28.json',
                                                  food_db='SurveyFoods')
    survey_data['data_source'] = 'survey22'
    # survey_21_data, survey_21_units = extract_food_data('food_data/FoodData_Central_survey_food_json_2021-10-28.json',
    # food_db='SurveyFoods')
    # survey_21_data['data_source'] = 'survey21'
    legacy_data, legacy_units = extract_food_data('food_data/FoodData_Central_sr_legacy_food_json_2021-10-28.json',
                                                  food_db='SRLegacyFoods')
    legacy_data['data_source'] = 'legacy'
    food_data_all = pd.concat([foundation_data, survey_data, legacy_data])
    food_data_all['bow'] = food_data_all['name'].str.split(', ')
    food_data_all = food_data_all.drop_duplicates(subset='name')
    food_data_all['title_simple'] = food_data_all.name.str.lower().str.split(', ').apply(lambda l: ' '.join(l))

    def join_with_append(tokens):
        tokens_normal, append_tokens = list(reversed(tokens)), []
        for t in tokens:
            if RE_POSTFIX.match(t) is not None:
                append_tokens.append(t)
                tokens_normal.remove(t)
        return ' '.join(tokens_normal + append_tokens).strip()

    food_data_all['title_simple_reversed'] = food_data_all.name.str.lower().str.split(', ').apply(join_with_append)
    food_data_all = food_data_all[~pd.isna(food_data_all.Energy)]
    # food_data_cooked = food_data_cooked.groupby(['title_simple_reversed', 'title_simple', 'name']).agg()
    return food_data_all if not filter_uncooked else \
        food_data_all[~food_data_all.name.str.contains(' raw| uncooked| unheated')]


def calc_errors(df_err, col, orig_col='kcal_100g', print_out=True, return_err=False):
    df_err[f'diff_{col}'] = df_err[orig_col] - df_err[col]
    df_err[f'diff_abs_{col}'] = np.abs(df_err[f'diff_{col}'])
    df_err[f'diff_squ_{col}'] = df_err[f'diff_{col}'] ** 2
    if print_out:
        print(f'MedAE for {orig_col} and {col}', np.nanmedian(df_err[f'diff_abs_{col}']))
        print(f'MSE for {orig_col} and {col}', np.nanmean(df_err[f'diff_squ_{col}']))
    return {'all_errors': df_err, 'mae': np.nanmedian(df_err[f'diff_abs_{col}']),
            'mse': np.nanmean(df_err[f'diff_squ_{col}'])} if return_err else None


def merge_and_calc_errors(df_full, df_groundtruth, df_cj, n, thresh, metric, col_dict=col_dict,
                          thresh_minmax=(30, 900), agg_columns=None):
    df_nutres_thresh = nutrition_df_apply_thresholds(df_full, n=n, sim_thresh=thresh)
    df_nutres_metrics = compute_metric_df(df_nutres_thresh, metric, agg_columns)
    df = create_comparison_df(df_groundtruth, df_cj, df_nutres_metrics, threshold=thresh_minmax)
    error_res = [[thresh, n, metric.__name__]]
    error_res_cj = [['-', '-', 'cj']]
    for c, gt_c in col_dict.items():
        dict_err = calc_errors(df, c, orig_col=gt_c, print_out=False, return_err=True)
        error_res[-1].extend([dict_err['mae'], dict_err['mse']])

        dict_err_cj = calc_errors(df, col_dict_cj[c], orig_col=gt_c, print_out=False, return_err=True)
        error_res_cj[-1].extend([dict_err_cj['mae'], dict_err_cj['mse']])

    return df, pd.DataFrame(error_res + error_res_cj, columns=['sim_thresh', 'n', 'metric'] +
                                                              flatten([(f'mae_{c}', f'mse_{c}') for c in col_dict]))


def plot_hist(df, col_comp=['kcal_100g', 'Energy', 'calories'], bins=100):
    import seaborn as sns
    all_cols = []
    df_plot = df[col_comp].dropna()
    for col in col_comp:
        df_part = df_plot[[col]].copy()
        df_part['value'] = df_plot[col]
        df_part['label'] = col
        all_cols.append(df_part[['value', 'label']])
    df_cols_combed = pd.concat(all_cols).reset_index(drop=True)
    sns.histplot(data=df_cols_combed, x='value', hue='label', bins=bins)


def grid_search_n_thresh(df_full, df_groundtruth, df_cj, n_l, thresh_l, metric_l, col_dict=col_dict):
    error_res = []
    for thresh in thresh_l:
        for n in n_l:
            for metric in metric_l:
                df_nutres_thresh = nutrition_df_apply_thresholds(df_full, n=n, sim_thresh=thresh)
                df_nutres_metrics = compute_metric_df(df_nutres_thresh, metric)
                df = create_comparison_df(df_groundtruth, df_cj, df_nutres_metrics)
                error_res.append([thresh, n, metric.__name__])
                for c, gt_c in col_dict.items():
                    dict_err = calc_errors(df, c, orig_col=gt_c, print_out=False, return_err=True)
                    error_res[-1].extend([dict_err['mae'], dict_err['mse']])
    return pd.DataFrame(error_res, columns=['sim_thresh', 'n', 'metric'] +
                                           flatten([(f'mae_{c}', f'mse_{c}') for c in col_dict]))


def create_comparison_df(df_groundtruth, df_calninja, df_retrieved, threshold=(30, 900)):
    df_mrgd = df_groundtruth.merge(df_retrieved, left_on='clean_title', right_on='match_title', how='left').merge(
        df_calninja[['clean_title', 'calories', 'carbohydrates_total_g', 'protein_g', 'fat_total_g',
                     'sugar_g', 'cholesterol_mg']].drop_duplicates(), on='clean_title', how='left').copy()
    return df_mrgd[(df_mrgd.kcal_100g < threshold[1]) & (df_mrgd.kcal_100g > threshold[0])]


def generate_dict_matches(texts, embedded_db_loc, model_name='all-mpnet-base-v2'):
    model = SentenceTransformer(model_name)
    span_vectors = model.encode(texts)
    db_np_vectors = np.asarray(list(embedded_db_loc.values()))
    similarities = cos_sim(span_vectors, db_np_vectors).numpy()
    matched_dict = dict(zip(texts, similarities))  # first helper dict
    matched_dict = {key: dict(zip(list(embedded_db_loc.keys()), val)) for key, val in matched_dict.items()}
    return matched_dict


def retrieve_nutrition(texts, df_food, embedded_db_loc, n=5, title_col='title_simple', model_name='all-mpnet-base-v2'):
    # print(span)
    matched_dict = generate_dict_matches(texts, embedded_db_loc, model_name)

    # print(matched_dict)
    result_dfs = []
    for text, similarities in matched_dict.items():
        df_match = pd.DataFrame.from_dict(similarities, orient='index').reset_index()
        df_match.columns = [title_col, 'similarity']
        df_match = df_match.nlargest(n, 'similarity')
        # print(df_match)
        df_food_match = df_food.merge(df_match, on=title_col, how='right')
        df_food_match['match_title'] = text
        result_dfs.append(df_food_match)
    return pd.concat(result_dfs)


'''
def transform_nut_dict_to_df(nut_dict, columns=['Energy', 'Protein'], prefix=''):
    df_res = pd.concat([pd.Series(title, index=['clean_title']).append(
        nuts['nutrition'][columns]) if nuts and title and (nuts['nutrition'] is not None) else pd.Series(
        [None, None], index=columns) for title, nuts in nut_dict.items()], axis=1).transpose()
    df_res.columns = ['clean_title'] + [f'{prefix}{col}' for col in columns]

    return df_res
'''


def nutrition_df_apply_thresholds(df_nutres, n=100, sim_thresh=0):
    df_threshold = df_nutres[df_nutres.similarity > sim_thresh]
    return df_threshold[df_threshold.groupby('match_title')['similarity'].rank(ascending=False) <= n]


def compute_metric_df(df_nutres, metric, columns=USDA_COLUMNS):
    return df_nutres.groupby('match_title')[columns].agg(metric).reset_index()


def compute_metric_vals_for_dict(mean_dict, columns=None, metric='mean'):
    median_dict = mean_dict.copy()
    for title in mean_dict:
        if median_dict[title]:
            median_dict[title]['nutrition'] = mean_dict[title]['matches'][
                USDA_COLUMNS if not columns else USDA_COLUMNS].agg(metric)
    return median_dict


def cross_val_dataset(df_food, db_emb, test_size=0.2, n_tries=10, n_folds=10, n_sim=50,
                      title_col='title_simple_reversed', model_name='all-mpnet-base-v2', agg_metric=np.mean,
                      columns=USDA_COLUMNS):
    df_full_train, df_test = train_test_split(
        df_food.groupby(title_col).mean().reset_index().sample(frac=1), test_size=test_size)
    return cross_val(df_full_train, df_test, db_emb, n_tries, n_folds, test_size, n_sim,
                     title_col, model_name, agg_metric, columns)


def cross_val_goldstandard(df_groundtruth, df_food, db_food_emb, test_size=0.2, n_tries=10, n_folds=10, n_sim=50,
                           food_name_col='clean_title', title_col='title_simple_reversed',
                           model_name='all-mpnet-base-v2', agg_metric=np.mean, columns=FOODCOM_COLUMNS):
    df_full_train, df_test = train_test_split(
        df_groundtruth.groupby(food_name_col).mean().reset_index().sample(frac=1).rename({
            food_name_col: title_col}, axis=1), test_size=test_size)
    return cross_val(df_full_train, df_test, db_food_emb, n_tries, n_folds, test_size, n_sim,
                     title_col, model_name, agg_metric, columns)


def cross_val(df_train_and_val, df_test, db_emb, n_tries, n_folds, val_size, n_sim=50,
              title_col='title_simple_reversed', model_name='all-mpnet-base-v2', agg_metric=np.mean,
              columns=None):
    results, n_it = defaultdict(dict), 0

    for i in KFold(n_folds).split(df_train_and_val) if n_folds else range(n_tries):
        if n_folds:
            # n_fold with test set
            df_train, df_val = df_train_and_val.iloc[i[0]], df_train_and_val.iloc[i[1]]
        else:
            # random sampling, just a test set
            df_train, df_val = train_test_split(df_train_and_val, test_size=val_size)
        # Inner-Data Validation
        results[n_it]['matches'] = retrieve_nutrition(df_val[title_col].values, df_train,
                                                      {key: val for key, val in db_emb.items()
                                                       if key in df_train[title_col].values}, n=n_sim,
                                                      title_col=title_col, model_name=model_name)
        results[n_it]['agg'] = compute_metric_df(results[n_it]['matches'], agg_metric, columns)
        n_it += 1

    results['df_train'] = df_train_and_val
    results['df_test'] = df_test
    results['df_food_db'] = df_train_and_val
    return results


def cross_val_with_db(df_groundtruth, df_food, db_food_emb, n_folds=10, val_size=.2, n_sim=50,
                      gt_title_col='clean_title', db_title_col='title_simple_reversed',
                      model_name='all-mpnet-base-v2', agg_metric=np.mean, columns=USDA_COLUMNS):
    df_train_and_val, df_test = train_test_split(
        df_groundtruth.groupby(gt_title_col).mean().reset_index().sample(frac=1).rename({
            gt_title_col: db_title_col}, axis=1), test_size=val_size)
    results, n_it = defaultdict(dict), 0
    # fold on train/val, retrieve via usda evaluate on food-com
    # for i in KFold(n_folds).split(df_train_and_val):
    #    print(len(i[0]), len(i[1]))
    # df_train, df_val = df_train_and_val.iloc[i[0]], df_train_and_val.iloc[i[1]]
    # Out-of-USDA-Data Validation
    results[0]['matches'] = retrieve_nutrition(df_train_and_val[db_title_col].values, df_food,
                                               {key: val for key, val in db_food_emb.items()}, n=n_sim,
                                               title_col=db_title_col, model_name=model_name)
    results[0]['agg'] = compute_metric_df(results[0]['matches'], agg_metric, columns)

    results['df_train'], results['df_test'], results['df_food_db'] = df_train_and_val, df_test, df_food
    return results


def mean_ape(true, pred):
    return skmetrics.mean_absolute_percentage_error(true, pred)


def median_ae(true, pred):
    # median absolute error
    return skmetrics.median_absolute_error(true, pred)


def mean_ae(true, pred):
    # mean absolute error
    return skmetrics.mean_absolute_error(true, pred)


def mean_se(true, pred):
    return skmetrics.mean_squared_error(true, pred, squared=False)


def mean_sle(true, pred):
    return skmetrics.mean_squared_log_error(true, pred)


def calc_performance_fold(vals_dict, df_groundtruth, title_col, col):
    df_reltruth = df_groundtruth[[title_col, col]].copy().rename({col: f'{col}_gt'}, axis=1)
    df_mrg = df_reltruth.merge(vals_dict['agg'][['match_title', col]], left_on=title_col, right_on='match_title',
                               how='left').dropna()
    # print(f'Fold {i}: Lost {lost_dp} datapoints')
    y_true, y_pred = df_mrg[f'{col}_gt'].values, df_mrg[col].values
    vals_dict['errors'] = {'median_absolute_error': median_ae(y_true, y_pred),
                           'mean_abs_perc_error': mean_ape(y_true, y_pred),
                           'mean_absolute_error': mean_ae(y_true, y_pred),
                           'mean_squared_error': mean_se(y_true, y_pred),
                           'mean_squared_log_error': mean_sle(y_true, y_pred),
                           'dropped_entries': len(vals_dict["agg"]) - len(df_mrg)}


def compute_cross_val_metrics(crossval_dict, df_groundtruth, title_col='title_simple_reversed', col='Energy'):
    agg_med_ae, agg_mean_ae, agg_mean_se, agg_mean_sle, agg_mean_ape, lost_dp = 0, 0, 0, 0, 0, 0

    for i, vals in crossval_dict.items():
        if not isinstance(i, numbers.Number):
            continue
        calc_performance_fold(vals, df_groundtruth, title_col, col)
        agg_med_ae, agg_mean_ae, agg_mean_se, agg_mean_sle, agg_mean_ape, lost_dp = \
            agg_med_ae + crossval_dict[i]['errors']['median_absolute_error'], \
            agg_mean_ae + crossval_dict[i]['errors']['mean_absolute_error'], \
            agg_mean_se + crossval_dict[i]['errors']['mean_squared_error'], \
            agg_mean_sle + crossval_dict[i]['errors']['mean_squared_log_error'], \
            agg_mean_ape + crossval_dict[i]['errors']['mean_abs_perc_error'], \
            lost_dp + crossval_dict[i]['errors']['dropped_entries']

    crossval_dict['errors'] = {'median_absolute_error': agg_med_ae / len(crossval_dict),
                               'mean_absolute_error': agg_mean_ae / len(crossval_dict),
                               'mean_abs_perc_error': agg_mean_ape / len(crossval_dict),
                               'mean_squared_error': agg_mean_se / len(crossval_dict),
                               'mean_squared_log_error': agg_mean_sle / len(crossval_dict),
                               'dropped_entries': lost_dp / len(crossval_dict)}
    return crossval_dict


def compute_test_metrics(crossval_dict, db_emb, title_col='title_simple_reversed',
                         col='Energy', min_error_metric='mean_squared_error', agg_metric=np.mean, n_sim=None,
                         thresh_sim=None, model_name='all-mpnet-base-v2', columns=USDA_COLUMNS):
    df_full_train, df_test, df_food_db = crossval_dict['df_train'], \
                                         crossval_dict['df_test'], \
                                         crossval_dict['df_food_db']
    if n_sim is None or thresh_sim is None:
        n_sim = crossval_dict['min_errors'][min_error_metric][1][0]
        thresh_sim = crossval_dict['min_errors'][min_error_metric][1][1]  # not used right now
        print(f'Calc test set for minimum {min_error_metric} at: n={n_sim}, thresh={thresh_sim}')

    crossval_dict['test']['matches'] = retrieve_nutrition(df_test[title_col].values, df_food_db,
                                                          {key: val for key, val in db_emb.items()
                                                           if key in df_food_db[title_col].values}, n=n_sim,
                                                          title_col=title_col, model_name=model_name)
    # apply thresholds here if applicable
    crossval_dict['test']['agg'] = compute_metric_df(crossval_dict['test']['matches'], agg_metric, columns)
    calc_performance_fold(crossval_dict['test'], df_test, title_col, col)
    print(crossval_dict['test']['errors'])
    return crossval_dict


def check_min_error(ov_dict, key, err_str):
    if ov_dict['min_errors'][err_str][0] > ov_dict[key]['errors'][err_str]:
        ov_dict['min_errors'][err_str] = (ov_dict[key]['errors'][err_str], key)


def compute_cross_val_metrics_gridsearch(crossval_dict, df_groundtruth, title_col='title_simple_reversed',
                                         gt_title_col='title_simple_reversed', col='Energy', gt_col='Energy',
                                         n_s=(1, 3, 4, 5, 7, 10, 15, 20, 25, 30, 40, 50),
                                         thresh_s=(.0, .1, .3, .5, .6, .7, .8, .85),
                                         agg_metric=np.mean, agg_columns=None):
    # we already retrieved all items for n=50 and thresh=0.0 in the previous step, now just filter it down from here.
    grid_cv_results = defaultdict(dict)
    crossval_dict['df_test'].rename({gt_col: col}, axis=1, inplace=True)
    crossval_dict['df_train'].rename({gt_col: col}, axis=1, inplace=True)
    grid_cv_results['df_train'], grid_cv_results['df_test'], grid_cv_results['df_food_db'] = \
        crossval_dict['df_train'], crossval_dict['df_test'], crossval_dict['df_food_db']
    df_gt = df_groundtruth.rename({gt_col: col, gt_title_col: title_col}, axis=1)

    columns = agg_columns if agg_columns else USDA_COLUMNS
    for n, sim_thresh in itertools.product(n_s, thresh_s):
        thresh_crossval_dict = dict(crossval_dict)
        for i, vals in thresh_crossval_dict.items():
            if not isinstance(i, numbers.Number):
                continue
            vals['thresh_matches'] = nutrition_df_apply_thresholds(vals['matches'], n=n, sim_thresh=sim_thresh)
            vals['agg'] = compute_metric_df(vals['thresh_matches'], agg_metric, columns)
        grid_cv_results[(n, sim_thresh)] = compute_cross_val_metrics(
            thresh_crossval_dict, df_gt, title_col, col)

    grid_cv_results['min_errors'] = {'median_absolute_error': (np.inf,),
                                     'mean_abs_perc_error': (np.inf,),
                                     'mean_absolute_error': (np.inf,),
                                     'mean_squared_error': (np.inf,),
                                     'mean_squared_log_error': (np.inf,),
                                     'dropped_entries': (np.inf,)}

    for key in grid_cv_results:
        if key in ['min_errors', 'df_train', 'df_test', 'df_food_db']:
            continue
        check_min_error(grid_cv_results, key, 'median_absolute_error')
        check_min_error(grid_cv_results, key, 'mean_abs_perc_error')
        check_min_error(grid_cv_results, key, 'mean_squared_error')
        check_min_error(grid_cv_results, key, 'mean_squared_log_error')
        check_min_error(grid_cv_results, key, 'mean_absolute_error')
        check_min_error(grid_cv_results, key, 'dropped_entries')

    return grid_cv_results
