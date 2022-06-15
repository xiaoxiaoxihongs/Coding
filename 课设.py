## 1.导入第三方包
import pandas as pd
import numpy as np

import lightgbm as lgb

from sklearn.model_selection import StratifiedKFold, KFold
from sklearn.preprocessing import OneHotEncoder

import warnings

warnings.filterwarnings('ignore')
import multiprocessing

import re


def gen_tsfresh_features():
    # 数据读取
    data_train = pd.read_csv("./data/train.csv")

    # 对训练数据处理

    # 对心电特征进行行转列处理，同时为每个心电信号加入时间步特征time
    train_heartbeat_df = data_train["heartbeat_signals"].str.split(",", expand=True).stack()
    train_heartbeat_df = train_heartbeat_df.reset_index()
    train_heartbeat_df = train_heartbeat_df.set_index("level_0")
    train_heartbeat_df.index.name = None
    train_heartbeat_df.rename(columns={"level_1": "time", 0: "heartbeat_signals"}, inplace=True)
    train_heartbeat_df["heartbeat_signals"] = train_heartbeat_df["heartbeat_signals"].astype(float)

    # 将处理后的心电特征加入到训练数据中，同时将训练数据label列单独存储
    data_train_label = data_train["label"]
    data_train = data_train.drop("label", axis=1)
    data_train = data_train.drop("heartbeat_signals", axis=1)
    data_train = data_train.join(train_heartbeat_df)

    print(data_train.info())
    print(data_train.tail())
    # 减少内存
    data_train = reduce_mem_usage(data_train)
    data_train.heartbeat_signals = data_train.heartbeat_signals.astype(np.float32)  # extract_features 中有函数不支持 float16
    print('data_train done Memory usage of dataframe is {:.2f} MB'.format(data_train.memory_usage().sum() / 1024 ** 2))
    print(data_train.info())
    print(data_train.tail())

    # 特征提取
    from tsfresh.feature_extraction import ComprehensiveFCParameters
    settings = ComprehensiveFCParameters()
    from tsfresh.feature_extraction import extract_features
    train_features = extract_features(data_train, default_fc_parameters=settings, column_id='id', column_sort='time')

    from tsfresh.utilities.dataframe_functions import impute

    # 去除抽取特征中的NaN值
    impute(train_features)

    from tsfresh import select_features

    # 按照特征和数据label之间的相关性进行特征选择
    train_features_filtered = select_features(train_features, data_train_label)

    # 对测试数据处理

    data_test_A = pd.read_csv("./data/testA.csv")

    # 对心电特征进行行转列处理，同时为每个心电信号加入时间步特征time
    test_heartbeat_df = data_test_A["heartbeat_signals"].str.split(",", expand=True).stack()
    test_heartbeat_df = test_heartbeat_df.reset_index()
    test_heartbeat_df = test_heartbeat_df.set_index("level_0")
    test_heartbeat_df.index.name = None
    test_heartbeat_df.rename(columns={"level_1": "time", 0: "heartbeat_signals"}, inplace=True)
    test_heartbeat_df["heartbeat_signals"] = test_heartbeat_df["heartbeat_signals"].astype(float)

    # 将处理后的心电特征加入到训练数据中，同时将训练数据label列单独存储
    data_test_A = data_test_A.drop("heartbeat_signals", axis=1)
    data_test_A = data_test_A.join(test_heartbeat_df)

    # 减少内存
    data_test_A = reduce_mem_usage(data_test_A)
    data_test_A.heartbeat_signals = data_test_A.heartbeat_signals.astype(np.float32)  # extract_features 中有函数不支持 float16
    print('data_test_A done Memory usage of dataframe is {:.2f} MB'.format(data_test_A.memory_usage().sum() / 1024 ** 2))
    print(data_test_A.info())
    print(data_test_A.tail())

    # 特征提取
    from tsfresh.feature_extraction import ComprehensiveFCParameters
    settings = ComprehensiveFCParameters()
    from tsfresh.feature_extraction import extract_features
    test_features = extract_features(data_test_A, default_fc_parameters=settings, column_id='id', column_sort='time')

    from tsfresh.utilities.dataframe_functions import impute

    # 去除抽取特征中的NaN值
    impute(test_features)
    # 测试数据的特征列与训练数据最终筛选出来的列对齐
    test_features_filtered = test_features[train_features_filtered.columns]

    return train_features_filtered, data_train_label, test_features_filtered


def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024 ** 2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df


def lightgbm_train_test(train, label, test):
    # 简单预处理
    train = reduce_mem_usage(train)
    test = reduce_mem_usage(test)

    ## 4.训练数据/测试数据准备

    x_train = train
    x_train.reset_index(drop=True, inplace=True)

    y_train = label
    y_train.reset_index(drop=True, inplace=True)

    x_test = test
    x_test.reset_index(drop=True, inplace=True)

    x_train = x_train.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))
    x_test = x_test.rename(columns=lambda x: re.sub('[^A-Za-z0-9_]+', '', x))

    print(x_train.shape, x_test.shape, y_train.shape)

    ## 5.模型训练

    def abs_sum(y_pre, y_tru):
        y_pre = np.array(y_pre)
        y_tru = np.array(y_tru)
        loss = sum(sum(abs(y_pre - y_tru)))
        return loss

    def cv_model(clf, train_x, train_y, test_x, clf_name):
        folds = 5
        seed = 2021
        kf = KFold(n_splits=folds, shuffle=True, random_state=seed)
        test = np.zeros((test_x.shape[0], 4))

        cv_scores = []
        onehot_encoder = OneHotEncoder(sparse=False)
        for i, (train_index, valid_index) in enumerate(kf.split(train_x, train_y)):
            print('************************************ {} ************************************'.format(str(i + 1)))
            trn_x, trn_y, val_x, val_y = train_x.iloc[train_index], train_y[train_index], train_x.iloc[valid_index], \
                                         train_y[valid_index]
            if clf_name == "lgb":
                train_matrix = clf.Dataset(trn_x, label=trn_y)
                valid_matrix = clf.Dataset(val_x, label=val_y)

                params = {
                    'boosting_type': 'gbdt',
                    'objective': 'multiclass',
                    'num_class': 4,
                    'num_leaves': 2 ** 5,
                    'feature_fraction': 0.8,
                    'bagging_fraction': 0.8,
                    'bagging_freq': 4,
                    'learning_rate': 0.1,
                    'seed': seed,
                    'n_jobs': 40,
                    'verbose': -1,
                }

                model = clf.train(params,
                                  train_set=train_matrix,
                                  valid_sets=valid_matrix,
                                  num_boost_round=2200,
                                  verbose_eval=100,
                                  early_stopping_rounds=150)
                val_pred = model.predict(val_x, num_iteration=model.best_iteration)
                test_pred = model.predict(test_x, num_iteration=model.best_iteration)

            print("val_y:", val_y.shape)
            val_y = np.array(val_y).reshape(-1, 1)
            val_y = onehot_encoder.fit_transform(val_y)
            print("val_y:", val_y.shape)
            print('预测的概率矩阵为：')
            print(test_pred)
            test += test_pred
            score = abs_sum(val_y, val_pred)
            cv_scores.append(score)
            print(cv_scores)
        print("%s_scotrainre_list:" % clf_name, cv_scores)
        print("%s_score_mean:" % clf_name, np.mean(cv_scores))
        print("%s_score_std:" % clf_name, np.std(cv_scores))
        test = test / kf.n_splits

        return test

    def lgb_model(x_train, y_train, x_test):
        lgb_test = cv_model(lgb, x_train, y_train, x_test, "lgb")
        return lgb_test

    lgb_test = lgb_model(x_train, y_train, x_test)

    # 6.预测结果

    temp = pd.DataFrame(lgb_test)
    result = pd.read_csv('./data/sample_submit.csv')
    result['label_0'] = temp[0]
    result['label_1'] = temp[1]
    result['label_2'] = temp[2]
    result['label_3'] = temp[3]
    result.to_csv('lightgbm_tsfresh_submit.csv', index=False)


if __name__ == '__main__':
    multiprocessing.freeze_support()

    # 2.读取数据
    train, label, test = gen_tsfresh_features()

    # 3.数据预处理
    lightgbm_train_test(train, label, test)
