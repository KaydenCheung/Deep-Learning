import pandas as pd
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


def sparse_feat(feat, feat_num):
    return {'feat': feat, 'feat_num': feat_num}


def dense_feat(feat):
    return {'feat': feat}


def process_dense_feats(data, feats):
    data[feats] = data[feats].fillna(0)
    for f in tqdm(feats, desc='process_dense_feats'):
        mean = data[f].mean()
        std = data[f].std()
        data[f] = (data[f] - mean) / (std + 1e-12)
    return data


def process_sparse_feats(data, feats):
    data[feats] = data[feats].fillna('-1')
    for f in tqdm(feats, desc='process_sparse_feats'):
        label_encoder = LabelEncoder()
        data[f] = label_encoder.fit_transform(data[f])
    return data


def create_dataset(file='./data/criteo_sampled_data.csv'):
    data = pd.read_csv(file)

    dense_feats = [col_name for col_name in data.columns if col_name[0] == 'I']    # dense_feats = ['I1', 'I2',...,'I13']
    sparse_feats = [col_name for col_name in data.columns if col_name[0] == 'C']   # sparse_feats = ['C1', 'C2',...,'C26']

    data = process_dense_feats(data, dense_feats)             # 处理缺失值并标准化
    data = process_sparse_feats(data, sparse_feats)           # 处理缺失值并映射
    feat_columns = [[dense_feat(feat) for feat in dense_feats]] + [[sparse_feat(feat, len(data[feat].unique())) for feat in sparse_feats]]

    return data, feat_columns, dense_feats, sparse_feats
