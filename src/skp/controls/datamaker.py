import ast
import numpy as np
import pandas as pd
import os, os.path as osp

from ..builder import build_dataset, build_dataloader


def get_train_val_test_splits(cfg, df): 
    if 'split' in df.columns and cfg.data.use_fixed_splits:
        train_df = df[df.split == 'train']
        valid_df = df[df.split == 'valid']
        test_df  = df[df.split == 'test']
        return train_df, valid_df, test_df

    i, o = cfg.data.inner_fold, cfg.data.outer_fold
    if isinstance(i, (int,float)):
        print(f'<inner fold> : {i}')
        print(f'<outer fold> : {o}')
        test_df = df[df.outer == o]
        df = df[df.outer != o]
        train_df = df[df[f'inner{o}'] != i]
        valid_df = df[df[f'inner{o}'] == i]
        valid_df = valid_df.drop_duplicates().reset_index(drop=True)
        test_df = test_df.drop_duplicates().reset_index(drop=True)
    else:
        print('No inner fold specified ...')
        print(f'<outer fold> : {o}')
        train_df = df[df.outer != o]
        valid_df = df[df.outer == o]
        valid_df = valid_df.drop_duplicates().reset_index(drop=True)
        test_df = valid_df
    return train_df, valid_df, test_df


def prepend_filepath(lst, prefix): 
    return np.asarray([osp.join(prefix, item) for item in lst])


def get_train_val_datasets(cfg):
    INPUT_COL = cfg.data.input
    LABEL_COL = cfg.data.target

    df = pd.read_csv(cfg.data.annotations) 
    if cfg.data.positives_only:
        assert not cfg.data.naive_pe_pseudolabel
        print(f"Using positive studies only ...")
        print(f"Note that this option only works for PE .")
        positive_series = df[df.pe_present_on_image == 1].SeriesInstanceUID.unique().tolist()
        df = df[df.SeriesInstanceUID.isin(positive_series)]

    train_df, valid_df, _ = get_train_val_test_splits(cfg, df)

    if LABEL_COL == "pe_pseudolabel":
        print(f"Using PE pseudolabels for training ...")
        if cfg.data.naive_pe_pseudolabel:
            print(f">> Naive approach specified . All slices from positive studies will be labeled as positive.")
            positive_series = train_df[train_df.pe_present_on_image == 1].SeriesInstanceUID.unique().tolist()
            train_df.loc[train_df.SeriesInstanceUID.isin(positive_series), "pe_pseudolabel"] = 1
        print(f"Ground truth for validation ...")
        valid_df["pe_pseudolabel"] = valid_df["pe_present_on_image"]

    if "any_pseudo" in LABEL_COL:
        print(f"Using ICH pseudolabels for training ...")
        print(f"Ground truth for validation ...")
        for each_class in LABEL_COL:
            valid_df[each_class] = valid_df[each_class.replace("_pseudo", "")]

    if cfg.data.use_cas:
        # For now, single class of any ICH
        print("Using class activation sequence pseudo-labels ...")
        assert isinstance(cfg.data.top_threshold, (int,float)), f"cfg.data.top_threshold is not defined !"
        if not isinstance(cfg.data.bot_threshold, (int,float)):
            print("cfg.data.bot_threshold is not defined ...\n")
            print(f"Using single threshold [cfg.data.top_threshold] {cfg.data.top_threshold} ...")
            cfg.data.bot_threshold = cfg.data.top_threshold
        for each_class in ['edh', 'sdh', 'iph', 'sah', 'ivh', 'any']:
            train_df[each_class] = train_df[f"cas_{each_class}"] >= cfg.data.top_threshold
            valid_df[each_class] = valid_df[f"gt_{each_class}"]
        # train_df_pos = train_df[train_df.cas >= cfg.data.top_threshold]
        # train_df_neg = train_df[train_df.cas <  cfg.data.bot_threshold]
        # train_df_pos["final_label"] = 1
        # train_df_neg["final_label"] = 0
        # train_df = pd.concat([train_df_pos, train_df_neg])
        # # For validation, use the true slice labels
        # valid_df["final_label"] = valid_df["any"]

    data_dir = cfg.data.data_dir

    train_inputs = [osp.join(data_dir, _) for _ in train_df[INPUT_COL]]
    valid_inputs = [osp.join(data_dir, _) for _ in valid_df[INPUT_COL]]

    train_labels = train_df[LABEL_COL].values
    valid_labels = valid_df[LABEL_COL].values

    train_data_info = dict(inputs=train_inputs, labels=train_labels)
    valid_data_info = dict(inputs=valid_inputs, labels=valid_labels)

    if cfg.data.tumor_only:
        train_data_info.update({'start': train_df.start.values, 'stop': train_df.stop.values})
        valid_data_info.update({'start': valid_df.start.values, 'stop': valid_df.stop.values})

    train_dataset = build_dataset(cfg, 
        data_info=train_data_info,
        mode='train')
    valid_dataset = build_dataset(cfg, 
        data_info=valid_data_info,
        mode='valid')

    print(f'TRAIN : n={len(train_dataset)}')
    print(f'VALID : n={len(valid_dataset)}')

    return train_dataset, valid_dataset


