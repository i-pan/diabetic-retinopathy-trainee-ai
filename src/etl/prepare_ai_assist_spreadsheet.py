import glob
import numpy as np
import os, os.path as osp
import pandas as pd


preds = np.sort(glob.glob("../predictions/mk000_n1000*")[:3])
dfs = [pd.read_csv(p) for p in preds]
proba_cols = [col for col in dfs[0].columns if "p" in col]
for df in dfs:
    df["y_pred"] = np.argmax(df[proba_cols].values, axis=1)
    df["pseudo_id"] = df.imgfile.apply(lambda x: x.split("/")[-1].replace(".png", "")).astype("int")


mean_df = dfs[0].copy() 
for df in dfs[1:]:
    mean_df[proba_cols] = mean_df[proba_cols].values + df[proba_cols].values

mean_df[proba_cols] /= 3.

mean_df["y_pred"] = np.argmax(mean_df[proba_cols].values, axis=1)

with open("prp_removed.txt", "r") as f: 
    prp_removed = [line.strip() for line in f.readlines()]

prp_removed = [int(i.replace(".png", "")) for i in prp_removed]
mean_df = mean_df[mean_df.pseudo_id.isin(prp_removed)]

# Generate new pseudo IDs
np.random.seed(88)
mean_df["new_pseudo_id"] = np.random.permutation(np.arange(len(mean_df)))

# Make new folder
if not osp.exists("../data/sample-300-round2/"):
    os.makedirs("../data/sample-300-round2/")


for row in mean_df.itertuples():
    _ = os.system(f"cp {row.imgfile} ../data/sample-300-round2/{row.new_pseudo_id:04d}.png")

mean_df.to_csv("../data/ai_pred.csv", index=False)