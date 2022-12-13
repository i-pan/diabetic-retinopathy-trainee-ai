import numpy as np
import os, os.path as osp
import pandas as pd


def create_dir(d):
    if not osp.exists(d):
        os.makedirs(d)


def rescale(dist, p=1/2.):
    dist = dist ** p
    return dist / np.sum(dist)


N_sample = 300
df = pd.read_csv('train.csv')
dist = df.diagnosis.value_counts(normalize=True)
dist = rescale(dist, 1/5)
dist = (dist * N_sample).round().astype('int')
dist = dist.reset_index()

sample_list = []
for each_class, num_sample in zip(dist['index'], dist['diagnosis']):
    sample_list += [df[df.diagnosis == each_class].sample(n=num_sample, replace=False, random_state=0)]


sample = pd.concat(sample_list)
sample.shape
sample.diagnosis.value_counts(normalize=True)


sample = sample.sample(n=len(sample), replace=False, random_state=0)
sample['pseudo_id'] = [f'{i:04d}' for i in range(len(sample))]

create_dir('sample-300/')

for row in sample.itertuples():
    _ = os.system(f'cp train_images/{row.id_code}.png sample-300/{row.pseudo_id}.png')


sample.to_csv('sample-300.csv', index=False)