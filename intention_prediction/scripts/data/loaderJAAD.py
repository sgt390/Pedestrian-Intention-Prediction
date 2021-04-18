from torch.utils.data import DataLoader
from intention_prediction.scripts.data.trajectories import JAADDataset, JAADLoader, JAADcollate

import torch
import numpy as np


def data_loader(args, path, dtype):
    # build the train set
    if dtype == "train":
        df = JAADDataset(path, args.min_obs_len, args.max_obs_len, args.timestep)
        dataset = JAADLoader(df, path, dtype, args.max_obs_len)
        #print(dtype, " has ", neg_sample_size, " negative samples and ", pos_sample_size, " positive samples")

        # build the train iterator
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            collate_fn=JAADcollate,
            shuffle=True)
            #sampler=sampler)  

    # build the val set
    # validation set should never use the weighted random sampler
    if dtype == "val":
        df = JAADDataset(path, args.min_obs_len, args.max_obs_len, args.timestep)
        dataset = JAADLoader(df, path, dtype, args.max_obs_len)
        #print(dtype, " has ", neg_sample_size, " negative samples and ", pos_sample_size, " positive samples")

        # build the val iterator
        loader  = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=1,
            collate_fn=JAADcollate)

    return len(df), loader
