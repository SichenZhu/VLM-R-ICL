import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import torch
import pickle
from tqdm import tqdm
import numpy as np
from vlmeval.smp import *

class JICES:
    def __init__(
        self,
        dataset,
        query_dataset,
        eval_model,
        device,
        batch_size,
        cached_features_path=None,
        query_cached_features_path=None,
        **kwargs,
    ):
        self.dataset = dataset
        self.query_dataset = query_dataset
        self.device = device
        self.batch_size = batch_size

        self.query_cached_features_path = query_cached_features_path
        self.cached_features_path = cached_features_path

        self.rank, self.world_size = get_rank_and_world_size()

        # Load the model and processor
        self.model = eval_model

        self.use_ans_feat = kwargs.get("use_ans_feat", False)

        # Precompute features
        if os.path.exists(cached_features_path):
            with open(cached_features_path, 'rb') as f:
                self.features = pickle.load(f)
        else:
            self.features = self._precompute_features(dataset, use_answer=True)
            if self.rank == 0:
                os.makedirs(os.path.dirname(cached_features_path), exist_ok=True)
                with open(cached_features_path, 'wb') as f:
                    pickle.dump(self.features, f)
            # Synchronize all processes
            torch.distributed.barrier()
            # Now all ranks load the features
            if self.rank != 0:
                with open(cached_features_path, 'rb') as f:
                    self.features = pickle.load(f)
        
        if os.path.exists(query_cached_features_path):
            with open(query_cached_features_path, 'rb') as f:
                self.query_features = pickle.load(f)
        else:
            self.query_features = self._precompute_features(query_dataset)
            if self.rank == 0:
                os.makedirs(os.path.dirname(query_cached_features_path), exist_ok=True)
                with open(query_cached_features_path, 'wb') as f:
                    pickle.dump(self.query_features, f)
            torch.distributed.barrier()
            if self.rank != 0:
                with open(query_cached_features_path, 'rb') as f:
                    self.query_features = pickle.load(f)
        
        self.prompts, self.idx = self.features["prompts"], self.features["idx"]
        self.query_prompts, self.query_idx = self.query_features["prompts"], self.query_features["idx"]
        
        self.features = self.features["features"].to(self.device)
        self.query_features = self.query_features['features'].to(self.device)

        assert len(self.features) == len(dataset)
        assert len(self.query_features) == len(query_dataset)

        self.query_index2id = {idx: i for i, idx in enumerate(self.query_idx)}
        self.query_id2index = {i: idx for i, idx in enumerate(self.query_idx)}
        self.index2id = {idx: i for i, idx in enumerate(self.idx)}
        self.id2index = {i: idx for i, idx in enumerate(self.idx)}

    def _precompute_features(self, dataset, use_answer=False):
        dataset_name = dataset.dataset_name
        
        sheet_indices = list(range(self.rank, len(dataset), self.world_size))
        lt = len(sheet_indices)
        data = dataset.data.iloc[sheet_indices]

        features_rank = []
        for i in tqdm(range(lt), desc="Precomputing features for JICES"):
            idx = data.iloc[i]['index']

            if hasattr(self.model, 'use_custom_prompt') and self.model.use_custom_prompt(dataset_name):
                if self.use_ans_feat:
                    struct = self.model.build_prompt(data.iloc[i], dataset=dataset_name, use_answer=True)
                else:
                    struct = self.model.build_prompt(data.iloc[i], dataset=dataset_name)
                if use_answer:
                    struct_answer = self.model.build_prompt(data.iloc[i], dataset=dataset_name, use_answer=True)
            else:
                if self.use_ans_feat:
                    struct = dataset.build_prompt(data.iloc[i], use_answer=True)
                else:
                    struct = dataset.build_prompt(data.iloc[i])
                if use_answer:
                    struct_answer = dataset.build_prompt(data.iloc[i], use_answer=True)

            # response = self.model.generate(message=struct, dataset=dataset_name)
            joint_features = self.model.__call__(message=struct,
                                                 dataset=dataset_name, 
                                                 output_hidden_states=True,
                                                ).hidden_states[-1][:, -1, :].clone()
        
            joint_features /= joint_features.norm(dim=-1, keepdim=True)
            # TypeError: Got unsupported ScalarType BFloat16
            joint_features = joint_features.to(torch.float16).cpu().detach().numpy()  # For NCCL-based processed groups, internal tensor representations of objects must be moved to the GPU device before communication takes place.

            if use_answer:
                features_rank.append({"feature": joint_features, "idx": idx, "prompt": struct_answer})
            else:
                features_rank.append({"feature": joint_features, "idx": idx, "prompt": struct})

        # all gather
        features = [None for _ in range(self.world_size)]
        torch.distributed.all_gather_object(features, features_rank)  # list of lists

        if self.rank != 0:
            return None

        # sort by idx: features is a list of jsons, each json has a feature and an idx
        features = sorted([item for sublist in features for item in sublist], key=lambda x: x["idx"])
        idx = [item["idx"] for item in features]
        prompts = [item["prompt"] for item in features]
        features = [torch.from_numpy(item["feature"]) for item in features]
        
        # remove duplicates in idx and corresponding features
        # Initialize an empty set to track seen indices
        seen = set()
        # Initialize empty lists to store unique indices and corresponding features
        unique_idx = []
        unique_features = []
        unique_prompts = []

        # Iterate over the idx and features lists simultaneously
        for i, feature, p in zip(idx, features, prompts):
            if i not in seen:
                # If the index hasn't been seen, add it to the set and append the index and feature to the unique lists
                seen.add(i)
                unique_idx.append(i)
                unique_features.append(feature)
                unique_prompts.append(p)

        # Update the original lists to the unique lists
        idx = unique_idx
        # if idx != np.arange(lt).tolist():
        #     print(f"idx: {idx}")
        #     print(f"lt: {lt}")
        features = unique_features
        prompts = unique_prompts
        
        features = torch.cat(features)
        # print("idx", idx)
        # print("features.shape", features.shape)
        # print("prompts", prompts)

        return {"features": features, "idx": idx, "prompts": prompts}

    def find(self, query_idx, num_examples):
        """
        Get the top num_examples most similar examples to the images.
        """

        with torch.no_grad():
            query_id = self.query_index2id[query_idx]
            query_feature = self.query_features[query_id].unsqueeze(0)

            # Compute the similarity of the input image to the precomputed features
            similarity = (query_feature @ self.features.T).squeeze()
            # (b, 1, d) @ (t, 1, d).T -> (b, 1, d) @ (d, 1, t) -> (b, 1, t) -> (b, t)

            if similarity.ndim == 1:
                similarity = similarity.unsqueeze(0)

            indices = similarity.argsort(dim=-1, descending=True)[:, :num_examples]

        # Return with the most similar images last
        return [self.prompts[i] for i in reversed(indices[0])]