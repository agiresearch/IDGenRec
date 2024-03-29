import numpy as np
import random
from itertools import combinations
from sklearn.cluster import SpectralClustering
from utils import utils
from collections import defaultdict
import os
from scipy.sparse import csr_matrix
import pdb
import tqdm
import json
import re
import torch


def sequential_indexing(data_path, dataset, user_sequence_dict, order):
    """
    Use sequential indexing method to index the given user seuqnece dict.
    """
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, f'item_sequential_indexing_{order}.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_sequential_indexing_{order}.txt')
    
    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
        
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
        
        return construct_user_sequence_dict(user_sequence), item_map
    
    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)
        
        
    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        item_map = dict()
        if order == 'original':
            user_list = user_sequence_dict.keys()
        elif order == 'short2long':
            user_list = sorted(user_sequence_dict, key=lambda x: len(user_sequence_dict[x]), reverse=False)
        elif order == 'long2short':
            user_list = sorted(user_sequence_dict, key=lambda x: len(user_sequence_dict[x]), reverse=True)
            
        for user in user_list:
            items = user_sequence_dict[user][:-2]
            for item in items:
                if item not in item_map:
                    item_map[item] = str(len(item_map) + 1001)
        for user in user_list:
            items = user_sequence_dict[user][-2:]
            for item in items:
                if item not in item_map:
                    item_map[item] = str(len(item_map) + 1001)
        utils.WriteDictToFile(item_index_file, item_map)
        
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map
        

def generative_indexing_id(data_path, dataset, user_sequence_dict, phase=0):
    """
    Use generative indexing method to index the given user seuqnece dict.
    """

    # user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    user_index_file = os.path.join(data_path, dataset, f'user_generative_index_phase_{phase}.txt')
    item_text_file = os.path.join(data_path, dataset, 'item_plain_text.txt')
    user_sequence_file = os.path.join(data_path, dataset, 'user_sequence.txt')

    user_info = utils.ReadLineFromFile(user_index_file)
    user_map = get_dict_from_lines(user_info)

    item_info = utils.ReadLineFromFile(item_text_file)
    item_map = get_dict_from_lines(item_info)

    user_sequence_info = utils.ReadLineFromFile(user_sequence_file)
    user_sequence = get_dict_from_lines(user_sequence_info)

    # 
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    return reindex_user_sequence_dict, item_map


def generative_indexing_rec(data_path, dataset, user_sequence_dict, model_gen, tokenizer, regenerate=True, phase=0):
    """
    Use generative indexing method to index the given user seuqnece dict.
    Generate ID and save to local first
    regenerate: if regenerate id file
    """

    # user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    user_index_file = os.path.join(data_path, dataset, f'user_generative_index_phase_{phase}.txt')
    item_text_file = os.path.join(data_path, dataset, 'item_plain_text.txt')
    user_sequence_file = os.path.join(data_path, dataset, 'user_sequence.txt')

    item_index_file = os.path.join(data_path, dataset, f'item_generative_indexing_phase_{phase}.txt')
    user_index_file = os.path.join(data_path, dataset, f'user_generative_index_phase_{phase}.txt')  # TODO

    # generate item id file
    if (phase == 0 and not os.path.exists(item_index_file)) or (phase != 0 and regenerate):
        print(f"(re)generate textual id with id generator phase {phase}!")
        generate_item_id_from_text(item_text_file, item_index_file, model_gen, tokenizer)

    item_info = utils.ReadLineFromFile(item_index_file)
    item_map = get_dict_from_lines(item_info)

    # generate user id file
    if (phase == 0 and not os.path.exists(user_index_file)) or (phase != 0 and regenerate):
        print(f"(re)generate user id with id generator phase {phase}!")
        generate_user_id_from_text(item_map, user_index_file, user_sequence_file, model_gen, tokenizer)

    user_info = utils.ReadLineFromFile(user_index_file)
    user_map = get_dict_from_lines(user_info)

    # item_info = utils.ReadLineFromFile(item_index_file)
    # item_map = get_dict_from_lines(item_info)

    user_sequence_info = utils.ReadLineFromFile(user_sequence_file)
    user_sequence = get_dict_from_lines(user_sequence_info)

    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    return reindex_user_sequence_dict, item_map


def random_indexing(data_path, dataset, user_sequence_dict):
    """
    Use random indexing method to index the given user seuqnece dict.
    """
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, 'item_random_indexing.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_random_indexing.txt')
    
    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
        
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
        
        return construct_user_sequence_dict(user_sequence), item_map
    
    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)
        
        
    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        item_map = dict()
        items = set()
        for user in user_sequence_dict:
            items.update(user_sequence_dict[user])
        items = list(items)
        random.shuffle(items)
        for item in items:
            if item not in item_map:
                item_map[item] = str(len(item_map) + 1001)
        utils.WriteDictToFile(item_index_file, item_map)
        
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map

def collaborative_indexing(data_path, dataset, user_sequence_dict, token_size, cluster_num, last_token, float32):
    """
    Use collaborative indexing method to index the given user seuqnece dict.
    """
    user_index_file = os.path.join(data_path, dataset, 'user_indexing.txt')
    item_index_file = os.path.join(data_path, dataset, f'item_collaborative_indexing_{token_size}_{cluster_num}_{last_token}.txt')
    reindex_sequence_file = os.path.join(data_path, dataset, f'user_sequence_collaborative_indexing_{token_size}_{cluster_num}_{last_token}.txt')
    
    if os.path.exists(reindex_sequence_file):
        user_sequence = utils.ReadLineFromFile(reindex_sequence_file)
        
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
        
        return construct_user_sequence_dict(user_sequence), item_map
    
    # For user index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(user_index_file):
        user_info = utils.ReadLineFromFile(user_index_file)
        user_map = get_dict_from_lines(user_info)
    else:
        user_map = generate_user_map(user_sequence_dict)
        utils.WriteDictToFile(user_index_file, user_map)
        
        
    # For item index, load from txt file if already exists, otherwise generate from user sequence and save.
    if os.path.exists(item_index_file):
        item_info = utils.ReadLineFromFile(item_index_file)
        item_map = get_dict_from_lines(item_info)
    else:
        item_map = generate_collaborative_id(user_sequence_dict, token_size, cluster_num, last_token, float32)
        utils.WriteDictToFile(item_index_file, item_map)
        
    reindex_user_sequence_dict = reindex(user_sequence_dict, user_map, item_map)
    utils.WriteDictToFile(reindex_sequence_file, reindex_user_sequence_dict)
    return reindex_user_sequence_dict, item_map
        
def generate_collaborative_id(user_sequence_dict, token_size, cluster_num, last_token, float32):
    """
    Generate collaborative index for items.
    """
    # get the items in training data and all data.
    all_items = set()
    train_items = set()
    for user in user_sequence_dict:
        all_items.update(set(user_sequence_dict[user]))
        train_items.update(set(user_sequence_dict[user][:-2]))
        
    # reindex all training items for calculating the adjacency matrix
    item2id = dict()
    id2item = dict()
    for item in train_items:
        item2id[item] = len(item2id)
        id2item[len(id2item)] = item
        
    
    # calculate the co-occurrence of items in the training data as an adjacency matrix
    if float32 > 0:
        adj_matrix = np.zeros((len(item2id), len(item2id)), dtype=np.float32)
    else:
        adj_matrix = np.zeros((len(item2id), len(item2id)))
    for user in user_sequence_dict:
        interactions = user_sequence_dict[user][:-2]
        for pairs in combinations(interactions, 2):
            adj_matrix[item2id[pairs[0]]][item2id[pairs[1]]] += 1
            adj_matrix[item2id[pairs[1]]][item2id[pairs[0]]] += 1
    
    
    # get the clustering results for the first layer
    clustering = SpectralClustering(
        n_clusters=cluster_num,
        assign_labels="cluster_qr",
        random_state=0,
        affinity="precomputed",
    ).fit(adj_matrix)
    labels = clustering.labels_.tolist()
    
    # count the clustering results
    grouping = defaultdict(list)
    for i in range(len(labels)):
        grouping[labels[i]].append((id2item[i],i))
    
    item_map = dict()
    index_now = 0
    
    # add current clustering information into the item indexing results.
    item_map, index_now = add_token_to_indexing(item_map, grouping, index_now, token_size)
    
    # add current clustering info into a queue for BFS
    queue = []
    for group in grouping:
        queue.append(grouping[group])
    
    # apply BFS to further use spectral clustering for large groups (> token_size)
    while queue:
        group_items = queue.pop(0)
        
        # if current group is small enough, add the last token to item indexing
        if len(group_items) <= token_size:
            item_list = [items[0] for items in group_items]
            if last_token == 'sequential':
                item_map = add_last_token_to_indexing_sequential(item_map, item_list, token_size)
            elif last_token == 'random':
                item_map = add_last_token_to_indexing_random(item_map, item_list, token_size)
        else:
            # calculate the adjacency matrix for current group
            if float32 > 0:
                sub_adj_matrix = np.zeros((len(group_items), len(group_items)), dtype=np.float32)
            else:
                sub_adj_matrix = np.zeros((len(group_items), len(group_items)))
            for i in range(len(group_items)):
                for j in range(i+1, len(group_items)):
                    sub_adj_matrix[i][j] = adj_matrix[group_items[i][1]][group_items[j][1]]
                    sub_adj_matrix[j][i] = adj_matrix[group_items[j][1]][group_items[i][1]]
                    
            # get the clustering results for current group        
            clustering = SpectralClustering(
                n_clusters=cluster_num,
                assign_labels="cluster_qr",
                random_state=0,
                affinity="precomputed",
            ).fit(sub_adj_matrix)
            labels = clustering.labels_.tolist()
            
            # count current clustering results
            grouping = defaultdict(list)
            for i in range(len(labels)):
                grouping[labels[i]].append(group_items[i])
                
            # add current clustering information into the item indexing results.
            item_map, index_now = add_token_to_indexing(item_map, grouping, index_now, token_size)
            
            # push current clustering info into the queue
            for group in grouping:
                queue.append(grouping[group])
                
    # if some items are not in the training data, assign an index for them
    remaining_items = list(all_items - train_items)
    if len(remaining_items) > 0:
        if last_token == 'sequential':
            item_map = add_last_token_to_indexing_sequential(item_map, remaining_items, token_size)
        elif last_token == 'random':
            item_map = add_last_token_to_indexing_random(item_map, remaining_items, token_size)
                
    return item_map
                
    
    
def add_token_to_indexing(item_map, grouping, index_now, token_size):
    for group in grouping:
        index_now = index_now % token_size
        for (item, idx) in grouping[group]:
            if item not in item_map:
                item_map[item] = ''
            item_map[item] += f'<CI{index_now}>'
        index_now += 1
    return item_map, index_now

def add_last_token_to_indexing_random(item_map, item_list, token_size):
    last_tokens = random.sample([i for i in range(token_size)], len(item_list))
    for i in range(len(item_list)):
        item = item_list[i]
        if item not in item_map:
            item_map[item] = ''
        item_map[item] += f'<CI{last_tokens[i]}>'
    return item_map

def add_last_token_to_indexing_sequential(item_map, item_list, token_size):
    for i in range(len(item_list)):
        item = item_list[i]
        if item not in item_map:
            item_map[item] = ''
        item_map[item] += f'<CI{i}>'
    return item_map
    
    
def get_dict_from_lines(lines):
    """
    Used to get user or item map from lines loaded from txt file.
    """
    index_map = dict()
    for line in lines:
        info = line.split(" ", 1)
        index_map[info[0]] = info[1]
    return index_map
        
        
        
        
def generate_user_map(user_sequence_dict):
    """
    generate user map based on user sequence dict.
    """
    user_map = dict()
    for user in user_sequence_dict.keys():
        user_map[user] = str(len(user_map) + 1)
    return user_map


def reindex(user_sequence_dict, user_map, item_map):
    """
    reindex the given user sequence dict by given user map and item map
    """
    reindex_user_sequence_dict = dict()
    for user in user_sequence_dict:
        uid = user_map[user]
        items = user_sequence_dict[user]
        reindex_user_sequence_dict[uid] = [item_map[i] for i in items]
        
    return reindex_user_sequence_dict
    
    
def construct_user_sequence_dict(user_sequence):
    """
    Convert a list of string to a user sequence dict. user as key, item list as value.
    """
    user_seq_dict = dict()
    for line in user_sequence:
        user_seq = line.split(" ")
        user_seq_dict[user_seq[0]] = user_seq[1:]
    return user_seq_dict


def construct_user_sequence_dict_generative(user_sequence):
    """
    Convert a list of string to a user sequence dict. user as key, item list as value.
    """
    user_seq_dict = dict()
    for line in user_sequence:
        t = line.split(' item ')
        key = t[0]
        items = t[1:]
        items = [item.strip() for item in items if item.strip()]
        # print(items)
        # exit(1)
        # items = ['item ' + item for item in items]
        user_seq_dict[key] = items
    return user_seq_dict


def generate_item_id_from_text(item_text_file_dir, item_id_file_dir, model_gen, tokenizer, device="cpu"):
    """
    generate item id file from item text file
    """



    device = next(model_gen.parameters()).device
    model_gen.to("cpu")
    item_text_dict = {}
    with open(item_text_file_dir, 'r') as file:
            for line in file:
                id_, text = line.split(' ', 1)
                item_text_dict[id_] = text.strip()  # Add to dictionary

    id_set = set()  # ensure no duplication
    item_id_dict = {}
    count = 0
    max_dp = 0

    # # Batch size can be set according to your GPU capacity
    # TODO:
    # BATCH_SIZE = 32

    # # Create batches
    # keys = list(item_text_dict.keys())
    # values = list(item_text_dict.values())
    # num_batches = len(keys) // BATCH_SIZE + int(len(keys) % BATCH_SIZE > 0)

    # for i in tqdm.tqdm(range(num_batches)):
    #     start_idx = i * BATCH_SIZE
    #     end_idx = start_idx + BATCH_SIZE

    #     batch_keys = keys[start_idx:end_idx]
    #     batch_texts = values[start_idx:end_idx]

    #     # Initialize found flags for each item in batch
    #     found_flags = [False] * len(batch_keys)
    #     dp_values = [1.] * len(batch_keys)
    #     min_l_values = [1] * len(batch_keys)

    #     while not all(found_flags):
    #         # Convert batch_texts to inputs and move to GPU
    #         inputs = tokenizer(batch_texts, max_length=512, truncation=True, return_tensors="pt", padding='max_length').to(device)

    #         # Generate outputs
    #         outputs = model_gen.generate(**inputs, num_beams=10, num_beam_groups=10, do_sample=False, 
    #                                     min_length=min(min_l_values), max_length=max(min_l_values) + 10, 
    #                                     diversity_penalty=max(dp_values), num_return_sequences=10)

    #         # Decode outputs
    #         decoded_outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)

    #         # Split decoded_outputs for each item in batch
    #         # split_decoded_outputs = torch.chunk(decoded_outputs, len(batch_keys), dim=0)
    #         split_decoded_outputs = [decoded_outputs[j:j + 10] for j in range(0, len(decoded_outputs), 10)]
    #         for j, (key, decoded_output) in enumerate(zip(batch_keys, split_decoded_outputs)):
    #             if found_flags[j]:
    #                 continue
    #             for output in decoded_output:
    #                 tags = re.findall(r'\b\w+\b', output)
    #                 id = ' '.join(tags)
    #                 if id not in id_set:
    #                     found_flags[j] = True
    #                     id_set.add(id)
    #                     item_id_dict[key] = id
    #                     break
    #             if not found_flags[j]:
    #                 dp_values[j] += 1
    #                 if dp_values[j] == 10:
    #                     min_l_values[j] += 10
    #                     dp_values[j] = 1.

    for iid, text in tqdm.tqdm(item_text_dict.items()):
        found = False
        dp = 1.  # penalty for diversity
        min_l = 1
        while not found:  # keep trying until generating an uniq id
            inputs = tokenizer([text], max_length=256, truncation=True, return_tensors="pt")
            if hasattr(model_gen, "module"):
                output = model_gen.module.generate(**inputs, num_beams=10, num_beam_groups=10, do_sample=False, min_length=min_l,
                                                max_length=min_l + 10, diversity_penalty=dp, num_return_sequences=10)
            else:
                output = model_gen.generate(**inputs, num_beams=10, num_beam_groups=10, do_sample=False, min_length=min_l,
                                                max_length=min_l + 10, diversity_penalty=dp, num_return_sequences=10)
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)
            # print('ids: ', decoded_output)
            for output in decoded_output:
                tags = re.findall(r'\b\w+\b', output)
                id = ' '.join(tags)
                if id not in id_set:
                    found = True
                    id_set.add(id)
                    if dp > max_dp:
                        max_dp = dp
                    break  # if found a new id, use it
            dp += 1
            if dp >= 10:
                min_l += 10
                dp = 1.
        item_id_dict[iid] = id

    with open(item_id_file_dir, "w") as f:
        for key, value in item_id_dict.items():
            f.write(f"{key} {value}\n")
    
    model_gen.to(device)
    # print('max_dp: ', max_dp)
    return True

def generate_user_id_from_text(item_map, user_index_file, user_sequence_file, model_gen, tokenizer):
    """
    item map: dictionary
    """

    device = next(model_gen.parameters()).device
    model_gen.to("cpu")

    user_seq_dict = {}
    with open(user_sequence_file, 'r') as file:
        for line in file:
            words = line.strip().split()
            if words:
                user_seq_dict[words[0]] = words[1:]

    
    for user, items in user_seq_dict.items():
        user_seq_dict[user] = [item_map[item] for item in items]

    id_set = set()  # no duplication
    user_id_dict = {}
    id_count_dict = {}
    count = 0
    max_dp = 0

    for uid, text in tqdm.tqdm(user_seq_dict.items()):
        text = " ".join(text)
        found = False
        dp = 1.
        min_l = 1
        while not found:  # keep trying until generating an uniq id
            inputs = tokenizer([text], max_length=256, truncation=True, return_tensors="pt")
            if hasattr(model_gen, "module"):
                # Use the underlying module for generation
                output = model_gen.module.generate(**inputs, num_beams=10, num_beam_groups=10, do_sample=False, min_length=min_l,
                                                max_length=min_l + 10, diversity_penalty=dp, num_return_sequences=10)
            else:
                # Model is not wrapped with DDP, use it directly
                output = model_gen.generate(**inputs, num_beams=10, num_beam_groups=10, do_sample=False, min_length=min_l,
                                                max_length=min_l + 10, diversity_penalty=dp, num_return_sequences=10)
            decoded_output = tokenizer.batch_decode(output, skip_special_tokens=True)

            for output in decoded_output:
                tags = re.findall(r'\b\w+\b', output)
                id = ' '.join(tags)
                if id not in id_set:
                    found = True
                    id_set.add(id)
                    if dp > max_dp:
                        max_dp = dp
                    break  # if found a new id, use it
            dp += 1
            if dp >= 10:  # increase length
                min_l += 10
                dp = 1.
        user_id_dict[uid] = id

    for key, value in user_seq_dict.items():
        if key in user_id_dict:
            user_seq_dict[key] = user_id_dict[key]
        else:
            raise ValueError(f"no user {key}")

    with open(user_index_file, "w") as f:
        for key, value in user_seq_dict.items():
            f.write(f"{key} {value}\n")
    model_gen.to(device)
    return True