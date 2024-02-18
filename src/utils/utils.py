import numpy as np
import os
import pickle
import argparse
import inspect
import logging
import sys
import random
import torch


def parse_global_args(parser):
    parser.add_argument("--seed", type=int, default=2023, help="Random seed")
    parser.add_argument("--model_dir", type=str, default='../model', help='The model directory')
    parser.add_argument("--checkpoint_dir", type=str, default='../checkpoint', help='The checkpoint directory')
    parser.add_argument("--model_name", type=str, default='model.pt', help='The model name')
    parser.add_argument("--log_dir", type=str, default='../log', help='The log directory')
    parser.add_argument("--distributed", type=int, default=1, help='use distributed data parallel or not.')
    parser.add_argument("--gpu", type=str, default='0,1,2,3', help='gpu ids, if not distributed, only use the first one.')
    parser.add_argument("--master_addr", type=str, default='localhost', help='Setup MASTER_ADDR for os.environ')
    parser.add_argument("--master_port", type=str, default='12345', help='Setup MASTER_PORT for os.environ')
    parser.add_argument('--logging_level', type=int, default=logging.INFO,help='Logging Level, 0, 10, ..., 50')
    # parser.add_argument("--training_strategy", type=int, default=0, help=(
    #         "only used for generative ID. 0: train ID generator only, "
    #         "1: train recommendation model only. 2: start with training ID generator. "
    #         "3: start with training recommendation model"
    #     )
    # )
    parser.add_argument("--id_epochs", type=int, default=10, help=("train id for certain num of epochs"))
    parser.add_argument("--rec_epochs", type=int, default=10, help=("train rec for certain num of epochs"))
    parser.add_argument("--id_batch_size", type=int, default=4, help="batch size for id generator")
    parser.add_argument("--rec_batch_size", type=int, default=64, help="batch size for rec model")
    parser.add_argument("--rec_model_path", type=str, help="path to rec model")
    parser.add_argument("--id_model_path", type=str, help="path to id model")
    parser.add_argument("--id_lr", type=float, default=1e-3, help="learning rate for recommendation model")
    parser.add_argument("--rec_lr", type=float, default=1e-5, help="learning rate for generation model")
    parser.add_argument("--alt_style", type=str, default="id_first", help="choose from rec_first or id_first")
    parser.add_argument("--test_epoch_id", type=int, default=1)
    parser.add_argument("--test_epoch_rec", type=int, default=5)
    parser.add_argument("--rounds", type=int, default=3, help="number of iterations")
    return parser

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False


def load_pickle(filename):
    with open(filename, "rb") as f:
        return pickle.load(f)


def save_pickle(data, filename):
    with open(filename, "wb") as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        
def ReadLineFromFile(path):
    if not os.path.exists(path):
        raise FileNotFoundError
    lines = []
    with open(path,'r') as fd:
        for line in fd:
            lines.append(line.rstrip('\n'))
    return lines

def WriteDictToFile(path, write_dict):
    with open(path, 'w') as out:
        for user, items in write_dict.items():
            if type(items) == list:
                out.write(user + ' ' + ' '.join(items) + '\n')
            else:
                out.write(user + ' ' + str(items) + '\n')

                        
def get_init_paras_dict(class_name, paras_dict):
    base_list = inspect.getmro(class_name)
    paras_list = []
    for base in base_list:
        paras = inspect.getfullargspec(base.__init__)
        paras_list.extend(paras.args)
    paras_list = sorted(list(set(paras_list)))
    out_dict = {}
    for para in paras_list:
        if para == 'self':
            continue
        out_dict[para] = paras_dict[para]
    return out_dict

def setup_logging(args):
    args.log_name = log_name(args)
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    folder = os.path.join(args.log_dir, folder_name)
    if not os.path.exists(folder):
        os.makedirs(folder)
    log_file = os.path.join(args.log_dir, folder_name, args.log_name + '.log')
    
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(filename=log_file, level=args.logging_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    
    return
    

def log_name(args):
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    params = [str(args.distributed), str(args.sample_prompt), str(args.his_prefix), str(args.skip_empty_his), str(args.max_his), str(args.master_port), folder_name, args.tasks, args.backbone, args.item_indexing, str(args.lr), str(args.epochs), str(args.batch_size), args.sample_num, args.prompt_file[3:-4]]
    return '_'.join(params)

def setup_model_path(args):
    import datetime
    if len(args.datasets.split(',')) > 1:
        folder_name = 'SP5'
    else:
        folder_name = args.datasets
    # Get the current timestamp
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    args.model_path = os.path.join(args.model_dir, f"{folder_name}_id_{args.id_epochs}_rec_{args.rec_epochs}_{timestamp}")
    from pathlib import Path
    Path(args.model_path).mkdir(parents=True, exist_ok=True)
    return
    
def save_model(model, path):
    torch.save(model.state_dict(), path)
    return
    
def load_model(model, path, args, loc=None):
    if loc is None and hasattr(args, 'gpu'):
        gpuid = args.gpu.split(',')
        loc = f'cuda:{gpuid[0]}'
    state_dict = torch.load(path, map_location=loc)
    model.load_state_dict(state_dict, strict=False)
    return model
