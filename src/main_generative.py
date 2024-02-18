import torch
import os
import argparse
import logging
import datetime
from transformers import AutoTokenizer, T5ForConditionalGeneration, AutoModelForSeq2SeqLM
from torch.utils.data import ConcatDataset, DataLoader
import torch.distributed as dist
import torch.multiprocessing as mp
from data.MultiTaskDataset_gen import MultiTaskDatasetGen
from data.MultiTaskDataset_rec import MultiTaskDatasetRec
from runner.SingleRunner import SingleRunner
from runner.DistributedRunner_gen import DistributedRunner
# from runner.DistributedRunner import DistributedRunner
from processor.Collator import CollatorGen, Collator, TestCollator
from processor.SingleMultiDataTaskSampler import SingleMultiDataTaskSampler
from processor.DistMultiDataTaskSampler import DistMultiDataTaskSampler
from torch.utils.data.distributed import DistributedSampler
from transformers import T5Config
from model.P5 import P5
from utils import utils
from utils import initialization
from torch.nn.parallel import DistributedDataParallel as DDP
from utils.dataset_utils import get_dataset_generative, get_loader
import pdb
from undecorated import undecorated
from types import MethodType

    
    
    
def distributed_launch():
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_global_args(parser)
    parser = MultiTaskDatasetGen.parse_dataset_args(parser)
    parser = DistMultiDataTaskSampler.parse_sampler_args(parser)
    parser = DistributedRunner.parse_runner_args(parser)
    args, extras = parser.parse_known_args()
    
    ngpus_per_node = torch.cuda.device_count()
    args.world_size = ngpus_per_node
        
    mp.spawn(
        distributed_main, args=(args, ), nprocs=ngpus_per_node, join=True
    )
    
    
def distributed_main(local_rank, args):
    # distributed learning
    args.rank = local_rank
    utils.set_seed(args.seed)
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    torch.cuda.set_device(local_rank)
    dist.init_process_group(
        backend="nccl", timeout=datetime.timedelta(seconds=18000), world_size=args.world_size, rank=local_rank
    )
    utils.setup_logging(args)
    utils.setup_model_path(args)

    if args.rank == 0:
        logging.info(vars(args))
    
    device = f"cuda:{local_rank}"
    args.gpu = local_rank

    if 't5' in args.backbone:
        config = T5Config.from_pretrained(args.backbone)
        if local_rank == 0:
            logging.info(f"Use {args.backbone} backbone model")
    else:
        raise NotImplementError
    model_rec = T5ForConditionalGeneration.from_pretrained(args.backbone, config=config)

    # generate with gradient
    generate_with_grad = undecorated(model_rec.generate)
    model_rec.generate_with_grad = MethodType(generate_with_grad, model_rec)
    model_rec.to(device)
    
    model_gen = AutoModelForSeq2SeqLM.from_pretrained("nandakishormpai/t5-small-machine-articles-tag-generation")
    # generate with gradient
    generate_with_grad = undecorated(model_gen.generate)
    model_gen.generate_with_grad = MethodType(generate_with_grad, model_gen)
    model_gen.to(device)

    tokenizer = AutoTokenizer.from_pretrained(args.backbone)

    model_rec.resize_token_embeddings(len(tokenizer))
    model_gen.resize_token_embeddings(len(tokenizer))
    # load models
    if args.rec_model_path:
        if local_rank == 0:
            logging.info(f"Load model from {args.rec_model_path}")
        model_rec = utils.load_model(model_rec, args.rec_model_path, args, loc=device)
        model_rec.to(device)
    
    if args.id_model_path:
        if local_rank == 0:
            logging.info(f"Load model from {args.id_model_path}")
        model_gen = utils.load_model(model_gen, args.id_model_path, args, loc=device)
        model_gen.to(device)


    TrainSetID, TrainSetRec, ValidSet = get_dataset_generative(args, model_gen, tokenizer)
    train_loader_id, train_loader_rec, valid_loader = get_loader(args, tokenizer, TrainSetID, TrainSetRec, ValidSet, local_rank)


    if args.random_initialize == 1:
        if local_rank == 0:
            logging.info("Random initialize number related tokens")
        initialization.random_initialization(model_rec, tokenizer)
        initialization.random_initialization(model_gen, tokenizer)

    runner = DistributedRunner(model_rec, model_gen, tokenizer, train_loader_id, train_loader_rec, valid_loader, device, args, local_rank)  # change loader

    if args.train:
        if local_rank == 0:
            logging.info("Start training")
        runner.train_generator()
    dist.barrier()
    
    return
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='OpenP5')
    parser = utils.parse_global_args(parser)
    init_args, extras = parser.parse_known_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = init_args.gpu
    ngpus_per_node = torch.cuda.device_count()

    if init_args.distributed and ngpus_per_node > 1:
        distributed_launch()