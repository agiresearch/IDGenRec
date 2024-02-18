import torch
from data.MultiTaskDataset_gen import MultiTaskDatasetGen
from data.MultiTaskDataset_rec import MultiTaskDatasetRec
from torch.utils.data import ConcatDataset, DataLoader
from processor.SingleMultiDataTaskSampler import SingleMultiDataTaskSampler
from processor.DistMultiDataTaskSampler import DistMultiDataTaskSampler
from torch.utils.data.distributed import DistributedSampler
from processor.Collator import CollatorGen, Collator, TestCollator


def get_dataset_generative(args, model_gen, tokenizer, phase=0, regenerate=True):
    # init dataset
    datasets = args.datasets.split(',')
    train_all_datasets_id = []
    train_all_datasets_rec = []
    valid_all_datasets = []
    test_all_datasets = []
    for data in datasets:
        TrainDatasetRec = MultiTaskDatasetRec(args, data, 'train', model_gen, tokenizer, phase, regenerate=regenerate)
        TrainDatasetID = MultiTaskDatasetGen(args, data, 'train', phase)
        
        train_all_datasets_id.append(TrainDatasetID)
        train_all_datasets_rec.append(TrainDatasetRec)
        if args.valid_select > 0:
            ValidDataset = MultiTaskDatasetRec(args, data, 'validation')
            valid_all_datasets.append(ValidDataset)
        
    TrainSetID = ConcatDataset(train_all_datasets_id)
    TrainSetRec = ConcatDataset(train_all_datasets_rec)
    if args.valid_select > 0:
        ValidSet = ConcatDataset(valid_all_datasets)
    else:
        ValidSet = None
    
    return TrainSetID, TrainSetRec, ValidSet

def get_loader(args, tokenizer, TrainSetID, TrainSetRec, ValidSet, rank=0):
    
    # generate training validation loader.
    ngpus_per_node = torch.cuda.device_count()
    if ngpus_per_node == 1:
        args.distributed = 0

    if args.dist_sampler == 0:
        train_sampler_id = DistMultiDataTaskSampler(TrainSetID, args.id_batch_size, args.world_size, rank, args.seed, shuffle=True) if args.distributed else SingleMultiDataTaskSampler(TrainSetID, args.id_batch_size, args.seed, shuffle=True)
        train_sampler_rec = DistMultiDataTaskSampler(TrainSetRec, args.rec_batch_size, args.world_size, rank, args.seed, shuffle=True) if args.distributed else SingleMultiDataTaskSampler(TrainSetRec, args.rec_batch_size, args.seed, shuffle=True)
    else:
        train_sampler_id = DistributedSampler(TrainSetID) if args.distributed else None
        train_sampler_rec = DistributedSampler(TrainSetRec) if args.distributed else None
    if args.valid_select > 0:
        valid_sampler = DistributedSampler(ValidSet) if args.distributed else None
    
    collator_gen = CollatorGen(tokenizer)
    collator_rec = Collator(tokenizer)
    train_loader_id = DataLoader(dataset=TrainSetID, sampler=train_sampler_id, batch_size=args.id_batch_size, collate_fn=collator_gen, shuffle=False)
    train_loader_rec = DataLoader(dataset=TrainSetRec, sampler=train_sampler_rec, batch_size=args.rec_batch_size, collate_fn=collator_rec, shuffle=False)
    # train_loader = DataLoader(dataset=TrainSet, sampler=train_sampler, batch_size=args.batch_size, shuffle=False)
    if args.valid_select > 0:
        valid_loader = DataLoader(dataset=ValidSet, sampler=valid_sampler, batch_size=args.rec_batch_size, collate_fn=collator_rec, shuffle=False)
    else:
        valid_loader = None
    
    return train_loader_id, train_loader_rec, valid_loader