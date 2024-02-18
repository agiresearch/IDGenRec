from runner.SingleRunner import SingleRunner
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import AdamW, get_linear_schedule_with_warmup
import torch.distributed as dist
import logging
from tqdm import tqdm
from utils import utils
import torch
import utils.generation_trie as gt
import utils.evaluate as evaluate
from torch.utils.data.distributed import DistributedSampler
from data.TestDataset import TestDatasetGen
from torch.utils.data import DataLoader
from processor.Collator import Collator, TestCollator
import time
import numpy as np
import random
import torch.nn.functional as F
import pdb
import os
import string
from utils.dataset_utils import get_dataset_generative, get_loader
# from undecorated import undecorated
# from types import MethodType


class DistributedRunner(SingleRunner):
    def __init__(self, model_rec, model_gen, tokenizer, train_loader_id, train_loader_rec, valid_loader, device, args, rank):
        super().__init__(model_rec, tokenizer, train_loader_rec, valid_loader, device, args)
        self.rank = rank
        self.train_loader_id = train_loader_id
        self.train_loader_rec = train_loader_rec
        self.model_gen = DDP(model_gen, device_ids=[self.args.gpu], find_unused_parameters=True)
        # self.model_rec = DDP(self.model, device_ids=[self.args.gpu], find_unused_parameters=True)
        self.model_rec = DDP(self.model, device_ids=[self.args.gpu], find_unused_parameters=True)
        # assert self.args.epochs % (self.args.id_epochs + self.args.rec_epochs) == 0, \
        #     f"total_epochs {self.args.epochs} cannot be evenly divided by the sum of id_epochs {self.args.id_epochs} and rec_epochs {self.args.rec_epochs}"
        self.num_alternations = self.args.rounds
        # self.num_alternations = self.args.epochs // (self.args.id_epochs + self.args.rec_epochs)
        (self.id_optimizer, self.id_scheduler, 
        self.rec_optimizer, self.rec_scheduler) = self.create_optimizer_and_scheduler_2()
        self.get_testloader(regenerate=False, phase=0)

    def create_optimizer_and_scheduler_2(self):
        if self.args.rank == 0:
            logging.info("Building Optimizer and Scheduler")
        batch_per_epoch_id = len(self.train_loader_id)
        batch_per_epoch_rec = len(self.train_loader_rec)
        id_total_steps = batch_per_epoch_id // self.args.gradient_accumulation_steps * self.args.id_epochs * self.num_alternations
        id_warmup_steps = int(id_total_steps * self.args.warmup_prop)
        
        rec_total_steps = batch_per_epoch_rec // self.args.gradient_accumulation_steps * self.args.rec_epochs * self.num_alternations
        rec_warmup_steps = int(rec_total_steps * self.args.warmup_prop)
        if self.args.rank == 0:
            logging.info(f'Batch per epoch id: {batch_per_epoch_id}')
            logging.info(f'Warmup proportion: {self.args.warmup_prop}')
            logging.info(f'Total id generator steps: {id_total_steps}')
            logging.info(f'Warm up id generator steps: {id_warmup_steps}')
            logging.info(f'Batch per epoch rec: {batch_per_epoch_rec}')
            logging.info(f'Warmup proportion: {self.args.warmup_prop}')
            logging.info(f'Total rec generator steps: {rec_total_steps}')
            logging.info(f'Warm up rec generator steps: {rec_warmup_steps}')

        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters_id = [
            {
                "params": [
                    p
                    for n, p in self.model_gen.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model_gen.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        optimizer_grouped_parameters_rec = [
            {
                "params": [
                    p
                    for n, p in self.model_rec.named_parameters()
                    if not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in self.model_rec.named_parameters()
                    if any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]

        if self.args.rank == 0:
            logging.info(f"Building Optimizer {self.args.optim}")

        if self.args.optim.lower() == 'adamw':
            optimizer_id = AdamW(optimizer_grouped_parameters_id, lr=self.args.id_lr, eps=self.args.adam_eps)
            optimizer_rec = AdamW(optimizer_grouped_parameters_rec, lr=self.args.rec_lr, eps=self.args.adam_eps)
        else:
            raise NotImplementError
        scheduler_id = get_linear_schedule_with_warmup(optimizer_id, id_warmup_steps, id_total_steps)
        scheduler_rec = get_linear_schedule_with_warmup(optimizer_rec, rec_warmup_steps, rec_total_steps)

        return optimizer_id, scheduler_id, optimizer_rec, scheduler_rec
    
    
    def train_generator(self):
        self.model_gen.zero_grad()
        self.model_rec.zero_grad()
        global_epoch = 0  # save the global training epoch, used for sampler
        total_id_epoch = 0
        train_losses = []
        if self.test_before_train > 0:
            self.test()
        
        if self.args.alt_style == "id_first":
            for alter in range(self.num_alternations):
                # only train generator
                if self.rank == 0:
                    logging.info(f'Training ID Generator phase {alter+1}')
                # fix rec model param
                for param in self.model_rec.parameters():
                    param.requires_grad = False
                for param in self.model_gen.parameters():
                    param.requires_grad = True
                for id_epoch in range(self.args.id_epochs):
                    if self.rank == 0:
                        logging.info(f"Start training generator for phase {alter+1}, epoch {id_epoch+1}")
                    dist.barrier()
                    self.train_loader_id.sampler.set_epoch(global_epoch)
                    # self.train_loader_rec.sampler.set_epoch(global_epoch)
                    dist.barrier()
                    self.model_gen.train()
                    self.model_rec.train()
                    losses = []  # loss for current eopch
                    batch_count= 0
                    for batch in tqdm(self.train_loader_id):
                        input_prompt_ids = batch[0].to(self.device)
                        input_prompt_positions = batch[1].to(self.device)
                        hist_ids = batch[2].to(self.device)
                        hist_att = batch[3].to(self.device)
                        output_ids = batch[4].to(self.device)
                        output_att = batch[5].to(self.device)

                        batch_size = hist_ids.shape[0]
                        hist_size = hist_ids.shape[1]

                        input_tensor = hist_ids.view(-1, hist_ids.shape[-1])

                        output = self.model_gen.module.generate_with_grad(
                            input_tensor,
                            attention_mask=hist_att.view(-1, hist_att.shape[-1]),
                            max_length=10,
                            min_length=1,
                            num_beams=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_hidden_states=False,
                            renormalize_logits=True,
                            early_stopping=True,
                        )

                        probabilities = torch.cat([score.unsqueeze(1) for score in output['scores']], dim=1)
                        train_id_token_size = probabilities.shape[1]

                        token_embeddings = self.model_rec.module.shared.weight  # use rec models' embedding as input
                        hist_embeddings = torch.einsum('bsv,ve->bse', probabilities, token_embeddings)
                        hist_embeddings = hist_embeddings.view(batch_size, hist_size, train_id_token_size, -1)  # [bs, hist_size, id_token_size, xxx]

                        # Remove punctuation embeddings
                        temp_ids = output['sequences'][:, 1:]

                        punctuation_tokens = [self.tokenizer.encode(p, add_special_tokens=False)[0] for p in string.punctuation]

                        punctuation_tokens_tensor = torch.tensor(punctuation_tokens).to(self.device)
                        punctuation_mask = torch.isin(temp_ids, punctuation_tokens_tensor)
                        # reshape
                        batch_size_, hist_size_, seq_length_minus_one_, embedding_dim_ = hist_embeddings.shape
                        punctuation_mask = punctuation_mask.view(batch_size_, hist_size_, seq_length_minus_one_)

                        hist_embeddings[punctuation_mask.unsqueeze(-1).expand_as(hist_embeddings)] = 0

                        input_prompt_embeddings = token_embeddings[input_prompt_ids]

                        # calculate the max sequence size
                        max_prompt_size = input_prompt_embeddings.shape[1]
                        max_hist_num = hist_ids.shape[1]
                        max_input_len = max_prompt_size + max_hist_num * train_id_token_size
                        final_input = self.insert_phrases_batch(input_prompt_embeddings, 
                                                            input_prompt_positions, 
                                                            hist_embeddings, 
                                                            max_input_len)
    
                        norms = torch.norm(final_input, dim=-1)
                        attention_mask = (norms > 1e-6).long()

                        output = self.model_rec.module(
                            inputs_embeds=final_input,
                            attention_mask=attention_mask,
                            labels=output_ids,
                            return_dict=True,
                        )

                        # compute loss masking padded tokens
                        loss = output["loss"]

                        # update
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_gen.parameters(), self.args.clip)

                        dist.barrier()
                        
                        self.id_optimizer.step()
                        self.id_scheduler.step()
                        self.model_gen.zero_grad()
                        self.model_rec.zero_grad()
                        
                        dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()
                        
                        dist.barrier()
                        if self.rank == 0:
                            losses.append(loss.detach())
                        batch_count += 1 

                    # save model
                    if self.rank == 0:
                        cur_path = os.path.join(self.args.model_path, f"model_gen_phase_{alter+1}_epoch_{id_epoch+1}.pt")
                        torch.save(self.model_gen.module.state_dict(), cur_path)
                        logging.info(f"Save the current ID model to {cur_path}")

                    if self.rank == 0:
                        train_epoch_loss = sum(losses)/len(losses)
                        train_losses.append(train_epoch_loss)
                        logging.info(f"The average training loss for id phase {alter+1} epoch {id_epoch+1} is {train_epoch_loss}")

                    global_epoch += 1
                    total_id_epoch += 1

                    if self.args.test_epoch_id > 0:
                        if (id_epoch + 1) % self.args.test_epoch_id == 0:
                            self.model_gen.eval()
                            self.model_rec.eval()
                            self.get_testloader(model_gen=self.model_gen, tokenizer=self.tokenizer, regenerate=True, phase=total_id_epoch)
                            self.test()
                # only train recommender
                if self.rank == 0:
                    logging.info(f'Training Recommender phase {alter+1}')
                # regenerate rec train loader
                TrainSetID, TrainSetRec, ValidSet = get_dataset_generative(self.args, self.model_gen, self.tokenizer, total_id_epoch, regenerate=False)
                train_loader_id, self.train_loader_rec, valid_loader = get_loader(self.args, self.tokenizer, TrainSetID, TrainSetRec, ValidSet, self.rank)

                # fix generator model param
                for param in self.model_rec.parameters():
                    param.requires_grad = True
                for param in self.model_gen.parameters():
                    param.requires_grad = False
                # only train generator
                for rec_epoch in range(self.args.rec_epochs):
                    self.model_gen.train()
                    self.model_rec.train()
                    if self.rank == 0:
                        logging.info(f"Start training recommender for phase {alter+1}, epoch {rec_epoch+1}")
                    dist.barrier()

                    if self.regenerate_candidate:
                        for ds in self.train_loader.dataset.datasets:
                            ds.generate_candidates()
                            ds.construct_sentence()
                    elif self.reconstruct_data:
                        for ds in self.train_loader.dataset.datasets:
                            ds.construct_sentence()


                    self.train_loader_rec.sampler.set_epoch(global_epoch)
                    # self.train_loader_rec.sampler.set_epoch(global_epoch)
                    dist.barrier()
                    losses = []
                    for batch in tqdm(self.train_loader_rec):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        whole_input_ids = batch[2].to(self.device)  # remove
                        output_ids = batch[3].to(self.device)
                        output_attention = batch[4].to(self.device)

                        output = self.model_rec.module(
                            input_ids=input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            return_dict=True,
                        )

                        # compute loss masking padded tokens
                        loss = output["loss"]
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_rec.parameters(), self.args.clip)

                        dist.barrier()
                        self.rec_optimizer.step()
                        self.rec_scheduler.step()
                        self.model_gen.zero_grad()
                        self.model_rec.zero_grad()
                        
                        dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()
                        
                        dist.barrier()
                        
                        if self.rank == 0:
                            losses.append(loss.detach())

                    learning_rate = self.rec_optimizer.param_groups[0]['lr']

                    # save model
                    if self.rank == 0 and (rec_epoch+1) % 10 == 0:
                        cur_path = os.path.join(self.args.model_path, f"model_rec_phase_{alter+1}_epoch_{rec_epoch+1}.pt")
                        torch.save(self.model_rec.module.state_dict(), cur_path)
                        logging.info(f"Save the current rec model to {cur_path}")

                    if self.rank == 0:
                        train_epoch_loss = sum(losses)/len(losses)
                        train_losses.append(train_epoch_loss)
                        logging.info(f"The average training loss for rec phase {alter+1} epoch {rec_epoch+1} is {train_epoch_loss}")

                    if self.args.test_epoch_rec > 0:
                        if (rec_epoch + 1) % self.args.test_epoch_rec == 0:
                            self.model_gen.eval()
                            self.model_rec.eval()
                            # regenerate text sampler
                            self.get_testloader(model_gen=self.model_gen, tokenizer=self.tokenizer, regenerate=False, phase=total_id_epoch)
                            self.test()
                    global_epoch += 1

        elif self.args.alt_style == "rec_first":
            for alter in range(self.num_alternations):
                # only train recommender
                if self.rank == 0:
                    logging.info(f'Training Recommender phase {alter+1}')
                # regenerate rec train loader
                TrainSetID, TrainSetRec, ValidSet = get_dataset_generative(self.args, self.model_gen, self.tokenizer, total_id_epoch, regenerate=False)
                train_loader_id, self.train_loader_rec, valid_loader = get_loader(self.args, self.tokenizer, TrainSetID, TrainSetRec, ValidSet, self.rank)

                # fix generator model param
                for param in self.model_rec.parameters():
                    param.requires_grad = True
                for param in self.model_gen.parameters():
                    param.requires_grad = False

                for rec_epoch in range(self.args.rec_epochs):
                    self.model_gen.train()
                    self.model_rec.train()
                    if self.rank == 0:
                        logging.info(f"Start training recommender for phase {alter+1}, epoch {rec_epoch+1}")
                    dist.barrier()

                    if self.regenerate_candidate:
                        for ds in self.train_loader.dataset.datasets:
                            ds.generate_candidates()
                            ds.construct_sentence()
                    elif self.reconstruct_data:
                        for ds in self.train_loader.dataset.datasets:
                            ds.construct_sentence()

                    self.train_loader_rec.sampler.set_epoch(global_epoch)
                    # self.train_loader_rec.sampler.set_epoch(global_epoch)
                    dist.barrier()
                    losses = []  # loss for current eopch
                    for batch in tqdm(self.train_loader_rec):
                        input_ids = batch[0].to(self.device)
                        attn = batch[1].to(self.device)
                        whole_input_ids = batch[2].to(self.device)  # remove
                        output_ids = batch[3].to(self.device)
                        output_attention = batch[4].to(self.device)

                        output = self.model_rec.module(
                            input_ids=input_ids,
                            attention_mask=attn,
                            labels=output_ids,
                            return_dict=True,
                        )

                        loss = output["loss"]
                        # update
                        loss.backward()

                        torch.nn.utils.clip_grad_norm_(self.model_rec.parameters(), self.args.clip)

                        dist.barrier()
                        self.rec_optimizer.step()
                        self.rec_scheduler.step()
                        self.model_gen.zero_grad()
                        self.model_rec.zero_grad()
                        
                        dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()
                        
                        dist.barrier()
                        
                        if self.rank == 0:
                            losses.append(loss.detach())

                    learning_rate = self.rec_optimizer.param_groups[0]['lr']

                    # save model
                    if self.rank == 0 and (rec_epoch+1) % 10 == 0:
                        cur_path = os.path.join(self.args.model_path, f"model_rec_phase_{alter+1}_epoch_{rec_epoch+1}.pt")
                        torch.save(self.model_rec.module.state_dict(), cur_path)
                        logging.info(f"Save the current rec model to {cur_path}")

                    if self.rank == 0:
                        train_epoch_loss = sum(losses)/len(losses)
                        train_losses.append(train_epoch_loss)
                        logging.info(f"The average training loss for rec phase {alter+1} epoch {rec_epoch+1} is {train_epoch_loss}")

                    if self.args.test_epoch_rec > 0:
                        if (rec_epoch + 1) % self.args.test_epoch_rec == 0:
                            self.model_gen.eval()
                            self.model_rec.eval()
                            # regenerate text sampler
                            self.get_testloader(model_gen=self.model_gen, tokenizer=self.tokenizer, regenerate=False, phase=total_id_epoch)
                            self.test()
                    global_epoch += 1

                if self.rank == 0:
                    logging.info(f'Training ID Generator phase {alter+1}')
                # fix rec model param
                for param in self.model_rec.parameters():
                    param.requires_grad = False
                for param in self.model_gen.parameters():
                    param.requires_grad = True
                for id_epoch in range(self.args.id_epochs):
                    if self.rank == 0:
                        logging.info(f"Start training generator for phase {alter+1}, epoch {id_epoch+1}")
                    dist.barrier()
                    self.train_loader_id.sampler.set_epoch(global_epoch)
                    # self.train_loader_rec.sampler.set_epoch(global_epoch)
                    dist.barrier()
                    self.model_gen.train()
                    self.model_rec.train()

                    losses = []  # loss for current eopch
                    batch_count= 0
                    for batch in tqdm(self.train_loader_id):
                        input_prompt_ids = batch[0].to(self.device)
                        input_prompt_positions = batch[1].to(self.device)
                        hist_ids = batch[2].to(self.device)
                        hist_att = batch[3].to(self.device)
                        output_ids = batch[4].to(self.device)
                        output_att = batch[5].to(self.device)

                        batch_size = hist_ids.shape[0]
                        hist_size = hist_ids.shape[1]

                        input_tensor = hist_ids.view(-1, hist_ids.shape[-1])

                        # generated_ids = self.model_gen.module.generate_with_grad(input_tensor, attention_mask=hist_att.view(-1, hist_att.shape[-1]), )
                        output = self.model_gen.module.generate_with_grad(
                            input_tensor,
                            attention_mask=hist_att.view(-1, hist_att.shape[-1]),
                            max_length=10,
                            min_length=1,
                            num_beams=1,
                            return_dict_in_generate=True,
                            output_scores=True,
                            output_hidden_states=False,
                            renormalize_logits=True,
                            early_stopping=True
                        )

                        probabilities = torch.cat([score.unsqueeze(1) for score in output['scores']], dim=1)
                        train_id_token_size = probabilities.shape[1]


                        token_embeddings = self.model_rec.module.shared.weight  # use rec models' embedding as input
                        hist_embeddings = torch.einsum('bsv,ve->bse', probabilities, token_embeddings)
                        hist_embeddings = hist_embeddings.view(batch_size, hist_size, train_id_token_size, -1)  # [bs, hist_size, id_token_size, xxx]

                        # Remove punctuation embeddings
                        temp_ids = output['sequences'][:, 1:]

                        punctuation_tokens = [self.tokenizer.encode(p, add_special_tokens=False)[0] for p in string.punctuation]

                        punctuation_tokens_tensor = torch.tensor(punctuation_tokens).to(self.device)
                        punctuation_mask = torch.isin(temp_ids, punctuation_tokens_tensor)
                        # reshaoe
                        batch_size_, hist_size_, seq_length_minus_one_, embedding_dim_ = hist_embeddings.shape
                        punctuation_mask = punctuation_mask.view(batch_size_, hist_size_, seq_length_minus_one_)

                        hist_embeddings[punctuation_mask.unsqueeze(-1).expand_as(hist_embeddings)] = 0

                        input_prompt_embeddings = token_embeddings[input_prompt_ids]

                        # calculate the max sequence size
                        max_prompt_size = input_prompt_embeddings.shape[1]
                        max_hist_num = hist_ids.shape[1]
                        max_input_len = max_prompt_size + max_hist_num * train_id_token_size
                        final_input = self.insert_phrases_batch(input_prompt_embeddings, 
                                                            input_prompt_positions, 
                                                            hist_embeddings, 
                                                            max_input_len)
    
                        norms = torch.norm(final_input, dim=-1)
                        attention_mask = (norms > 1e-6).long()

                        output = self.model_rec.module(
                            inputs_embeds=final_input,
                            attention_mask=attention_mask,
                            labels=output_ids,
                            return_dict=True,
                        )

                        loss = output["loss"]

                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(self.model_gen.parameters(), self.args.clip)

                        dist.barrier()
                        
                        self.id_optimizer.step()
                        self.id_scheduler.step()
                        self.model_gen.zero_grad()
                        self.model_rec.zero_grad()
                        
                        dist.all_reduce(loss.detach(), op=dist.ReduceOp.SUM)
                        loss /= dist.get_world_size()
                        
                        dist.barrier()
                        if self.rank == 0:
                            losses.append(loss.detach())
                        batch_count += 1 


                    # save model
                    if self.rank == 0:
                        cur_path = os.path.join(self.args.model_path, f"model_gen_phase_{alter+1}_epoch_{id_epoch+1}.pt")
                        torch.save(self.model_gen.module.state_dict(), cur_path)
                        logging.info(f"Save the current ID model to {cur_path}")

                    if self.rank == 0:
                        train_epoch_loss = sum(losses)/len(losses)
                        train_losses.append(train_epoch_loss)
                        logging.info(f"The average training loss for id phase {alter+1} epoch {id_epoch+1} is {train_epoch_loss}")

                    global_epoch += 1
                    total_id_epoch += 1

                    if self.args.test_epoch_id > 0:
                        if (global_epoch + 1) % self.args.test_epoch_id == 0:
                            self.model_gen.eval()
                            self.model_rec.eval()
                            self.get_testloader(model_gen=self.model_gen, tokenizer=self.tokenizer, regenerate=True, phase=total_id_epoch)
                            self.test()
            
        else:
            raise NotImplementedError
        return True

    
    def get_testloader(self, model_gen=None, tokenizer=None, regenerate=False, phase=0):
        self.testloaders = []
        datasets = self.args.datasets.split(',')
        tasks = self.args.tasks.split(',')
        if self.test_filtered > 0:
            collator = TestCollator(self.tokenizer)
        else:
            collator = Collator(self.tokenizer)
        for dataset in datasets:
            for task in tasks:
                try:
                    if self.rank == 0:
                        testdata = TestDatasetGen(self.args, dataset, task, model_gen, tokenizer, regenerate, phase)
                        dist.barrier()
                    else:
                        dist.barrier()
                        testdata = TestDatasetGen(self.args, dataset, task, model_gen, tokenizer, False, phase)
                except AttributeError:
                    # If self.rank does not exist, execute this block
                    testdata = TestDatasetGen(self.args, dataset, task, model_gen, tokenizer, regenerate, phase)
                test_sampler = DistributedSampler(testdata)
                testloader = DataLoader(dataset=testdata, sampler=test_sampler, batch_size=self.args.eval_batch_size, collate_fn=collator, shuffle=False)
                self.testloaders.append(testloader)
                
    def test(self, path=None):
        self.model.eval()
        if path:
            self.model.module.load_state_dict(torch.load(path, map_location=self.device))
        for loader in self.testloaders:
            if self.test_filtered > 0:
                if self.test_filtered_batch > 0:
                    self.test_dataset_task_filtered_batch(loader)
                else:
                    assert self.args.eval_batch_size == 1
                    self.test_dataset_task_filtered(loader)
            else:
                self.test_dataset_task(loader)
    
    def test_dataset_task_filtered_batch(self, testloader):
        if self.rank == 0:
            logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)

            candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in candidates
                ]
                )
            
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = batch[5]
                
                prediction = self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num + testloader.dataset.max_positive,
                        num_return_sequences=self.generate_num + testloader.dataset.max_positive,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results_filtered(testloader.dataset.positive, testloader.dataset.id2user, user_idx.detach().cpu().numpy(), \
                                                            self.generate_num+testloader.dataset.max_positive, \
                                                            generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
    
    def test_dataset_task_filtered(self, testloader):
        if self.rank == 0:
            logging.info(f'testing filtered {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = set(testloader.dataset.all_items)
            
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)
                user_idx = int(batch[5][0])
                positive = testloader.dataset.positive[testloader.dataset.id2user[user_idx]]
                
                user_candidate = candidates - positive
                
                candidate_trie = gt.Trie(
                [
                    [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                    for candidate in user_candidate
                ]
                )
                prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
                
                prediction = self.model.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        whole_word_ids=whole_input_ids,
                        max_length=30,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')
            
    def test_dataset_task(self, testloader):
        if self.rank == 0:
            logging.info(f'testing {testloader.dataset.dataset} dataset on {testloader.dataset.task} task')
        test_total = 0
        with torch.no_grad():
            candidates = testloader.dataset.all_items
            if self.args.his_prefix:
                candidate_trie = gt.Trie(
                    [
                        [0] + self.tokenizer.encode(f"{testloader.dataset.dataset} item_{candidate}")
                        for candidate in candidates
                    ]
                )
            else:
                    candidate_trie = gt.Trie(
                    [
                        [0] + self.tokenizer.encode(f"{candidate}")
                        for candidate in candidates
                    ]
                )
            prefix_allowed_tokens = gt.prefix_allowed_tokens_fn(candidate_trie)
            
            metrics_res = np.array([0.0] * len(self.metrics))
            for batch in tqdm(testloader):
                input_ids = batch[0].to(self.device)
                attn = batch[1].to(self.device)
                whole_input_ids = batch[2].to(self.device)
                output_ids = batch[3].to(self.device)
                output_attention = batch[4].to(self.device)

                prediction = self.model_rec.module.generate(
                        input_ids=input_ids,
                        attention_mask=attn,
                        max_length=50,
                        prefix_allowed_tokens_fn=prefix_allowed_tokens,
                        num_beams=self.generate_num,
                        num_return_sequences=self.generate_num,
                        output_scores=True,
                        return_dict_in_generate=True,
                    )
                
                prediction_ids = prediction["sequences"]
                prediction_scores = prediction["sequences_scores"]
                
                gold_sents = self.tokenizer.batch_decode(
                    output_ids, skip_special_tokens=True
                )
                generated_sents = self.tokenizer.batch_decode(
                    prediction_ids, skip_special_tokens=True
                )
                
                rel_results = evaluate.rel_results(generated_sents, gold_sents, prediction_scores, self.generate_num)
                
                test_total += len(rel_results)
                
                metrics_res += evaluate.get_metrics_results(rel_results, self.metrics)
                
            dist.barrier()
            metrics_res = torch.tensor(metrics_res).to(self.device)
            test_total = torch.tensor(test_total).to(self.device)
            dist.all_reduce(metrics_res, op=dist.ReduceOp.SUM)
            dist.all_reduce(test_total, op=dist.ReduceOp.SUM)
            
            metrics_res /= test_total
            
            if self.rank == 0:
                for i in range(len(self.metrics)):
                    logging.info(f'{self.metrics[i]}: {metrics_res[i]}')


    def insert_phrases_batch(self, prompt, positions, hist, max_input_len):
        """
        prompt: [batch_size, seq_len, emb_size] - embedding of the template sentence
        hist: [batch_size, phrase_num, 10, emb_size] - embeddings of the hist
        positions: [batch_size, seq_len] - binary tensor where "1" indicates insertion points
        max_input_len: int - the maximum length after processing
        """
        batch_size, seq_len, emb_size = prompt.shape
        
        batch_results = []
        
        # Iterate through each example in the batch
        for b in range(batch_size):
            result = []
            hist_idx = 0

            for i in range(seq_len):
                if positions[b, i] == 1:
                    result.append(prompt[b, i].unsqueeze(0))  
                    result.append(hist[b, hist_idx])
                    hist_idx += 1
                else:
                    result.append(prompt[b, i].unsqueeze(0))

            result_tensor = torch.cat(result, dim=0)
            
            # Pad the tensor to max_input_len
            pad_size = max_input_len - result_tensor.shape[0]
            pad_tensor = torch.zeros((pad_size, emb_size)).to(self.device)
            result_tensor = torch.cat([result_tensor, pad_tensor], dim=0)
            
            batch_results.append(result_tensor)

        # Concatenate batch_results to get final tensor
        final_tensor = torch.stack(batch_results, dim=0)
        
        return final_tensor


    def insert_phrases_batch_target(self, prompt, positions, target, max_input_len):
        """
        prompt: [batch_size, seq_len] - embedding of the template sentence
        target: [batch_size, 10] - embeddings of the hist
        positions: [batch_size, seq_len] - binary tensor where "1" indicates insertion points
        max_input_len: int - the maximum length after processing
        """
        batch_size, seq_len = prompt.shape
        
        batch_results = []
        
        # Iterate through each example in the batch
        for b in range(batch_size):
            result = []
            target_idx = 0
            for i in range(seq_len):
                if positions[b, i+1] == 1:
                    result.extend(target[b, target_idx].tolist())
                    break
                else:
                    result.append(prompt[b, i].item())

            result += [self.tokenizer.pad_token_id] * (max_input_len - len(result))
            batch_results.append(result)

        final_tensor = torch.tensor(batch_results).to(self.device)
        return final_tensor


# def insert_phrases_batch_target(prompt, positions, target, max_input_len, device):
#     """
#     prompt: [batch_size, seq_len] - embedding of the template sentence
#     target: [batch_size, 10] - embeddings of the hist
#     positions: [batch_size, seq_len] - binary tensor where "1" indicates insertion points
#     max_input_len: int - the maximum length after processing
#     """
#     batch_size, seq_len = prompt.shape
#     # _, phrase_num, _, _ = hist.shape
    
#     batch_results = []
    
#     # Iterate through each example in the batch
#     for b in range(batch_size):
#         result = []
#         target_idx = 0
#         for i in range(seq_len):
#             if positions[b, i] == 1:
#                 result.append(target[b, target_idx])
#                 target_idx+=1
#             else:
#                 result.append(prompt[b, i].unsqueeze(0))

#         # Concatenate to form the tensor for the current example
#         result_tensor = torch.cat(result, dim=0)

#         # Pad the tensor to max_input_len
#         pad_size = max_input_len - result_tensor.shape[0]
#         pad_tensor = torch.zeros(pad_size).to(device)
#         result_tensor = torch.cat([result_tensor, pad_tensor], dim=0)
        
#         batch_results.append(result_tensor)

#     # Concatenate batch_results to get final tensor
#     final_tensor = torch.stack(batch_results, dim=0, dtype=torch.int64)

#     return final_tensor