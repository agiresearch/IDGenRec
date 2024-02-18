import torch
import numpy as np

class Collator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        input_texts = [input_text['input'] for input_text in batch]
        output_texts = [input_text['output'] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        whole_word_ids = []
        for input_id in input_ids:
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
            whole_word_ids.append(whole_word_id)
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(whole_word_ids),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )

    
class TestCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        user_idx = [input_text['user_idx'] for input_text in batch]
        input_texts = [input_text['input'] for input_text in batch]
        output_texts = [input_text['output'] for input_text in batch]

        inputs = self.tokenizer.batch_encode_plus(
            input_texts, padding="longest", truncation=True, max_length=512
        )
        input_ids = inputs["input_ids"]
        whole_word_ids = []
        for input_id in input_ids:
            tokenized_text = self.tokenizer.convert_ids_to_tokens(input_id)
            whole_word_id = calculate_whole_word_ids(tokenized_text, input_id)
            whole_word_ids.append(whole_word_id)
        input_attention = inputs["attention_mask"]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]

        return (
            torch.tensor(input_ids),
            torch.tensor(input_attention),
            torch.tensor(whole_word_ids),
            torch.tensor(output_ids),
            torch.tensor(output_attention),
            torch.tensor(user_idx),
        )

    
class CollatorGen:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        output_texts = [input_text['output_prompt'] for input_text in batch]
        outputs = self.tokenizer.batch_encode_plus(
            output_texts, padding="longest", truncation=True, max_length=512
        )
        output_ids = outputs["input_ids"]
        output_attention = outputs["attention_mask"]
        histories = [input_text['history'] for input_text in batch]
        input_prompt = [input_text['input_prompt'] for input_text in batch]
        hist_lengths = [len(hist) for hist in histories]

        # add placeholders to the input prompt
        input_prompt_ph = []  # with placeholders
        input_insert_positions = []
        tokenized_prompts = []  # Store tokenized prompts here
        for i, p in enumerate(input_prompt):
            length = hist_lengths[i]
            p_s = p.replace("{history}", " ; " * (length))
            tokens = self.tokenizer.tokenize(p_s)
            insert_p = [1 if token == ";" else 0 for token in tokens]
            tokenized_prompts.append(tokens)
            input_prompt_ph.append(p_s)
            input_insert_positions.append(insert_p)

        # process input prompt
        input_prompt_inputs = self.tokenizer.batch_encode_plus(
            tokenized_prompts, is_split_into_words=True, padding="longest", truncation=True, max_length=512
        )

        # pad input prompt insert positions
        input_prompt_len = len(input_prompt_inputs['input_ids'][0])
        for insert_p in input_insert_positions:
            while len(insert_p) < input_prompt_len:
                insert_p.append(0)

        # process history
        flattened_histories = [plain_text for hist in histories for plain_text in hist]

        # process input history, need two level of paddings, history level and plain text level
        history_inputs = self.tokenizer.batch_encode_plus(
            flattened_histories, padding="longest", truncation=True, max_length=256)
        max_hist_token = len(history_inputs['input_ids'][0])
        hist_lengths = [len(hist) for hist in histories] 

        # Apply padding at the history level
        padded_histories = []
        padded_attention_mask_histories = []
        max_hist_length = max(hist_lengths)
        current_index = 0

        for length in hist_lengths:
            padded_hist = torch.zeros((max_hist_length, max_hist_token), dtype=torch.long)
            padded_attention_mask = torch.zeros((max_hist_length, max_hist_token))
            padded_hist[:length] = torch.tensor(history_inputs['input_ids'][current_index:current_index+length], dtype=torch.long)
            padded_attention_mask[:length] = torch.tensor(history_inputs['attention_mask'][current_index:current_index+length])
            padded_histories.append(padded_hist)
            padded_attention_mask_histories.append(padded_attention_mask)
            current_index += length
        history_input_ids = torch.stack(padded_histories)
        history_input_attention = torch.stack(padded_attention_mask_histories)

        return (
            torch.tensor(input_prompt_inputs['input_ids']),
            torch.tensor(input_insert_positions),
            # torch.tensor(output_prompt_inputs['input_ids']),
            # torch.tensor(output_insert_positions),
            history_input_ids,
            history_input_attention,
            torch.tensor(output_ids),
            torch.tensor(output_attention),
        )


def calculate_whole_word_ids(tokenized_text, input_ids):
    whole_word_ids = []
    curr = 0
    for i in range(len(tokenized_text)):
        if tokenized_text[i] == "<pad>":
            curr = 0
        if tokenized_text[i].startswith("▁"):
            curr += 1
            whole_word_ids.append(curr)
        else:
            whole_word_ids.append(curr)
    return whole_word_ids[: len(input_ids) - 1] + [0]  # [0] for </s>

