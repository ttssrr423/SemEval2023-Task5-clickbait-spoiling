#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2021/11/12 14:00
# @Author  : ZMU

import math
import torch
import re
import os
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from transformers import AutoTokenizer, BertTokenizer
from torch.utils.data import (Dataset, DataLoader, SequentialSampler, RandomSampler)
from torch.utils.data import WeightedRandomSampler


label_class = {
    "taskA": 2,
    "taskB": 4,
    "taskC": 11,
}


label_dict_1 = {
    "not sexist": 0,
    "sexist": 1
}

label_dict_2 = {
    "none": 4,
    "1. threats, plans to harm and incitement": 0,
    "2. derogation": 1,
    "3. animosity": 2,
    "4. prejudiced discussions": 3,
}

label_dict_3 = {
    "none": 11,
    "1.1 threats of harm": 0,
    "1.2 incitement and encouragement of harm": 1,
    "2.1 descriptive attacks": 2,
    "2.2 aggressive and emotive attacks": 3,
    "2.3 dehumanising attacks & overt sexual objectification": 4,
    "3.1 casual use of gendered slurs, profanities, and insults": 5,
    "3.2 immutable gender differences and gender stereotypes": 6,
    "3.3 backhanded gendered compliments": 7,
    "3.4 condescending explanations or unwelcome advice": 8,
    "4.1 supporting mistreatment of individual women": 9,
    "4.2 supporting systemic discrimination against women as a group": 10,
}

"""
# 观察测试集可知，TaskB和TaskC计算指标时只计算TaskA中是歧视数据的那一部分
# 因此TaskB和TaskC分类类别为4,11
"""


def read_dataset(args):
    # 读取训练集和验证集数据
    train = pd.DataFrame()
    train_df = pd.read_csv(args.train_df, sep=',')
    train_df.columns = ["rewire_id", "text", "label_sexist", "label_category", "label_vector"]
    train["text"] = train_df["text"].apply(lambda x: clean_text(x))
    train["taskA_label"] = train_df["label_sexist"].apply(lambda x: label_dict_1[x])
    train["taskB_label"] = train_df["label_category"].apply(lambda x: label_dict_2[x])
    train["taskC_label"] = train_df["label_vector"].apply(lambda x: label_dict_3[x])
    train = train.sample(frac=1).reset_index(drop=True)
    train = train[train["taskC_label"] != 11]
    train = train.sample(frac=1).reset_index(drop=True)
    return train

def read_test_dataset(args):
    # 读取测试数据集
    test = pd.DataFrame()
    test_df = pd.read_csv(args.test_df, sep=',')
    test_df.columns = ["rewire_id", "text"]
    test["id"] = test_df["rewire_id"]
    test["text"] = test_df["text"].apply(lambda x: clean_text(x))
    return test


def clean_text(text):
    # 推特句柄
    text = re.sub('@[\w]*', '[user]', str(text))
    text = re.sub(r'(https?|ftp|file)?:\/\/(\w|\.|\/|\?|\=|\&|\%)*\b', '[url]', text, flags=re.MULTILINE)
    text = re.sub(r'(https?|ftp|file)?:\\/\\/(\w|\.|\\/|\?|\=|\&|\%)*\b', '[url]', text, flags=re.MULTILINE)
    text = re.sub(r'(https?|ftp|file)?\\/\\/(\w|\.|\\/|\?|\=|\&|\%)*\b', '[url]', text, flags=re.MULTILINE)
    text = re.sub(r'(https?|ftp|file)?:\\/\\/', '[url]', text, flags=re.MULTILINE)
    text = re.sub(r'(https?|ftp|file)?:', '[url]', text, flags=re.MULTILINE)

    text = re.sub('\n', '', text)

    # 添加一些缩略词替换
    text = re.sub(' u ', ' you ', text)
    text = re.sub(' ur', ' your', text)
    text = re.sub('btw', 'by the way', text)
    text = re.sub('gosh', 'god', text)
    text = re.sub('omg', 'oh my god', text)
    text = re.sub(' 4 ', ' for ', text)
    text = re.sub('sry', 'sorry', text)
    text = re.sub('idk', 'i do not know', text)


    text = re.sub(r"’s", " is", text)
    text = re.sub(r"'s", " is", text)
    text = re.sub(r"can\'t", " can not", text)
    text = re.sub(r"cannot", " can not", text)
    text = re.sub(r"ca 't", " can not", text)
    text = re.sub(r"wo n\'t", " can not", text)
    text = re.sub(r"wo n't", " can not", text)
    text = re.sub(r"what\'s", " what is", text)
    text = re.sub(r"What\'s", " What is", text)
    text = re.sub(r"what 's", " what is", text)
    text = re.sub(r"What 's", " what is", text)
    text = re.sub(r"how 's", " how is", text)
    text = re.sub(r"How 's", " How is", text)
    text = re.sub(r"how \'s", " how is", text)
    text = re.sub(r"How \'s", " How is", text)
    text = re.sub(r"it \'s", " it is", text)
    text = re.sub(r"it \'s", " it is", text)
    text = re.sub(r"i\'m", "i am", text)
    text = re.sub(r"I\'m", "i am", text)
    text = re.sub(r"i \'m", "i am", text)
    text = re.sub(r"i’m", "i am", text)
    text = re.sub(r"i'm", "i am", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"\'re", " are", text)
    text = re.sub(r"'re", " are", text)
    text = re.sub(r"\'d", " would", text)
    text = re.sub(r"'d", " would", text)
    text = re.sub(r"\'d", "would", text)
    text = re.sub(r"'d", "would", text)
    text = re.sub(r"\'ll", " will", text)
    text = re.sub(r"'ll", " will", text)
    text = re.sub(r" e mail ", " email ", text)
    text = re.sub(r" e \- mail ", " email ", text)
    text = re.sub(r" e\-mail ", " email ", text)
    text = re.sub(r"\'ve ", " have ", text)
    text = re.sub(r"'ve ", " have ", text)
    text = re.sub(r"n\'t", " not", text)
    text = re.sub(r"n’t", " not", text)
    text = re.sub(r"’ll", " will", text)
    text = re.sub(r"3m , h&amp:m and c&amp", " ", text)
    text = re.sub(r"&amp: #x27 : s", " ", text)
    text = re.sub(r"at&amp:", " ", text)
    text = re.sub(r"q&amp", " ", text)
    text = re.sub(r"&amp", " ", text)
    text = re.sub(r"ph\.d", "phd", text)
    text = re.sub(r"PhD", "phd", text)
    text = re.sub(r" e g ", " eg ", text)
    text = re.sub(r" fb ", "facebook", text)
    text = re.sub(r"facebooks", "facebook", text)
    text = re.sub(r"facebooking", "facebook", text)
    text = re.sub(r" usa ", " america ", text)
    text = re.sub(r" us ", " america ", text)
    text = re.sub(r" u s ", " america ", text)
    text = re.sub(r" U\.S\. ", " america ", text)
    text = re.sub(r" US ", " america ", text)
    text = re.sub(r" American ", " america ", text)
    text = re.sub(r" America ", " america ", text)
    text = re.sub(r" mbp ", " macbook-pro ", text)
    text = re.sub(r" mac ", " macbook ", text)
    text = re.sub(r"macbook pro", "macbook-pro", text)
    text = re.sub(r"macbook-pros", "macbook-pro", text)
    text = re.sub(r"googling", " google ", text)
    text = re.sub(r"googled", " google ", text)
    text = re.sub(r"googleable", " google ", text)
    text = re.sub(r"googles", " google ", text)
    text = re.sub(r"dollars", " dollar ", text)
    text = re.sub(r"donald trump", "trump", text)
    text = re.sub(r" u\.n\.", "un", text)
    text = re.sub(r" c\.i\.a\.", "cia", text)
    text = re.sub(r" d\.c\.", "dc", text)
    text = re.sub(r" n\.j\.", "nj", text)
    text = re.sub(r" f\.c\.", "fc", text)
    text = re.sub(r" h\.r\.", "hr", text)
    text = re.sub(r" l\.a\.", "la", text)
    text = re.sub(r" u\.k\.", "uk", text)
    text = re.sub(r" p\.f\.", "pf", text)
    text = re.sub(r" h\.w\.", "hw", text)
    text = re.sub(r" n\.f\.l\.", "nfl", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"\+", " + ", text)
    text = re.sub(r"'", "", text)
    text = re.sub(r"-", " - ", text)
    text = re.sub(r"/", " / ", text)
    text = re.sub(r"\\", " \ ", text)
    text = re.sub(r"=", " = ", text)
    text = re.sub(r"\^", " ^ ", text)
    text = re.sub(r" : ", " : ", text)
    text = re.sub(r"\.", " . ", text)
    text = re.sub(r",", " , ", text)
    text = re.sub(r"\?", " ? ", text)
    text = re.sub(r"!", " ! ", text)
    text = re.sub(r"\"", " \" ", text)
    text = re.sub(r"&", " & ", text)
    text = re.sub(r"\|", " | ", text)
    text = re.sub(r";", " ; ", text)
    text = re.sub(r"\(", " ( ", text)
    text = re.sub(r"\)", " ( ", text)
    text = re.sub(r"&", "and", text)
    text = re.sub(r"\|", " or ", text)
    text = re.sub(r"=", " equal ", text)
    text = re.sub(r"\+", " plus ", text)
    text = re.sub(r"\$", " dollar ", text)
    text = re.sub(r" [b-hB-H] ", " ", text)
    text = re.sub(r" [J-ZJ-Z] ", " ", text)
    text = re.sub(r"oooo", " ", text)
    text = re.sub(r"#", " #", text)

    return text


class SemEval2023_Task10_1_Dataset(torch.utils.data.Dataset):

    def __init__(self, args, df):
        self.args = args
        self.df = df
        self.labeled = "taskC_label" in df
        self.tokenizer = tokenizer
        self.max_length = args.max_sequence_length

    def __getitem__(self, index):
        row = self.df.iloc[index]
        if self.args.dynamic_padding == True:
            encoded_output = self.get_input_data_D(row)
        else:
            encoded_output = self.get_input_data(row)

        if self.labeled:
            label = self.get_label(row.taskC_label)
            encoded_output["label"] = label

        return encoded_output

    def __len__(self):
        return len(self.df)

    def get_input_data_D(self, data):

        input = data["text"]
        encoded_output = self.tokenizer.encode_plus(
            input,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True,
        )

        curr_sent = {}
        curr_sent["input_ids"] = encoded_output["input_ids"]
        curr_sent["token_type_ids"] = encoded_output["token_type_ids"]
        curr_sent["attention_mask"] = encoded_output["attention_mask"]

        return curr_sent

    def get_input_data(self, data):

        input = data["text"]
        encoded_output = self.tokenizer.encode_plus(
            input,
            return_attention_mask=True,
            return_token_type_ids=True,
            add_special_tokens=True,
            padding="max_length",
            max_length=self.max_length,
            truncation_strategy = "longest_first",
        )

        curr_sent = {}
        curr_sent["input_ids"] = torch.tensor(encoded_output["input_ids"], dtype=torch.long)
        curr_sent["token_type_ids"] = torch.tensor(encoded_output["token_type_ids"], dtype=torch.long)
        curr_sent["attention_mask"] = torch.tensor(encoded_output["attention_mask"], dtype=torch.long)
        return curr_sent

    def get_label(self, row):
        label = torch.tensor(int(row))
        return label


def get_train_val_dataloader(args, train_df, train_idx, val_idx):

    train_data = train_df.iloc[train_idx]
    valid_data = train_df.iloc[val_idx]
    # SampleWeight
    temp = train_data.groupby('taskC_label').count()['text'].reset_index().sort_values(by='text', ascending=False)
    label_frequency = {}
    sample_weights = []

    item_class = 11

    for item in range(item_class):
        label_frequency[temp["taskC_label"][item]] = temp["text"][item]

    for item in train_data["taskC_label"]:
        weight = label_frequency[item]
        samp_weight = math.sqrt(
            1 / weight
        )
        sample_weights.append(samp_weight)

    global tokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path, cache_dir=args.cache_dir)
    train_dataset = SemEval2023_Task10_1_Dataset(args, train_data)
    valid_dataset = SemEval2023_Task10_1_Dataset(args, valid_data)

    if args.use_weighted_sampler:
        train_sampler = WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)
        valid_sampler = SequentialSampler(valid_dataset)

    else:
        train_sampler = RandomSampler(train_dataset)
        valid_sampler = SequentialSampler(valid_dataset)

    if args.dynamic_padding == True:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_fn
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.eval_batch_size,
            sampler=valid_sampler,
            pin_memory=True,
            drop_last=False,
            collate_fn=collate_fn
        )

    else:
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.train_batch_size,
            sampler=train_sampler,
            pin_memory=True,
            drop_last=False,
            num_workers=0,
        )

        valid_loader = DataLoader(
            valid_dataset,
            batch_size=args.eval_batch_size,
            sampler=valid_sampler,
            pin_memory=True,
            drop_last=False,
        )
    return train_loader, valid_loader, tokenizer


def get_test_dataloader(args, test_df):

    global tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.pretrain_model_path, cache_dir=args.cache_dir)
    test_dataset = SemEval2023_Task10_1_Dataset(args, test_df)
    test_sampler = SequentialSampler(test_dataset)

    if args.dynamic_padding == True:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.test_batch_size,
            sampler=test_sampler,
            pin_memory=True,
            drop_last=False,
            num_workers=0,
            collate_fn=collate_fn_t
        )

    else:
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.eval_batch_size,
            sampler=test_sampler,
            pin_memory=True,
            drop_last=False,
            num_workers=0,
        )

    dataloaders_dict = {"test": test_loader}
    return dataloaders_dict, tokenizer


def collate_fn(batch):
    max_len = 0
    for i in batch:
        if len(i["input_ids"]) > max_len:
            max_len = len(i["input_ids"])

    input_ids = []
    token_type_ids = []
    attention_mask = []
    label = []
    id = []

    for j in batch:
        padding_length = max_len - len(j["input_ids"])
        input_ids.append(j['input_ids'] +[tokenizer.pad_token_id] * padding_length)
        token_type_ids.append(j['token_type_ids'] + [0] * padding_length)
        attention_mask.append(j['attention_mask'] + [0] * padding_length)
        label.append(j["label"])
        id.append(j["id"])

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
        'label': torch.tensor(label, dtype=torch.long),
        'id': torch.tensor(id, dtype=torch.long)
    }


def collate_fn_t(batch):

    max_len = 0
    for i in batch:
        if len(i["input_ids"]) > max_len:
            max_len = len(i["input_ids"])

    input_ids = []
    token_type_ids = []
    attention_mask = []

    for j in batch:
        padding_length = max_len - len(j["input_ids"])
        input_ids.append(j['input_ids'] +[tokenizer.pad_token_id] * padding_length)
        token_type_ids.append(j['token_type_ids'] + [0] * padding_length)
        attention_mask.append(j['attention_mask'] + [0] * padding_length)

    return {
        'input_ids': torch.tensor(input_ids, dtype=torch.long),
        'token_type_ids': torch.tensor(token_type_ids, dtype=torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
    }

def test_model(args, model, test_dataloader, n_splits):
    prediction_sum = []
    for fold in range(n_splits):
        model.cuda()
        output_dir = os.path.join(args.model_save_path, f"pytorch_model_{fold}.bin")
        model.load_state_dict(torch.load(output_dir))
        model.eval()
        prediction = []
        for batch in test_dataloader["test"]:
            model.eval()
            with torch.no_grad():
                input_ids = batch['input_ids'].cuda()
                attention_mask = batch['attention_mask'].cuda()
                token_type_ids = batch['token_type_ids'].cuda()
                output = model(input_ids, attention_mask, token_type_ids, labels=None)
                _, logits = output[0], output[1]
                logits = F.softmax(logits, dim=1).cpu().detach().numpy()
                if prediction == []:
                    prediction = logits
                else:
                    prediction = np.vstack((prediction, logits))
        if prediction_sum == []:
            prediction_sum = prediction
        else:
            prediction_sum += prediction

    logits = prediction_sum/n_splits
    pd.DataFrame(logits).to_csv("output/test_logits.csv")
    test_df = np.argmax(logits, axis=1).tolist()
    label_dict_3 = {
        0: "1.1 threats of harm",
        1: "1.2 incitement and encouragement of harm",
        2: "2.1 descriptive attacks",
        3: "2.2 aggressive and emotive attacks",
        4: "2.3 dehumanising attacks & overt sexual objectification",
        5: "3.1 casual use of gendered slurs, profanities, and insults",
        6: "3.2 immutable gender differences and gender stereotypes",
        7: "3.3 backhanded gendered compliments",
        8: "3.4 condescending explanations or unwelcome advice",
        9: "4.1 supporting mistreatment of individual women",
        10: "4.2 supporting systemic discrimination against women as a group",
    }

    test_submission = []
    for item in test_df:
        test_submission.append(label_dict_3[item])
    submission = pd.read_csv("data/official data/starting_ki/EXAMPLE_SUBMISSION_dev_task_c.csv")
    submission["label_pred"] = test_submission
    pd.DataFrame(submission).to_csv(args.submission_path, sep=',', index=False)

if __name__ == "__main__":
    pass