from hyperparams import args, PROJECT_ROOT, DEFAULT_MODEL
from data_process.sampler import Sampler
from modelling.postprocess import postprocess_qa_predictions, PredictionOutput, \
    postprocess_qa_predictions_using_score_metric
from pipeline import setup_model, input_data_to_squad_format, setup_features
from training_utils import MeasureTracker, adjust_lr_optimizer
import random
import json
import pandas as pd
import numpy as np
from data_process.utils import EvalPrediction
from modelling.losses import calc_metric_contrastive_losses, calc_modifier_losses, calc_metric_contrastive_losses2
import torch
import os
from datasets import Dataset
from rouge import Rouge
from nltk.translate.bleu_score import sentence_bleu
from utils.lookahead import Lookahead
from utils.at_training import FGM, PGD, AWP
from utils.function_utils import make_optimizer, make_scheduler
import datetime, time
import evaluate

rouge = Rouge()

data_from_pickle = False
TRN_BATCH_SIZE = 4
# TRN_MAX_EPOCH = 5
# E:\\UbuntuFile\\PycharmProjects\\Task5\\models\\deberta
eval_model_path = None  # "E:\\UbuntuFile\\PycharmProjects\\Task5\\models\\saved\\117-79-9-0.165661-0.453954-model.bin"
TRN_MAX_EPOCH = 8
use_abs_pos_embedding = True
use_pre_abs_pos = False

def run_training(train_filepath, val_filepath, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_inp = [json.loads(i) for i in open(train_filepath, "r", encoding="utf8")]
    val_inp = [json.loads(i) for i in open(val_filepath, "r", encoding="utf8")]

    trn_dataset = input_data_to_squad_format(train_inp, load_spoiler=True)
    val_dataset = input_data_to_squad_format(val_inp, load_spoiler=True)
    config, tokenizer, model, args_read = setup_model(
        ['--model_name_or_path', model_path, '--output_dir', os.path.join(PROJECT_ROOT, 'tmp/ignored')],
        using_absolute_position=use_pre_abs_pos)
    model_args, data_args, training_args = args_read
    model = model.to(device)
    if eval_model_path is not None:
        model.load_state_dict(torch.load(open(eval_model_path, mode="rb")))
    # optim = torch.optim.AdamW(model.parameters(), lr=1e-6)
    model_args.optimizer_type = "AdamW"
    model_args.hidden_size = config.hidden_size
    model_args.weight_decay = 1e-2
    model_args.learning_rate = 1e-5
    model_args.epsilon = 1e-8
    optim = Lookahead(make_optimizer(model_args, model, is_model_nested=False), k=5, alpha=0.5)

    if not data_from_pickle:
        features_val, features_trn = setup_features(tokenizer, val_dataset, train_dataset=trn_dataset)
        features_val.save_to_disk(os.path.join(PROJECT_ROOT, "dataset", "val_dataset"))
        features_trn.save_to_disk(os.path.join(PROJECT_ROOT, "dataset", "trn_dataset"))
    else:
        features_val = Dataset.load_from_disk(os.path.join(PROJECT_ROOT, "dataset", "val_dataset"))
        features_trn = Dataset.load_from_disk(os.path.join(PROJECT_ROOT, "dataset", "trn_dataset"))
    sampler_train = Sampler()
    sampler_train.load_dataset(features_trn, features_val)

    iter_max = TRN_MAX_EPOCH * len(sampler_train.train_dataset) // TRN_BATCH_SIZE
    scheduler = make_scheduler(optim, training_args, decay_name="linear_schedule_with_warmup", t_max=iter_max,
                               warmup_steps=60)

    adv_trainer = AWP(model, optim, False, adv_lr=1e-4, adv_eps=1e-2)

    def evaluate_training_model(save_model=False, ep_num=0):
        start_lgts = []
        end_lgts = []
        score_metrics = []
        is_pred_correct_all = []
        model.eval()
        for itm in sampler_train.val_batch_sampling(batch_size=1, dataset=sampler_train.val_dataset, only_first=True):
            input_ids = torch.tensor(itm["input_ids"], dtype=torch.long, device=device)
            token_type_ids = torch.tensor(itm["token_type_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(itm["attention_mask"], dtype=torch.long, device=device)
            paragraph_ids = torch.tensor(itm["paragraph_ids"], dtype=torch.long, device=device)
            absolute_position_ids = torch.tensor(itm["absolute_positions"], dtype=torch.long, device=device)
            absolute_position_ids_plus1 = torch.clip(torch.ones_like(absolute_position_ids) + absolute_position_ids, 0, config.max_position_embeddings-1)
            with torch.no_grad():
                predicted = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                  paragraph_ids=paragraph_ids, position_ids=None if not use_pre_abs_pos else absolute_position_ids_plus1, absolute_position_ids=absolute_position_ids, use_abs_pos_embedding=use_abs_pos_embedding)
            # start_lgts.append(predicted.start_logits.detach().to("cpu").numpy())
            # end_lgts.append(predicted.end_logits.detach().to("cpu").numpy())
            # score_metrics.append(predicted.scoring_metric.detach().to("cpu").numpy())

            if hasattr(predicted, "type_classify_logits") and predicted.type_classify_logits is not None:
                predicted_type_ids = torch.argmax(torch.softmax(predicted.type_classify_logits, dim=-1), dim=-1)
                if "tags" in itm:
                    tgt_tags = torch.tensor(itm["tags"], dtype=torch.long, device=device)
                    is_pred_correct = predicted_type_ids.eq(tgt_tags[:, 0])
                    is_pred_correct_all.append(is_pred_correct)

            # cheat_st = np.array(itm["start_target"],dtype=np.float)
            # cheat_ed = np.array(itm["end_target"],dtype=np.float)
            # start_lgts.append(cheat_st)
            # end_lgts.append(cheat_ed)
            # score_metrics.append(cheat_st[:,:,None]+cheat_ed[:, None, :])

        if len(is_pred_correct_all) > 0:
            type_is_correct = torch.stack(is_pred_correct_all, dim=0).float()
            type_accuracy = type_is_correct.sum() / type_is_correct.shape[0]
            print("type predict accuracy: ", type_accuracy)
            with open("result_file.txt", mode="a", encoding="utf8") as fw:
                fw.write("type predict accuracy: " + str(type_accuracy) + "\n")
            type_accuracy = float(type_accuracy.clone().detach().to("cpu"))
        else:
            type_accuracy = -1.0

        if save_model:
            dt = datetime.datetime.now()
            print("saving model")
            file_str = "{0}{1}-{2}{3}-{4}-{5}-type_pred_model.bin".format(dt.month, dt.day, dt.hour, dt.minute, ep_num, str(type_accuracy))
            torch.save(model.state_dict(), open(os.path.join(PROJECT_ROOT, "models", "saved", file_str), mode="wb"))
        model.train()
        return 1 #PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=None)

    def multi_softmax(inputs, target_labels):
        inp_exp = torch.exp(inputs)
        zero_partitions = inp_exp.sum(dim=1, keepdim=True).repeat([1, inputs.shape[1]])
        neg_exps = torch.where(target_labels.eq(1), torch.zeros_like(inp_exp), inp_exp)
        one_partitions = neg_exps.sum(dim=1, keepdim=True) + inp_exp
        partitions = torch.where(target_labels.eq(1), one_partitions, zero_partitions)
        probabilities = inp_exp / partitions
        return probabilities

    def is_nearby(ranking_coord, gt_coords_dict):
        for _gt_coord in gt_coords_dict.keys():
            if _gt_coord[1] - _gt_coord[0] >= 4 and (
                    abs(_gt_coord[0] - ranking_coord[0]) + abs(_gt_coord[1] - ranking_coord[1]) <= 2):
                return _gt_coord
        return None

    def calc_losses(pred_out, input_feat, tgt_info, negative_sample_num=-1):
        if hasattr(pred_out, "type_classify_logits") and pred_out.type_classify_logits is not None:
            type_prob = torch.softmax(pred_out.type_classify_logits, dim=-1)
            is_first_feature = input_feat["is_first_feature"]
            type_ids = tgt_info["tags"].view(-1, 1)
            # focal_gamma = 0.6 # * torch.pow((1-start_prob), focal_gamma) # * torch.pow((1-end_prob), focal_gamma)
            tags_one_hot_tgt = torch.zeros([type_ids.shape[0], 3], dtype=torch.float32, device=type_prob.device)
            tags_one_hot_tgt.scatter_(1, type_ids.data, 1.)
            ce_type = - tags_one_hot_tgt * torch.log(type_prob)
            ce_type_loss = ce_type.sum(dim=-1).mean()
        else:
            raise Exception("classify loss should be calculated...")
        return ce_type_loss, ""

    iter_i = 0
    grad_id = 0
    grad_accumulate_steps = 5
    if TRN_MAX_EPOCH > 0:
        model.train()
        for itm in sampler_train.train_batch_sampling(batch_size=TRN_BATCH_SIZE, max_epoch=TRN_MAX_EPOCH, only_first=True):
            input_ids = torch.tensor(itm["input_ids"], dtype=torch.long, device=device)
            token_type_ids = torch.tensor(itm["token_type_ids"], dtype=torch.long, device=device)
            attention_mask = torch.tensor(itm["attention_mask"], dtype=torch.long, device=device)
            paragraph_ids = torch.tensor(itm["paragraph_ids"], dtype=torch.long, device=device)
            is_first_feature = torch.tensor(itm["is_first_feature"], dtype=torch.float32, device=device)
            absolute_position_ids = torch.tensor(itm["absolute_positions"], dtype=torch.long, device=device)
            absolute_position_ids_plus1 = torch.clip(torch.ones_like(absolute_position_ids) + absolute_position_ids, 0,
                                                     config.max_position_embeddings - 1)
            predicted = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              paragraph_ids=paragraph_ids, position_ids=None if not use_pre_abs_pos else absolute_position_ids_plus1, absolute_position_ids=absolute_position_ids, use_abs_pos_embedding=use_abs_pos_embedding)

            targets = {"start": torch.tensor(itm["start_target"], dtype=torch.float32, device=device),
                       "end": torch.tensor(itm["end_target"], dtype=torch.float32, device=device),
                       "span_positions": itm["span_pos"],
                       "tags": torch.tensor(itm["tags"], dtype=torch.long, device=device)}
            *losses, track_info = calc_losses(predicted,
                                              {"token_type_ids": token_type_ids, "is_first_feature": is_first_feature},
                                              targets, negative_sample_num=20)

            loss = losses[0] / (grad_id + 1)
            # scaler.scale(loss).backward()
            loss.backward()
            # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model_args.max_grad_norm)

            if sampler_train.current_epoch >= 1:
                with torch.cuda.amp.autocast(enabled=False):
                    adv_trainer._save()
                    adv_trainer._attack_step()
                    predicted = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                      paragraph_ids=paragraph_ids, position_ids=None if not use_pre_abs_pos else absolute_position_ids_plus1,
                                      absolute_position_ids=absolute_position_ids, use_abs_pos_embedding=use_abs_pos_embedding)

                    targets = {"start": torch.tensor(itm["start_target"], dtype=torch.float32, device=device),
                               "end": torch.tensor(itm["end_target"], dtype=torch.float32, device=device),
                               "span_positions": itm["span_pos"],
                               "tags": torch.tensor(itm["tags"], dtype=torch.long, device=device)}
                    *losses, track_info = calc_losses(predicted, {"token_type_ids": token_type_ids,
                                                                  "is_first_feature": is_first_feature}, targets,
                                                      negative_sample_num=20)
                    loss = losses[0]
                    model.zero_grad()
                loss.backward()
                adv_trainer._restore()

            grad_id += 1
            if grad_id >= grad_accumulate_steps:
                # optim.step()
                optim.step()
                scheduler.step()
                model.zero_grad()
                grad_id = 0

            MeasureTracker.track_record(*losses, track_names=["loss"])
            iter_i += 1
            if iter_i % 20 == 0:
                print(iter_i, MeasureTracker.print_measures())
                # batch_ranking_info = track_info["ranking_info"]
                # print("\n".join(batch_ranking_info))

            if itm["is_new_epoch"]:  # or iter_i % 1000 == 0:
                evaluate_training_model(save_model=True, ep_num=sampler_train.current_epoch)

    evaluate_training_model(save_model=True if TRN_MAX_EPOCH > 0 else False, ep_num=sampler_train.current_epoch + 1)
    return


if __name__ == "__main__":
    train_path = os.path.join(PROJECT_ROOT, "dataset", "train.jsonl")
    val_path = os.path.join(PROJECT_ROOT, "dataset", "validation.jsonl")
    run_training(train_path, val_path, DEFAULT_MODEL)