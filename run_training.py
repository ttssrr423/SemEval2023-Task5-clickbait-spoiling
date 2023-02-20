from hyperparams import args, PROJECT_ROOT, DEFAULT_MODEL
from data_process.sampler import Sampler
from modelling.postprocess import postprocess_qa_predictions, PredictionOutput, postprocess_qa_predictions_using_score_metric, postprocess_multi_qa_predictions_using_score_metric
from pipeline import setup_model, input_data_to_squad_format, setup_features
from training_utils import MeasureTracker, adjust_lr_optimizer
import random
import json
import pandas as pd
import numpy as np
from data_process.utils import EvalPrediction
from modelling.losses import calc_metric_contrastive_losses, calc_modifier_losses, calc_metric_contrastive_losses2, calc_metric_contrastive_losses_for_multi
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

data_from_pickle = True
TRN_BATCH_SIZE = 4
# TRN_MAX_EPOCH = 5
# E:\\UbuntuFile\\PycharmProjects\\Task5\\models\\deberta
eval_model_path = None #"E:\\UbuntuFile\\PycharmProjects\\Task5\\models\\saved\\214-177-multi-11-0.305793-0.574750-model.bin"
# eval_model_path = "/home/st491/PycharmProjects/Task5/models/saved/214-054-mix-5-0.390304-0.462820-model.bin"
TRN_MAX_EPOCH = 10
USE_ONLY_CE_LOSS = False #True
USE_SOLID_SPAN_LOSS = False
SPOILER_TYPE = 2
ADV_THRESHOLD = 1
lstm_span_modifier = False # True if SPOILER_TYPE == 1 else False  # False if SPOILER_TYPE != 1 else True
use_antisymetric_modifier = True if SPOILER_TYPE == 2 else False
use_abs_pos_embedding = True if SPOILER_TYPE==2 else False
use_pre_absolute_pos = False
if SPOILER_TYPE == 0:
    spoiler_type_str = "phrase"
    MAX_SPOILER_LEN = 40
elif SPOILER_TYPE == 1:
    spoiler_type_str = "passage"
    MAX_SPOILER_LEN = 200
elif SPOILER_TYPE==2:
    spoiler_type_str = "multi"
    MAX_SPOILER_LEN = 40
else:
    assert SPOILER_TYPE is None
    spoiler_type_str = "mix"
    MAX_SPOILER_LEN = 200


classification_prediction_map = {}
if os.path.exists(os.path.join(PROJECT_ROOT, "models", "inference", "task1_tmp_out.jsonl")):
    task1_json_loaded = json.load(open(os.path.join(PROJECT_ROOT, "models", "inference", "task1_tmp_out.jsonl")))
    for pred_dict in task1_json_loaded:
        classification_prediction_map[pred_dict["uuid"]] = pred_dict

def run_training(train_filepath, val_filepath, model_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_inp = [json.loads(i) for i in open(train_filepath, "r", encoding="utf8")]
    val_inp = [json.loads(i) for i in open(val_filepath, "r", encoding="utf8")]

    trn_dataset = input_data_to_squad_format(train_inp, load_spoiler=True)
    val_dataset = input_data_to_squad_format(val_inp, load_spoiler=True)
    using_absolute_position = use_pre_absolute_pos # False #  False if SPOILER_TYPE != 2 else True
    config, tokenizer, model, args_read = setup_model(['--model_name_or_path', model_path, '--output_dir', os.path.join(PROJECT_ROOT, 'tmp/ignored')], using_absolute_position=using_absolute_position)
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
        print(" data loaded from pickle...")
    sampler_train = Sampler()
    sampler_train.load_dataset(features_trn, features_val)

    iter_max = TRN_MAX_EPOCH * len(sampler_train.train_dataset) // TRN_BATCH_SIZE
    scheduler = make_scheduler(optim, training_args, decay_name="linear_schedule_with_warmup", t_max=iter_max,
                               warmup_steps=60)

    adv_trainer = AWP(model, optim, False, adv_lr=model_args.learning_rate*10.0, adv_eps=1e-2)

    def is_spoiler_type_match(_itm):
        if "tags" in _itm:
            teat_tag_id = _itm["tags"][0][0]
            if spoiler_type_str == "mix" or SPOILER_TYPE == teat_tag_id:
                return 1
            else:
                return 0
        elif "example_id" in _itm:
            feat_uuid = _itm["example_id"][0]
            if feat_uuid in classification_prediction_map:
                task1_pred = classification_prediction_map[feat_uuid]
                feat_pred_type_str = task1_pred["spoilerType"]
                if task1_pred["confidence"] < 0.5:
                    feat_pred_type_str = "mix"
            else:
                feat_pred_type_str = "mix"
            if feat_pred_type_str == spoiler_type_str:
                return 1
            else:
                return 0
        else:
            if "mix" == spoiler_type_str:
                return 1
            else:
                return 0

    # # thresholds = 0.75,0.55,0.35
    # def is_spoiler_type_match(_itm):
    #     if SPOILER_TYPE is None:
    #         return 1
    #     if "example_id" in _itm:
    #         feat_uuid = _itm["example_id"][0]
    #         if feat_uuid in classification_prediction_map:
    #             task1_pred = classification_prediction_map[feat_uuid]
    #             feat_pred_type_str = task1_pred["spoilerType"]
    #             probs = task1_pred["confidences"]
    #             confidences_with_labels = [(_lid, _prob) for _lid, _prob in enumerate(probs)]
    #             sorted_confidences = list(sorted(confidences_with_labels, key=lambda x:x[1], reverse=True))
    #             max_pred = sorted_confidences[0]
    #             if max_pred[0] == 0 and max_pred[1] >= 0.75:
    #                 feat_pred_type_str = "phrase"
    #             elif max_pred[0]==1 and max_pred[1] >= 0.55:
    #                 feat_pred_type_str = "passage"
    #             elif max_pred[0]==2 and max_pred[1] >= 0.35:
    #                 feat_pred_type_str = "multi"
    #             else:
    #                 feat_pred_type_str = "mix"
    #         else:
    #             feat_pred_type_str = "mix"
    #         if feat_pred_type_str == spoiler_type_str:
    #             return 1
    #         else:
    #             return 0
    #     else:
    #         if "mix" == spoiler_type_str:
    #             return 1
    #         else:
    #             return 0

    def evaluate_training_model(save_model=False, ep_num=0):
        start_lgts = []
        end_lgts = []
        score_metrics = []
        is_pred_correct_all = []
        tags_are_matched = []
        model.eval()
        for itm in sampler_train.val_batch_sampling(batch_size=1, dataset=sampler_train.val_dataset):
            is_feature_match_spoiler = is_spoiler_type_match(itm)
            tags_are_matched.append(is_feature_match_spoiler)
            input_ids = torch.tensor(itm["input_ids"],dtype=torch.long,device=device)
            token_type_ids = torch.tensor(itm["token_type_ids"],dtype=torch.long,device=device)
            attention_mask = torch.tensor(itm["attention_mask"],dtype=torch.long,device=device)
            paragraph_ids = torch.tensor(itm["paragraph_ids"], dtype=torch.long, device=device)
            absolute_position_ids = torch.tensor(itm["absolute_positions"], dtype=torch.long, device=device)
            absolute_position_ids_plus1 = torch.clip(torch.ones_like(absolute_position_ids) + absolute_position_ids,
                                                     0, config.max_position_embeddings - 1)
            with torch.no_grad():
                predicted = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=absolute_position_ids_plus1 if use_pre_absolute_pos else None,
                                  paragraph_ids=paragraph_ids, absolute_position_ids=absolute_position_ids, using_modifier=use_antisymetric_modifier,
                                  lstm_span_modifier=lstm_span_modifier, use_abs_pos_embedding=use_abs_pos_embedding, solid_span=USE_SOLID_SPAN_LOSS)
            start_lgts.append(predicted.start_logits.detach().to("cpu").numpy())
            end_lgts.append(predicted.end_logits.detach().to("cpu").numpy())

            if USE_SOLID_SPAN_LOSS:
                solid_probs = predicted.modifier.detach().to("cpu").numpy()
                solid_probs_padded = np.concatenate([np.zeros([1, 1]), solid_probs, np.zeros([1, 1])], axis=1)
                start_raise_list = []
                end_fall_list = []
                for _t in range(solid_probs.shape[1]):
                    # start_raise = solid_probs[0, _t] - solid_probs_padded[0, _t]
                    # end_fall = solid_probs[0, _t] - solid_probs_padded[0, _t+2]
                    start_extension_length = min(_t+3, solid_probs.shape[1]-1) - _t + 1
                    start_raise = np.mean(solid_probs[0, _t:_t+start_extension_length]) - solid_probs_padded[0, _t]
                    end_extension_length = _t - max(0, _t-3)
                    end_fall = np.mean(solid_probs[0, _t-end_extension_length:_t+1]) - solid_probs_padded[0, _t+2]
                    start_raise_list.append(float(start_raise))
                    end_fall_list.append(float(end_fall))
                score_metric = np.expand_dims(np.array(start_raise_list), axis=1) + np.expand_dims(np.array(end_fall_list), axis=0)
                # score_metric *= predicted.scoring_metric[0].detach().to("cpu").numpy()
                score_metrics.append(np.expand_dims(score_metric, axis=0))
            else:
                score_metrics.append(predicted.scoring_metric.detach().to("cpu").numpy())

        scoring_array = np.concatenate(score_metrics, axis=0)
        prediction_tuple = (np.concatenate(start_lgts, axis=0), np.concatenate(end_lgts, axis=0), scoring_array)

        if SPOILER_TYPE != 2:
            post_processed = postprocess_qa_predictions_using_score_metric(val_dataset, features_val, prediction_tuple,
                                                        version_2_with_negative=data_args.version_2_with_negative,
                                                        n_best_size=data_args.n_best_size,
                                                        max_answer_length=MAX_SPOILER_LEN, # data_args.max_answer_length,
                                                        null_score_diff_threshold=data_args.null_score_diff_threshold,
                                                        output_dir=training_args.output_dir,
                                                        prefix="eval",
                                                        tag_restrictions=tags_are_matched)
        else:
            post_processed = postprocess_multi_qa_predictions_using_score_metric(val_dataset, features_val, prediction_tuple,
                                                                           version_2_with_negative=data_args.version_2_with_negative,
                                                                           n_best_size=data_args.n_best_size,
                                                                           max_answer_length=MAX_SPOILER_LEN,
                                                                           # data_args.max_answer_length,
                                                                           null_score_diff_threshold=data_args.null_score_diff_threshold,
                                                                           output_dir=training_args.output_dir,
                                                                           prefix="eval",
                                                                           tag_restrictions=tags_are_matched)
        # Format the result to the format the metric expects.
        if data_args.version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in post_processed.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in post_processed.items()]

        references = [{"id": val_dataset.iloc[eid]["id"], "answers": val_dataset.iloc[eid]["answers"].replace("\n", " ")} for eid in range(len(val_dataset))]
        # references = [{"id": val_dataset.iloc[eid]["id"], "answers": val_dataset.iloc[eid]["answers"]} for eid in range(len(val_dataset))]
        predictions = EvalPrediction(predictions=formatted_predictions, label_ids=references)

        id_map = {}
        for pred_dict in predictions.predictions:
            id_map[pred_dict["id"]] = [pred_dict["prediction_text"]]
        for lab_dict in predictions.label_ids:
            if lab_dict["id"] in id_map:
                id_map[lab_dict["id"]].append(lab_dict["answers"])
        refs = []
        hypos = []
        refs2 = []
        hypos2 = []
        for _k, _pair in id_map.items():
            if len(_pair)==2:
                hypos.append(" "+_pair[0])
                refs.append(" "+_pair[1])
                hypos2.append(_pair[0])
                refs2.append(_pair[1])

        rouge_res = rouge.get_scores(hypos, refs, avg=True)
        print(rouge_res)
        meteor = evaluate.load('meteor')
        meteor_score = meteor.compute(predictions=hypos2, references=refs2)["meteor"]
        print(meteor_score)

        # bleus = []
        # for _hypo, _ref in zip(hypos, refs):
        #     _ref_splited = [[x for x in _ref.split(" ") if len(x)>0]]
        #     _hypo_splited = [x for x in _hypo.split(" ") if len(x)>0]
        #     bleu_score = sentence_bleu(_ref_splited, _hypo_splited, weights=(0.25, 0.25, 0.25, 0.25))
        #     bleus.append(bleu_score)
        # print(sum(bleus)/len(bleus))
        bleus = []
        # bleu = evaluate.load("bleu")
        for _hypo, _ref in zip(hypos, refs):
            _ref_splited = [[x for x in _ref.split(" ") if len(x)>0]]
            _hypo_splited = [x for x in _hypo.split(" ") if len(x)>0]
            bleu_len = min(len(_ref.strip().split(" ")), 4)
            weights = tuple([1.0 / bleu_len] * bleu_len + [0.0]*(4-bleu_len))
            bleu_score = sentence_bleu(_ref_splited, _hypo_splited, weights=weights)
            # bleu_results = bleu.compute(predictions=[_hypo.strip()], references=[[_ref.strip()]], max_order=bleu_len)
            # bleu_score = float(bleu_results["bleu"])
            bleus.append(bleu_score)
        print(sum(bleus)/len(bleus))


        with open("result_file.txt", mode="a", encoding="utf8") as fw:
            fw.write("spoilerType={4}， datetime={3}, bleu={0}, meteor={1}, rouge={2}".format(sum(bleus)/len(bleus), meteor_score, str(rouge_res), datetime.datetime.now(), spoiler_type_str) + "\n")
        # GT: {'rouge-1': {'r': 0.8860411380085066, 'p': 0.9838491006763259, 'f': 0.9033035004796143}, 'rouge-2': {'r': 0.7543655498420224, 'p': 0.8502023739832764, 'f': 0.7685640460466481}, 'rouge-l': {'r': 0.8858669259358569, 'p': 0.9837229906308584, 'f': 0.9031720775906342}}
        # GT: 0.49630818582813824

        # 30 maxlen
        # {'rouge-1': {'r': 0.47197892803978886, 'p': 0.5895729870967438, 'f': 0.47336022368342845},
        #  'rouge-2': {'r': 0.32671174068956665, 'p': 0.41919362416091765, 'f': 0.3266982549164375},
        #  'rouge-l': {'r': 0.4683356709115797, 'p': 0.5845111226464796, 'f': 0.46953086506853753}}
        # 0.11728596960409562

        # 200 maxlen
        # {'rouge-1': {'r': 0.5071461541089293, 'p': 0.5796065094992003, 'f': 0.483601304181905},
        #  'rouge-2': {'r': 0.35917717246007824, 'p': 0.41524215822373944, 'f': 0.3390555221363232},
        #  'rouge-l': {'r': 0.5035038451451365, 'p': 0.5744714529683397, 'f': 0.4797898017603762}}
        # 0.13357444396815243
        model.train()
        if save_model:
            dt = datetime.datetime.now()
            file_str = "{0}{1}-{2}{3}-{7}-{4}-{5}-{6}-model.bin".format(dt.month,dt.day,dt.hour,dt.minute, ep_num, format(sum(bleus)/len(bleus), ".6f"), format(meteor_score, ".6f"), spoiler_type_str)
            torch.save(model.state_dict(), open(os.path.join(PROJECT_ROOT, "models", "saved", file_str), mode="wb"))
        return PredictionOutput(predictions=predictions.predictions, label_ids=predictions.label_ids, metrics=None)

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
            if _gt_coord[1]-_gt_coord[0] >= 4 and (abs(_gt_coord[0]-ranking_coord[0])+abs(_gt_coord[1]-ranking_coord[1]) <= 2):
                return _gt_coord
        return None


    def calc_losses(pred_out, input_feat, tgt_info, negative_sample_num=-1):
        # if hasattr(pred_out, "type_classify_logits") and pred_out.type_classify_logits is not None:
        #     type_prob = torch.softmax(pred_out.type_classify_logits, dim=-1)
        #     is_first_feature = input_feat["is_first_feature"]
        #     type_ids = tgt_info["tags"].view(-1, 1)
        #     tags_one_hot_tgt = torch.zeros([type_ids.shape[0], 3], dtype=torch.float32,device=type_prob.device)
        #     tags_one_hot_tgt.scatter_(1, type_ids.data, 1.)
        #     ce_type = - tags_one_hot_tgt * torch.log(type_prob)
        #     ce_type_loss = ce_type.sum(dim=-1).mean()

        token_type_ids = input_feat["token_type_ids"]

        if SPOILER_TYPE == 2 or SPOILER_TYPE is None:
            start_prob = multi_softmax(pred_out.start_logits, tgt_info["start"])
            end_prob = multi_softmax(pred_out.end_logits, tgt_info["end"])
        else:
            start_prob = torch.softmax(pred_out.start_logits, dim=-1)
            end_prob = torch.softmax(pred_out.end_logits, dim=-1)

        # focal_gamma = 0.6 # * torch.pow((1-start_prob), focal_gamma) # * torch.pow((1-end_prob), focal_gamma)
        ce_start_loss = - tgt_info["start"] * torch.log(start_prob)
        ce_end_loss = - tgt_info["end"] * torch.log(end_prob)
        avg_ce_losses = ce_start_loss.sum(dim=1) + ce_end_loss.sum(dim=1)
        avg_ce_loss = avg_ce_losses.sum() / (tgt_info["start"].sum()+tgt_info["end"].sum())

        if USE_ONLY_CE_LOSS:
            total_loss = avg_ce_loss
            return total_loss, avg_ce_loss, torch.zeros_like(total_loss), {"ranking_info": [""]*TRN_BATCH_SIZE}
        elif USE_SOLID_SPAN_LOSS:
            modifier_prob_seq = pred_out.modifier
            start_id = torch.argmax(tgt_info["start"], dim=1).detach().to("cpu").numpy()
            end_id = torch.argmax(tgt_info["end"], dim=1).detach().to("cpu").numpy()
            solid_targets = torch.zeros_like(modifier_prob_seq)
            solid_weights = torch.ones_like(modifier_prob_seq)
            for bi in range(start_id.shape[0]):
                for _t in range(start_id[bi], end_id[bi]+1):
                    solid_targets[bi, _t].fill_(1.0)
                solid_weights[bi, start_id[bi]].fill_(5.0)
                solid_weights[bi, end_id[bi]].fill_(5.0)

            focal_gamma =0.2 # torch.pow((1-modifier_prob_seq), focal_gamma) * # torch.pow(modifier_prob_seq, focal_gamma) *
            # balance_factor = min(solid_targets.shape[0]*solid_targets.shape[1] / float(torch.sum(solid_targets).detach().to("cpu")), 100.0)
            solid_ce_losses = - solid_targets * torch.pow((1-modifier_prob_seq), focal_gamma) * torch.log(modifier_prob_seq) - (1-solid_targets) * torch.pow(modifier_prob_seq, focal_gamma) * torch.log(torch.ones_like(modifier_prob_seq)-modifier_prob_seq)

            # total_loss = solid_ce_losses.mean() # + 0.1 * avg_ce_loss
            total_loss = (solid_ce_losses * solid_weights).sum() / solid_weights.sum()
            return total_loss, avg_ce_loss, torch.zeros_like(total_loss), {"ranking_info": [""] * TRN_BATCH_SIZE}

        null_scores = list((pred_out.start_logits[:, 0] + pred_out.end_logits[:, 0]).clone().detach().to("cpu").numpy())
        advantages_over_null = []

        advantage_losses = []
        batch_ranking_info = []

        for bi in range(tgt_info["start"].shape[0]):
            type_ids = list(token_type_ids[bi].clone().detach().to("cpu").numpy())
            span_poses = [x.split(",") for x in tgt_info["span_positions"][bi].split("|") if len(x) > 3]
            # scoring_metric = pred_out.start_logits[bi].unsqueeze(1) + pred_out.end_logits[bi].unsqueeze(0)
            scoring_metric = pred_out.scoring_metric[bi]
            modifier_metric = pred_out.modifier[bi]
            # tmp1 = (torch.arange(pred_out.start_logits.shape[1]).unsqueeze(1) * pred_out.end_logits.shape[1]) + torch.arange(pred_out.end_logits.shape[1]).unsqueeze(0)
            tgt_span_print_info = tgt_info["span_positions"][bi]

            # advantage_loss, batch_print_info = calc_metric_contrastive_losses(pred_out, scoring_metric, span_poses, tgt_span_print_info, negative_sample_num=negative_sample_num)
            if SPOILER_TYPE != 2:
                use_whitelist = False if (SPOILER_TYPE == 0 or SPOILER_TYPE is None) else True
                advantage_loss, batch_print_info = calc_metric_contrastive_losses2(pred_out, scoring_metric, span_poses, tgt_span_print_info, negative_sample_num=negative_sample_num, use_whitelist=use_whitelist)
            else:
                advantage_loss, batch_print_info = calc_metric_contrastive_losses_for_multi(pred_out, scoring_metric, span_poses, tgt_span_print_info, negative_sample_num=negative_sample_num)
            advantage_losses.append(advantage_loss)
            batch_ranking_info.append(batch_print_info)

            # modifier_loss, batch_print_info = calc_modifier_losses(pred_out, modifier_metric, span_poses, tgt_span_print_info, negative_sample_num=20)
            # advantage_losses.append(modifier_loss)
            # batch_ranking_info.append(batch_print_info)


        contrastive_loss = torch.stack(advantage_losses).mean()
        # total_loss = contrastive_loss # + avg_ce_loss
        # contrastive_loss = 0.0

        if SPOILER_TYPE == 2:
            total_loss = avg_ce_loss * 0.5 + contrastive_loss
        elif SPOILER_TYPE == 1:
            total_loss = avg_ce_loss + contrastive_loss
        elif SPOILER_TYPE == 0:
            total_loss = avg_ce_loss # + contrastive_loss * 0.5
        else:
            total_loss = avg_ce_loss + contrastive_loss * 0.5  # * 0.1 + ce_type_loss

        return total_loss, avg_ce_loss, contrastive_loss, {"ranking_info":batch_ranking_info}

    def cosine_decay_lr(_i, _imax):
        max_lr = 5e-5
        min_lr = 1e-6
        num_oscillations = 4
        if _i < 10:
            lr = min_lr + (max_lr - min_lr) / 10.0 * _i
        else:
            progress = _i / _imax
            decayed_cosine = (0.5 * (1 + np.cos(num_oscillations * progress * np.pi)) * (max_lr - min_lr) + min_lr) * (1 - progress)
            lr = max(decayed_cosine, min_lr)
        return lr

    iter_i = 0
    grad_id = 0
    grad_accumulate_steps = 5
    if TRN_MAX_EPOCH > 0:
        model.train()
        for itm in sampler_train.train_batch_sampling(batch_size=TRN_BATCH_SIZE, max_epoch=TRN_MAX_EPOCH, filter_spoiler_type=SPOILER_TYPE):
            try:
                input_ids = torch.tensor(itm["input_ids"], dtype=torch.long, device=device)
                token_type_ids = torch.tensor(itm["token_type_ids"], dtype=torch.long, device=device)
                attention_mask = torch.tensor(itm["attention_mask"], dtype=torch.long, device=device)
                paragraph_ids = torch.tensor(itm["paragraph_ids"], dtype=torch.long, device=device)
                is_first_feature = torch.tensor(itm["is_first_feature"], dtype=torch.float32, device=device)
                absolute_position_ids = torch.tensor(itm["absolute_positions"], dtype=torch.long, device=device)
                absolute_position_ids_plus1 = torch.clip(torch.ones_like(absolute_position_ids) + absolute_position_ids,
                                                         0, config.max_position_embeddings - 1)

                predicted = model(input_ids, token_type_ids=token_type_ids, position_ids=absolute_position_ids_plus1 if use_pre_absolute_pos else None,
                                  attention_mask=attention_mask, paragraph_ids=paragraph_ids, absolute_position_ids=absolute_position_ids, using_modifier=use_antisymetric_modifier,
                                  lstm_span_modifier=lstm_span_modifier, use_abs_pos_embedding=use_abs_pos_embedding, solid_span=USE_SOLID_SPAN_LOSS)

                targets = {"start":torch.tensor(itm["start_target"],dtype=torch.float32, device=device),
                           "end":torch.tensor(itm["end_target"],dtype=torch.float32, device=device),
                           "span_positions":itm["span_pos"], "tags":torch.tensor(itm["tags"],dtype=torch.long, device=device)}
                *losses, track_info = calc_losses(predicted, {"token_type_ids":token_type_ids,"is_first_feature":is_first_feature}, targets, negative_sample_num=20)
                # with adjust_lr_optimizer(optim, cosine_decay_lr, iter_i, iter_max) as scheduled_optimizer:
                #     scheduled_optimizer.zero_grad()
                #     loss = losses[0]
                #     loss.backward()
                #     scheduled_optimizer.step()

                loss = losses[0] / (grad_id + 1)
                # scaler.scale(loss).backward()
                loss.backward()
                # grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), model_args.max_grad_norm)

                if sampler_train.current_epoch >= ADV_THRESHOLD: # iter_i > 100: #
                    with torch.cuda.amp.autocast(enabled=False):
                        adv_trainer._save()
                        adv_trainer._attack_step()
                        predicted = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask, position_ids=absolute_position_ids_plus1 if use_pre_absolute_pos else None,
                                          paragraph_ids=paragraph_ids, absolute_position_ids=absolute_position_ids,
                                          using_modifier=use_antisymetric_modifier, lstm_span_modifier=lstm_span_modifier,
                                          use_abs_pos_embedding=use_abs_pos_embedding, solid_span=USE_SOLID_SPAN_LOSS)

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

                MeasureTracker.track_record(*losses, track_names=["loss", "ce", "contrastive"])
                iter_i += 1
                if iter_i % 20 == 0:
                    print(iter_i, MeasureTracker.print_measures())
                    batch_ranking_info = track_info["ranking_info"]
                    print("\n".join(batch_ranking_info))
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print("|WARNING: ran out of memory, retrying batch")
                    for p in model.parameters():
                        if p.grad is not None:
                            del p.grad
                    torch.cuda.empty_cache()
                    grad_id = 0
                    pass
                else:
                    raise e
            if itm["is_new_epoch"]: # or iter_i % 1000 == 0:
                evaluate_training_model(save_model=True, ep_num=sampler_train.current_epoch)

    evaluate_training_model(save_model=True if TRN_MAX_EPOCH>0 else False, ep_num=sampler_train.current_epoch+1)
    return

if __name__ == "__main__":
    train_path = os.path.join(PROJECT_ROOT, "dataset", "train.jsonl")
    val_path = os.path.join(PROJECT_ROOT, "dataset", "validation.jsonl")
    run_training(train_path, val_path, DEFAULT_MODEL)