import sys
import os
if os.path.exists("/"):
    sys.path.append("/")
import argparse
import torch
import os
from pipeline import HfArgumentParser, ModelArguments, DataTrainingArguments, TrainingArguments
from modelling.postprocess import postprocess_qa_predictions, PredictionOutput, postprocess_qa_predictions_using_score_metric, postprocess_multi_qa_predictions_using_score_metric
from transformers import AutoConfig, AutoTokenizer
from modelling.modified import DebertaAddon
import logging
import numpy as np
import time
import pandas as pd
from clickbait_spoiling_task_1 import run_evaluation as run_task1
from pipeline import input_data_to_squad_format, setup_features
from data_process.sampler import Sampler
import json

script_path = os.path.dirname(__file__)
gt_path = os.path.join(script_path, "dataset", "validation.jsonl")
_val_inp = [json.loads(i) for i in open(gt_path, "r", encoding="utf8")]
gt_oracle_type = {x["uuid"]: x["tags"][0] for x in _val_inp}

LOADING_BUCKET_MODELS = {
    # "phrase": os.path.join(script_path, "models", "inference", "130-2012-phrase-5-0.646810-0.599000-model.bin"),
    "phrase": os.path.join(script_path, "models", "inference", "214-1951-phrase-4-0.643935-0.610233-model.bin"),
    # "passage": os.path.join(script_path, "models", "inference", "118-2032-passage-5-0.352349-0.487054-model.bin"),
    "passage": os.path.join(script_path, "models", "inference", "215-37-passage-8-0.342452-0.500634-model.bin"),
    # "multi": os.path.join(script_path, "models", "inference", "118-2150-multi-7-0.168945-0.390739-model.bin"),
    "multi": os.path.join(script_path, "models", "inference", "215-2323-multi-9-0.274307-0.570845-model.bin"),
    # "mix": os.path.join(script_path, "models", "inference", "119-028-mix-5-0.158897-0.454500-model.bin")
    "mix": os.path.join(script_path, "models", "inference", "214-054-mix-5-0.390304-0.462820-model.bin")
}
# LOADING_MODEL = os.path.join(script_path, "models", "inference", "117-644-7-0.168163-0.454150-model.bin")
LOADING_MODEL_PATH = os.path.join(script_path, "models", "inference")
DEBERTA_PATH = os.path.join(script_path, "models", "deberta")


logger = logging.getLogger(__name__)

print(str(LOADING_BUCKET_MODELS))
print(LOADING_MODEL_PATH)
print(os.path.exists(LOADING_MODEL_PATH))
print(",".join([str(x) for x in os.listdir("/")]))
logger.info(str(LOADING_BUCKET_MODELS))
logger.info(LOADING_MODEL_PATH)
logger.info(os.path.exists(LOADING_MODEL_PATH))
logger.info(",".join([str(x) for x in os.listdir("/")]))

classification_prediction_map = {}
if os.path.exists(os.path.join(script_path, "models", "inference", "task1_tmp_out.jsonl")):
    task1_json_loaded = json.load(open(os.path.join(script_path, "models", "inference", "task1_tmp_out.jsonl")))
    for pred_dict in task1_json_loaded:
        classification_prediction_map[pred_dict["uuid"]] = pred_dict

# MODEL setup:
# PHRASE: no modifier, no paragraph_ids, no absolute_pos
# PASSAGE: bilstm modifier, no paragraph_ids, no_absolute_pos
# MULTI: antisymmetric modifier, paragraph_ids, absolute_pos, contrastive score using null as threshold
# MIX: same as PHRASE, with longer max_spoiler_length
def run_evaluation(args, model_args, data_args, training_args, bucket="mix", gt_path=None, use_abs_embedding=False):
    print("evaluating bucket="+bucket)
    if bucket == "phrase":
        MAX_SPOILER_LEN = 40
    else:
        MAX_SPOILER_LEN = 200

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)# if training_args.should_log else logging.WARN

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else False,
    )
    config.position_biased_input = False # False if bucket != "multi" else True

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else False,
    )
    if os.path.exists(LOADING_BUCKET_MODELS[bucket]):
        model = DebertaAddon(config)
        init_state_dict = model.state_dict()
        should_have = list(init_state_dict.keys())
        state_dict = torch.load(open(LOADING_BUCKET_MODELS[bucket], mode="rb"), map_location=device)
        missing_keys = []
        for var_key in should_have:
            if var_key in state_dict:
                pass
            else:
                missing_keys.append(var_key)
                state_dict[var_key] = init_state_dict[var_key]
        logging.info("variables not loaded from state, reset from init: "+ (",".join(missing_keys)))
        print("variables not loaded from state, reset from init: "+ (",".join(missing_keys)))
        model.load_state_dict(state_dict)
        logging.info(bucket+"MODEL LOADED FROM STATE DICT: "+LOADING_BUCKET_MODELS[bucket])
    else:
        logging.info("MODEL STATE DICT NOT FOUND, USING INITIAL DEBERTA...")
        model = DebertaAddon.from_pretrained(
            model_args.model_name_or_path,
            from_tf=bool(".ckpt" in model_args.model_name_or_path),
            config=config,
            cache_dir=model_args.cache_dir,
            revision=model_args.model_revision,
            use_auth_token=True if model_args.use_auth_token else False,
        )
    model = model.to(device)
    model.eval()
    val_inp = [json.loads(i) for i in open(args.input, "r", encoding="utf8")]
    val_dataset = input_data_to_squad_format(val_inp, load_spoiler=False)
    features_val = setup_features(tokenizer, val_dataset, train_dataset=None, args_parsed=(model_args, data_args, training_args))

    sampler_train = Sampler()
    sampler_train.load_dataset(None, features_val)

    # # thresholds = 0.75,0.55,0.35
    # def is_spoiler_type_match(_itm):
    #     if bucket == "mix":
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
    #         if feat_pred_type_str == bucket:
    #             return 1
    #         else:
    #             return 0
    #     else:
    #         if "mix" == bucket:
    #             return 1
    #         else:
    #             return 0

    def is_spoiler_type_match(_itm):
        if bucket == "mix":
                return 1
        if "example_id" in _itm:
            feat_uuid = _itm["example_id"][0]
            return 1 if gt_oracle_type[feat_uuid] == bucket else 0

    tags_are_matched = []
    start_lgts = []
    end_lgts = []
    score_metrics = []
    for itm in sampler_train.val_batch_sampling(batch_size=1, dataset=sampler_train.val_dataset):
        is_feature_match_spoiler = is_spoiler_type_match(itm)
        tags_are_matched.append(is_feature_match_spoiler)
        input_ids = torch.tensor(itm["input_ids"], dtype=torch.long, device=device)
        token_type_ids = torch.tensor(itm["token_type_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(itm["attention_mask"], dtype=torch.long, device=device)
        paragraph_ids = torch.tensor(itm["paragraph_ids"], dtype=torch.long, device=device)
        absolute_position_ids = torch.tensor(itm["absolute_positions"], dtype=torch.long, device=device)
        lstm_span_modifier = False # False if bucket != "passage" else True
        using_modifier = False if bucket != "multi" else True
        with torch.no_grad():
            predicted = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              paragraph_ids=paragraph_ids, absolute_position_ids=absolute_position_ids, using_modifier=using_modifier,
                              lstm_span_modifier=lstm_span_modifier, use_abs_pos_embedding=use_abs_embedding)
        start_lgts.append(predicted.start_logits.detach().to("cpu").numpy())
        end_lgts.append(predicted.end_logits.detach().to("cpu").numpy())
        score_metrics.append(predicted.scoring_metric.detach().to("cpu").numpy())

    scoring_array = np.concatenate(score_metrics, axis=0)
    prediction_tuple = (np.concatenate(start_lgts, axis=0), np.concatenate(end_lgts, axis=0), scoring_array)

    if bucket != "multi":
        post_processed = postprocess_qa_predictions_using_score_metric(val_dataset, features_val, prediction_tuple,
                                                    version_2_with_negative=data_args.version_2_with_negative,
                                                    n_best_size=data_args.n_best_size,
                                                    max_answer_length=MAX_SPOILER_LEN, # data_args.max_answer_length,
                                                    null_score_diff_threshold=data_args.null_score_diff_threshold,
                                                    output_dir=None,
                                                    prefix="eval",
                                                    tag_restrictions=tags_are_matched)
    else:
        post_processed = postprocess_multi_qa_predictions_using_score_metric(val_dataset, features_val,
                                                                             prediction_tuple,
                                                                             version_2_with_negative=data_args.version_2_with_negative,
                                                                             n_best_size=data_args.n_best_size,
                                                                             max_answer_length=MAX_SPOILER_LEN,
                                                                             # data_args.max_answer_length,
                                                                             null_score_diff_threshold=data_args.null_score_diff_threshold,
                                                                             output_dir=None,
                                                                             prefix="eval",
                                                                             tag_restrictions=tags_are_matched)

    two_stage_results = []
    for _uid, _text in post_processed.items():
        two_stage_results.append({"uuid": _uid, "spoilerType": bucket, "spoiler": _text})

    with open(os.path.join(LOADING_MODEL_PATH, "task2_tmp_out_{0}.jsonl".format(bucket)), "w") as writer:
        writer.write(json.dumps(two_stage_results, indent=4) + "\n")

    if gt_path is not None and os.path.exists(gt_path):
        bucket_analyse_tmp_file = open(os.path.join(script_path, "tmp", bucket+"_oracle.txt"), encoding="utf8", mode="w")
        val_inp = [json.loads(i) for i in open(gt_path, "r", encoding="utf8")]
        gt_mapping = {x["uuid"]: " ".join(x["spoiler"]) for x in val_inp}
        acc_list = []
        hypos = []
        refs = []
        for _uid, _text in post_processed.items():
            if _uid in gt_mapping:
                is_relevent = 1 if (_text==gt_mapping[_uid]) else 0 # or gt_mapping[_uid].__contains__(_text) or _text.__contains__(gt_mapping[_uid])) else 0
                acc_list.append(is_relevent)
                bucket_analyse_tmp_file.write("{0} :{1} | {2} | {3}\n".format(is_relevent, _uid, _text, gt_mapping[_uid]))
                hypos.append(_text)
                refs.append(gt_mapping[_uid])
        print(sum(acc_list)/len(acc_list))

        from nltk.translate.bleu_score import sentence_bleu
        bleus = []
        for _hypo, _ref in zip(hypos, refs):
            _ref_splited = [[x for x in _ref.split(" ") if len(x)>0]]
            _hypo_splited = [x for x in _hypo.split(" ") if len(x)>0]
            bleu_len = min(len(_ref.strip().split(" ")), 4)
            weights = tuple([1.0 / bleu_len] * bleu_len + [0.0]*(4-bleu_len))
            bleu_score = sentence_bleu(_ref_splited, _hypo_splited, weights=weights)
            bleus.append(bleu_score)
        print(sum(bleus)/len(bleus))

        import evaluate
        meteor = evaluate.load('meteor')
        meteor_score = meteor.compute(predictions=hypos, references=refs)["meteor"]
        print(meteor_score)

        bucket_analyse_tmp_file.write("\n"+str(sum(acc_list)/len(acc_list)) + "\t"+str(sum(bleus)/len(bleus))+"\t"+str(meteor_score))
        bucket_analyse_tmp_file.close()
    return

def merge_predictions(args_parsed, gt_path=None):
    buckets = ["phrase", "passage", "multi", "mix"]
    combined_predictions = {}
    for bucket in buckets:
        task_output_file = os.path.join(LOADING_MODEL_PATH, "task2_tmp_out_{0}.jsonl".format(bucket))
        pred_outs = json.load(open(task_output_file, "r", encoding="utf8"))
        for pred_info in pred_outs:
            if pred_info["uuid"] not in combined_predictions:
                combined_predictions[pred_info["uuid"]] = pred_info
            if pred_info["spoilerType"] == "mix":
                if pred_info["uuid"] in classification_prediction_map:
                    pred_type = classification_prediction_map[pred_info["uuid"]]["spoilerType"]
                    combined_predictions[pred_info["uuid"]]["spoilerType"] = pred_type

    final_output_list = list(combined_predictions.values())
    # print("outputing final results: " + str(len(final_output_list)))
    # with open(args_parsed.output, "w") as writer:
    #     # writer.write(json.dumps(final_output_list, indent=4) + "\n")
    #     for _eid, pred_line in enumerate(final_output_list):
    #         endl = "" if _eid == len(final_output_list) - 1 else "\n"
    #         writer.write(json.dumps(pred_line) + endl)

    if gt_path is not None and os.path.exists(gt_path):
        bucket_analyse_tmp_file = open(os.path.join(script_path, "tmp", "combined"+"_predictions.txt"), encoding="utf8", mode="w")
        val_inp = [json.loads(i) for i in open(gt_path, "r", encoding="utf8")]
        gt_mapping = {x["uuid"]: " ".join(x["spoiler"]) for x in val_inp}
        acc_list = []
        hypos = []
        refs = []
        for _uid, output_info in combined_predictions.items():
            _text = output_info["spoiler"]
            if _uid in gt_mapping:
                is_relevent = 1 if (_text==gt_mapping[_uid] or gt_mapping[_uid].__contains__(_text) or _text.__contains__(gt_mapping[_uid])) else 0
                acc_list.append(is_relevent)
                bucket_analyse_tmp_file.write("{0} :{1} | {2} | {3}\n".format(is_relevent, _uid, _text, gt_mapping[_uid]))
                hypos.append(_text)
                refs.append(gt_mapping[_uid])
            else:
                hypos.append(_text)
                refs.append("unknown")
        print(sum(acc_list)/len(acc_list))

        from nltk.translate.bleu_score import sentence_bleu
        bleus = []
        for _hypo, _ref in zip(hypos, refs):
            _ref_splited = [[x for x in _ref.split(" ") if len(x)>0]]
            _hypo_splited = [x for x in _hypo.split(" ") if len(x)>0]
            bleu_len = min(len(_ref.strip().split(" ")), 4)
            weights = tuple([1.0 / bleu_len] * bleu_len + [0.0]*(4-bleu_len))
            bleu_score = sentence_bleu(_ref_splited, _hypo_splited, weights=weights)
            bleus.append(bleu_score)
        print(sum(bleus)/len(bleus))

        import evaluate
        meteor = evaluate.load('meteor')
        meteor_score = meteor.compute(predictions=hypos, references=refs)["meteor"]
        print(meteor_score)

        bucket_analyse_tmp_file.write("\n"+str(sum(acc_list)/len(acc_list)) + "\t"+str(sum(bleus)/len(bleus))+"\t"+str(meteor_score))
        bucket_analyse_tmp_file.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=LOADING_MODEL_PATH)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)

    args_parsed = parser.parse_args()
    args = ["--model_name_or_path", args_parsed.model_name_or_path, "--output", args_parsed.output]
    logger.info("model_name_or_path="+str(args_parsed.model_name_or_path))
    if args_parsed.model_name_or_path is None:
        args_parsed.model_name_or_path = LOADING_MODEL_PATH
    model_args, data_args, training_args = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)).parse_args_into_dataclasses(args=args)
    # run_task1(args_parsed, model_args, data_args, training_args, gt_path=gt_path)
    # time.sleep(1.0)
    # run_evaluation(args_parsed, model_args, data_args, training_args, bucket="phrase", gt_path=gt_path, use_abs_embedding=False)
    # time.sleep(1.0)
    # run_evaluation(args_parsed, model_args, data_args, training_args, bucket="passage", gt_path=gt_path)
    # time.sleep(1.0)
    # run_evaluation(args_parsed, model_args, data_args, training_args, bucket="multi", gt_path=gt_path, use_abs_embedding=True)
    # time.sleep(1.0)
    run_evaluation(args_parsed, model_args, data_args, training_args, bucket="mix", gt_path=gt_path)
    time.sleep(1.0)
    # merge_predictions(args_parsed, gt_path=gt_path)