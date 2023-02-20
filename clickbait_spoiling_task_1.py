import sys
import os
if os.path.exists("/"):
    sys.path.append("/")
import argparse
import torch
from pipeline import HfArgumentParser, ModelArguments, DataTrainingArguments, TrainingArguments
from transformers import AutoConfig, AutoTokenizer
from modelling.modified import DebertaAddon
import logging
script_path = os.path.dirname(__file__)
LOADING_MODEL = os.path.join(script_path, "models", "inference", "214-1842-6-0.762499988079071-type_pred_model.bin")
LOADING_MODEL_PATH = os.path.join(script_path, "models", "inference")
DEBERTA_PATH = os.path.join(script_path, "models", "deberta")
from pipeline import input_data_to_squad_format, setup_features
from data_process.sampler import Sampler
import json

logger = logging.getLogger(__name__)

def run_evaluation(args, model_args, data_args, training_args, gt_path=None):

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
    config.position_biased_input=False
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else False,
    )

    if os.path.exists(LOADING_MODEL):
        model = DebertaAddon(config)
        init_state_dict = model.state_dict()
        should_have = list(init_state_dict.keys())
        state_dict = torch.load(open(LOADING_MODEL, mode="rb"), map_location=device)
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

        logging.info("MODEL LOADED FROM STATE DICT: "+LOADING_MODEL)
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

    type_predict_res = []
    for itm in sampler_train.val_batch_sampling(batch_size=1, dataset=sampler_train.val_dataset, only_first=True):
        input_ids = torch.tensor(itm["input_ids"], dtype=torch.long, device=device)
        token_type_ids = torch.tensor(itm["token_type_ids"], dtype=torch.long, device=device)
        attention_mask = torch.tensor(itm["attention_mask"], dtype=torch.long, device=device)
        paragraph_ids = torch.tensor(itm["paragraph_ids"], dtype=torch.long, device=device)
        absolute_position_ids = torch.tensor(itm["absolute_positions"], dtype=torch.long, device=device)
        eid = itm["example_id"][0]
        with torch.no_grad():
            predicted = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                              paragraph_ids=paragraph_ids, absolute_position_ids=absolute_position_ids, use_abs_pos_embedding=True)
        predicted_prob = torch.softmax(predicted.type_classify_logits, dim=-1)
        predicted_type_ids = torch.argmax(predicted_prob, dim=-1)
        # confidence = torch.gather(predicted_prob, 1, predicted_type_ids.unsqueeze(1))
        type_id = int(predicted_type_ids.reshape([-1])[0].detach().to("cpu"))
        # confidence_score = float(confidence.detach().to("cpu"))
        confidences = [float(x) for x in list(predicted_prob[0, :].detach().to("cpu").numpy())]

        if type_id == 0:
            spoiler_type_str = "phrase"
        elif type_id == 1:
            spoiler_type_str = "passage"
        else:
            spoiler_type_str = "multi"
        type_predict_res.append({"uuid":eid, "spoilerType":spoiler_type_str, "confidences":confidences})

    with open(args.output, "w") as writer:
        # writer.write(json.dumps(type_predict_res, indent=4) + "\n")
        for _eid, pred_line in enumerate(type_predict_res):
            endl = "" if _eid == len(type_predict_res) - 1 else "\n"
            writer.write(json.dumps(pred_line) + endl)
    with open(os.path.join(LOADING_MODEL_PATH, "task1_tmp_out.jsonl"), "w") as writer:
        writer.write(json.dumps(type_predict_res, indent=4) + "\n")

    if gt_path is not None:
        classify_analyse_fw = open(os.path.join(script_path, "tmp", "cls_predictions.txt"), encoding="utf8", mode="w")
        val_inp = [json.loads(i) for i in open(gt_path, "r", encoding="utf8")]
        gt_mapping = {x["uuid"]: x["tags"][0] for x in val_inp}
        accurate_list = []
        for prd_info in type_predict_res:
            if prd_info["uuid"] in gt_mapping:
                is_accurate = 1 if prd_info["spoilerType"]==gt_mapping[prd_info["uuid"]] else 0
                gt_type = gt_mapping[prd_info["uuid"]]
            else:
                is_accurate = 0
                gt_type = "UNK"
            classify_analyse_fw.write("{0} | {3} | {1} | {2}\n".format(is_accurate, prd_info["spoilerType"], gt_type, prd_info["uuid"]))
            accurate_list.append(is_accurate)
        classify_analyse_fw.write("\n" + str(sum(accurate_list)/len(accurate_list)))
        classify_analyse_fw.close()
    return

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, default=LOADING_MODEL_PATH)
    parser.add_argument("--input", type=str)
    parser.add_argument("--output", type=str)
    args_parsed = parser.parse_args()
    args = ["--model_name_or_path", args_parsed.model_name_or_path, "--output", args_parsed.output]
    model_args, data_args, training_args = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)).parse_args_into_dataclasses(args=args)
    gt_path = os.path.join(script_path, "dataset", "test.jsonl")
    run_evaluation(args_parsed, model_args, data_args, training_args, gt_path=gt_path)
