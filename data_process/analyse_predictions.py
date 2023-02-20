import os
import json
script_path = os.path.dirname(__file__)
classification_prediction_map = {}
if os.path.exists(os.path.join(script_path, "..", "models", "inference", "task1_tmp_out.jsonl")):
    task1_json_loaded = json.load(open(os.path.join(script_path, "..", "models", "inference", "task1_tmp_out.jsonl")))
    for pred_dict in task1_json_loaded:
        classification_prediction_map[pred_dict["uuid"]] = pred_dict


# thresholds = 0.85,0.55,0.35, # 0.75,0.45,0.7
def is_spoiler_type_match(_itm, bucket):
    if bucket == "mix":
        return 1
    if "example_id" in _itm:
        feat_uuid = _itm["example_id"][0]
        if feat_uuid in classification_prediction_map:
            task1_pred = classification_prediction_map[feat_uuid]
            feat_pred_type_str = task1_pred["spoilerType"]
            probs = task1_pred["confidences"]
            confidences_with_labels = [(_lid, _prob) for _lid, _prob in enumerate(probs)]
            sorted_confidences = list(sorted(confidences_with_labels, key=lambda x:x[1], reverse=True))
            max_pred = sorted_confidences[0]
            if max_pred[0] == 0 and max_pred[1] >= 0.75:
                feat_pred_type_str = "phrase"
            elif max_pred[0]==1 and max_pred[1] >= 0.45:
                feat_pred_type_str = "passage"
            elif max_pred[0]==2 and max_pred[1] >= 0.7:
                feat_pred_type_str = "multi"
            else:
                feat_pred_type_str = "mix"
        else:
            feat_pred_type_str = "mix"
        if feat_pred_type_str == bucket:
            return 1
        else:
            return 0
    else:
        if "mix" == bucket:
            return 1
        else:
            return 0

def make_confusion_matrix(load_file_path):
    data_entities = [json.loads(i) for i in open(load_file_path, "r", encoding="utf8")]
    from collections import Counter
    confusion_stat = {}
    non_match_ct = 0
    for item in data_entities:
        gt_type = item["tags"][0]
        if gt_type not in confusion_stat:
            confusion_stat[gt_type] = {"ct":0, "prd_passage":0, "prd_phrase":0, "prd_multi":0, "prd_mix":0}

        uuid = item["uuid"]
        if uuid in classification_prediction_map:
            cls_pred_info = classification_prediction_map[uuid]
            probs = cls_pred_info["confidences"]
            confidences_with_labels = [(_lid, _prob) for _lid, _prob in enumerate(probs)]
            sorted_confidences = list(sorted(confidences_with_labels, key=lambda x:x[1], reverse=True))
            max_pred = sorted_confidences[0]
            confusion_stat[gt_type]["ct"] += 1
            # if max_pred[0] == 0 and max_pred[1] >= 0.75:
            #     confusion_stat[gt_type]["prd_phrase"] += 1
            # elif max_pred[0]==1 and max_pred[1] >= 0.45:
            #     confusion_stat[gt_type]["prd_passage"] += 1
            # elif max_pred[0]==2 and max_pred[1] >= 0.7:
            #     confusion_stat[gt_type]["prd_multi"] += 1
            # else:
            #     confusion_stat[gt_type]["prd_mix"] += 1
            if max_pred[0] == 0:
                confusion_stat[gt_type]["prd_phrase"] += 1
            elif max_pred[0]==1:
                confusion_stat[gt_type]["prd_passage"] += 1
            elif max_pred[0]==2:
                confusion_stat[gt_type]["prd_multi"] += 1
            else:
                confusion_stat[gt_type]["prd_mix"] += 1
        else:
            non_match_ct += 1
    for _k, _itm in confusion_stat.items():
        print("gt: ", _k)
        print(_itm)
        print("--------------")

if __name__ == "__main__":
    # os.path.join(script_path, "..", "dataset", "test.jsonl")
    # os.path.join(script_path, "..", "dataset", "validation.jsonl")
    file_path = os.path.join(script_path, "..", "dataset", "validation.jsonl")
    make_confusion_matrix(file_path)