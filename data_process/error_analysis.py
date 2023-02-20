import os
import json
script_path = os.path.dirname(__file__)

from nltk.translate.bleu_score import sentence_bleu
import evaluate


def analyse_on_results(filepath, gt_file):

    gt_data_entities = [json.loads(i) for i in open(gt_file, "r", encoding="utf8")]
    gt_types = {}
    for item in gt_data_entities:
        gt_type = item["tags"][0]
        uuid = item["uuid"]
        gt_types[uuid] = gt_type

    output_fw = open(os.path.join(script_path, "error_csv.csv"), mode="w", encoding="utf8")

    pred_info_list = [x.split("|") for x in open(filepath, encoding="utf8").read().splitlines()]
    for pred_info in pred_info_list:
        if len(pred_info) != 3:
            continue
        uuid = (pred_info[0].split(":")[1]).strip()
        if uuid in gt_types:
            gt_type = gt_types[uuid]
        else:
            print("dataset may not match, check for dataset to make sure uuid fits...")
            exit()

        predicted = pred_info[1].strip()
        gt_text = pred_info[2].strip()

        _ref_splited = [[x for x in gt_text.split(" ") if len(x) > 0]]
        _hypo_splited = [x for x in predicted.split(" ") if len(x) > 0]
        bleu_len = min(len(gt_text.strip().split(" ")), 4)
        weights = tuple([1.0 / bleu_len] * bleu_len + [0.0] * (4 - bleu_len))
        bleu_score = sentence_bleu(_ref_splited, _hypo_splited, weights=weights)

        meteor = evaluate.load('meteor')
        meteor_score = meteor.compute(predictions=[predicted], references=[gt_text])["meteor"]
        # print(meteor_score)

        result_line = "{0}\t{1}\t{2}\t{3}\t{4}\t{5}\n".format(uuid, gt_type, float(bleu_score), float(meteor_score), predicted, gt_text)
        output_fw.write(result_line)
        aaa = 1
    output_fw.close()

if __name__ == "__main__":
    type_name = "phrase"
    gt_file = os.path.join(script_path, "..", "dataset", "validation.jsonl")
    prediction_out_filepath = os.path.join(script_path, "..", "tmp", type_name+"_predictions.txt")
    analyse_on_results(prediction_out_filepath, gt_file)