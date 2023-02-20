import collections
import json
import logging
import os
from typing import Optional, Tuple
from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import numpy as np
from tqdm.auto import tqdm
import torch
from data_process.utils import lcs
import re
from collections import OrderedDict
import copy

logger = logging.getLogger(__name__)

class PredictionOutput(NamedTuple):
    predictions: Union[np.ndarray, Tuple[np.ndarray]]
    label_ids: Optional[Union[np.ndarray, Tuple[np.ndarray]]]
    metrics: Optional[Dict[str, float]]

def postprocess_qa_predictions(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert len(predictions) == 2, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits = predictions

    assert len(predictions[0]) == len(features), f"Got {len(predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, _ in enumerate(tqdm(range(len(examples)))):
        example = examples.iloc[example_index]
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            token_type_ids_in_feature = features[feature_index]["token_type_ids"]
            start_indexes = np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            end_indexes = np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.

                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    # for non-null spoilers, restrict prelim_prediction to locate within context positions, where token_type_ids are 1s
                    if (start_index >0 and end_index>0) and (token_type_ids_in_feature[start_index] == 0 or token_type_ids_in_feature[end_index]==0):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                        }
                    )
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        context_into_lines = example["context"].split("\n")
        context = context_into_lines[0] + " - " + (" ".join(context_into_lines[1:]))
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = (context[offsets[0] : offsets[1]]).strip()

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
            if len(predictions[0]["text"]) < 3:
                print("??empty spoiler", example_index, example["id"], predictions[0]["text"], predictions[0]["start_logit"], )
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions



def postprocess_qa_predictions_using_score_metric(
    examples,
    features,
    tuple_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    null_score_diff_threshold: float = 0.0,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
    tag_restrictions = None,
):
    """
    Post-processes the predictions of a question-answering model to convert them to answers that are substrings of the
    original contexts. This is the base postprocessing functions for models that only return start and end logits.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        null_score_diff_threshold (:obj:`float`, `optional`, defaults to 0):
            The threshold used to select the null answer: if the best answer has a score that is less than the score of
            the null answer minus this threshold, the null answer is selected for this example (note that the score of
            the null answer for an example giving several features is the minimum of the scores for the null answer on
            each feature: all features must be aligned on the fact they `want` to predict a null answer).
            Only useful when :obj:`version_2_with_negative` is :obj:`True`.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert len(tuple_predictions) == 3, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits, all_scores = tuple_predictions

    assert len(tuple_predictions[0]) == len(features), f"Got {len(tuple_predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, _ in enumerate(tqdm(range(len(examples)))):
        example = examples.iloc[example_index]
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        context_into_lines = example["context"].split("\n")
        context = context_into_lines[0] + " - " + (" ".join(context_into_lines[1:]))

        # Looping through all the features associated to the current example.
        tgt_scores = {}
        for feature_index in feature_indices:
            if tag_restrictions[feature_index] == 0:
                continue

            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            scoring_metric = all_scores[feature_index]

            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction.
            feature_null_score = scoring_metric[0, 0] # start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }

            # Go through all possibilities for the `n_best_size` greater start and end logits.
            token_type_ids_in_feature = features[feature_index]["token_type_ids"]
            start2maxes = np.max(scoring_metric, axis=1)
            start_indices = np.argsort(start2maxes)[-1: -n_best_size-1: -1].tolist()

            # start_indexes = list(range(scoring_metric.shape[0])) # np.argsort(start_logits)[-1 : -n_best_size - 1 : -1].tolist()
            # end_indexes = list(range(scoring_metric.shape[1])) # np.argsort(end_logits)[-1 : -n_best_size - 1 : -1].tolist()

            if "start_target" in features[feature_index]:
                start_poses = [(_sid, st_lab) for _sid, st_lab in enumerate(features[feature_index]["start_target"]) if st_lab==1]
                ed_poses = [(_eid, ed_lab) for _eid, ed_lab in enumerate(features[feature_index]["end_target"]) if ed_lab==1]
                for _sid, st_lab in start_poses:
                    for _eid, ed_lab in ed_poses:
                        if st_lab==1 and ed_lab==1 and _sid<=_eid:
                            # tgt_scores[(_sid, _eid)] = scoring_metric[_sid, _eid]
                            tgt_st_offset, tgt_ed_offset = offset_mapping[_sid], offset_mapping[_eid]
                            if tgt_st_offset is not None and tgt_ed_offset is not None:
                                tgt_st_offset = tgt_st_offset[0]
                                tgt_ed_offset = tgt_ed_offset[1]
                                tgt_string = context[tgt_st_offset:tgt_ed_offset].strip()
                                if feature_index not in tgt_scores:
                                    tgt_scores[feature_index] = []
                                tgt_scores[feature_index].append((scoring_metric[_sid, _eid], start_logits[_sid], end_logits[_eid], _sid, _eid, tgt_st_offset, tgt_ed_offset, tgt_string))

            # for start_index in start_indexes:
            for start_index in start_indices:
                end_indexes = np.argsort(scoring_metric[start_index], axis=0)[-1: -n_best_size-1: -1].tolist()
                for end_index in end_indexes:
                    # Don't consider out-of-scope answers, either because the indices are out of bounds or correspond
                    # to part of the input_ids that are not in the context.

                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length that is either < 0 or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue

                    # for non-null spoilers, restrict prelim_prediction to locate within context positions, where token_type_ids are 1s
                    if (start_index >0 and end_index>0) and (token_type_ids_in_feature[start_index] == 0 or token_type_ids_in_feature[end_index]==0):
                        continue

                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": scoring_metric[start_index, end_index], # start_logits[start_index] + end_logits[end_index],
                            "start_logit": start_logits[start_index],
                            "end_logit": end_logits[end_index],
                            "feature_index": feature_index,
                            "modifier": scoring_metric[start_index, end_index]-(start_logits[start_index]+end_logits[end_index])
                        }
                    )

        if len(prelim_predictions)==0:
            continue
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        def fix_text(_text, _text_st, _text_ed, _symbol):
            if _text.startswith(_symbol) and len(_text[1:])-len(_text[1:].replace(_symbol,"")) % 2==1:
                return _text
            if _text.endswith(_symbol) and len(_text[:-1])-len(_text[:-1].replace(_symbol,"")) % 2==1:
                return _text
            if _text.startswith(_symbol):
                if _text_ed<len(context) and context[_text_ed]==_symbol:
                    return _text+_symbol
                # else:
                #     _text = _text[1:]
            if _text.endswith(_symbol):
                if _text_st>0 and context[_text_st-1]==_symbol:
                    return _symbol+_text
                # else:
                #     _text = _text[:-1]
            return _text

        for pred in predictions:
            offsets = pred.pop("offsets")
            text_raw = (context[offsets[0] : offsets[1]]).strip()
            text_raw = fix_text(text_raw, offsets[0],offsets[1], '"')
            text_raw = fix_text(text_raw, offsets[0],offsets[1], "'")
            pred["text"] = text_raw

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
            predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        if not version_2_with_negative:
            all_predictions[example["id"]] = predictions[0]["text"]
            if len(predictions[0]["text"]) < 3:
                print("??empty spoiler", example_index, example["id"], predictions[0]["text"], predictions[0]["start_logit"], )
        else:
            # Otherwise we first need to find the best non-empty prediction.
            i = 0
            while predictions[i]["text"] == "":
                i += 1
            best_non_null_pred = predictions[i]

            # Then we compare to the null prediction using the threshold.
            score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
            scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
            if score_diff > null_score_diff_threshold:
                all_predictions[example["id"]] = ""
            else:
                all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.
        if "answers" in example:
            spoiler_poses = [[int(y) for y in x.split(",")] for x in example["spoiler_poses"].split(" ") if len(x)>0]
            spoiler_context_lines = example["context"].split("\n")
            spoiler_seg_texts = []
            for pid, st_offset, ed_offset in spoiler_poses:
                try:
                    gt_text_part = spoiler_context_lines[pid][st_offset:ed_offset]
                except:
                    gt_text_part = ""
                spoiler_seg_texts.append(gt_text_part)
            gt_text = " ".join(spoiler_seg_texts)
            gt_detail = []
            for _fid, tgt_score_infos in tgt_scores.items():
                for _score, st_lgt, ed_lgt, _sid, _eid, tgt_st_offset, tgt_ed_offset, tgt_string in tgt_score_infos:
                    gt_detail.append({"coord":[int(_fid), int(_sid), int(_eid), int(tgt_st_offset), int(tgt_ed_offset)], "scores":[float(_score), float(st_lgt), float(ed_lgt)], "text":tgt_string})
            gt_json_ent = [{"gt_fulltext":gt_text, "list":gt_detail}]
        else:
            gt_json_ent = []

        all_nbest_json[example["id"]] = gt_json_ent + [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions


def postprocess_multi_qa_predictions_using_score_metric(
        examples,
        features,
        tuple_predictions: Tuple[np.ndarray, np.ndarray, np.ndarray],
        version_2_with_negative: bool = False,
        n_best_size: int = 20,
        max_answer_length: int = 30,
        null_score_diff_threshold: float = 0.0,
        output_dir: Optional[str] = None,
        prefix: Optional[str] = None,
        is_world_process_zero: bool = True,
        tag_restrictions=None,
):
    assert len(tuple_predictions) == 3, "`predictions` should be a tuple with two elements (start_logits, end_logits)."
    all_start_logits, all_end_logits, all_scores = tuple_predictions

    assert len(tuple_predictions[0]) == len(features), f"Got {len(tuple_predictions[0])} predictions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    if version_2_with_negative:
        scores_diff_json = collections.OrderedDict()

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, _ in enumerate(tqdm(range(len(examples)))):
        example = examples.iloc[example_index]
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_prediction = None
        prelim_predictions = []

        context_into_lines = example["context"].split("\n")
        context = context_into_lines[0] + " - " + (" ".join(context_into_lines[1:]))

        # Looping through all the features associated to the current example.
        tgt_scores = {}
        example_candidates = []
        for feature_index in feature_indices:
            if tag_restrictions[feature_index] == 0:
                continue

            # We grab the predictions of the model for this feature.
            start_logits = all_start_logits[feature_index]
            end_logits = all_end_logits[feature_index]
            scoring_metric = all_scores[feature_index]

            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)
            paragraph_ids = features[feature_index].get("paragraph_ids", 1)

            # Update minimum null prediction.
            feature_null_score = scoring_metric[0, 0] # start_logits[0] + end_logits[0]
            if min_null_prediction is None or min_null_prediction["score"] > feature_null_score:
                min_null_prediction = {
                    "offsets": (0, 0),
                    "score": feature_null_score,
                    "start_logit": start_logits[0],
                    "end_logit": end_logits[0],
                }
            soft_null_threshold = 0.9
            above_threshold_flag = np.where(scoring_metric > feature_null_score*soft_null_threshold, np.ones_like(scoring_metric), np.zeros_like(scoring_metric))
            positive_sorting_scores = scoring_metric - (np.ones_like(above_threshold_flag) - above_threshold_flag) * 100.0
            top_score_indices = np.argsort(positive_sorting_scores.reshape([-1]))
            max_seq_len = positive_sorting_scores.shape[0]

            def has_overlap(_candidate_spans, _st, _ed):
                for cand_info in _candidate_spans:
                    _cst, _ced = cand_info["span"][0], cand_info["span"][1]
                    overlap_length = max(max(_ced - _st+1, 0), max(_ed-_cst+1, 0))
                    overlap_proportion = overlap_length / max(min(_ed-_st+1, _ced-_cst+1), 1)
                    if overlap_proportion > 0.3:
                        return True
                return False

            candidates = []
            start2maxes = np.max(scoring_metric, axis=1)
            start_indices = np.argsort(start2maxes)[-1: -n_best_size-1: -1].tolist()
            for top_st in start_indices:
                end_indexes = np.argsort(scoring_metric[top_st], axis=0)[-1: -n_best_size-1: -1].tolist()
                for top_ed in end_indexes:
                    if (
                            top_st >= len(offset_mapping)
                            or top_ed >= len(offset_mapping)
                            or offset_mapping[top_st] is None
                            or offset_mapping[top_ed] is None
                    ):
                        continue
                    if top_st > top_ed or offset_mapping[top_ed] is None or offset_mapping[top_st] is None:
                        continue
                    # if has_overlap(candidates, top_st, top_ed):
                    #     continue
                    spoiler_text = context[offset_mapping[top_st][0]: offset_mapping[top_ed][1]]
                    # if len(spoiler_text) > 100:
                    #     continue
                    if top_ed-top_st+1 > 40:
                        continue

                    candidate_info = {
                                "offsets": (offset_mapping[top_st][0], offset_mapping[top_ed][1]),
                                "score": scoring_metric[top_st, top_ed], # start_logits[start_index] + end_logits[end_index],
                                "start_logit": start_logits[top_st],
                                "end_logit": end_logits[top_ed],
                                "feature_index": feature_index,
                                "modifier": float(scoring_metric[top_st, top_ed]-(start_logits[top_st]+end_logits[top_ed])),
                                "score_with_null_threshold": float(positive_sorting_scores[top_st, top_ed]),
                                "span": (top_st, top_ed),
                                "pid":int(paragraph_ids[top_st])
                            }
                    candidates.append(candidate_info)
                    # if len(candidates) >= 6:
                    #     break

            candidates = list(sorted(candidates, key=lambda x:x["score_with_null_threshold"], reverse=True))
            top_30_cands = candidates[:min(30, len(candidates))]

            candidate_groups = []
            def is_in_group(_st_offset, _ed_offset, group_poses):
                group_starts = [x[0] for x in group_poses]
                group_ends = [x[1] for x in group_poses]
                avg_st = sum(group_starts)/len(group_starts)
                avg_ed = sum(group_ends)/len(group_ends)
                if avg_st <= _st_offset and avg_ed >=_st_offset:
                    intersect_len = min(avg_ed, _ed_offset) - _st_offset
                    union_len = max(avg_ed, _ed_offset) - avg_st
                    iou = intersect_len / union_len
                elif _st_offset < avg_st and _ed_offset >=  avg_st:
                    intersect_len = min(avg_ed, _ed_offset) - avg_st
                    union_len = max(avg_ed, _ed_offset) - _st_offset
                    iou = intersect_len/union_len
                else:
                    iou = 0.0
                if iou > 0.6:
                    return True
                else:
                    return False

            for cid, top_cand in enumerate(top_30_cands):
                # cand_text = context[top_cand["offsets"][0]:top_cand["offsets"][1]]
                if len(candidate_groups)>0:
                    cand_has_a_group = False
                    for gid in range(len(candidate_groups)):
                        c_group = candidate_groups[gid]
                        if is_in_group(top_cand["offsets"][0], top_cand["offsets"][1], c_group):
                            candidate_groups[gid].append((top_cand["offsets"][0], top_cand["offsets"][1], cid, top_cand["score_with_null_threshold"]))
                            cand_has_a_group = True
                            break
                        else:
                            continue
                    if not cand_has_a_group:
                        candidate_groups.append([(top_cand["offsets"][0], top_cand["offsets"][1], cid, top_cand["score_with_null_threshold"])])
                else:
                    candidate_groups.append([(top_cand["offsets"][0], top_cand["offsets"][1], cid, top_cand["score_with_null_threshold"])])

            group_elected = [list(sorted(gp,key=lambda x:x[3],reverse=True))[0] for gp in candidate_groups]
            non_overlap_candidates = [top_30_cands[x[2]] for x in group_elected]
            for itm in non_overlap_candidates:
                itm["text"] = context[itm["offsets"][0]:itm["offsets"][1]]
            example_candidates.extend(non_overlap_candidates[:min(6, len(non_overlap_candidates))])
            # example_candidates.extend(candidates[:min(6, len(candidates))])

            if "start_target" in features[feature_index]:
                start_poses = [(_sid, st_lab) for _sid, st_lab in enumerate(features[feature_index]["start_target"]) if st_lab==1]
                ed_poses = [(_eid, ed_lab) for _eid, ed_lab in enumerate(features[feature_index]["end_target"]) if ed_lab==1]
                for _sid, st_lab in start_poses:
                    for _eid, ed_lab in ed_poses:
                        if st_lab==1 and ed_lab==1 and _sid<=_eid:
                            # tgt_scores[(_sid, _eid)] = scoring_metric[_sid, _eid]
                            tgt_st_offset, tgt_ed_offset = offset_mapping[_sid], offset_mapping[_eid]
                            if tgt_st_offset is not None and tgt_ed_offset is not None:
                                tgt_st_offset = tgt_st_offset[0]
                                tgt_ed_offset = tgt_ed_offset[1]
                                tgt_string = context[tgt_st_offset:tgt_ed_offset].strip()
                                if feature_index not in tgt_scores:
                                    tgt_scores[feature_index] = []
                                tgt_scores[feature_index].append((scoring_metric[_sid, _eid], start_logits[_sid], end_logits[_eid], _sid, _eid, tgt_st_offset, tgt_ed_offset, tgt_string))


        prelim_predictions = example_candidates
        if len(prelim_predictions)==0:
            continue
        if version_2_with_negative:
            # Add the minimum null prediction
            prelim_predictions.append(min_null_prediction)
            null_score = min_null_prediction["score"]

        # Only keep the best `n_best_size` predictions.

        predictions = sorted(prelim_predictions, key=lambda x: x["score_with_null_threshold"], reverse=True)
        above_null_checked = [x for x in predictions if x["score_with_null_threshold"] > -50.0]

        example_cand_groups = []
        def is_in_example_group(_st_offset, _ed_offset, group_poses):
            group_starts = [x[0] for x in group_poses]
            group_ends = [x[1] for x in group_poses]
            avg_st = sum(group_starts) / len(group_starts)
            avg_ed = sum(group_ends) / len(group_ends)
            if avg_st <= _st_offset and avg_ed >= _st_offset:
                intersect_len = min(avg_ed, _ed_offset) - _st_offset
                union_len = max(avg_ed, _ed_offset) - avg_st
                iou = intersect_len / union_len
            elif _st_offset < avg_st and _ed_offset >= avg_st:
                intersect_len = min(avg_ed, _ed_offset) - avg_st
                union_len = max(avg_ed, _ed_offset) - _st_offset
                iou = intersect_len / union_len
            else:
                iou = 0.0
            if iou > 0.6:
                return True
            elif _st_offset in group_starts:
                return True
            else:
                return False

        for cid, top_cand in enumerate(above_null_checked):
            # cand_text = context[top_cand["offsets"][0]:top_cand["offsets"][1]]
            if len(example_cand_groups) > 0:
                cand_has_a_group = False
                for gid in range(len(example_cand_groups)):
                    c_group = example_cand_groups[gid]
                    if is_in_example_group(top_cand["offsets"][0], top_cand["offsets"][1], c_group):
                        example_cand_groups[gid].append((top_cand["offsets"][0], top_cand["offsets"][1], cid,
                                                      top_cand["score_with_null_threshold"]))
                        cand_has_a_group = True
                        break
                    else:
                        continue
                if not cand_has_a_group:
                    example_cand_groups.append(
                        [(top_cand["offsets"][0], top_cand["offsets"][1], cid, top_cand["score_with_null_threshold"])])
            else:
                example_cand_groups.append(
                    [(top_cand["offsets"][0], top_cand["offsets"][1], cid, top_cand["score_with_null_threshold"])])
        example_group_elected = [list(sorted(gp,key=lambda x:x[3],reverse=True))[0] for gp in example_cand_groups]
        non_overlap_above_null_checked = [above_null_checked[x[2]] for x in example_group_elected]

        above_null_checked = non_overlap_above_null_checked[:min(6, len(non_overlap_above_null_checked))]
        if len(above_null_checked) == 0:
            above_null_checked = predictions[:2]
            predictions = above_null_checked
        else:
            predictions = above_null_checked

        # Add back the minimum null prediction if it was removed because of its low score.
        if version_2_with_negative and not any(p["offsets"] == (0, 0) for p in predictions):
            predictions.append(min_null_prediction)

        # Use the offsets to gather the answer text in the original context.
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = (context[offsets[0] : offsets[1]]).strip()

        # # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # # failure.
        # if len(predictions) == 0 or (len(predictions) == 1 and predictions[0]["text"] == ""):
        #     predictions.insert(0, {"text": "empty", "start_logit": 0.0, "end_logit": 0.0, "score": 0.0})
        #
        # # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # # the LogSumExp trick).
        # scores = np.array([pred.pop("score") for pred in predictions])
        # exp_scores = np.exp(scores - np.max(scores))
        # probs = exp_scores / exp_scores.sum()
        #
        # # Include the probabilities in our predictions.
        # for prob, pred in zip(probs, predictions):
        #     pred["probability"] = prob

        # Pick the best prediction. If the null answer is not possible, this is easy.
        # if not version_2_with_negative:
        #     all_predictions[example["id"]] = predictions[0]["text"]
        #     if len(predictions[0]["text"]) < 3:
        #         print("??empty spoiler", example_index, example["id"], predictions[0]["text"], predictions[0]["start_logit"], )
        # else:
        #     # Otherwise we first need to find the best non-empty prediction.
        #     i = 0
        #     while predictions[i]["text"] == "":
        #         i += 1
        #     best_non_null_pred = predictions[i]
        #
        #     # Then we compare to the null prediction using the threshold.
        #     score_diff = null_score - best_non_null_pred["start_logit"] - best_non_null_pred["end_logit"]
        #     scores_diff_json[example["id"]] = float(score_diff)  # To be JSON-serializable.
        #     if score_diff > null_score_diff_threshold:
        #         all_predictions[example["id"]] = ""
        #     else:
        #         all_predictions[example["id"]] = best_non_null_pred["text"]

        # Make `predictions` JSON-serializable by casting np.float back to float.

        predictions = list(sorted(predictions, key=lambda x:x["pid"]))
        distinct_texts = OrderedDict()
        for _text in [x["text"] for x in predictions]:
            if _text not in distinct_texts:
                distinct_texts[_text] =_text

        joined_text = " ".join(list(distinct_texts.values()))
        all_predictions[example["id"]] = joined_text # " ".join([x["text"] for x in predictions])
        if "answers" in example:
            spoiler_poses = [[int(y) for y in x.split(",")] for x in example["spoiler_poses"].split(" ") if len(x)>0]
            spoiler_context_lines = example["context"].split("\n")
            spoiler_seg_texts = []
            for pid, st_offset, ed_offset in spoiler_poses:
                try:
                    gt_text_part = spoiler_context_lines[pid][st_offset:ed_offset]
                except:
                    gt_text_part = ""
                spoiler_seg_texts.append(gt_text_part)
            gt_text = " ".join(spoiler_seg_texts)
            gt_detail = []
            for _fid, tgt_score_infos in tgt_scores.items():
                for _score, st_lgt, ed_lgt, _sid, _eid, tgt_st_offset, tgt_ed_offset, tgt_string in tgt_score_infos:
                    gt_detail.append({"coord":[int(_fid), int(_sid), int(_eid), int(tgt_st_offset), int(tgt_ed_offset)], "scores":[float(_score), float(st_lgt), float(ed_lgt)], "text":tgt_string})
            gt_json_ent = [{"gt_fulltext":gt_text, "list":gt_detail}]
        else:
            gt_json_ent = []

        all_nbest_json[example["id"]] = gt_json_ent + [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        logger.info(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        logger.info(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            logger.info(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions

def postprocess_qa_predictions_with_beam_search(
    examples,
    features,
    predictions: Tuple[np.ndarray, np.ndarray],
    version_2_with_negative: bool = False,
    n_best_size: int = 20,
    max_answer_length: int = 30,
    start_n_top: int = 5,
    end_n_top: int = 5,
    output_dir: Optional[str] = None,
    prefix: Optional[str] = None,
    is_world_process_zero: bool = True,
):
    """
    Post-processes the predictions of a question-answering model with beam search to convert them to answers that are substrings of the
    original contexts. This is the postprocessing functions for models that return start and end logits, indices, as well as
    cls token predictions.
    Args:
        examples: The non-preprocessed dataset (see the main script for more information).
        features: The processed dataset (see the main script for more information).
        predictions (:obj:`Tuple[np.ndarray, np.ndarray]`):
            The predictions of the model: two arrays containing the start logits and the end logits respectively. Its
            first dimension must match the number of elements of :obj:`features`.
        version_2_with_negative (:obj:`bool`, `optional`, defaults to :obj:`False`):
            Whether or not the underlying dataset contains examples with no answers.
        n_best_size (:obj:`int`, `optional`, defaults to 20):
            The total number of n-best predictions to generate when looking for an answer.
        max_answer_length (:obj:`int`, `optional`, defaults to 30):
            The maximum length of an answer that can be generated. This is needed because the start and end predictions
            are not conditioned on one another.
        start_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top start logits too keep when searching for the :obj:`n_best_size` predictions.
        end_n_top (:obj:`int`, `optional`, defaults to 5):
            The number of top end logits too keep when searching for the :obj:`n_best_size` predictions.
        output_dir (:obj:`str`, `optional`):
            If provided, the dictionaries of predictions, n_best predictions (with their scores and logits) and, if
            :obj:`version_2_with_negative=True`, the dictionary of the scores differences between best and null
            answers, are saved in `output_dir`.
        prefix (:obj:`str`, `optional`):
            If provided, the dictionaries mentioned above are saved with `prefix` added to their names.
        is_world_process_zero (:obj:`bool`, `optional`, defaults to :obj:`True`):
            Whether this process is the main process or not (used to determine if logging/saves should be done).
    """
    assert len(predictions) == 5, "`predictions` should be a tuple with five elements."
    start_top_log_probs, start_top_index, end_top_log_probs, end_top_index, cls_logits = predictions

    assert len(predictions[0]) == len(
        features
    ), f"Got {len(predictions[0])} predicitions and {len(features)} features."

    # Build a map example to its corresponding features.
    example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
    features_per_example = collections.defaultdict(list)
    for i, feature in enumerate(features):
        features_per_example[example_id_to_index[feature["example_id"]]].append(i)

    # The dictionaries we have to fill.
    all_predictions = collections.OrderedDict()
    all_nbest_json = collections.OrderedDict()
    scores_diff_json = collections.OrderedDict() if version_2_with_negative else None

    # Logging.
    logger.setLevel(logging.INFO if is_world_process_zero else logging.WARN)
    logger.info(f"Post-processing {len(examples)} example predictions split into {len(features)} features.")

    # Let's loop over all the examples!
    for example_index, example in enumerate(tqdm(examples)):
        # Those are the indices of the features associated to the current example.
        feature_indices = features_per_example[example_index]

        min_null_score = None
        prelim_predictions = []

        # Looping through all the features associated to the current example.
        for feature_index in feature_indices:
            # We grab the predictions of the model for this feature.
            start_log_prob = start_top_log_probs[feature_index]
            start_indexes = start_top_index[feature_index]
            end_log_prob = end_top_log_probs[feature_index]
            end_indexes = end_top_index[feature_index]
            feature_null_score = cls_logits[feature_index]
            # This is what will allow us to map some the positions in our logits to span of texts in the original
            # context.
            offset_mapping = features[feature_index]["offset_mapping"]
            # Optional `token_is_max_context`, if provided we will remove answers that do not have the maximum context
            # available in the current feature.
            token_is_max_context = features[feature_index].get("token_is_max_context", None)

            # Update minimum null prediction
            if min_null_score is None or feature_null_score < min_null_score:
                min_null_score = feature_null_score

            # Go through all possibilities for the `n_start_top`/`n_end_top` greater start and end logits.
            for i in range(start_n_top):
                for j in range(end_n_top):
                    start_index = int(start_indexes[i])
                    j_index = i * end_n_top + j
                    end_index = int(end_indexes[j_index])
                    # Don't consider out-of-scope answers (last part of the test should be unnecessary because of the
                    # p_mask but let's not take any risk)
                    if (
                        start_index >= len(offset_mapping)
                        or end_index >= len(offset_mapping)
                        or offset_mapping[start_index] is None
                        or offset_mapping[end_index] is None
                    ):
                        continue
                    # Don't consider answers with a length negative or > max_answer_length.
                    if end_index < start_index or end_index - start_index + 1 > max_answer_length:
                        continue
                    # Don't consider answer that don't have the maximum context available (if such information is
                    # provided).
                    if token_is_max_context is not None and not token_is_max_context.get(str(start_index), False):
                        continue
                    prelim_predictions.append(
                        {
                            "offsets": (offset_mapping[start_index][0], offset_mapping[end_index][1]),
                            "score": start_log_prob[i] + end_log_prob[j_index],
                            "start_log_prob": start_log_prob[i],
                            "end_log_prob": end_log_prob[j_index],
                        }
                    )

        # Only keep the best `n_best_size` predictions.
        predictions = sorted(prelim_predictions, key=lambda x: x["score"], reverse=True)[:n_best_size]

        # Use the offsets to gather the answer text in the original context.
        context = example["context"]
        for pred in predictions:
            offsets = pred.pop("offsets")
            pred["text"] = context[offsets[0] : offsets[1]]

        # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid
        # failure.
        if len(predictions) == 0:
            predictions.insert(0, {"text": "", "start_logit": -1e-6, "end_logit": -1e-6, "score": -2e-6})

        # Compute the softmax of all scores (we do it with numpy to stay independent from torch/tf in this file, using
        # the LogSumExp trick).
        scores = np.array([pred.pop("score") for pred in predictions])
        exp_scores = np.exp(scores - np.max(scores))
        probs = exp_scores / exp_scores.sum()

        # Include the probabilities in our predictions.
        for prob, pred in zip(probs, predictions):
            pred["probability"] = prob

        # Pick the best prediction and set the probability for the null answer.
        all_predictions[example["id"]] = predictions[0]["text"]
        if version_2_with_negative:
            scores_diff_json[example["id"]] = float(min_null_score)

        # Make `predictions` JSON-serializable by casting np.float back to float.
        all_nbest_json[example["id"]] = [
            {k: (float(v) if isinstance(v, (np.float16, np.float32, np.float64)) else v) for k, v in pred.items()}
            for pred in predictions
        ]

    # If we have an output_dir, let's save all those dicts.
    if output_dir is not None:
        assert os.path.isdir(output_dir), f"{output_dir} is not a directory."

        prediction_file = os.path.join(
            output_dir, "predictions.json" if prefix is None else f"{prefix}_predictions.json"
        )
        nbest_file = os.path.join(
            output_dir, "nbest_predictions.json" if prefix is None else f"{prefix}_nbest_predictions.json"
        )
        if version_2_with_negative:
            null_odds_file = os.path.join(
                output_dir, "null_odds.json" if prefix is None else f"{prefix}_null_odds.json"
            )

        print(f"Saving predictions to {prediction_file}.")
        with open(prediction_file, "w") as writer:
            writer.write(json.dumps(all_predictions, indent=4) + "\n")
        print(f"Saving nbest_preds to {nbest_file}.")
        with open(nbest_file, "w") as writer:
            writer.write(json.dumps(all_nbest_json, indent=4) + "\n")
        if version_2_with_negative:
            print(f"Saving null_odds to {null_odds_file}.")
            with open(null_odds_file, "w") as writer:
                writer.write(json.dumps(scores_diff_json, indent=4) + "\n")

    return all_predictions, scores_diff_json