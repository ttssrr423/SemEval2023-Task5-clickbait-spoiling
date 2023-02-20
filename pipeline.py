import sys
import logging
from data_process.utils import forward_nudge_search, backward_nudge_search, find_start_poses, find_spoiler_pid, lcs
import pandas as pd
import re
from dataclasses import dataclass, field
from datasets import load_dataset, load_metric, Dataset
from typing import Optional
from modelling.modified import AutoConfigMod, DebertaAddon
from transformers import (
    # AutoConfig,
    AutoModelForQuestionAnswering,
    AutoTokenizer,
    DataCollatorWithPadding,
    EvalPrediction,
    HfArgumentParser,
    PreTrainedTokenizerFast,
    TrainingArguments,
    default_data_collator,
    set_seed,
    XLNetTokenizerFast)

logger = logging.getLogger(__name__)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to directory to store the pretrained models downloaded from huggingface.co"},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=True,
        metadata={
            "help": "Whether to pad all samples to `max_seq_length`. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch (which can "
            "be faster on GPU but will be slower on TPU)."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    version_2_with_negative: bool = field(
        default=False, metadata={"help": "If true, some of the examples do not have an answer."}
    )
    null_score_diff_threshold: float = field(
        default=0.0,
        metadata={
            "help": "The threshold used to select the null answer: if the best answer has a score that is less than "
            "the score of the null answer minus this threshold, the null answer is selected for this example. "
            "Only useful when `version_2_with_negative=True`."
        },
    )
    doc_stride: int = field(
        default=128,
        metadata={"help": "When splitting up a long document into chunks, how much stride to take between chunks."},
    )
    n_best_size: int = field(
        default=20,
        metadata={"help": "The total number of n-best predictions to generate when looking for an answer."},
    )
    max_answer_length: int = field(
        default=30,
        metadata={
            "help": "The maximum length of an answer that can be generated. This is needed because the start "
            "and end predictions are not conditioned on one another."
        },
    )

def input_data_to_squad_format(inp, load_spoiler=False):
    #     return pd.DataFrame([{'id': i['uuid'], 'title': i['targetTitle'], 'question': ' '.join(i['postText']),
    #                           'context': i['targetTitle'] + ' - ' + (' '.join(i['targetParagraphs'])),
    #                           'answers': " ".join(i["spoiler"])} for i in inp])
    def get_coords(start_coord, ed_coord, context=None):
        pid1, pid2 = start_coord[0], ed_coord[0]
        if pid1 != pid2 or start_coord[1]>ed_coord[1]:
            assert context is not None
            lined_contexts = context.split("\n")
            if pid1 > pid2:
                return "-1,-1,-1"
            prev_paragraph = lined_contexts[pid1 + 1]
            spoiler_upper_len = len(prev_paragraph)
            upper_spoiler = "{0},{1},{2}".format(pid1+1, start_coord[1], spoiler_upper_len)
            lower_spoiler = "{0},{1},{2}".format(pid2+1, 0, ed_coord[1])
            return upper_spoiler+" "+lower_spoiler
        else:
            # title adding 1 to existing pids
            return "{0},{1},{2}".format(pid1+1, start_coord[1], ed_coord[1])

    #
    if not load_spoiler:
        return pd.DataFrame([{'id': i['uuid'], 'title': i['targetTitle'].replace("\n", " "), 'question': ' '.join([x.replace("\n", " ") for x in i['postText']]),
                              'context': i['targetTitle'].replace("\n", " ") + '\n' + ('\n'.join([x.replace("\n", " ") for x in i['targetParagraphs']])),
                              'answers': 'not available for predictions', 'spoiler_poses': ""} for i in inp])

    else:
        spoiler_pids_checked = [find_spoiler_pid(i, eid=_eid) for _eid, i in enumerate(inp)]
        for eid, i in enumerate(inp):
            i["spoiler"] = spoiler_pids_checked[eid]["spoiler"]
            i["spoilerPositions"] = spoiler_pids_checked[eid]["spoilerPositions"]
        return pd.DataFrame([{'id': i['uuid'], 'title': i['targetTitle'].replace("\n", " "), 'question': ' '.join([x.replace("\n", " ") for x in i['postText']]),
                          'context': i['targetTitle'].replace("\n", " ") + '\n' + ('\n'.join([x.replace("\n", " ") for x in i['targetParagraphs']])), 'tags':i["tags"][0],
                          'answers': "\n".join(i["spoiler"]), 'spoiler_poses': " ".join([get_coords(itm[0], itm[1], context=i['targetTitle'].replace("\n", " ") + '\n' + ('\n'.join([x.replace("\n", " ") for x in i['targetParagraphs']]))) for itm in i["spoilerPositions"] if len(itm)==2])} for i in inp])

global args_tuples
args_tuples = [None]

def setup_model(args, using_absolute_position=False):
    model_args, data_args, training_args = HfArgumentParser((ModelArguments, DataTrainingArguments, TrainingArguments)).parse_args_into_dataclasses(args=args)
    args_tuples[0] = (model_args, data_args, training_args)
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    logger.setLevel(logging.INFO)# if training_args.should_log else logging.WARN

    config = AutoConfigMod.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    config.position_biased_input=using_absolute_position
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=True,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    # model = AutoModelForQuestionAnswering.from_pretrained(
    #     model_args.model_name_or_path,
    #     from_tf=bool(".ckpt" in model_args.model_name_or_path),
    #     config=config,
    #     cache_dir=model_args.cache_dir,
    #     revision=model_args.model_revision,
    #     use_auth_token=True if model_args.use_auth_token else None,
    # )
    model = DebertaAddon.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    return config, tokenizer, model, (model_args, data_args, training_args)

def setup_features(tokenizer, val_dataset, train_dataset=None, args_parsed=None):
    if args_parsed is not None:
        model_args, data_args, training_args = args_parsed
    elif args_tuples[0] is None:
        model_args, data_args = HfArgumentParser(
            (ModelArguments, DataTrainingArguments)).parse_args_into_dataclasses(args=[])
    else:
        model_args, data_args, training_args = args_tuples[0]

    question_column_name = "question"
    context_column_name = "context"
    answer_column_name = "answers"
    pad_on_right = tokenizer.padding_side == "right"
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    # Validation preprocessing
    has_target_info = [False]

    def prepare_validation_features(examples):
        # Tokenize our examples with truncation and maybe padding, but keep the overflows using a stride. This results
        # in one example possible giving several features when a context is long, each of those features having a
        # context that overlaps a bit the context of the previous feature.

        question_texts = examples[question_column_name]
        context_texts = examples[context_column_name]
        texts_with_pids = [x.split("\n") for x in context_texts]
        assert pad_on_right

        recovered_tokenize_text = [x[0] + " - " + (" ".join(x[1:])) for x in texts_with_pids]
        pid_offset_full_positions = [[0] for _eid in range(len(texts_with_pids))]
        for _eid in range(len(texts_with_pids)):
            for _pid in range(len(texts_with_pids[_eid])):
                if _pid == 0:
                    paragraph_text = texts_with_pids[_eid][_pid] + " -"
                else:
                    paragraph_text = " "+texts_with_pids[_eid][_pid]
                accumulated_len = pid_offset_full_positions[_eid][-1]
                pid_offset_full_positions[_eid].append(accumulated_len+len(paragraph_text))

        tokenized_examples = tokenizer(
            examples[question_column_name if pad_on_right else context_column_name],
            # examples[context_column_name if pad_on_right else question_column_name],
            recovered_tokenize_text,
            truncation="only_second" if pad_on_right else "only_first",
            max_length=max_seq_length,
            stride=data_args.doc_stride,
            return_overflowing_tokens=True,
            return_offsets_mapping=True,
            padding="max_length" if data_args.pad_to_max_length else False,
        )
        assert pad_on_right
        # Since one example might give us several features if it has a long context, we need a map from a feature to
        # its corresponding example. This key gives us just that.
        sample_mapping = tokenized_examples.pop("overflow_to_sample_mapping")
        context_start_poses_via_type = [len(x)-sum(x) for x in tokenized_examples["token_type_ids"]]
        neighbours = [[] for _ in range(len(examples[question_column_name]))]
        for _fid, _eid in enumerate(sample_mapping):
            neighbours[_eid].append(_fid)
        first_fids = [min(x) for x in neighbours]

        # context_offset_positions = [len(x[0] + " -") for x in texts_with_pids]
        # context_offset_positions_by_feature = [context_offset_positions[sample_mapping[_fid]] for _fid in range(len(sample_mapping))]
        question_min_lengths = [len(question_texts[sample_mapping[fid]].split(" ")) for fid in range(len(sample_mapping))]
        context_start_poses = [find_start_poses(tokenized_examples["offset_mapping"][fid], context_start_poses_via_type[fid], min_len=question_min_lengths[fid]) for fid in range(len(sample_mapping))]

        for fid, cont_st_pos in enumerate(context_start_poses):
            input_offset = tokenized_examples["offset_mapping"][fid]
            questions = input_offset[:context_start_poses[fid]]
            final_pos = max(questions[-5:], key=lambda x:x[1])[1]
            q_text = question_texts[sample_mapping[fid]]
            q_text_seg_by_len = q_text[:final_pos]
            # if len(q_text) != final_pos:
            #     print("question length not passed...")
            #     print(q_text, q_text_seg_by_len)
            # if len([x for x in tokenized_examples["token_type_ids"][fid] if x==0]) != context_start_poses[fid]:
            #     print("mismatch input_type_ids, check for type")
            #     # tokenized_examples["token_type_ids"][fid] = [0]*context_start_poses[fid] + [1]*(len(tokenized_examples["token_type_ids"][fid])-context_start_poses[fid])


        def locate_pid_via_full_offset(_eid, prev_shift, token_offsets):
            result_pids = []
            for token_pos, (i, j) in enumerate(token_offsets):
                full_i, full_j = prev_shift + i, prev_shift + j
                for _pid, paragraph_start_pos in enumerate(pid_offset_full_positions[_eid]):
                    next_start = pid_offset_full_positions[_eid][_pid+1]
                    if i==0 and j==0:
                        result_pids.append(-1)
                        break
                    elif full_i >=paragraph_start_pos and full_j <=next_start:
                        result_pids.append(_pid)
                        break
                    else:
                        continue
                assert token_pos+1 == len(result_pids)
            return result_pids

        accumulated_fail = 0
        eid_segs = {}
        sample_paragraph_ids = []
        is_first_feature = []
        eid2fids = {}
        for fid in range(len(context_start_poses)):
            eid = sample_mapping[fid]
            if fid==first_fids[eid]:
                is_first_feature.append(1)
            else:
                is_first_feature.append(0)
            if eid not in eid2fids: eid2fids[eid] = []
            eid2fids[eid].append(fid)
            input_offset = tokenized_examples["offset_mapping"][fid]
            context_offset = input_offset[context_start_poses[fid]:]
            context_text = recovered_tokenize_text[eid]
            text_tokenized_for_check = [context_text[i:j] for i,j in context_offset]

            if eid not in eid_segs:
                eid_segs[eid] = "".join([x for x in text_tokenized_for_check if len(x)>1])
            else:
                eid_segs[eid] = eid_segs[eid] + " " + ("".join([x for x in text_tokenized_for_check if len(x)>1]))

            token_pids = locate_pid_via_full_offset(eid, 0, context_offset)
            sample_paragraph_ids.append([-1]*context_start_poses[fid] + token_pids)
            prev_pid = -1
            paragraph_start_ids = [start_id for start_id in pid_offset_full_positions[eid]]

            # print([recovered_tokenize_text[eid][paragraph_start_ids[x]:paragraph_start_ids[x] + 20] for x in
            #        range(len(paragraph_start_ids))])
            # for pos_id, _pid in enumerate(token_pids):
            #     if (prev_pid != _pid):
            #         print(_pid, context_offset[pos_id][0], pos_id)
            #         print("".join([recovered_tokenize_text[eid][st:ed] for st,ed in context_offset[pos_id:pos_id+10]]))
            #         print(texts_with_pids[eid][_pid][:20])
            #         print("------------------")
            #     prev_pid = _pid

        tokenized_examples["is_first_feature"] = is_first_feature

        absolute_position_ids = []
        for eid in range(len(texts_with_pids)):
            fids = eid2fids[eid]
            positions_on_each_paragraph = -1

            prev_pid = -1
            cursor_pos = 0
            for fid_in_e in fids:
                abs_position_ids = []
                pids_in_f = sample_paragraph_ids[fid_in_e]
                for _pid in pids_in_f:
                    if _pid < 0:
                        abs_position_ids.append(-1)
                    else:
                        if prev_pid < _pid:
                            cursor_pos = 0
                            abs_position_ids.append(cursor_pos)
                        else:
                            abs_position_ids.append(cursor_pos)
                        cursor_pos += 1
                        prev_pid = _pid
                absolute_position_ids.append(abs_position_ids)
        tokenized_examples["absolute_positions"] = absolute_position_ids

        if has_target_info[0]:
            spoiler_poses = [[[int(z) for z in y.split(",") if z != ""] for y in x.split(" ") if y!=""] for x in examples["spoiler_poses"]]
            gt_spoilers = [x.split("\n") for x in examples["answers"]]

            eid2fids = {}
            for fid, eid in enumerate(sample_mapping):
                if eid not in eid2fids:
                    eid2fids[eid] = [fid]
                else:
                    eid2fids[eid].append(fid)

            spoiler_coordinates = []
            spoiler_texts_lined = [x.split("\n") for x in examples["answers"]]

            for eid, example_poses in enumerate(spoiler_poses):
                uuid = examples["id"][eid]
                # if uuid == "8bd60f06-a4cb-4694-91ed-e6c33cbc6183":
                #     print("debug use")

                pid2spoilers = {}
                sid2pid = {}
                for sid, example_spoil_pos in enumerate(example_poses):
                    pid, st_pos, ed_pos = example_spoil_pos
                    # try:
                    #     matched_spoilers = False
                    #     if pid+st_pos+ed_pos != -3:
                    #         tgt_paragraph = texts_with_pids[eid][pid]
                    #         tgt_spoiler = tgt_paragraph[st_pos:ed_pos]
                    #         if tgt_spoiler == gt_spoilers[eid][sid]:
                    #             matched_spoilers = True
                    #             continue
                    # except:
                    #     matched_spoilers = False
                    if pid not in pid2spoilers:
                        pid2spoilers[pid] = []
                    pid2spoilers[pid].append((st_pos, ed_pos, sid))
                    sid2pid[sid] = pid
                for _pid in pid2spoilers.keys(): # sort by ed pos to make next window continue safe
                    pid2spoilers[_pid] = list(sorted(pid2spoilers[_pid], key=lambda x:x[1]))

                gt_texts = examples["answers"][eid]
                gt_texts_splited = [x for x in gt_texts.split("\n") if len(x)>0]
                gt_texts_group_by_pid = {}
                for _sid, s_text in enumerate(gt_texts_splited):
                    pid = sid2pid[_sid]
                    if pid not in gt_texts_group_by_pid: gt_texts_group_by_pid[pid]=[]
                    gt_texts_group_by_pid[pid].append((_sid, s_text))


                archiving_new_starts = {}
                for f_order, fid in enumerate(eid2fids[eid]):
                    para_ids = sample_paragraph_ids[fid]
                    token_ids = tokenized_examples["input_ids"][fid]
                    context_token_id_shift = context_start_poses[fid]
                    input_offset = tokenized_examples["offset_mapping"][fid]
                    context_offset = input_offset[context_token_id_shift:]
                    start_paragraph_offset = 0
                    new_starts = []
                    start_paragraph_id = -1
                    for _t in range(len(para_ids)):
                        if start_paragraph_id < para_ids[_t]:
                            if para_ids[_t] in archiving_new_starts:
                                start_paragraph_offset = archiving_new_starts[para_ids[_t]]
                            else:
                                start_paragraph_offset = input_offset[_t][0]
                                archiving_new_starts[para_ids[_t]] = start_paragraph_offset
                            # print(tokenizer.decode(token_ids[_t:_t+10]))
                            if para_ids[_t] in pid2spoilers:
                                # print(_t, para_ids[_t])
                                lookingfor_next_window = False
                                for spoiler_para_st, spoiler_para_ed, _sid in pid2spoilers[para_ids[_t]]:
                                    if para_ids[_t] == 0:
                                        modified_shift = 0
                                    else:
                                        modified_shift = 1
                                    spoiler_start = start_paragraph_offset + spoiler_para_st + modified_shift
                                    spoiler_end = start_paragraph_offset + spoiler_para_ed + modified_shift
                                    spoiler_texts = recovered_tokenize_text[eid][spoiler_start:spoiler_end]
                                    input_raw_text = tokenizer.decode(token_ids[context_start_poses[fid]:], skip_special_tokens=True)

                                    expected_pos_id_st_candidates = [(_ind, x[0]-spoiler_start) for _ind, x in enumerate(input_offset) if x[0]-spoiler_start<=0 and _ind>=context_start_poses[fid]]
                                    expected_pos_id_ed_candidates = [(_ind, x[1]-spoiler_end) for _ind, x in enumerate(input_offset) if x[1]-spoiler_end>=0 and _ind>=context_start_poses[fid]]
                                    if len(expected_pos_id_st_candidates)==0 or len(expected_pos_id_ed_candidates)==0:
                                        # print("next window slide should have the spoiler")
                                        if len(expected_pos_id_ed_candidates)==0:
                                            lookingfor_next_window = True
                                        continue
                                    expected_pos_id_st = max(expected_pos_id_st_candidates, key=lambda x:x[1])
                                    expected_pos_id_ed = min(expected_pos_id_ed_candidates, key=lambda x:x[1])
                                    position_span = (fid, para_ids[_t], expected_pos_id_st[0], expected_pos_id_ed[0])
                                    tgt_span_text = tokenizer.decode(token_ids[position_span[2]:position_span[3]+1])
                                    if expected_pos_id_st[0] > expected_pos_id_ed[0]:
                                        # start after end, possibly due to residual of previous paragraph, with prev paragraph having a spoiler
                                        continue
                                    tgt_validate_pass = False
                                    paragraph_limited_spoilers = gt_texts_group_by_pid[para_ids[_t]] if para_ids[_t] in gt_texts_group_by_pid else []
                                    paragraph_spoiler_text_only = [x[1] for x in paragraph_limited_spoilers]
                                    if tgt_span_text.strip() not in paragraph_spoiler_text_only:
                                        print("fixing mismatch: ", tgt_span_text.strip(), "|", " ".join(paragraph_spoiler_text_only), "|", texts_with_pids[eid][para_ids[_t]])
                                        # pattern_gt = gt_texts.replace("\.", ".").replace(".", "\.")
                                        current_partial_context_text = tokenizer.decode(token_ids[context_token_id_shift:], skip_special_tokens=True)
                                        nearest_commons = []
                                        context_start_offset = context_offset[0][0]
                                        for single_spoiler_gt in paragraph_spoiler_text_only:
                                            largest_common, common_st, common_ed = lcs(current_partial_context_text, single_spoiler_gt)
                                            if common_ed - common_st > len(single_spoiler_gt) * 0.6:
                                                shifted_st = context_start_offset + common_st + modified_shift
                                                shifted_ed = context_start_offset + common_ed + modified_shift
                                                dist = abs(spoiler_start - shifted_st) + abs(spoiler_end - shifted_ed)
                                                nearest_commons.append((dist, shifted_st, shifted_ed))
                                        if len(nearest_commons) > 0:
                                            nearest_commons = list(sorted(nearest_commons, key=lambda x:x[0]))
                                            offset_st_by_lcs, offset_ed_by_lcs = nearest_commons[0][1], nearest_commons[0][2]

                                            expected_pos_id_st_candidates = [(_ind, x[0] - offset_st_by_lcs) for _ind, x in
                                                                             enumerate(input_offset) if
                                                                             x[0] - offset_st_by_lcs <= 0 and _ind >=
                                                                             context_start_poses[fid]]
                                            expected_pos_id_ed_candidates = [(_ind, x[1] - offset_ed_by_lcs) for _ind, x in
                                                                             enumerate(input_offset) if
                                                                             x[1] - offset_ed_by_lcs >= 0 and _ind >=
                                                                             context_start_poses[fid]]
                                            if len(expected_pos_id_st_candidates) == 0 or len(
                                                    expected_pos_id_ed_candidates) == 0:
                                                # print("next window slide should have the spoiler")
                                                tgt_validate_pass = False
                                            else:
                                                expected_pos_id_st = max(expected_pos_id_st_candidates, key=lambda x: x[1])
                                                expected_pos_id_ed = min(expected_pos_id_ed_candidates, key=lambda x: x[1])
                                                position_span = [fid, para_ids[_t], expected_pos_id_st[0], expected_pos_id_ed[0]]
                                                assert expected_pos_id_st <= expected_pos_id_ed
                                                tgt_validate_pass = True
                                        else:
                                            tgt_validate_pass = False
                                    else:
                                        tgt_validate_pass = True

                                    if tgt_validate_pass:
                                        tgt_span_text = tokenizer.decode(token_ids[position_span[2]:position_span[3]+1], skip_special_tokens=True).strip()
                                        if len(tgt_span_text) < 3:
                                            print("???? short gt")
                                        else:
                                            lcs_with_gt, lcs_st, lcs_ed = lcs(gt_texts, tgt_span_text)
                                            assert len(lcs_with_gt)>1
                                        spoiler_coordinates.append((position_span, tgt_span_text, gt_texts))
                                    if len(tgt_span_text) == "":
                                        print("??? empty span")

                                if lookingfor_next_window:
                                    pass
                                    # break
                            new_starts.append((_t, input_offset[_t][0]))
                        start_paragraph_id = max(para_ids[_t], start_paragraph_id)

        # For evaluation, we will need to convert our predictions to substrings of the context, so we keep the
        # corresponding example_id and we will store the offset mappings.
        tokenized_examples["example_id"] = []

        for i in range(len(tokenized_examples["input_ids"])):
            # Grab the sequence corresponding to that example (to know what is the context and what is the question).
            sequence_ids = tokenized_examples.sequence_ids(i)
            context_index = 1 if pad_on_right else 0

            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = sample_mapping[i]
            tokenized_examples["example_id"].append(examples["id"][sample_index])

            # Set to None the offset_mapping that are not part of the context so it's easy to determine if a token
            # position is part of the context or not.
            tokenized_examples["offset_mapping"][i] = [
                (o if sequence_ids[k] == context_index else None)
                for k, o in enumerate(tokenized_examples["offset_mapping"][i])
            ]

            tokenized_examples["paragraph_ids"] = sample_paragraph_ids

        if has_target_info[0]:
            spoiler_types = examples["tags"]
            # tag_is_first_feature = {}
            tag_ids = []
            for i in range(len(tokenized_examples["input_ids"])):
                eid = sample_mapping[i]
                spoiler_type_str = spoiler_types[eid]
                # if eid not in tag_is_first_feature:
                if spoiler_type_str == "phrase":
                    type_id = 0
                elif spoiler_type_str == "passage":
                    type_id = 1
                else:
                    type_id = 2
                tag_ids.append(type_id)
                # else:
                #     tag_ids.append(-1)
            tokenized_examples["tags"] = tag_ids

            dataset_starts = []
            dataset_ends = []
            span_poses = []
            for i in range(len(tokenized_examples["input_ids"])):
                feature_is_start = len(tokenized_examples["input_ids"][i]) * [0]
                feature_is_end = len(tokenized_examples["input_ids"][i]) * [0]
                feature_is_start[0] = 1
                feature_is_end[0] = 1
                dataset_starts.append(feature_is_start)
                dataset_ends.append(feature_is_end)
                span_poses.append("")
            tokenized_examples["start_target"] = dataset_starts
            tokenized_examples["end_target"] = dataset_ends
            tokenized_examples["span_pos"] = span_poses

            feature_tgt_spoiler_texts = ["" for _ in range(len(tokenized_examples["input_ids"]))]
            for coord_info, tgt_span, _ in spoiler_coordinates:
                fid, pid, st_pos, ed_pos = coord_info
                tokenized_examples["start_target"][fid][st_pos] = 1
                tokenized_examples["end_target"][fid][ed_pos] = 1
                tokenized_examples["start_target"][fid][0] = 0
                tokenized_examples["end_target"][fid][0] = 0
                tokenized_examples["span_pos"][fid] = tokenized_examples["span_pos"][fid] + "|" + (",".join([str(fid), str(pid), str(st_pos), str(ed_pos)]))
                feature_tgt_spoiler_texts[fid] = feature_tgt_spoiler_texts[fid] + (tgt_span if (len(tgt_span)>0 and tgt_span[0]==" ") else " "+tgt_span)

            tokenized_examples["target_partial_spoiler"] = feature_tgt_spoiler_texts

        return tokenized_examples

    if train_dataset is not None:
        has_target_info[0] = True

    predict_examples = Dataset.from_pandas(val_dataset)
    predict_dataset = predict_examples.map(
        prepare_validation_features,
        batched=True,
        num_proc=data_args.preprocessing_num_workers,
        remove_columns=predict_examples.column_names,
        load_from_cache_file=not data_args.overwrite_cache,
    )

    if train_dataset is not None:
        has_target_info[0] = True
        train_examples = Dataset.from_pandas(train_dataset)
        train_dataset_prepared = train_examples.map(
            prepare_validation_features,
            batched=True,
            num_proc=data_args.preprocessing_num_workers,
            remove_columns=predict_examples.column_names,
            load_from_cache_file=not data_args.overwrite_cache,
        )

        return predict_dataset, train_dataset_prepared
    return predict_dataset

