from typing import Any, Dict, List, NamedTuple, Optional, Tuple, Union
import re
import numpy as np

def forward_nudge_search(tokenizer, token_ids, position_span, gt_texts_splited):
    ed_forward_nudge = 0
    gt_single_spoiler_found = None
    for nudge_forward in range(4):
        try_text_span = tokenizer.decode(token_ids[position_span[2]:position_span[3] + nudge_forward + 1])
        should_break = False
        for gt_single_spoiler in gt_texts_splited:
            if try_text_span.endswith(gt_single_spoiler):
                ed_forward_nudge = nudge_forward
                should_break = True
                gt_single_spoiler_found = gt_single_spoiler
                break
        if should_break:
            break

    if ed_forward_nudge >= 0 and gt_single_spoiler_found is not None:
        st_forward_nudge = 0
        min_length = len(gt_single_spoiler_found.split(" "))
        nudge_dist = position_span[3] + ed_forward_nudge - position_span[2] - min_length
        for nudge_forward in range(nudge_dist):
            try_text_span = tokenizer.decode(
                token_ids[position_span[2] + nudge_forward:position_span[3] + ed_forward_nudge + 1])
            if len(try_text_span) < len(gt_single_spoiler_found):
                break
            if try_text_span.strip() == gt_single_spoiler_found:
                st_forward_nudge = nudge_forward
                break
        # old_span = tgt_span_text.strip()
        tgt_span_text = tokenizer.decode(
            token_ids[position_span[2] + st_forward_nudge:position_span[3] + ed_forward_nudge + 1])
        print("found fixed nudge: {0}, {1}, {2}, {3}".format(st_forward_nudge, ed_forward_nudge, tgt_span_text.strip(),
                                                             gt_single_spoiler_found))
        old_fid, old_pid, old_st, old_ed = position_span
        nudged_span = (old_fid, old_pid, old_st + st_forward_nudge, old_ed + ed_forward_nudge)

        success = (st_forward_nudge != 0 or ed_forward_nudge != 0)
        return nudged_span, success

    return position_span, False

def backward_nudge_search(tokenizer, token_ids, position_span, gt_texts_splited):
    st_backward_nudge = 0
    gt_single_spoiler_found = None
    for nudge_backward in range(4):
        try_text_span = tokenizer.decode(token_ids[position_span[2]-nudge_backward:position_span[3] + 1])
        should_break = False
        for gt_single_spoiler in gt_texts_splited:
            if try_text_span.startswith(gt_single_spoiler):
                st_backward_nudge = nudge_backward
                should_break = True
                gt_single_spoiler_found = gt_single_spoiler
                break
        if should_break:
            break

    if st_backward_nudge >= 0 and gt_single_spoiler_found is not None:
        ed_backward_nudge = 0
        min_length = len(gt_single_spoiler_found.split(" "))
        nudge_dist = position_span[3] - (position_span[2] - st_backward_nudge) - min_length
        for nudge_backward in range(nudge_dist):
            try_text_span = tokenizer.decode(
                token_ids[position_span[2]-st_backward_nudge: position_span[3] - nudge_backward + 1])
            if len(try_text_span) < len(gt_single_spoiler_found):
                break
            if try_text_span.strip() == gt_single_spoiler_found:
                ed_backward_nudge = nudge_backward
                break
        # old_span = tgt_span_text.strip()
        tgt_span_text = tokenizer.decode(
            token_ids[position_span[2] - st_backward_nudge:position_span[3] - ed_backward_nudge + 1])
        print("found fixed nudge: {0}, {1}, {2}, {3}".format(st_backward_nudge, ed_backward_nudge, tgt_span_text.strip(),
                                                             gt_single_spoiler_found))
        old_fid, old_pid, old_st, old_ed = position_span
        nudged_span = (old_fid, old_pid, old_st - st_backward_nudge, old_ed - ed_backward_nudge)

        success = (st_backward_nudge != 0 or ed_backward_nudge != 0)
        return nudged_span, success

    return position_span, False

def search():
    return

def find_start_poses(offsets, len_by_input_type, min_len=3):
    prev_st = None
    for _pos, (st, ed) in enumerate(offsets):
        if _pos > min_len and prev_st==0:
            return _pos
        prev_st = st
    return len_by_input_type

class EvalPrediction:
    """
    Evaluation output (always contains labels), to be used to compute metrics.

    Parameters:
        predictions (`np.ndarray`): Predictions of the model.
        label_ids (`np.ndarray`): Targets to be matched.
        inputs (`np.ndarray`, *optional*)
    """

    def __init__(
        self,
        predictions: Union[np.ndarray, Tuple[np.ndarray]],
        label_ids: Union[np.ndarray, Tuple[np.ndarray]],
        inputs: Optional[Union[np.ndarray, Tuple[np.ndarray]]] = None,
    ):
        self.predictions = predictions
        self.label_ids = label_ids
        self.inputs = inputs

    def __iter__(self):
        if self.inputs is not None:
            return iter((self.predictions, self.label_ids, self.inputs))
        else:
            return iter((self.predictions, self.label_ids))

    def __getitem__(self, idx):
        if idx < 0 or idx > 2:
            raise IndexError("tuple index out of range")
        if idx == 2 and self.inputs is None:
            raise IndexError("tuple index out of range")
        if idx == 0:
            return self.predictions
        elif idx == 1:
            return self.label_ids
        elif idx == 2:
            return self.inputs

def check_spoilers(json_ent):
    lined_spoilers = []
    position_index = 0
    spoiler_poses = json_ent["spoilerPositions"]
    title_plus_paragraph = [json_ent["postText"][0]] + json_ent["targetParagraphs"]
    for single_spoiler in json_ent["spoiler"]:
        st_coord, ed_coord = spoiler_poses[position_index]
        if st_coord[0] == ed_coord[0]:
            paragraph_text = title_plus_paragraph[st_coord[0]+1]
        else:
            paragraph_text = "\n".join(title_plus_paragraph[st_coord[0]+1:ed_coord[0]+2])
        possible_lines = [x for x in single_spoiler.split("\n") if x != ""]
        for line_spoiler in possible_lines:
            matched = re.search(line_spoiler, paragraph_text)
            aaa = 1
        position_index += 1
    return

def lcs(s1, s2):
    m = [[0] * (1 + len(s2)) for i in range(1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):

        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] > longest:
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest], x_longest - longest, x_longest

def find_spoiler_pid(json_ent, eid=-1):
    lined_spoilers = []
    position_index = 0
    # if json_ent["uuid"] == "8bd60f06-a4cb-4694-91ed-e6c33cbc6183":
    #     print("debug use")

    spoiler_poses = json_ent["spoilerPositions"]
    title_plus_paragraph = [json_ent["targetTitle"]] + json_ent["targetParagraphs"]
    spoiler_texts = json_ent["spoiler"]
    verified_spoilers = []
    verified_positions = []
    if len(spoiler_poses) == len(spoiler_texts):
        for single_coord, single_text in zip(spoiler_poses, spoiler_texts):
            if len(single_coord) == 1:
                single_coord = [single_coord[0], single_coord[0]]
                single_coord[1][1] = single_coord[1][0] + len(single_text)

            if single_coord[0][0] == single_coord[1][0]:
                pids = [single_coord[0][0]+ 1]
            else:
                min_pid = min(single_coord[0][0], single_coord[1][0]) + 1
                max_pid = max(single_coord[0][0], single_coord[1][0]) + 1
                pids = list(range(min_pid, max_pid+1))

            spoiler_text_spans = [x for x in single_text.split("\n") if x != ""]
            if len(pids) == len(spoiler_text_spans):
                if len(pids) == 1:
                    existing_paragraph = title_plus_paragraph[pids[0]]
                    if len(existing_paragraph.split(spoiler_text_spans[0])) >= 2:
                        verified_spoilers.append(single_text)
                        verified_positions.append(single_coord)
                    else:
                        search_p_start, search_p_end = max(pids[0] - 3, 0), min(pids[0]+3, len(title_plus_paragraph))
                        for _pid in range(search_p_start, search_p_end):
                            if len(title_plus_paragraph[_pid].split(spoiler_text_spans[0])) >= 2:
                                verified_spoilers.append(single_text)
                                modified_coord = [[_pid-1, single_coord[0][1]], [_pid-1, single_coord[1][1]]]
                                verified_positions.append(modified_coord)
                                break
                else:
                    # across mlti lines
                    print("???")
            else:
                # mismatch spoiler position line_ct and spoiler text line_ct
                single_line_gt_text = " ".join(spoiler_text_spans)
                for _pid in pids:
                    try:
                        lcs_in_line, offset_st, offset_ed = lcs(title_plus_paragraph[_pid], single_line_gt_text)
                        if len(lcs_in_line) > 1:
                            modified_coord = [[_pid - 1, offset_st], [_pid - 1, offset_ed]]
                            verified_positions.append(modified_coord)
                            verified_spoilers.append(lcs_in_line)
                    except:
                        continue
                print("reshape spoiler positions which across lines="+str(pids))
    else:
        # mismatch spoiler number
        print("???")
    return {"spoiler":verified_spoilers, "spoilerPositions":verified_positions}