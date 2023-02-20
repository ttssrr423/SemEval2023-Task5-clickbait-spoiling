import torch
import random

upper_triangle = []

def calc_modifier_losses(pred_out, modifier, span_poses, tgt_span_print_info, negative_sample_num=20):
    if len(upper_triangle)==0:
        triangle_mask = torch.zeros_like(modifier,dtype=torch.float32, device=modifier.device)
        for _i in range(modifier.shape[0]):
            for _j in range(modifier.shape[1]):
                if _i <= _j:
                    triangle_mask[_i, _j].fill_(1.0)
        upper_triangle.append(triangle_mask)

    maxlen = modifier.shape[0]
    positive_mask = torch.zeros_like(modifier,dtype=torch.float32)
    whitelist = torch.zeros_like(modifier, dtype=torch.float32)
    gt_coordinates = {}
    if len(span_poses) > 0:
        for span_coord in span_poses:
            _fid, _pid, _st, _ed = [int(x) for x in span_coord]
            gt_coordinates[(_st, _ed)] = False
            positive_mask[_st, _ed].fill_(1.0)
            neighbour_margin = 2
            neighbour_start, neighbour_end = max(0, _st-neighbour_margin), min(_ed+neighbour_margin, maxlen-1)
            for t1 in range(neighbour_start, neighbour_end+1, 1):
                for t2 in range(t1, neighbour_end+1, 1):
                    whitelist[t1, t2].fill_(1.0)
    else:
        gt_coordinates[(0, 0)] = False
        positive_mask[0, 0].fill_(1.0)
        whitelist[0,0].fill_(1.0)

    positive_weight = 10.0
    positive_weights = positive_weight * torch.ones_like(modifier)
    positive_weights[0, 0].fill_(1.0)
    additional_ct = 0.0 if (0,0) in gt_coordinates else (positive_weight-1.0)*len(gt_coordinates)
    modifier_losses = torch.where(positive_mask.eq(1.0), -positive_weights * modifier, torch.relu(modifier))
    positive_modifiers = torch.where(positive_mask.eq(1.0), modifier, torch.zeros_like(modifier))
    positive_mean = positive_modifiers.sum() / positive_mask.sum()

    negative_modifier_flag = (torch.ones_like(whitelist)-whitelist) * upper_triangle[0]
    negative_modifiers = torch.where(negative_modifier_flag.eq(1.0), modifier, torch.zeros_like(modifier))
    negative_mean = negative_modifiers.sum() / ((modifier.shape[0]*modifier.shape[1])-len(gt_coordinates))

    weighted_loss = modifier_losses.sum() / (positive_mask.shape[0]*positive_mask.shape[1] + additional_ct)
    prt_str_info = "+modifier: " + str(float(positive_mean.clone().detach().to("cpu"))) + " | -modifier: "+ str(float(negative_mean.clone().detach().to("cpu"))) + " | tgts:" + str(tgt_span_print_info)
    return weighted_loss, prt_str_info

def calc_metric_contrastive_losses(pred_out, scoring_metric, span_poses, tgt_span_print_info, negative_sample_num=20):
    descending_ids = torch.argsort(scoring_metric.reshape([-1]), descending=True)
    argsort_score_ids = [(x // pred_out.end_logits.shape[1], x % pred_out.end_logits.shape[1]) for x in
                         list(descending_ids.clone().detach().to("cpu").numpy())]
    gt_coordinates = {}
    if len(span_poses) > 0:
        for span_coord in span_poses:
            _fid, _pid, _st, _ed = [int(x) for x in span_coord]
            gt_coordinates[(_st, _ed)] = False
    else:
        gt_coordinates[(0, 0)] = False

    negative_samples = []
    examined_positive_ct = 0
    positive_coordinates = []
    coord2rank = {}
    for rank, coord in enumerate(argsort_score_ids):
        # if (type_ids[coord[0]] == 0 and coord[0]>0) or (type_ids[coord[1]] == 0 and coord[1]>0):
        #     # coordinate start or end position is not index=0(CLS) and not within context region
        #     continue
        if examined_positive_ct >= len(gt_coordinates):
            break
        if coord in gt_coordinates:
            positive_coordinates.append(coord)
            coord2rank[coord] = rank
            examined_positive_ct += 1
        else:
            # nearby_coord = is_nearby(coord, gt_coordinates)
            # if nearby_coord is not None:
            #     positive_coordinates.append(coord)
            #     coord2rank[coord] = rank
            #     examined_positive_ct += 1
            # else:
            negative_samples.append(coord)
            coord2rank[coord] = rank

    if negative_sample_num > 0 and len(negative_samples) > 0:
        partial_scores = [negative_samples[0]]
        decay_prob = 0.9
        neg_samp_prob = 0.9
        sampled_ni = set([0])
        for ni in range(min(negative_sample_num - 1, len(negative_samples) - 1)):
            if random.random() < neg_samp_prob:
                partial_scores.append(negative_samples[ni + 1])
                neg_samp_prob = neg_samp_prob * decay_prob
                sampled_ni.add(ni)
        if len(partial_scores) < negative_sample_num:
            for ni in range(1, len(negative_samples)):
                if ni not in sampled_ni:
                    partial_scores.append(negative_samples[ni])
                    if len(partial_scores) >= negative_sample_num:
                        break
        negative_samples = partial_scores

    positive_samp_scores = [scoring_metric[coord[0], coord[1]] for coord in positive_coordinates]
    positive_ranks = [coord2rank[coord] for coord in positive_coordinates]
    positive_tensor_scores = torch.stack(positive_samp_scores, dim=0)
    positive_weighted_score = 0.5 * (
                positive_tensor_scores.max() + positive_tensor_scores.mean())  # positive_tensor_scores.min() # 0.5 * (positive_tensor_scores.mean()+positive_tensor_scores.min())

    if len(negative_samples) > 0:
        negative_samp_scores = [scoring_metric[coord[0], coord[1]] for coord in negative_samples]
        negative_tensor_scores = torch.stack(negative_samp_scores, dim=0)
        negative_ranks = [coord2rank[coord] for coord in negative_samples]
        # advantage_loss_unnormalized_all = torch.nn.ELU(alpha=6.0)(negative_tensor_scores - positive_tensor_scores.mean().unsqueeze(0))
        # advantage_loss_unnormalized = 0.5 * (advantage_loss_unnormalized_all.max() + advantage_loss_unnormalized_all.mean())
        advantage_loss_unnormalized_all = torch.relu(negative_tensor_scores - positive_weighted_score.unsqueeze(0))
        advantage_loss_unnormalized = 0.5 * (
                    advantage_loss_unnormalized_all.sum() + advantage_loss_unnormalized_all.max())
        normalizing_factor = scoring_metric.abs().mean()  # .detach()
        advantage_loss = advantage_loss_unnormalized / normalizing_factor
    else:
        positive_ct = len(gt_coordinates)
        neg_top_num = max(4, negative_sample_num)
        negative_tops = [scoring_metric[coord[0], coord[1]] for coord in
                         argsort_score_ids[positive_ct:positive_ct + neg_top_num]]
        negative_top_scores = torch.stack(negative_tops, dim=0)
        negative_coords = [coord for coord in argsort_score_ids[positive_ct:positive_ct + neg_top_num]]
        negative_ranks = [coord2rank[coord] for coord in negative_coords if coord in coord2rank]
        advantage_losses_unnormalized = torch.nn.ELU(alpha=6.0)(
            negative_top_scores - (positive_weighted_score.unsqueeze(0)))
        advantage_loss_unnormalized = 0.5 * (advantage_losses_unnormalized.mean() + advantage_losses_unnormalized.max())
        normalizing_factor = scoring_metric.abs().mean()
        advantage_loss = advantage_loss_unnormalized / normalizing_factor

    negative_rank_mean = "NA" if len(negative_ranks) == 0 else str(sum(negative_ranks) / len(negative_ranks))
    negative_rank_min = "NA" if len(negative_ranks) == 0 else str(min(negative_ranks))
    print_info_str = "#spoiler avg={0}, max={1}, min={4}| #-ve avg={2}, min={3}| coord={5}".format(
        sum(positive_ranks) / len(positive_ranks), max(positive_ranks), negative_rank_mean, negative_rank_min,
        min(positive_ranks), tgt_span_print_info)
    return advantage_loss, print_info_str

def calc_metric_contrastive_losses2(pred_out, scoring_metric, span_poses, tgt_span_print_info, negative_sample_num=20, use_whitelist=True):
    if len(upper_triangle)==0:
        triangle_mask = torch.zeros_like(scoring_metric,dtype=torch.float32, device=scoring_metric.device)
        for _i in range(scoring_metric.shape[0]):
            for _j in range(scoring_metric.shape[1]):
                if _i <= _j:
                    triangle_mask[_i, _j].fill_(1.0)
        upper_triangle.append(triangle_mask)

    maxlen = scoring_metric.shape[0]
    positive_mask = torch.zeros_like(scoring_metric,dtype=torch.float32)
    whitelist = torch.zeros_like(scoring_metric, dtype=torch.float32)
    gt_coordinates = {}
    if len(span_poses) > 0:
        for span_coord in span_poses:
            _fid, _pid, _st, _ed = [int(x) for x in span_coord]
            gt_coordinates[(_st, _ed)] = False
            positive_mask[_st, _ed].fill_(1.0)
            neighbour_margin = 2
            neighbour_start, neighbour_end = max(0, _st-neighbour_margin), min(_ed+neighbour_margin, maxlen-1)
            if use_whitelist:
                for t1 in range(neighbour_start, neighbour_end+1, 1):
                    for t2 in range(t1, neighbour_end+1, 1):
                        whitelist[t1, t2].fill_(1.0)
            else:
                whitelist[_st, _ed].fill_(1.0)
    else:
        gt_coordinates[(0, 0)] = False
        positive_mask[0, 0].fill_(1.0)
        whitelist[0, 0].fill_(1.0)


    positive_metric = torch.where(positive_mask.eq(1.0), scoring_metric, torch.zeros_like(scoring_metric))
    positive_metric_without_limit = torch.where(positive_mask.eq(1.0), scoring_metric, -100.0*torch.ones_like(scoring_metric))
    mean_positive = positive_metric.sum() / positive_mask.sum()
    max_positive = positive_metric_without_limit.max()

    # negative_metric = torch.where(positive_mask.eq(0.0), scoring_metric, torch.zeros_like(scoring_metric))
    # penalty_with_max = torch.nn.ELU(alpha=0.2)((scoring_metric - max_positive))
    penalty_with_max = torch.relu((scoring_metric - max_positive))
    contrastive_losses = torch.where(whitelist.eq(1.0), -100.0*torch.ones_like(positive_mask), penalty_with_max)
    top_losses, top_indices = torch.topk(contrastive_losses.reshape([-1]), k=negative_sample_num)
    # top_negative_mean = top_losses.mean()
    print_info = "+ve mean={0},top={1} | -ve tops={2}, max={3}, | tgt={4}".format(float(mean_positive.clone().detach().to("cpu")), float(max_positive.clone().detach().to("cpu")), float(top_losses.mean().clone().detach().to("cpu")), float(top_losses.max().clone().detach().to("cpu")), tgt_span_print_info)
    return top_losses.max(), print_info

def calc_metric_contrastive_losses_for_multi(pred_out, scoring_metric, span_poses, tgt_span_print_info, negative_sample_num=20):
    null_score = scoring_metric[0, 0]
    positive_scores = []
    if len(span_poses) > 0:
        maxlen = scoring_metric.shape[0]
        whitelist = torch.zeros_like(scoring_metric, dtype=torch.float32)
        for span_coord in span_poses:
            _fid, _pid, _st, _ed = [int(x) for x in span_coord]
            positive_scores.append(scoring_metric[_st, _ed])
            neighbour_margin = 2
            neighbour_start, neighbour_end = max(0, _st-neighbour_margin), min(_ed+neighbour_margin, maxlen-1)
            for t1 in range(neighbour_start, neighbour_end+1, 1):
                for t2 in range(t1, neighbour_end+1, 1):
                    whitelist[t1, t2].fill_(1.0)
        masked_scores = scoring_metric - whitelist * 100.0
        top_negative_sores, top_negative_indices = torch.topk(masked_scores.reshape([-1]), negative_sample_num)
        max_false_positive_loss = torch.relu(top_negative_sores - null_score.unsqueeze(0)).max()
        mean_false_positive_loss = torch.relu(top_negative_sores - null_score.unsqueeze(0)).mean()
        contrastive_loss_negative = 0.5 * (mean_false_positive_loss + max_false_positive_loss)

        all_positives = torch.stack(positive_scores, dim=0)
        advantages = torch.nn.ELU(alpha=1)(null_score.unsqueeze(0)-all_positives)
        contrastive_loss_positive = 0.5 * (advantages.max() + advantages.mean())
        contrastive_loss = contrastive_loss_negative + contrastive_loss_positive
        print_info = "positive advantage="+str(float(contrastive_loss_positive.clone().detach().to("cpu"))) + " false positive penalty=" + str(float(contrastive_loss_negative.clone().detach().to("cpu")))
    else:
        top_scores, top_indices = torch.topk(scoring_metric.reshape([-1]), negative_sample_num)
        max_false_positive_loss = torch.relu(top_scores - null_score.unsqueeze(0)).max()
        mean_false_positive_loss = torch.relu(top_scores - null_score.unsqueeze(0)).mean()
        contrastive_loss = 0.5 * (mean_false_positive_loss + max_false_positive_loss)
        print_info = "max false positive=" + str(float(max_false_positive_loss.clone().detach().to("cpu")))
    return contrastive_loss, print_info