from transformers import AutoConfig
from transformers.models.deberta.modeling_deberta import DebertaForQuestionAnswering
from collections.abc import Sequence
from typing import Optional, Tuple, Union
import math
import torch
import torch.utils.checkpoint
from torch import nn
from transformers.models.deberta.modeling_deberta import DebertaPreTrainedModel, DebertaEncoder, DebertaEmbeddings, DebertaModel, DebertaLayerNorm, StableDropout
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss

from transformers.activations import ACT2FN
from transformers.modeling_outputs import (
    BaseModelOutput,
    MaskedLMOutput,
    QuestionAnsweringModelOutput,
    SequenceClassifierOutput,
    TokenClassifierOutput,
)
from dataclasses import dataclass
from transformers.utils import ModelOutput
MAX_PARAGRAPH_NUM = 50

@dataclass
class QuestionAnsweringModelOutputWithMetric(ModelOutput):
    """
    Base class for outputs of question answering models.

    Args:
        loss (`torch.FloatTensor` of shape `(1,)`, *optional*, returned when `labels` is provided):
            Total span extraction loss is the sum of a Cross-Entropy for the start and end positions.
        start_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-start scores (before SoftMax).
        end_logits (`torch.FloatTensor` of shape `(batch_size, sequence_length)`):
            Span-end scores (before SoftMax).
        hidden_states (`tuple(torch.FloatTensor)`, *optional*, returned when `output_hidden_states=True` is passed or when `config.output_hidden_states=True`):
            Tuple of `torch.FloatTensor` (one for the output of the embeddings, if the model has an embedding layer, +
            one for the output of each layer) of shape `(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the optional initial embedding outputs.
        attentions (`tuple(torch.FloatTensor)`, *optional*, returned when `output_attentions=True` is passed or when `config.output_attentions=True`):
            Tuple of `torch.FloatTensor` (one for each layer) of shape `(batch_size, num_heads, sequence_length,
            sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    start_logits: torch.FloatTensor = None
    end_logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    scoring_metric: Optional[torch.FloatTensor] =None
    modifier: Optional[torch.FloatTensor] = None
    type_classify_logits: Optional[torch.FloatTensor] = None

class AutoConfigMod(AutoConfig):
    def modify_params(self):
        return

class TypeClassifyHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.proj1 = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.act1 = nn.Tanh()
        self.proj_out = torch.nn.Linear(config.hidden_size, 3)
    def forward(self, cls_feature):
        return self.proj_out(self.act1(self.proj1(cls_feature)))

class DebertaEmbeddingModified(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings."""

    def __init__(self, config):
        super().__init__()
        pad_token_id = getattr(config, "pad_token_id", 0)
        self.embedding_size = getattr(config, "embedding_size", config.hidden_size)
        self.word_embeddings = nn.Embedding(config.vocab_size, self.embedding_size, padding_idx=pad_token_id)

        self.position_biased_input = getattr(config, "position_biased_input", True)
        if not self.position_biased_input:
            self.position_embeddings = None
        else:
            self.position_embeddings = nn.Embedding(config.max_position_embeddings, self.embedding_size)

        if config.type_vocab_size > 0:
            self.token_type_embeddings = nn.Embedding(config.type_vocab_size, self.embedding_size)

        if MAX_PARAGRAPH_NUM>0:
            self.paragraph_embeddings = nn.Embedding(MAX_PARAGRAPH_NUM, self.embedding_size)
            if self.embedding_size != config.hidden_size:
                self.paragraph_embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)

        if self.embedding_size != config.hidden_size:
            self.embed_proj = nn.Linear(self.embedding_size, config.hidden_size, bias=False)
        self.LayerNorm = DebertaLayerNorm(config.hidden_size, config.layer_norm_eps)
        self.dropout = StableDropout(config.hidden_dropout_prob)
        self.config = config

        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))

    def forward(self, input_ids=None, token_type_ids=None, position_ids=None, mask=None, inputs_embeds=None, paragraph_ids=None):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if paragraph_ids is not None:
            pp1_ids = (torch.ones_like(paragraph_ids)+paragraph_ids)
            clipped_paragraph_ids = torch.where(pp1_ids.ge(MAX_PARAGRAPH_NUM), (MAX_PARAGRAPH_NUM-1)*torch.ones_like(pp1_ids), pp1_ids)
            paragraph_embeds = self.paragraph_embeddings(clipped_paragraph_ids)
            if self.embedding_size != self.config.hidden_size:
                paragraph_embeds = self.paragraph_embed_proj(paragraph_embeds)

        if position_ids is None:
            position_ids = self.position_ids[:, :seq_length]

        if token_type_ids is None:
            token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)

        if self.position_embeddings is not None:
            position_embeddings = self.position_embeddings(position_ids.long())
        else:
            position_embeddings = torch.zeros_like(inputs_embeds)

        embeddings = inputs_embeds
        if self.position_biased_input:
            embeddings += position_embeddings
        if self.config.type_vocab_size > 0:
            token_type_embeddings = self.token_type_embeddings(token_type_ids)
            embeddings += token_type_embeddings
        if paragraph_ids is not None:
            embeddings += paragraph_embeds

        if self.embedding_size != self.config.hidden_size:
            embeddings = self.embed_proj(embeddings)

        embeddings = self.LayerNorm(embeddings)

        if mask is not None:
            if mask.dim() != embeddings.dim():
                if mask.dim() == 4:
                    mask = mask.squeeze(1).squeeze(1)
                mask = mask.unsqueeze(2)
            mask = mask.to(embeddings.dtype)

            embeddings = embeddings * mask

        embeddings = self.dropout(embeddings)
        return embeddings


class DebertaEncoderModified(DebertaEncoder):

    def forward(
        self,
        hidden_states,
        attention_mask,
        output_hidden_states=True,
        output_attentions=False,
        query_states=None,
        relative_pos=None,
        return_dict=True,
        paragraph_embeddings=None,
        paragraph_inject_layer_num=-2
    ):
        attention_mask = self.get_attention_mask(attention_mask)
        relative_pos = self.get_rel_pos(hidden_states, query_states, relative_pos)

        all_hidden_states = () if output_hidden_states else None
        all_attentions = () if output_attentions else None

        if isinstance(hidden_states, Sequence):
            next_kv = hidden_states[0]
        else:
            next_kv = hidden_states
        rel_embeddings = self.get_rel_embedding()
        for i, layer_module in enumerate(self.layer):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            # next_kv =

            if self.gradient_checkpointing and self.training:

                def create_custom_forward(module):
                    def custom_forward(*inputs):
                        return module(*inputs, output_attentions)

                    return custom_forward

                hidden_states = torch.utils.checkpoint.checkpoint(
                    create_custom_forward(layer_module),
                    next_kv,
                    attention_mask,
                    query_states,
                    relative_pos,
                    rel_embeddings,
                )
            else:
                hidden_states = layer_module(
                    next_kv,
                    attention_mask,
                    query_states=query_states,
                    relative_pos=relative_pos,
                    rel_embeddings=rel_embeddings,
                    output_attentions=output_attentions,
                )

            if output_attentions:
                hidden_states, att_m = hidden_states

            if query_states is not None:
                query_states = hidden_states
                if isinstance(hidden_states, Sequence):
                    next_kv = hidden_states[i + 1] if i + 1 < len(self.layer) else None
            else:
                next_kv = hidden_states

            if output_attentions:
                all_attentions = all_attentions + (att_m,)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        if not return_dict:
            return tuple(v for v in [hidden_states, all_hidden_states, all_attentions] if v is not None)
        return BaseModelOutput(
            last_hidden_state=hidden_states, hidden_states=all_hidden_states, attentions=all_attentions
        )

class DebertaModelModified(DebertaModel):
        def __init__(self, config):
            super().__init__(config)
            # self.embeddings = DebertaEmbeddingModified(config)
            self.encoder = DebertaEncoderModified(config)
            self.post_init()


        def forward(
                self,
                input_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                token_type_ids: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                output_attentions: Optional[bool] = None,
                output_hidden_states: Optional[bool] = None,
                return_dict: Optional[bool] = None,
                paragraph_ids:Optional[torch.Tensor] = None,
        ) -> Union[Tuple, BaseModelOutput]:
            output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
            output_hidden_states = (
                output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
            )
            return_dict = return_dict if return_dict is not None else self.config.use_return_dict

            if input_ids is not None and inputs_embeds is not None:
                raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
            elif input_ids is not None:
                input_shape = input_ids.size()
            elif inputs_embeds is not None:
                input_shape = inputs_embeds.size()[:-1]
            else:
                raise ValueError("You have to specify either input_ids or inputs_embeds")

            device = input_ids.device if input_ids is not None else inputs_embeds.device

            if attention_mask is None:
                attention_mask = torch.ones(input_shape, device=device)
            if token_type_ids is None:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

            embedding_output = self.embeddings(
                input_ids=input_ids,
                token_type_ids=token_type_ids,
                position_ids=position_ids,
                mask=attention_mask,
                inputs_embeds=inputs_embeds,
                paragraph_ids=paragraph_ids,
            )

            encoder_outputs = self.encoder(
                embedding_output,
                attention_mask,
                output_hidden_states=True,
                output_attentions=output_attentions,
                return_dict=return_dict,
            )
            encoded_layers = encoder_outputs[1]

            if self.z_steps > 1:
                hidden_states = encoded_layers[-2]
                layers = [self.encoder.layer[-1] for _ in range(self.z_steps)]
                query_states = encoded_layers[-1]
                rel_embeddings = self.encoder.get_rel_embedding()
                attention_mask = self.encoder.get_attention_mask(attention_mask)
                rel_pos = self.encoder.get_rel_pos(embedding_output)
                for layer in layers[1:]:
                    query_states = layer(
                        hidden_states,
                        attention_mask,
                        output_attentions=False,
                        query_states=query_states,
                        relative_pos=rel_pos,
                        rel_embeddings=rel_embeddings,
                    )
                    encoded_layers.append(query_states)

            sequence_output = encoded_layers[-1]

            if not return_dict:
                return (sequence_output,) + encoder_outputs[(1 if output_hidden_states else 2):]

            return BaseModelOutput(
                last_hidden_state=sequence_output,
                hidden_states=encoder_outputs.hidden_states if output_hidden_states else None,
                attentions=encoder_outputs.attentions,
            )


class DebertaAddon(DebertaForQuestionAnswering):
    def __init__(self, config):
        super().__init__(config)

        self.deberta = DebertaModel(config)
        # self.deberta = DebertaModelModified(config)
        self.post_init()

        self.rel_paragraph_kernel_layer = torch.nn.Linear(config.hidden_size, config.hidden_size)
        self.mid_pos = 6
        self.rel_hidden_size = 256
        self.relative_project_st = torch.nn.Linear(config.hidden_size, self.rel_hidden_size)
        self.relative_project_ed = torch.nn.Linear(config.hidden_size, self.rel_hidden_size)
        self.paragraph_relative_position_embeddings = torch.nn.Embedding(self.mid_pos*2, self.rel_hidden_size)
        self.null_position_embeddings = torch.nn.Embedding(1, self.rel_hidden_size)
        self.type_classify_head = TypeClassifyHead(config)
        # self.score_gate_proj1 = torch.nn.Linear(config.hidden_size+self.rel_hidden_size, config.hidden_size)
        # self.score_gate_proj2 = torch.nn.Linear(config.hidden_size, 1)

        self.paragraph_clip_maxnum = 30

        self.lstm = torch.nn.LSTM(input_size=config.hidden_size, hidden_size=self.rel_hidden_size, num_layers=1, batch_first=True, bidirectional=True)
        self.span_project = torch.nn.Linear(2 * self.rel_hidden_size, 1)
        # self.span_kernel = torch.nn.Linear(2*self.rel_hidden_size, 2*self.rel_hidden_size)
        # self.paragraph_absolute_position_embeddings = torch.nn.Embedding(self.paragraph_clip_maxnum, self.rel_hidden_size)
        self.absolute_position_embedding = torch.nn.Embedding(config.max_position_embeddings, config.hidden_size)
        return

    def antisymmetric_score(self, hidden_features, paragraph_ids):
        bsz, max_step = hidden_features.shape[0], hidden_features.shape[1]
        # token_type_ids.unsqueeze(2) * token_type_ids.unsqueeze(1)
        # scoring_mask = token_type_ids.clone().detach()
        # scoring_mask[:, 0].fill_(1)
        # scoring_metric_mask = (1 - scoring_mask.unsqueeze(2) * scoring_mask.unsqueeze(1)) * (-1000.0)

        is_question_null = (paragraph_ids < 0)
        null_mask = torch.logical_or(is_question_null.unsqueeze(2), is_question_null.unsqueeze(1))
        rel_positions = paragraph_ids.unsqueeze(2) - paragraph_ids.unsqueeze(1)
        rel_positions_uppper_clipped = torch.where(rel_positions.ge(self.mid_pos), (self.mid_pos-1)*torch.ones_like(rel_positions), rel_positions)
        rel_positions_clipped = torch.where(rel_positions_uppper_clipped.le(-self.mid_pos), -(self.mid_pos)*torch.ones_like(rel_positions), rel_positions_uppper_clipped)
        rel_pos_embeddings = self.paragraph_relative_position_embeddings(self.mid_pos*torch.ones_like(rel_positions)+rel_positions_clipped)
        null_rel_pos_embeddings = self.null_position_embeddings(torch.zeros_like(rel_positions_clipped, dtype=torch.long))
        relative_position_embeddings = torch.where(null_mask.unsqueeze(3), null_rel_pos_embeddings, rel_pos_embeddings)
        # query_view_mask = (torch.ones_like(null_mask,dtype=torch.float32)-null_mask.float()) #
        # query_view_feature = (relative_position_embeddings * query_view_mask.unsqueeze(3)).sum(dim=2) / (0.00001+query_view_mask.sum(dim=2,keepdim=True))
        # hidden_consistancy = torch.square(query_view_feature.sum(dim=1)) / (query_view_feature*query_view_feature).sum(dim=2).sum(dim=1,keepdim=True)

        # absolute_pids = paragraph_ids + torch.ones_like(paragraph_ids, dtype=torch.long)
        # clipped_absolute_pids = torch.where(absolute_pids.ge(self.paragraph_clip_maxnum), (self.paragraph_clip_maxnum-1)*torch.ones_like(absolute_pids), absolute_pids)
        # absolute_paragraph_embeddings = self.paragraph_absolute_position_embeddings(clipped_absolute_pids)
        # zero_embedding = torch.zeros([bsz, max_step, self.rel_hidden_size], dtype=torch.float32, device=absolute_paragraph_embeddings.device)
        # absolute_paragraph_embeddings = torch.where(is_question_null.unsqueeze(2), zero_embedding, absolute_paragraph_embeddings)
        # hidden_consistancy = torch.square(absolute_paragraph_embeddings.sum(dim=1)) / (absolute_paragraph_embeddings*absolute_paragraph_embeddings).sum(dim=2).sum(dim=1,keepdim=True)

        query = key = hidden_features
        query_rel_pos = self.relative_project_st(hidden_features)
        key_rel_pos = self.relative_project_ed(hidden_features)

        relative_position_scores_query = torch.einsum("bld,blrd->blr", query_rel_pos, relative_position_embeddings)
        relative_position_scores_key = torch.einsum("brd,blrd->blr", key_rel_pos, relative_position_embeddings)

        qTWk = torch.bmm(query, self.rel_paragraph_kernel_layer(key).transpose(2, 1))
        attention_scores = qTWk / math.sqrt(self.config.hidden_size) + (relative_position_scores_query + relative_position_scores_key) / math.sqrt(self.rel_hidden_size)
        # attention_scores = attention_scores / math.sqrt(self.config.hidden_size)
        assymetric_scores = 0.5 * (attention_scores - attention_scores.transpose(2, 1))

        # except_null_mask = torch.ones_like(attention_scores)
        # except_null_mask[:, 0, 0].fill_(0.0)
        # assymetric_scores = except_null_mask * assymetric_scores
        return assymetric_scores, None#hidden_consistancy

    def calc_lstm_span_score(self, hidden_feature):
        encoded_feature, (hs, cs) = self.lstm(hidden_feature)
        span_diff = (encoded_feature.unsqueeze(2) - encoded_feature.unsqueeze(1))
        span_modifier = self.span_project(span_diff)[:, :, :, 0]
        # span_multiply = encoded_feature.unsqueeze(2) * self.span_kernel(encoded_feature.unsqueeze(1))
        # span_modifier = span_multiply.sum(dim=-1)/math.sqrt(self.rel_hidden_size)
        return span_modifier

    def calc_lstm_solid_span(self, hidden_feature):
        encoded_feature, (hs, cs) = self.lstm(hidden_feature)
        span_character_lgt = self.span_project(encoded_feature)[:, :, 0]
        span_character_prob = torch.sigmoid(span_character_lgt)
        return span_character_prob

    # def antisymmetric_score(self, hidden_features,  mean_feat_0_using_absolute=False):
    #     tmp_pids = torch.arange(cls_features.shape[0]).to(cls_features.device)
    #     rel_positions = tmp_pids.unsqueeze(1) - tmp_pids.unsqueeze(0)
    #     rel_pos_ids = self.mid_pos * torch.ones_like(rel_positions) + rel_positions
    #     rel_pos_embeddings = self.paragraph_relative_embedding(rel_pos_ids)
    #
    #     # if mean_feat_0_using_absolute:
    #     #     mean_padding = self.paragraph_relative_embedding(torch.zeros_like(rel_pos_ids, dtype=torch.long))
    #     #     using_relative = torch.logical_and((tmp_pids.unsqueeze(1) > 0), tmp_pids.unsqueeze(0) > 0).unsqueeze(2)
    #     #     rel_pos_embeddings = torch.where(using_relative, rel_pos_embeddings, mean_padding)
    #
    #     query = hidden_features
    #     key = hidden_features
    #     # value = hidden_features
    #     relative_position_scores_query = torch.einsum("bld,lrd->blr", query, rel_pos_embeddings)
    #     relative_position_scores_key = torch.einsum("brd,lrd->blr", key, rel_pos_embeddings)
    #
    #     qTWk = torch.bmm(query, self.rel_paragraph_kernel_layer(key).transpose(2, 1))
    #     attention_scores = qTWk + relative_position_scores_query + relative_position_scores_key
    #     attention_scores = attention_scores / math.sqrt(self.config.hidden_size)
    #
    #     assymetric_scores = 0.5 * (attention_scores - attention_scores.transpose(2, 1))
    #     return assymetric_scores

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        paragraph_ids: Optional[torch.Tensor] = None,
        absolute_position_ids: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        start_positions: Optional[torch.Tensor] = None,
        end_positions: Optional[torch.Tensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        using_modifier: bool = False,
        lstm_span_modifier: bool = False,
        solid_span: bool = False,
        use_abs_pos_embedding: bool = False,
    ) -> Union[Tuple, QuestionAnsweringModelOutput]:
        r"""
        start_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the start of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        end_positions (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for position (index) of the end of the labelled span for computing the token classification loss.
            Positions are clamped to the length of the sequence (`sequence_length`). Position outside of the sequence
            are not taken into account for computing the loss.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # position_ids = None

        outputs = self.deberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            # paragraph_ids=paragraph_ids,
        )

        sequence_output = outputs[0]
        if use_abs_pos_embedding:
            position_ids_tmp = absolute_position_ids + torch.ones_like(absolute_position_ids)
            position_ids_inp = torch.where(position_ids_tmp.ge(self.config.max_length),
                                           (self.config.max_length - 1) * torch.ones_like(position_ids_tmp),
                                           position_ids_tmp)
            absolute_pos_embedding = self.absolute_position_embedding(position_ids_inp)
            sequence_output += absolute_pos_embedding
        bsz, seq_len = sequence_output.shape[0], sequence_output.shape[1]

        type_classify_logits = self.type_classify_head(sequence_output[:, 0, :]) # torch.zeros([sequence_output.shape[0], 3],dtype=torch.float32,device=sequence_output.device) #

        logits = self.qa_outputs(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1).contiguous()
        end_logits = end_logits.squeeze(-1).contiguous()

        scoring_metric = start_logits.unsqueeze(2)+end_logits.unsqueeze(1)
        if using_modifier:
            score_modifier, relative_feature_consistancy = self.antisymmetric_score(sequence_output, paragraph_ids)
            # gate_feature = torch.cat([relative_feature_consistancy, sequence_output[:, 0, :]], dim=-1)
            # gate_lgt = self.score_gate_proj2(torch.tanh(self.score_gate_proj1(gate_feature)))
            # gate = torch.relu(gate_lgt.unsqueeze(2)) # torch.sigmoid(gate_lgt).unsqueeze(2)
            # modifier = (0.1+gate) * score_modifier
            # modifier = (0.1+gate) * score_modifier
            modifier = score_modifier
            scoring_metric = scoring_metric + modifier
        elif lstm_span_modifier:
            modifier = self.calc_lstm_span_score(sequence_output)
            scoring_metric = scoring_metric + modifier
        elif solid_span:
            modifier = self.calc_lstm_solid_span(sequence_output)
        else:
            modifier = torch.zeros_like(scoring_metric)

        total_loss = None
        if start_positions is not None and end_positions is not None:
            # If we are on multi-GPU, split add a dimension
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            # sometimes the start/end positions are outside our model inputs, we ignore these terms
            ignored_index = start_logits.size(1)
            start_positions = start_positions.clamp(0, ignored_index)
            end_positions = end_positions.clamp(0, ignored_index)

            loss_fct = CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2

        if not return_dict:
            output = (start_logits, end_logits) + outputs[1:]
            return ((total_loss,) + output) if total_loss is not None else output

        return QuestionAnsweringModelOutputWithMetric(
            loss=total_loss,
            start_logits=start_logits,
            end_logits=end_logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            scoring_metric=scoring_metric,
            modifier=modifier,
            type_classify_logits=type_classify_logits,
        )
