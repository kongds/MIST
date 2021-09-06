import logging
import math
import os

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss

from bert import BertModel, BertOnlyMLMHead
# from transformers.models.bert.modeling_bert import BertOnlyMLMHead
# from transformers.modeling_bert import BertOnlyMLMHead


# BERTModel.forward extend_attention_mask [batch_size, from_seq_length, to_seq_length]

def select_worst_as_mask(token_probs, num_mask):
    bsz, seq_len = token_probs.size()
    masks = [token_probs[batch, :].topk(max(1, num_mask[batch]), largest=False, sorted=False)[1] for batch in range(bsz)]
    masks = [torch.cat([mask, mask.new(seq_len - mask.size(0)).fill_(mask[0])], dim=0) for mask in masks]
    return torch.stack(masks, dim=0)

def assign_single_value_long(x, i, y):
    b, l = x.size()
    i = i + torch.arange(0, b*l, l, device=i.device).unsqueeze(1)
    x.view(-1)[i.view(-1)] = y

class LabelSmoothingLoss(_Loss):
    """
    With label smoothing,
    KL-divergence between q_{smoothed ground truth prob.}(w)
    and p_{prob. computed by model}(w) is minimized.
    """

    def __init__(self, label_smoothing=0, tgt_vocab_size=0, ignore_index=0,
                 size_average=None, reduce=None, reduction='mean'):
        assert 0.0 < label_smoothing <= 1.0
        self.ignore_index = ignore_index
        super(LabelSmoothingLoss, self).__init__(
            size_average=size_average, reduce=reduce, reduction=reduction)

        assert label_smoothing > 0
        assert tgt_vocab_size > 0

        smoothing_value = label_smoothing / (tgt_vocab_size - 2)
        one_hot = torch.full((tgt_vocab_size,), smoothing_value)
        one_hot[self.ignore_index] = 0
        self.register_buffer('one_hot', one_hot.unsqueeze(0))
        self.confidence = 1.0 - label_smoothing
        self.tgt_vocab_size = tgt_vocab_size

    def forward(self, output, target):
        """
        output (FloatTensor): batch_size * num_pos * n_classes
        target (LongTensor): batch_size * num_pos
        """
        assert self.tgt_vocab_size == output.size(2)
        batch_size, num_pos = target.size(0), target.size(1)
        output = output.view(-1, self.tgt_vocab_size)
        target = target.view(-1)
        model_prob = self.one_hot.float().repeat(target.size(0), 1)
        model_prob.scatter_(1, target.unsqueeze(1), self.confidence)
        model_prob.masked_fill_((target == self.ignore_index).unsqueeze(1), 0)

        return F.kl_div(output, model_prob, reduction='none').view(batch_size, num_pos, -1).sum(2)

class UnilmNAT(nn.Module):
    def __init__(self, unilm_path, label_smoothing=0.1, use_glat=False, multi_layer=1, glat_random_prob=None, two_state=False,
                 remove_sep_in_target=False, multi_layer_attention=False, use_ctc=False, glat_f=0.5, use_multi_glat=False, freeze_source_embeds=True,
                 sep_word_id=102, mask_word_id=103, pad_word_id=0, decode_offset=True, decoder_relative_position=False,
                 two_unilm_mask_mask_first_layer_num=6, feed_embeds=False, add_masks=False, half_full_attention=False,
                 clear_bert_weight=False,
                 add_masks_in_pseudo=False, add_decoder_length_loss=False,
                 set_mask_inf=False, mask_target_loss=False, half_layer_nums=False, same_bert=False):
        super(UnilmNAT, self).__init__()
        self.source_type_id = 0
        self.target_type_id = 1

        self.swa_state = {}

        self.freeze_source_embeds = freeze_source_embeds
        self.two_state = two_state
        self.decoder_relative_position = decoder_relative_position
        self.glat_random_prob = glat_random_prob
        self.use_multi_glat = use_multi_glat
        self.multi_glat_easy = True
        self.glat_f = glat_f
        self.multi_layer = multi_layer
        self.multi_layer_attention = multi_layer_attention
        self.decode_offset = decode_offset
        self.remove_sep_in_target = remove_sep_in_target
        self.label_smoothing = label_smoothing
        self.mask_word_id = mask_word_id
        self.sep_word_id = sep_word_id
        self.pad_word_id = pad_word_id
        self.mask_target_loss = mask_target_loss
        self.set_mask_inf = set_mask_inf
        self.use_ctc = use_ctc
        self.feed_embeds = feed_embeds
        self.add_masks = add_masks
        self.add_masks_in_pseudo = add_masks_in_pseudo
        self.add_decoder_length_loss = add_decoder_length_loss

        self.bert_encoder = BertModel.from_pretrained(unilm_path)
        if same_bert:
            self.bert_decoder = self.bert_encoder
        else:
            self.bert_decoder = BertModel.from_pretrained(unilm_path)

        if clear_bert_weight:
            self.bert_encoder.init_weights()
            self.bert_decoder.init_weights()

        self.half_layer_nums = half_layer_nums
        self.half_full_attention = half_full_attention
        if half_layer_nums:
            self.bert_encoder.encoder.layer = self.bert_encoder.encoder.layer[:two_unilm_mask_mask_first_layer_num]
            self.bert_decoder.encoder.layer = self.bert_decoder.encoder.layer[12-two_unilm_mask_mask_first_layer_num:]

        self.config = self.bert_encoder.config
        self.config.__dict__['label_smoothing'] = label_smoothing

        self.encoder_embed_dim = self.bert_encoder.config.hidden_size
        self.embed_length = nn.Embedding(512, self.encoder_embed_dim * self.multi_layer, None)
        #init cls decoder weight with embedding
        self.cls = BertOnlyMLMHead(self.bert_encoder.config)
        self.cls.predictions.decoder.weight = nn.Parameter(self.bert_encoder.embeddings.word_embeddings.weight.clone())

        if self.config.label_smoothing > 0:
            self.crit_mask_lm_smoothed = LabelSmoothingLoss(
                self.config.label_smoothing, self.config.vocab_size, ignore_index=0, reduction='none')
            self.crit_mask_lm = None
        else:
            self.crit_mask_lm_smoothed = None
            self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')

        self.use_glat = use_glat

    @staticmethod
    def create_mask_and_position_ids(num_tokens, max_len, offset=None):
        base_position_matrix = torch.arange(
            0, max_len, dtype=num_tokens.dtype, device=num_tokens.device).view(1, -1)
        mask = (base_position_matrix < num_tokens.view(-1, 1)).type_as(num_tokens)
        if offset is not None:
            base_position_matrix = base_position_matrix + offset.view(-1, 1)
        position_ids = base_position_matrix * mask
        return mask, position_ids

    @staticmethod
    def _create_attention_mask(source_mask, target_mask, source_position_ids, target_span_ids):
        weight = torch.cat((torch.zeros_like(source_position_ids), target_span_ids, -target_span_ids), dim=1)
        from_weight = weight.unsqueeze(-1)
        to_weight = weight.unsqueeze(1)

        true_tokens = (0 <= to_weight) & (torch.cat((source_mask, target_mask, target_mask), dim=1) == 1).unsqueeze(1)
        # true_tokens_mask[b][i][j] = (from_weight[b][i][0] >= 0) & true_tokens[b][0][j] & (to_weight[b][0][j] <= from_weight[b][i][0])
        true_tokens_mask = (from_weight >= 0) & true_tokens & (to_weight <= from_weight)
        pseudo_tokens_mask = (from_weight < 0) & true_tokens & (-to_weight > from_weight)
        pseudo_tokens_mask = pseudo_tokens_mask | ((from_weight < 0) & (to_weight == from_weight))

        return (true_tokens_mask | pseudo_tokens_mask).type_as(source_mask)

    @staticmethod
    def create_attention_mask(source_mask, target_mask):
        b = source_mask.shape[0]
        sl = source_mask.shape[1]
        tl = target_mask.shape[1]
        weight = torch.cat((torch.ones_like(source_mask), torch.zeros_like(target_mask)), dim=1)
        from_weight = weight.unsqueeze(-1)
        to_weight = weight.unsqueeze(1)

        mask = torch.cat((source_mask, target_mask), dim=1) == 1
        mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
        # w[i][j] = f[i][0] == 1 or t[0][j] == 0
        return (((from_weight == 0) | (to_weight == 1)) & mask).type_as(source_mask)

    @staticmethod
    def create_relative_positionn_mask(source_mask, target_mask):
        b = source_mask.shape[0]
        sl = source_mask.shape[1]
        tl = target_mask.shape[1]
        weight = torch.cat((torch.ones_like(source_mask), torch.zeros_like(target_mask)), dim=1)
        from_weight = weight.unsqueeze(-1)
        to_weight = weight.unsqueeze(1)

        mask = torch.cat((source_mask, target_mask), dim=1) == 1
        mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
        # w[i][j] = f[i][0] == 1 or t[0][j] == 0
        return (((from_weight == 0) & (to_weight == 0)) & mask).type_as(source_mask)

    @staticmethod
    def create_layer_attention_mask(a1_layer_num, a2_layer_num, source_mask,
                                    target_mask, target_position_ids, include_sep=True):
        weight = torch.cat((-source_mask, target_position_ids), dim=1)
        from_weight = weight.unsqueeze(-1)
        to_weight = weight.unsqueeze(1)
        mask = torch.cat((source_mask, target_mask), dim=1) == 1
        mask = mask.unsqueeze(-1) & mask.unsqueeze(1)
        # import pdb;pdb.set_trace()
        a1 = (((from_weight > -1) | (to_weight == -1)) & mask).type_as(source_mask)
        a2 = (((from_weight > -2) & (to_weight == -1)) | (from_weight == to_weight) & mask).type_as(source_mask)
        if include_sep:
            t_l = target_mask.sum(-1) + source_mask.shape[-1] - 1
            m2 = torch.zeros_like(a1)
            m2[torch.arange(m2.shape[0]), :, t_l] = 1
            m2 = a1 & m2
            a2 = a1 & a2 | m2
        else:
            a2 = a1 & a2

        a1 = a1.unsqueeze(0).repeat(a1_layer_num, 1, 1, 1)
        a2 = a2.unsqueeze(0).repeat(a2_layer_num, 1, 1, 1)
        plt.imshow(a1[1][0].numpy())
        plt.show()
        plt.imshow(a2[1][0].numpy())
        plt.show()
        return torch.cat([a1, a2], dim=0).type_as(source_mask)

    def forward_length(self, enc_feats, src_masks):
        # enc_feats: B x T x C
        # src_masks: B x T or None
        enc_feats = enc_feats.transpose(0, 1)
        src_masks = src_masks.transpose(0, 1)
        #src_masks = (~src_masks).type_as(enc_feats)
        src_masks = src_masks.type_as(enc_feats)
        enc_feats = (
            (enc_feats / src_masks.sum(0)[None, :, None]) * src_masks[:, :, None]
        ).sum(0)
        length_out = F.linear(enc_feats, self.embed_length.weight)
        return F.log_softmax(length_out, -1)

    def feed_bert_half(self, input_ids, source_mask, target_mask,
                      token_type_ids, position_ids, target_position_ids, 
                      target_ids=None, return_all_hidden_states=False, decoding=False):
        attention_mask = self.create_attention_mask(source_mask, target_mask)
        attention_masks = None
        source_len = source_mask.size(1)
        # source_embeds = self.bert_encoder(input_ids[:, :sl])[0]
        encoder_outputs = self.bert_encoder(
                          input_ids=input_ids, keep_source_unmodified=-1, attention_mask=attention_mask,
                          attention_masks=attention_masks, token_type_ids=token_type_ids,
                          output_hidden_states=self.multi_layer > 1 or return_all_hidden_states,
                          position_ids=position_ids)

        encoder_sequence_output = encoder_outputs[0]
        pseudo_sequence_output = encoder_sequence_output[:, source_len:, ]
        source_embeds = encoder_sequence_output[:, :source_len, ]

        source_len = source_mask.shape[1]
        if decoding:
            source_len += pseudo_sequence_output.shape[1]
            if source_len + pseudo_sequence_output.shape[1] > 512:
                source_len = 512 - pseudo_sequence_output.shape[1]

        new_source_embeds = []
        new_source_masks = []
        for b in range(pseudo_sequence_output.shape[0]):
            pt = pseudo_sequence_output[b][target_mask.bool()[b]]
            source_embed = source_embeds[b][source_mask.bool()[b]]
            if source_embed.shape[0] + pt.shape[0] > source_len:
                s_len = source_len - pt.shape[0]
                source_embed = torch.cat([source_embed[:s_len-1], source_embed[-1].unsqueeze(0), pt], dim=0)
                _source_mask = torch.ones(source_embed.shape[0]).to(source_embed.device)
            else:
                pads_embs = self.bert_encoder.embeddings.word_embeddings(torch.tensor(self.pad_word_id).to(input_ids.device)).view(1, -1)
                pads_embs = pads_embs.repeat(source_len - pt.shape[0] - source_embed.shape[0], 1)
                source_embed = torch.cat([source_embed, pt, pads_embs], dim=0)
                _source_mask = torch.ones(source_embed.shape[0]).to(source_embed.device)
                _source_mask[-pads_embs.shape[0]:] = 0
            new_source_embeds.append(source_embed)
            new_source_masks.append(_source_mask)
        new_source_embeds = torch.stack(new_source_embeds, dim=0)
        new_source_masks = torch.stack(new_source_masks, dim=0)

        new_input_embeds = torch.cat([new_source_embeds, pseudo_sequence_output], dim=1)
        if self.half_full_attention:
            weight = torch.cat([new_source_masks, target_mask], dim=-1).long()
            from_weight = weight.unsqueeze(-1)
            to_weight = weight.unsqueeze(1)
            new_attention_mask = ((from_weight > 0) & (to_weight > 0)).bool()
        else:
            new_attention_mask = self.create_attention_mask(new_source_masks, target_mask)

        extend_attention_mask = self.bert_decoder.get_extended_attention_mask(new_attention_mask,
                                                                              new_input_embeds.size()[:-1],
                                                                              new_input_embeds.device)

        decoder_sequence_output = self.bert_decoder.encoder(new_input_embeds,
                                                    attention_mask=extend_attention_mask)[0]
        prediction_scores_masked = self.cls(pseudo_sequence_output)
        return encoder_sequence_output, decoder_sequence_output, prediction_scores_masked

    def feed_bert(self, input_ids, source_mask, target_mask,
                  token_type_ids, position_ids, target_position_ids, 
                  target_ids=None, return_all_hidden_states=False, decoding=False):

        if self.half_layer_nums:
            return self.feed_bert_half(input_ids, source_mask, target_mask,
                                        token_type_ids, position_ids, target_position_ids, 
                                        target_ids, return_all_hidden_states, decoding)
        if self.multi_layer_attention:
            a1_layer_num = 3
            a2_layer_num = self.config.num_hidden_layers - 3
            #import pdb;pdb.set_trace()
            attention_masks = self.create_layer_attention_mask(a1_layer_num,  a2_layer_num, source_mask,
                                                              target_mask, target_position_ids)
            attention_masks = attention_masks.to(input_ids.device)
            attention_mask = None
        else:
            attention_mask = self.create_attention_mask(source_mask, target_mask)
            attention_masks = None

        decoder_relative_position_mask = None
        if self.decoder_relative_position:
            decoder_relative_position_mask = self.create_relative_positionn_mask(source_mask, target_mask)

        sl = source_mask.shape[1]
        source_len = source_mask.size(1)
        # source_embeds = self.bert_encoder(input_ids[:, :sl])[0]
        encoder_outputs = self.bert_encoder(
                          input_ids=input_ids, keep_source_unmodified=-1, attention_mask=attention_mask,
                          attention_masks=attention_masks, token_type_ids=token_type_ids,
                          output_hidden_states=self.multi_layer > 1 or return_all_hidden_states,
                          position_ids=position_ids, decoder_relative_position_mask=decoder_relative_position_mask) 
        encoder_sequence_output = encoder_outputs[0]
        pseudo_sequence_output = encoder_sequence_output[:, source_len:, ]
        prediction_scores_masked = self.cls(pseudo_sequence_output)
        prediction_scores, prediction_tokens = prediction_scores_masked.max(-1)

        prediction_tokens[~target_mask.bool()] = self.pad_word_id

        source_ids = []
        source_len = source_mask.shape[1]
        old_source_ids = input_ids[:, :source_len]
        pseudo_ids = input_ids[:, source_len:]
        if decoding:
            source_len += prediction_tokens.shape[1]
            if source_len + pseudo_ids.shape[1] > 512:
                source_len = 512 - pseudo_ids.shape[1]

        new_pseudo_ids = pseudo_ids.clone()
        if self.add_masks or self.add_masks_in_pseudo:
            num_mask = (target_mask.sum(-1)*0.5).long()
            mask_ind = select_worst_as_mask(prediction_scores, num_mask)
            if self.add_masks:
                assign_single_value_long(prediction_tokens, mask_ind, self.mask_word_id)
            else:
                new_pseudo_ids = prediction_tokens.clone()
                assign_single_value_long(new_pseudo_ids, mask_ind, self.mask_word_id)

        for b in range(prediction_tokens.shape[0]):
            pt = prediction_tokens[b][target_mask.bool()[b]]

            # remove sep in pt_masks
            if pt[-1] != self.sep_word_id:
                pt = torch.cat([pt, torch.tensor([self.sep_word_id]).to(pt.device)])
            source_id = torch.cat([input_ids[b][0].view(1),
                                   pt,
                                   old_source_ids[b][source_mask.bool()[b]][1:],], # remove cls
                                   dim=0)
            if source_id.shape[0] >= source_len:
                source_id = source_id[:source_len]
            else:
                pads = torch.zeros(source_len - source_id.shape[0]).fill_(self.pad_word_id).long()
                source_id = torch.cat([source_id, pads.to(source_id.device)], dim=0)
            source_ids.append(source_id)
        source_ids = torch.stack(source_ids, dim=0)
        if self.add_masks_in_pseudo:
            new_input_ids = torch.cat((source_ids, new_pseudo_ids), dim=1)
        else:
            new_input_ids = torch.cat((source_ids, pseudo_ids), dim=1)

        if self.feed_embeds:
            new_input_embeds = self.bert_decoder.embeddings.word_embeddings(new_input_ids)
            _target_mask = target_mask.bool()
            _target_mask = (_target_mask & (prediction_tokens != self.sep_word_id))
            pt_masks = torch.cat([torch.zeros(_target_mask.shape[0], 1).to(_target_mask.device), _target_mask],
                                 dim=1).bool()
            new_input_embeds[:, :pt_masks.shape[1]][pt_masks] = pseudo_sequence_output[_target_mask]

        new_source_mask = (source_ids != self.pad_word_id).type_as(source_ids)
        new_attention_mask = self.create_attention_mask(new_source_mask, target_mask)

        num_source_tokens, source_len = (source_ids != self.pad_word_id).sum(-1), source_ids.shape[1]
        num_pseudo_tokens, pseudo_len = (pseudo_ids != self.pad_word_id).sum(-1), pseudo_ids.shape[1]

        source_mask, source_position_ids = \
            self.create_mask_and_position_ids(num_source_tokens, source_len)
        pseudo_mask, pseudo_position_ids = \
            self.create_mask_and_position_ids(num_pseudo_tokens, pseudo_len, offset=num_source_tokens if self.decode_offset else None)

        new_position_ids = torch.cat((source_position_ids, pseudo_position_ids), dim=1)

        if decoding:
            token_type_ids = torch.cat(
                (torch.ones_like(source_ids) * self.source_type_id,
                 torch.ones_like(pseudo_ids) * self.target_type_id), dim=1).long()

        if self.feed_embeds:
            outputs = self.bert_decoder(
                inputs_embeds = new_input_embeds, keep_source_unmodified=-1, attention_mask=new_attention_mask,
                attention_masks=None, token_type_ids=token_type_ids,
                output_hidden_states=self.multi_layer > 1 or return_all_hidden_states,
                position_ids=new_position_ids, decoder_relative_position_mask=decoder_relative_position_mask) #split_lengths=split_lengths) split_length for source, target ,pseudo
        else:
            outputs = self.bert_decoder(
                input_ids=new_input_ids, keep_source_unmodified=-1, attention_mask=new_attention_mask,
                attention_masks=None, token_type_ids=token_type_ids,
                output_hidden_states=self.multi_layer > 1 or return_all_hidden_states,
                position_ids=new_position_ids, decoder_relative_position_mask=decoder_relative_position_mask) #split_lengths=split_lengths) split_length for source, target ,pseudo

        if return_all_hidden_states: return outputs['hidden_states']
        if self.multi_layer > 1:
            sequence_output = outputs['hidden_states'][-self.multi_layer:]
            sequence_output = torch.cat(sequence_output[-self.multi_layer:], dim=-1)
        else:
            sequence_output = outputs['last_hidden_state']
        return encoder_sequence_output, sequence_output, prediction_scores_masked

    def forward(self, source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, target_kd_ids=None,
                decode=False, decoder_all_layer=False, target_span_ids=None, use_gd_len=False):
        if decode:
            source_mask = source_ids != self.pad_word_id
            position_ids = torch.arange(source_ids.shape[1]).repeat(source_ids.shape[0], 1).to(source_ids.device)
            position_ids.masked_fill_(~source_mask, 0)
            token_type_ids = torch.zeros_like(source_ids).to(source_ids.device)
            token_type_ids.masked_fill_(~source_mask, 1)

            length_out = (target_ids != self.pad_word_id).sum(-1)
            if use_gd_len:
                length_out.to(source_ids.device)
                prediction_tokens, prediction_tokens_before, pred_length_out = self.forward_decode(source_ids, token_type_ids,
                                                                      position_ids, source_mask, length_out=length_out, decoder_all_layer=decoder_all_layer)
            else:
                prediction_tokens, prediction_tokens_before, pred_length_out = self.forward_decode(source_ids, token_type_ids,
                                                                      position_ids, source_mask, decoder_all_layer=decoder_all_layer)
            len_acc = (length_out == pred_length_out).sum()
            min_size = min(target_ids.shape[-1], prediction_tokens.shape[-1])
            _target_mask = (target_ids != self.pad_word_id)[:, :min_size]
            tokens_acc = (prediction_tokens[:, :min_size] == target_ids[:, :min_size]).masked_fill(~_target_mask, 0).sum()
            tokens_acc = torch.true_divide(tokens_acc, _target_mask.sum())
            len_acc = torch.true_divide(len_acc, length_out.shape[0])
            return prediction_tokens, prediction_tokens_before, pred_length_out, len_acc, tokens_acc

        if self.use_glat:
            with torch.no_grad():
                pseudo_ids, N = self.forward_glat(source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens)
        source_len = source_ids.size(1)
        target_len = target_ids.size(1)
        pseudo_len = pseudo_ids.size(1)
        assert target_len == pseudo_len
        assert source_len > 0 and target_len > 0
        split_lengths = (source_len, target_len, pseudo_len)

        input_ids = torch.cat((source_ids, pseudo_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones_like(source_ids) * self.source_type_id,
             torch.ones_like(pseudo_ids) * self.target_type_id), dim=1)

        source_mask, source_position_ids = \
            self.create_mask_and_position_ids(num_source_tokens, source_len)
        target_mask, target_position_ids = \
            self.create_mask_and_position_ids(num_target_tokens, target_len, offset=num_source_tokens if self.decode_offset else None)

        if self.remove_sep_in_target:
            indices = (target_mask.sum(-1) - 1).view(-1, 1)
            pseudo_ids.scatter_(1, indices, self.pad_word_id)
            target_ids.scatter_(1, indices, self.pad_word_id)
            target_mask.scatter_(1, indices, 0)
            target_position_ids.scatter_(1, indices, 0)

       # pseudo_ids.scatter_(1, (target_mask.sum(-1) - 1).view(-1, 1), self.sep_word_id)

        position_ids = torch.cat((source_position_ids, target_position_ids), dim=1)
        if target_span_ids is None:
            target_span_ids = target_position_ids

        outputs = self.feed_bert(input_ids, source_mask, target_mask,
                                 token_type_ids, position_ids, target_position_ids,
                                 target_ids=target_ids)
        encoder_sequence_output, decoder_sequence_output, encoder_prediction_scores_masked = outputs[:3]

        # only use encoder to predict length
        length_tgt = target_mask.sum(-1)
        encoder_length_out = self.forward_length(encoder_sequence_output[:, :source_len], source_mask)
        encoder_length_loss = F.cross_entropy(encoder_length_out, length_tgt)
        length_loss = encoder_length_loss #+ decoder_length_loss

        if self.add_decoder_length_loss:
            decoder_length_out = self.forward_length(decoder_sequence_output[:, :source_len], source_mask)
            decoder_length_loss = F.cross_entropy(decoder_length_out, length_tgt)
            length_loss += decoder_length_loss


        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        decoder_prediction_scores_masked = self.cls(decoder_sequence_output[:, source_len:])


        pseudo_lm_losses = []
        for index, prediction_scores_masked in enumerate([encoder_prediction_scores_masked, decoder_prediction_scores_masked]):
            if index == 0 and target_kd_ids is not None: 
                g_ids = target_kd_ids
                tl = target_ids.shape[1]
                gl = g_ids.shape[1]
                if gl > tl: 
                    g_ids = g_ids[:, :tl]
                elif gl < tl: 
                    _pad = torch.ones(g_ids.shape[0], tl - gl).to(target_ids.device).fill_(self.pad_word_id)
                    g_ids = torch.cat([g_ids, _pad], dim = 1)
                #g_masks = g_ids != self.pad_word_id
            else:
                g_ids = target_ids
                #g_masks = target_mask
            if self.crit_mask_lm_smoothed:
                masked_lm_loss = self.crit_mask_lm_smoothed(
                    F.log_softmax(prediction_scores_masked.float(), dim=-1), g_ids)
            else:
                masked_lm_loss = self.crit_mask_lm(
                    prediction_scores_masked.transpose(1, 2).float(), g_ids)

            pseudo_lm_losses.append(loss_mask_and_normalize(masked_lm_loss.float(), target_mask))

        encoder_pseudo_lm_loss, decoder_pseudo_lm_loss = pseudo_lm_losses

        if self.use_glat:
            return encoder_pseudo_lm_loss, decoder_pseudo_lm_loss, length_loss, torch.mean(N.float())
        else:
            return encoder_pseudo_lm_loss, decoder_pseudo_lm_loss, length_loss

    def forward_glat(self, source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, target_span_ids=None):
        source_len = source_ids.size(1)
        target_len = target_ids.size(1)
        pseudo_len = pseudo_ids.size(1)
        assert target_len == pseudo_len
        assert source_len > 0 and target_len > 0
        split_lengths = (source_len, target_len, pseudo_len)

        input_ids = torch.cat((source_ids, pseudo_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones_like(source_ids) * self.source_type_id,
             torch.ones_like(pseudo_ids) * self.target_type_id), dim=1)

        source_mask, source_position_ids = \
            self.create_mask_and_position_ids(num_source_tokens, source_len)
        target_mask, target_position_ids = \
            self.create_mask_and_position_ids(num_target_tokens, target_len, offset=num_source_tokens if self.decode_offset else None)

        if self.remove_sep_in_target:
            indices = (target_mask.sum(-1) - 1).view(-1, 1)
            pseudo_ids.scatter_(1, indices, self.pad_word_id)
            target_ids.scatter_(1, indices, self.pad_word_id)
            target_mask.scatter_(1, indices, 0)
            target_position_ids.scatter_(1, indices, 0)

        #pseudo_ids.scatter_(1, (target_mask.sum(-1) - 1).view(-1, 1), self.sep_word_id)

        position_ids = torch.cat((source_position_ids, target_position_ids), dim=1)
        if target_span_ids is None:
            target_span_ids = target_position_ids

        outputs = self.feed_bert(input_ids, source_mask, target_mask,
                                 token_type_ids, position_ids, target_position_ids)
        encoder_sequence_output, decoder_sequence_output, encoder_prediction_scores_masked = outputs[:3]

        # pseudo_sequence_output = sequence_output[:, source_len:, ]

        prediction_scores_masked = self.cls(decoder_sequence_output[:, source_len:])
        prediction_tokens = prediction_scores_masked.max(-1)[-1]
        N = ((prediction_tokens != target_ids) & (target_mask == 1)).sum(-1) * self.glat_f
        N = N.long()

        _, indices = torch.sort(torch.rand(pseudo_ids.shape), descending=True)
        indices = indices.to(source_ids.device)
        if self.glat_random_prob:
            ind_masks = torch.rand_like(indices.float()) > self.glat_random_prob
            ind_masks.to(indices.device)
        for i, indice in enumerate(indices):
            indice = indice[indice < target_mask[i].sum()]
            n = N[i].item()
            if self.glat_random_prob:
                ind = indice[:n]
                ind_mask = ind_masks[i][:n]
                pseudo_ids[i, ind[ind_mask]]  = target_ids[i, ind[ind_mask]]
                rn = (n - ind_mask.sum()).item()
                if rn > 0:
                    pseudo_ids[i, ind[~ind_mask]]  = torch.randint(0, self.config.vocab_size-1, (rn,)).long().to(ind.device)
            else:
                pseudo_ids[i, indice[:n]] = target_ids[i, indice[:n]]

        return pseudo_ids, N
        # pseudo_lm_loss, length_loss = self.forward(source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, target_span_ids)
        # return pseudo_lm_loss, length_loss, torch.mean(N.float())

    def forward_decode(self, input_ids, token_type_ids, position_ids, input_mask, length_out=None, decoder_all_layer=False):
        source_len = input_ids.shape[1]
        token_type_ids = token_type_ids[:, :source_len]
        position_ids = position_ids[:, :source_len]
        input_mask = input_mask[:, :source_len]
        source_ids = input_ids
        source_mask, source_position_ids = (input_ids != self.pad_word_id).int(), position_ids

        if length_out is None:
            weight = torch.ones_like(input_ids)
            weight[input_ids == self.pad_word_id] = 0
            from_weight = weight.unsqueeze(-1)
            to_weight = weight.unsqueeze(1)
            attention_mask = ((from_weight > 0) & (to_weight > 0)).bool()

            outputs = self.bert_encoder(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                        position_ids=position_ids, output_hidden_states=self.multi_layer > 1)

            sequence_output = outputs['last_hidden_state']

            length_out = self.forward_length(sequence_output, source_mask)
            length_out = length_out.max(-1)[1]
            length_out[length_out > 48] = 48
            length_out[length_out < 7] = 7
        else:
            length_out += 1

        target_len = length_out.max().item()
        if target_len + source_len > 512:
            source_len = 512-target_len
            source_ids = source_ids[:, :source_len]
            source_position_ids = source_position_ids[:, :source_len]
            source_mask = source_mask[:, :source_len]

        pseudo_ids = torch.empty(length_out.shape[0], target_len).fill_(self.mask_word_id).to(input_ids.device)
        base_position_matrix = torch.arange(target_len, dtype=input_ids.dtype,
                                            device=input_ids.device).view(1, -1)

        pseudo_mask = base_position_matrix < length_out.view(-1, 1)
        #pseudo_ids.scatter_(1, (pseudo_mask.sum(-1) - 1).view(-1, 1), self.sep_word_id)

        if self.decode_offset:
            pseudo_position_ids = base_position_matrix * pseudo_mask + source_mask.sum(-1).view(-1, 1)
        else:
            pseudo_position_ids = base_position_matrix * pseudo_mask

        pseudo_ids[~pseudo_mask] = self.pad_word_id
        input_ids = torch.cat((source_ids, pseudo_ids), dim=1).long()

        position_ids = torch.cat((source_position_ids, pseudo_position_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones_like(source_ids) * self.source_type_id,
             torch.ones_like(pseudo_ids) * self.target_type_id), dim=1).long()

        # import pdb; pdb.set_trace()
        outputs = self.feed_bert(input_ids, source_mask, pseudo_mask,
                                token_type_ids, position_ids, pseudo_position_ids,
                                decoding=True,
                                return_all_hidden_states=decoder_all_layer)
        encoder_sequence_output, decoder_sequence_output, encoder_prediction_scores_masked = outputs[:3]

        pseudo_len = encoder_prediction_scores_masked.shape[1]
        decoder_prediction_scores_masked = self.cls(decoder_sequence_output[:, -pseudo_len:, ])
        decoder_prediction_tokens = decoder_prediction_scores_masked.max(-1)[-1]
        decoder_prediction_tokens[~pseudo_mask] = self.pad_word_id

        encoder_prediction_tokens = encoder_prediction_scores_masked.max(-1)[-1]
        encoder_prediction_tokens[~pseudo_mask] = self.pad_word_id
        return decoder_prediction_tokens, encoder_prediction_tokens, length_out

    def swa_init(self):
        self.swa_state = {'models_num': 1}
        for n, p in self.named_parameters():
            self.swa_state[n] = p.data.cpu().clone().detach()

    def swa_step(self):
        if 'models_num' not in self.swa_state:
            return
        self.swa_state['models_num'] += 1
        beta = 1.0 / self.swa_state['models_num']
        with torch.no_grad():
            for n, p in self.named_parameters():
                self.swa_state[n].mul_(1.0 - beta).add_(beta*p.data.cpu())

    def swa_swap_params(self):
        if 'models_num' not in self.swa_state:
            return
        for n, p in self.named_parameters():
            device = p.data.device
            self.swa_state[n], p.data =  self.swa_state[n].cpu(), p.data.cpu()
            self.swa_state[n], p.data =  p.data.cpu(), self.swa_state[n].to(device)

if __name__ == '__main__':
    unilm_path = './models/unilm1.2-base-uncased/'
    un = UnilmNAT(unilm_path, use_glat=False, multi_layer_attention=False, decoder_relative_position=True,
                multi_layer=1, glat_random_prob=0.0, use_multi_glat=False, mask_target_loss=True)

    source_ids = torch.zeros(3, 10).long()
    target_ids = torch.ones(3, 10).long()*torch.tensor([233,234, 235]).view(-1, 1).long()
    pseudo_ids = torch.zeros(3, 10).long()
    num_source_tokens = torch.tensor([5, 5, 5]).long()
    num_target_tokens = torch.tensor([5, 5, 5]).long()
    # print(un.forward_decode(source_ids, token_type_ids,

    print(un.forward(source_ids, target_ids, torch.ones_like(source_ids),
                     num_source_tokens, num_target_tokens))
