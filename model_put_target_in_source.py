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


class MISTNAT(nn.Module):
    def __init__(self, unilm_path, use_glat=False, glat_random_prob=None, glat_f=0.5,
                 sep_word_id=102, mask_word_id=103, pad_word_id=0, clear_bert_weight=False,):
        super(MISTNAT, self).__init__()
        self.source_type_id = 0
        self.target_type_id = 1


        self.use_glat = use_glat
        self.glat_random_prob = glat_random_prob
        self.glat_f = glat_f
        self.mask_word_id = mask_word_id
        self.sep_word_id = sep_word_id
        self.pad_word_id = pad_word_id

        self.bert = BertModel.from_pretrained(unilm_path)

        if clear_bert_weight:
            self.bert.init_weights()

        self.config = self.bert.config
        self.encoder_embed_dim = self.bert.config.hidden_size
        self.embed_length = nn.Embedding(512, self.encoder_embed_dim , None)
        #init cls decoder weight with embedding
        self.cls = BertOnlyMLMHead(self.bert.config)
        self.cls.predictions.decoder.weight = nn.Parameter(self.bert.embeddings.word_embeddings.weight.clone())

        self.crit_mask_lm = nn.CrossEntropyLoss(reduction='none')


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


    def feed_bert(self, input_ids, source_mask, target_mask,
                  token_type_ids, position_ids, target_position_ids, 
                  target_ids=None, decoding=False):

        attention_mask = self.create_attention_mask(source_mask, target_mask)
        decoder_relative_position_mask = None
        source_len = source_mask.size(1)

        outputs = self.bert(input_ids=input_ids, keep_source_unmodified=-1, attention_mask=attention_mask,
                            token_type_ids=token_type_ids,
                            output_hidden_states=False,
                            position_ids=position_ids)
        sequence_output = outputs[0]
        pseudo_sequence_output = sequence_output[:, source_len:, ]
        prediction_scores = self.cls(pseudo_sequence_output)
        _, prediction_tokens = prediction_scores.max(-1)

        prediction_tokens[~target_mask.bool()] = self.pad_word_id

        source_ids = []
        source_len = source_mask.shape[1]
        old_source_ids = input_ids[:, :source_len]
        pseudo_ids = input_ids[:, source_len:]
        if decoding:
            source_len += prediction_tokens.shape[1]
            if source_len + pseudo_ids.shape[1] > 512:
                source_len = 512 - pseudo_ids.shape[1]

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
        new_input_ids = torch.cat((source_ids, pseudo_ids), dim=1)

        new_source_mask = (source_ids != self.pad_word_id).type_as(source_ids)
        new_attention_mask = self.create_attention_mask(new_source_mask, target_mask)

        num_source_tokens, source_len = (source_ids != self.pad_word_id).sum(-1), source_ids.shape[1]
        num_pseudo_tokens, pseudo_len = (pseudo_ids != self.pad_word_id).sum(-1), pseudo_ids.shape[1]

        source_mask, source_position_ids = \
            self.create_mask_and_position_ids(num_source_tokens, source_len)
        pseudo_mask, pseudo_position_ids = \
            self.create_mask_and_position_ids(num_pseudo_tokens, pseudo_len, offset=num_source_tokens)

        new_position_ids = torch.cat((source_position_ids, pseudo_position_ids), dim=1)

        if decoding:
            token_type_ids = torch.cat(
                (torch.ones_like(source_ids) * self.source_type_id,
                 torch.ones_like(pseudo_ids) * self.target_type_id), dim=1).long()

        mist_outputs = self.bert(
            input_ids=new_input_ids, keep_source_unmodified=-1, attention_mask=new_attention_mask,
            attention_masks=None, token_type_ids=token_type_ids,
            output_hidden_states=False,
            position_ids=new_position_ids, decoder_relative_position_mask=decoder_relative_position_mask)

        mist_sequence_output = mist_outputs['last_hidden_state']
        mist_prediction_scores = self.cls(mist_sequence_output[:, source_len:])

        return sequence_output, mist_sequence_output, prediction_scores, mist_prediction_scores

    def forward(self, source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens, decode=False):
        if decode:
            source_mask = source_ids != self.pad_word_id
            position_ids = torch.arange(source_ids.shape[1]).repeat(source_ids.shape[0], 1).to(source_ids.device)
            position_ids.masked_fill_(~source_mask, 0)
            token_type_ids = torch.zeros_like(source_ids).to(source_ids.device)
            token_type_ids.masked_fill_(~source_mask, 1)

            length_out = (target_ids != self.pad_word_id).sum(-1)
            prediction_tokens, prediction_tokens_before, pred_length_out = self.forward_decode(source_ids, token_type_ids,
                                                                    position_ids, source_mask)
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

        input_ids = torch.cat((source_ids, pseudo_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones_like(source_ids) * self.source_type_id,
             torch.ones_like(pseudo_ids) * self.target_type_id), dim=1)

        source_mask, source_position_ids = \
            self.create_mask_and_position_ids(num_source_tokens, source_len)
        target_mask, target_position_ids = \
            self.create_mask_and_position_ids(num_target_tokens, target_len, offset=num_source_tokens)

        position_ids = torch.cat((source_position_ids, target_position_ids), dim=1)

        outputs = self.feed_bert(input_ids, source_mask, target_mask,
                                 token_type_ids, position_ids, target_position_ids,
                                 target_ids=target_ids)
        sequence_output, mist_sequence_output, prediction_scores, mist_prediction_scores = outputs[:4]

        length_tgt = target_mask.sum(-1)
        length_out = self.forward_length(sequence_output[:, :source_len], source_mask)
        length_loss = F.cross_entropy(length_out, length_tgt)
        length_loss = length_loss


        def loss_mask_and_normalize(loss, mask):
            mask = mask.type_as(loss)
            loss = loss * mask
            denominator = torch.sum(mask) + 1e-5
            return (loss / denominator).sum()

        pseudo_lm_losses = []
        for prediction_scores_masked in [prediction_scores, mist_prediction_scores]:
            masked_lm_loss = self.crit_mask_lm(
                prediction_scores_masked.transpose(1, 2).float(), target_ids)
            pseudo_lm_losses.append(loss_mask_and_normalize(masked_lm_loss.float(), target_mask))

        pseudo_lm_loss, mist_pseudo_lm_loss = pseudo_lm_losses

        if self.use_glat:
            return pseudo_lm_loss, mist_pseudo_lm_loss, length_loss, torch.mean(N.float())
        else:
            return pseudo_lm_loss, mist_pseudo_lm_loss, length_loss

    def forward_glat(self, source_ids, target_ids, pseudo_ids, num_source_tokens, num_target_tokens):
        source_len = source_ids.size(1)
        target_len = target_ids.size(1)
        pseudo_len = pseudo_ids.size(1)
        assert target_len == pseudo_len
        assert source_len > 0 and target_len > 0

        input_ids = torch.cat((source_ids, pseudo_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones_like(source_ids) * self.source_type_id,
             torch.ones_like(pseudo_ids) * self.target_type_id), dim=1)

        source_mask, source_position_ids = \
            self.create_mask_and_position_ids(num_source_tokens, source_len)
        target_mask, target_position_ids = \
            self.create_mask_and_position_ids(num_target_tokens, target_len, offset=num_source_tokens)

        #pseudo_ids.scatter_(1, (target_mask.sum(-1) - 1).view(-1, 1), self.sep_word_id)

        position_ids = torch.cat((source_position_ids, target_position_ids), dim=1)

        outputs = self.feed_bert(input_ids, source_mask, target_mask,
                                 token_type_ids, position_ids, target_position_ids)
        mist_sequence_output = outputs[1]

        # pseudo_sequence_output = sequence_output[:, source_len:, ]

        prediction_scores_masked = self.cls(mist_sequence_output[:, source_len:])
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

    def forward_decode(self, input_ids, token_type_ids, position_ids, input_mask, length_out=None):
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

            outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask,
                                        position_ids=position_ids, output_hidden_states=False)

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

        pseudo_position_ids = base_position_matrix * pseudo_mask + source_mask.sum(-1).view(-1, 1)

        pseudo_ids[~pseudo_mask] = self.pad_word_id
        input_ids = torch.cat((source_ids, pseudo_ids), dim=1).long()

        position_ids = torch.cat((source_position_ids, pseudo_position_ids), dim=1)
        token_type_ids = torch.cat(
            (torch.ones_like(source_ids) * self.source_type_id,
             torch.ones_like(pseudo_ids) * self.target_type_id), dim=1).long()

        outputs = self.feed_bert(input_ids, source_mask, pseudo_mask,
                                token_type_ids, position_ids, pseudo_position_ids,
                                decoding=True)
        sequence_output, mist_sequence_output, prediction_scores, mist_prediction_scores = outputs[:4]

        mist_prediction_tokens = mist_prediction_scores.max(-1)[-1]
        mist_prediction_tokens[~pseudo_mask] = self.pad_word_id

        prediction_tokens = prediction_scores.max(-1)[-1]
        prediction_tokens[~pseudo_mask] = self.pad_word_id
        return mist_prediction_tokens, prediction_tokens, length_out
