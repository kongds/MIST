# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn.functional as F
from fairseq import utils
from fairseq.criterions import FairseqCriterion, register_criterion
from torch import Tensor


@register_criterion("nat_loss")
class LabelSmoothedDualImitationCriterion(FairseqCriterion):
    def __init__(self, args, task):
        super().__init__(args, task)
        self.label_smoothing = args.label_smoothing
        self.mist = args.mist


    @staticmethod
    def add_args(parser):
        """Add criterion-specific arguments to the parser."""
        parser.add_argument('--mist', action='store_true')
        parser.add_argument(
            "--label-smoothing",
            default=0.0,
            type=float,
            metavar="D",
            help="epsilon for label smoothing, 0 means no label smoothing",
        )

    def _compute_loss(
        self, outputs, targets, masks=None, label_smoothing=0.0, name="loss", factor=1.0
    ):
        """
        outputs: batch x len x d_model
        targets: batch x len
        masks:   batch x len

        policy_logprob: if there is some policy
            depends on the likelihood score as rewards.
        """

        def mean_ds(x: Tensor, dim=None) -> Tensor:
            return (
                x.float().mean().type_as(x)
                if dim is None
                else x.float().mean(dim).type_as(x)
            )

        if masks is not None:
            outputs, targets = outputs[masks], targets[masks]

        if masks is not None and not masks.any():
            nll_loss = torch.tensor(0)
            loss = nll_loss
        else:
            logits = F.log_softmax(outputs, dim=-1)
            if targets.dim() == 1:
                losses = F.nll_loss(logits, targets.to(logits.device), reduction="none")

            else:  # soft-labels
                losses = F.kl_div(logits, targets.to(logits.device), reduction="none")
                losses = losses.sum(-1)

            nll_loss = mean_ds(losses)
            if label_smoothing > 0:
                loss = (
                    nll_loss * (1 - label_smoothing) - mean_ds(logits) * label_smoothing
                )
            else:
                loss = nll_loss

        loss = loss * factor
        return {"name": name, "loss": loss, "nll_loss": nll_loss, "factor": factor}

    def _custom_loss(self, loss, name="loss", factor=1.0):
        return {"name": name, "loss": loss, "factor": factor}

    def forward(self, model, sample, reduce=True, valid=True):
        """Compute the loss for the given sample.
        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        nsentences, ntokens = sample["nsentences"], sample["ntokens"]

        # B x T

        src_tokens, src_lengths = (
            sample["net_input"]["src_tokens"],
            sample["net_input"]["src_lengths"],
        )
        tgt_tokens = sample["target"]
        prev_tgt_tokens = sample["prev_target"]

        outputs = model(src_tokens, src_lengths, prev_tgt_tokens, tgt_tokens)

        if self.mist:
            decoder_out = outputs['word_ins']['out']
            prediction_ids = decoder_out.max(-1)[1].detach()

            source_ids = src_tokens
            target_mask = tgt_tokens != 0
            source_mask = src_tokens != 0
            prediction_ids[~target_mask] = 0

            new_source_ids = torch.zeros(source_ids.shape[0],
                                         min((target_mask.sum(-1) + source_mask.sum(-1)).max().item(), 512)).to(source_ids.device).long()
            for i in range(new_source_ids.shape[0]):
                pt = prediction_ids[i][target_mask[i]]
                pt[-1] = 102
                os = source_ids[i][source_mask[i]]
                pt_p = pt.shape[0]
                new_source_ids[i][-pt_p:] = pt
                s_p = pt.shape[0]+os.shape[0]
                if s_p > new_source_ids.shape[1]:
                    new_os_len = new_source_ids[i][pt_p:].shape[0]
                    new_source_ids[i][:-pt_p] = os[:new_os_len]
                    new_source_ids[i][-1] = 102
                else:
                    new_source_ids[i][-s_p:-pt_p] = os

            mist_outputs = model(new_source_ids, (new_source_ids != 0).sum(-1), prev_tgt_tokens, tgt_tokens)
            for obj in mist_outputs:
                outputs['mist_' + obj] = mist_outputs[obj]

        losses, nll_loss = [], []

        for obj in outputs:
            if outputs[obj].get("loss", None) is None:
                _losses = self._compute_loss(
                    outputs[obj].get("out"),
                    outputs[obj].get("tgt"),
                    outputs[obj].get("mask", None),
                    outputs[obj].get("ls", 0.0),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )
            else:
                _losses = self._custom_loss(
                    outputs[obj].get("loss"),
                    name=obj + "-loss",
                    factor=outputs[obj].get("factor", 1.0),
                )

            losses += [_losses]
            if outputs[obj].get("nll_loss", False):
                nll_loss += [_losses.get("nll_loss", 0.0)]

        loss = sum(l["loss"] for l in losses)
        nll_loss = sum(l for l in nll_loss) if len(nll_loss) > 0 else loss.new_tensor(0)

        # NOTE:
        # we don't need to use sample_size as denominator for the gradient
        # here sample_size is just used for logging
        sample_size = 1
        logging_output = {
            "loss": loss.data.item(),
            "nll_loss": nll_loss.data.item(),
            "ntokens": ntokens,
            "nsentences": nsentences,
            "sample_size": sample_size,
        }

        for l in losses:
            logging_output[l["name"]] = (
                utils.item(l["loss"].data / l["factor"])
                if reduce
                else l[["loss"]].data / l["factor"]
            )

        return loss, sample_size, logging_output

    @staticmethod
    def aggregate_logging_outputs(logging_outputs):
         """Aggregate logging outputs from data parallel training."""
         loss = sum(log.get("loss", 0) for log in logging_outputs)
         nll_loss = sum(log.get("nll_loss", 0) for log in logging_outputs)
         sample_size = sum(log.get('sample_size', 0) for log in logging_outputs)

         agg_output = {
             'loss': loss / sample_size / math.log(2) if sample_size > 0 else 0.,
             'ntokens': sum(log.get("ntokens", 0) for log in logging_outputs),
             'nsentences': sum(log.get("nsentences", 0) for log in logging_outputs),
             'nll_loss': nll_loss / sample_size / math.log(2) if sample_size > 0 else 0.,
             'sample_size': sample_size,
         }
         return agg_output

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
