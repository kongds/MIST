#from fairseq.data import BertDictionary
import torch
from fairseq.tasks import register_task
from fairseq.tasks.translation import TranslationTask

from .bert_dictionary import BertDictionary
from .iterative_refinement_generator import IterativeRefinementGenerator

def new_arange(x, *size):
    """
    Return a Tensor of `size` filled with a range function on the device of x.
    If size is empty, using the size of the variable x.
    """
    if len(size) == 0:
        size = x.size()
    return torch.arange(size[-1], device=x.device).expand(*size).contiguous()

@register_task('translation_nat_mass')
class NATTranslationMASSTask(TranslationTask):
    def __init__(self, args, src_dict, tgt_dict):
        super().__init__(args, src_dict, tgt_dict)

    @classmethod
    def load_dictionary(cls, filename):
        return BertDictionary.load_from_file(filename)

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        return (self.args.max_source_positions, self.args.max_target_positions)

    def inject_noise(self, target_tokens):
        def _random_mask(target_tokens):
            pad = self.tgt_dict.pad()
            bos = self.tgt_dict.bos()
            eos = self.tgt_dict.eos()
            unk = self.tgt_dict.unk()
            mask = self.src_dict.index('[MASK]')

            target_masks = (
                target_tokens.ne(pad) & target_tokens.ne(bos) & target_tokens.ne(eos)
            )
            target_score = target_tokens.clone().float().uniform_()
            target_score.masked_fill_(~target_masks, 2.0)
            target_length = target_masks.sum(1).float()
            target_length = target_length * 0.80 # mask
            target_length = target_length + 1  # make sure to mask at least one token.

            _, target_rank = target_score.sort(1)
            target_cutoff = new_arange(target_rank) < target_length[:, None].long()
            prev_target_tokens = target_tokens.masked_fill(
                target_cutoff.scatter(1, target_rank, target_cutoff), mask
            )
            return prev_target_tokens

        return _random_mask(target_tokens)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        #s = sample['target'].shape
        #sample['target'] = torch.cat([torch.zeros(s[0], 1, device=sample['target'].device).fill_(self.src_dict.bos()),
                                      #sample['target']], dim=1)
        sample["prev_target"] = self.inject_noise(sample["target"])
        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        #s = sample['target'].shape
        #sample['target'] = torch.cat([torch.zeros(s[0], 1, device=sample['target'].device).fill_(self.src_dict.bos()),
                                      #sample['target']], dim=1)
        with torch.no_grad():
            sample["prev_target"] = self.inject_noise(sample["target"])
            loss, sample_size, logging_output = criterion(model, sample, True)
        return loss, sample_size, logging_output

    def build_generator(self, args, **unused):
        # add models input to match the API for SequenceGenerator

        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, "iter_decode_eos_penalty", 0.0),
            max_iter=getattr(args, "iter_decode_max_iter", 0),
            beam_size=getattr(args, "iter_decode_with_beam", 1),
            reranking=getattr(args, "iter_decode_with_external_reranker", False),
            decoding_format=getattr(args, "decoding_format", None),
            adaptive=False,
            retain_history=getattr(args, "retain_iter_history", False),
        )
