import math

import torch

import submission_adapter as adapter
from tests.toy_lm import ToyCausalLM, ToyTokenizer


class NoPadTokenizer(ToyTokenizer):
    pad_token_id = None


def test_truncation_drops_eos_and_keeps_width():
    tok = ToyTokenizer()
    batch = adapter.build_sft_batch(tok, ["ab: "], ["abcdefgh"], max_length=8)
    assert batch["input_ids"].shape == (1, 8)
    assert tok.eos_token_id not in batch["labels"][0].tolist()
    assert batch["labels"][0, :4].eq(-100).all() and batch["labels"][0, 4:].ne(-100).all()


def test_prompt_longer_than_limit_scores_zero():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.5)
    logps = adapter.compute_response_logprobs(model, tok, ["abcdefghij"], ["xyz"], max_length=6)
    assert logps.item() == 0.0


def test_missing_pad_token_uses_eos():
    tok = NoPadTokenizer()
    batch = adapter.build_sft_batch(tok, ["a", "abc"], ["b", "c"], max_length=7)
    assert batch["input_ids"][0, -1].item() == tok.eos_token_id
    assert batch["attention_mask"][0, -1].item() == 0


def test_row_order_and_exact_values():
    tok = ToyTokenizer()
    model = ToyCausalLM(tok.vocab_size, bias_scale=0.9)
    prompts, replies = ["q: ", "hello: "], ["ab", "c"]
    fwd = adapter.compute_response_logprobs(model, tok, prompts, replies, max_length=32)
    rev = adapter.compute_response_logprobs(model, tok, prompts[::-1], replies[::-1], max_length=32)
    torch.testing.assert_close(fwd, rev.flip(0))
    table = torch.log_softmax(model.logit_bias.detach(), -1)
    expect = [sum(table[t].item() for t in tok.encode(r) + [tok.eos_token_id]) for r in replies]
    torch.testing.assert_close(fwd.detach(), torch.tensor(expect))


def test_dpo_value_matches_formula():
    pc, pr, rc, rr = torch.tensor([-1.0]), torch.tensor([-3.0]), torch.tensor([-2.0]), torch.tensor([-2.5])
    loss = adapter.compute_dpo_loss_from_logps(pc, pr, rc, rr, beta=0.5)
    gap = 0.5 * ((-1.0 + 3.0) - (-2.0 + 2.5))
    assert math.isclose(loss.item(), math.log1p(math.exp(-gap)), rel_tol=1e-6)
