from __future__ import annotations

from pocketalign.objectives import dpo_objective, dpo_step_terms, sft_objective, span_logps
from pocketalign.packing import pack_pairs, pack_rows


def build_sft_batch(tokenizer, prompts, responses, max_length):
    return pack_rows(tokenizer, prompts, responses, max_length, fixed_width=True)


def compute_response_logprobs(model, tokenizer, prompts, responses, max_length):
    return span_logps(model, pack_rows(tokenizer, prompts, responses, max_length))[0]


def compute_sft_loss(model, tokenizer, prompts, responses, max_length):
    return sft_objective(model, pack_rows(tokenizer, prompts, responses, max_length))


def build_dpo_batch(tokenizer, prompts, chosen_responses, rejected_responses, max_length):
    return pack_pairs(tokenizer, prompts, chosen_responses, rejected_responses, max_length, fixed_width=True)


def compute_dpo_loss_from_logps(
    policy_chosen_logps,
    policy_rejected_logps,
    reference_chosen_logps,
    reference_rejected_logps,
    beta,
):
    return dpo_objective(policy_chosen_logps, policy_rejected_logps, reference_chosen_logps, reference_rejected_logps, beta)


def compute_dpo_loss(
    policy_model,
    reference_model,
    tokenizer,
    prompts,
    chosen_responses,
    rejected_responses,
    beta,
    max_length,
):
    pair_batch = pack_pairs(tokenizer, prompts, chosen_responses, rejected_responses, max_length)
    return dpo_step_terms(policy_model, reference_model, pair_batch, beta)[0]
