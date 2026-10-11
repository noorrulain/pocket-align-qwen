import torch
import torch.nn.functional as F

from pocketalign.packing import MASK_ID


def _home(model):
    for weight in model.parameters():
        return weight.device
    return torch.device("cpu")


def span_logps(model, batch):
    dev = _home(model)
    ids = batch["input_ids"].to(dev)
    att = batch["attention_mask"].to(dev)
    lab = batch["labels"].to(dev)
    logits = model(input_ids=ids, attention_mask=att).logits[:, :-1].float()
    nxt = lab[:, 1:]
    live = nxt.ne(MASK_ID)
    picked = torch.gather(logits, -1, nxt.clamp_min(0).unsqueeze(-1)).squeeze(-1)
    token_lp = picked - torch.logsumexp(logits, dim=-1)
    return torch.where(live, token_lp, torch.zeros_like(token_lp)).sum(-1), live.sum(-1)


def sft_objective(model, batch):
    total, count = span_logps(model, batch)
    return -total.sum() / count.sum().clamp_min(1)


def dpo_objective(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta):
    gap = (pol_chosen - pol_rejected) - (ref_chosen - ref_rejected)
    return -F.logsigmoid(beta * gap).mean()


def implicit_rewards(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta):
    return beta * (pol_chosen - ref_chosen), beta * (pol_rejected - ref_rejected)


def pair_logps(model, pair_batch):
    joint = {k: torch.cat([pair_batch["chosen"][k], pair_batch["rejected"][k]]) for k in pair_batch["chosen"]}
    sums, _ = span_logps(model, joint)
    half = pair_batch["chosen"]["input_ids"].shape[0]
    return sums[:half], sums[half:]


def dpo_step_terms(policy, reference, pair_batch, beta):
    with torch.no_grad():
        ref_chosen, ref_rejected = pair_logps(reference, pair_batch)
    pol_chosen, pol_rejected = pair_logps(policy, pair_batch)
    ref_chosen = ref_chosen.to(pol_chosen.device)
    ref_rejected = ref_rejected.to(pol_chosen.device)
    loss = dpo_objective(pol_chosen, pol_rejected, ref_chosen, ref_rejected, beta)
    return loss, (pol_chosen.detach(), pol_rejected.detach(), ref_chosen, ref_rejected)
