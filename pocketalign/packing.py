import torch

MASK_ID = -100


def _encode(tok, text):
    return list(tok.encode(text, add_special_tokens=False))


def _filler(tok):
    for key in ("pad_token_id", "eos_token_id"):
        val = getattr(tok, key, None)
        if val is not None:
            return val
    return 0


def pack_rows(tok, prompts, replies, max_length, fixed_width=False):
    stop_id = getattr(tok, "eos_token_id", None)
    rows = []
    for ask, say in zip(prompts, replies):
        ask_ids = _encode(tok, ask)
        say_ids = _encode(tok, say)
        if stop_id is not None:
            say_ids.append(stop_id)
        tokens = (ask_ids + say_ids)[:max_length]
        targets = ([MASK_ID] * len(ask_ids) + say_ids)[:max_length]
        rows.append((tokens, targets))
    width = max_length if fixed_width else max([len(t) for t, _ in rows] + [1])
    count = len(rows)
    input_ids = torch.full((count, width), _filler(tok), dtype=torch.long)
    attention_mask = torch.zeros((count, width), dtype=torch.long)
    labels = torch.full((count, width), MASK_ID, dtype=torch.long)
    for idx, (tokens, targets) in enumerate(rows):
        span = len(tokens)
        if span:
            input_ids[idx, :span] = torch.tensor(tokens, dtype=torch.long)
            attention_mask[idx, :span] = 1
            labels[idx, :span] = torch.tensor(targets, dtype=torch.long)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}


def pack_pairs(tok, prompts, chosen, rejected, max_length, fixed_width=False):
    prompts = list(prompts)
    joint = pack_rows(tok, prompts + prompts, list(chosen) + list(rejected), max_length, fixed_width)
    half = len(prompts)
    return {
        "chosen": {k: v[:half] for k, v in joint.items()},
        "rejected": {k: v[half:] for k, v in joint.items()},
    }
