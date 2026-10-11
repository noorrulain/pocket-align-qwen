from datasets import load_dataset

from pocketalign.frame import frame_query


def _fits(tok, query, prompt_cap):
    return tok is None or len(tok.encode(frame_query(query), add_special_tokens=False)) <= prompt_cap


def ultrachat_rows(split, count, seed):
    pool = load_dataset("HuggingFaceH4/ultrachat_200k", split=split).shuffle(seed=seed)
    kept = []
    for rec in pool:
        turns = rec["messages"]
        if len(turns) < 2 or turns[0]["role"] != "user" or turns[1]["role"] != "assistant":
            continue
        query, reply = turns[0]["content"].strip(), turns[1]["content"].strip()
        if query and reply:
            kept.append({"uid": rec["prompt_id"], "query": query, "reply": reply})
        if len(kept) == count:
            break
    return kept


def _hh_single(text):
    if text.count("\n\nHuman:") != 1 or text.count("\n\nAssistant:") != 1:
        return None
    head, tail = text.split("\n\nAssistant:")
    if "\n\nHuman:" not in head:
        return None
    return head.split("\n\nHuman:", 1)[1].strip(), tail.strip()


def hh_rows(split, count, seed, tok=None, prompt_cap=256):
    pool = load_dataset("Anthropic/hh-rlhf", split=split).shuffle(seed=seed)
    kept = []
    for pos, rec in enumerate(pool):
        good, bad = _hh_single(rec["chosen"]), _hh_single(rec["rejected"])
        if not good or not bad or good[0] != bad[0]:
            continue
        query, win, lose = good[0], good[1], bad[1]
        if not query or not win or not lose or win == lose or not _fits(tok, query, prompt_cap):
            continue
        kept.append({"uid": f"hh-{split}-{pos}", "query": query, "chosen": win, "rejected": lose})
        if len(kept) == count:
            break
    return kept


def ultrafeedback_rows(split, count, seed, tok=None, prompt_cap=256):
    pool = load_dataset("HuggingFaceH4/ultrafeedback_binarized", split=split).shuffle(seed=seed)
    kept = []
    for rec in pool:
        if len(rec["chosen"]) != 2 or len(rec["rejected"]) != 2:
            continue
        if rec["score_chosen"] <= rec["score_rejected"]:
            continue
        query = rec["prompt"].strip()
        win, lose = rec["chosen"][1]["content"].strip(), rec["rejected"][1]["content"].strip()
        if not query or not win or not lose or win == lose or not _fits(tok, query, prompt_cap):
            continue
        kept.append({"uid": rec["prompt_id"], "query": query, "chosen": win, "rejected": lose})
        if len(kept) == count:
            break
    return kept
