import argparse
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from pocketalign.common import dump_json, load_json
from pocketalign.frame import STOP_MARK, frame_query


def cli():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--adapter_path", default="")
    ap.add_argument("--generator", required=True)
    ap.add_argument("--eval_file", required=True)
    ap.add_argument("--out_file", required=True)
    ap.add_argument("--style", choices=["frame", "chat"], default="frame")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--repetition_penalty", type=float, default=1.1)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--dtype", choices=["float16", "float32"], default="float16")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--show", action="store_true")
    return ap.parse_args()


def main():
    cfg = cli()
    tok = AutoTokenizer.from_pretrained(cfg.model_path)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(cfg.model_path, torch_dtype=getattr(torch, cfg.dtype)).to(cfg.device)
    if cfg.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cfg.adapter_path).merge_and_unload()
    model.eval()
    items = load_json(cfg.eval_file)
    if cfg.limit:
        items = items[: cfg.limit]
    if cfg.style == "frame":
        texts = [frame_query(it["instruction"]) for it in items]
        stops = [STOP_MARK]
    else:
        texts = [tok.apply_chat_template([{"role": "user", "content": it["instruction"]}], tokenize=False, add_generation_prompt=True) for it in items]
        stops = None
    outs = []
    clock = time.time()
    for lo in range(0, len(texts), cfg.batch):
        enc = tok(texts[lo : lo + cfg.batch], return_tensors="pt", padding=True, add_special_tokens=False).to(cfg.device)
        with torch.no_grad():
            made = model.generate(
                **enc,
                do_sample=False,
                max_new_tokens=cfg.max_new_tokens,
                repetition_penalty=cfg.repetition_penalty,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id,
                stop_strings=stops,
                tokenizer=tok if stops else None,
            )
        for row in made[:, enc["input_ids"].shape[1] :]:
            text = tok.decode(row, skip_special_tokens=True)
            if stops:
                text = text.split(STOP_MARK)[0]
            outs.append(text.strip())
        print(cfg.generator, len(outs), "/", len(texts), round(time.time() - clock), "s", flush=True)
    rows = []
    for it, text in zip(items, outs):
        rec = {"id": it["id"], "instruction": it["instruction"], "output": text, "generator": cfg.generator, "dataset": it.get("dataset", "")}
        if "category" in it:
            rec["category"] = it["category"]
        rows.append(rec)
    dump_json(cfg.out_file, rows)
    if cfg.show:
        for rec in rows:
            print("=" * 80, "\n", rec["instruction"], "\n", "-" * 40, "\n", rec["output"], flush=True)


if __name__ == "__main__":
    main()
