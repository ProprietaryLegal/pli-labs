#!/usr/bin/env python3
"""LC-CACHE experiment: cold vs warm prompt_ms with --cache-reuse on llama-server (the test GPU)."""
import json, sys, time, urllib.request

BASE = "http://127.0.0.1:8199"

def post(path, payload, timeout=600):
    req = urllib.request.Request(BASE + path, data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        body = json.loads(r.read().decode())
    return body, (time.time() - t0) * 1000.0

def tokenize_count(text):
    body, _ = post("/tokenize", {"content": text})
    return len(body["tokens"])

def build_prefix(target_tokens, seed_word):
    # Repeatable synthetic system prefix; distinct per seed_word so each COLD trial truly misses cache.
    sent = (f"System policy {seed_word}: The legal-drafting harness legal assistant must review South Carolina "
            f"family court procedure, discovery obligations, exhibit indexing, financial tracing, "
            f"and affidavit drafting standards with meticulous care and consistent citation cadence. ")
    text = sent
    n = tokenize_count(text)
    reps = max(1, int(target_tokens / n))
    text = sent * reps
    n = tokenize_count(text)
    while n < target_tokens:
        text += sent
        n = tokenize_count(text)
    return text, n

def completion(prompt, label):
    payload = {"prompt": prompt, "n_predict": 16, "temperature": 0.0, "cache_prompt": True}
    body, wall = post("/completion", payload)
    t = body.get("timings", {})
    row = {
        "label": label,
        "prompt_n": t.get("prompt_n"),
        "prompt_ms": t.get("prompt_ms"),
        "predicted_n": t.get("predicted_n"),
        "predicted_ms": t.get("predicted_ms"),
        "wall_ms": round(wall, 1),
        "tokens_cached": body.get("tokens_cached"),
    }
    print(json.dumps(row), flush=True)
    return row

def run_block(target_tokens, trials, tag):
    rows = []
    for i in range(trials):
        prefix, ntok = build_prefix(target_tokens, f"{tag}-trial{i}")
        print(f"# {tag} trial {i}: prefix ~{ntok} tokens", flush=True)
        qa = f"\n\nQuestion: What is the {i*2+1}th most important deadline rule for motion practice? Answer briefly."
        qb = f"\n\nQuestion: Explain exhibit numbering requirement number {i*2+2} in one sentence."
        rows.append(completion(prefix + qa, f"{tag}-cold-{i}"))
        rows.append(completion(prefix + qb, f"{tag}-warm-{i}"))
    return rows

if __name__ == "__main__":
    print("# health:", json.dumps(post("/health", {"":""})[0] if False else "skip"), flush=True)
    all_rows = []
    all_rows += run_block(6000, 3, "p6k")
    all_rows += run_block(12000, 2, "p12k")
    with open("lc_cache_raw.json", "w") as f:
        json.dump(all_rows, f, indent=1)
    print("# DONE", flush=True)
