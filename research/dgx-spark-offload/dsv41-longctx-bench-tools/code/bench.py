#!/usr/bin/env python3
"""Single-stream benchmark for the DSv4.1 TP2 serve.

Per run: actual prompt/completion tokens (server usage), TTFT, prefill tok/s (prompt_tokens / TTFT),
decode tok/s excluding prefill ((completion_tokens - 1) / (t_last - t_first)), DSpark acceptance from /metrics
deltas, min MemAvailable and max GPU temp / clock on both nodes (sampled over ssh every 2 s).
Cold prefill: every request gets a fresh cache_salt and a random nonce at the start of the prompt.

  ./bench.py --ctx 0 8000 32000 --workload legal --reps 3 --max-tokens 1500 --out logs/bench.jsonl
"""
import argparse, json, random, re, statistics, subprocess, threading, time, urllib.request, uuid

URL = "http://localhost:8000"
MODEL = "DeepSeek-V4.1-Flash-1017-EXL3-Spark-TP2"
NODES = ["spark", "spark-b"]

WORKLOADS = {
    "prose": "Write a detailed, well-organized analytical essay (at least 1,200 words) on how the rise of remote work "
             "has changed commercial real estate, labor markets, and city budgets. Use clear section headings.",
    "legal": "Draft a memorandum in support of a motion for temporary custody in a family court matter, based only on "
             "the record above (or, if no record is provided, on a fictional record you state up front). Use the "
             "structure: Introduction, Statement of Facts (separating undisputed from disputed facts), Issues, "
             "Argument, Conclusion. Do not cite real cases or statutes; mark any authority as [AUTHORITY NEEDED]. "
             "Write at least 1,200 words.",
    "summary": "Summarize the record above for a supervising attorney: a dated chronology of key events, the conflicting "
               "statements and who made them, the evidentiary details that matter most, and open questions. Quote "
               "exact figures and dates. Do not invent facts. At least 900 words.",
    "reason": "Reason step by step through this planning problem and then write up a recommendation of at least 1,000 "
              "words: a small law firm must decide whether to hire a second paralegal or adopt document-automation "
              "software, given caseload growth of 30% per year, current paralegal overtime of 12 hours per week, a "
              "software cost of $18,000/year, and a paralegal cost of $62,000/year fully loaded.",
    "synth": "The materials above are public court opinions and filings. Using ONLY these materials, write a memorandum "
             "(at least 1,200 words) for a family-law attorney that identifies the recurring legal standards the courts "
             "apply, compares how at least four of the cases applied them, and notes any tension between decisions. Name "
             "cases exactly as they appear in the materials; do not cite anything that is not in the materials.",
    "extract": "From the record above, extract every dated event into a JSON array of objects with keys date (ISO), "
               "actor, event, source_exhibit, and disputed (boolean). Output only the JSON.",
}

FIRST = ["Dana", "Marcus", "Elena", "Tobias", "Priya", "Grant", "Lucia", "Owen", "Hazel", "Reuben", "Mira", "Silas"]
LAST = ["Whitfield", "Okafor", "Brannigan", "Castellanos", "Pruitt", "Lindqvist", "Abernathy", "Moreau", "Halvorsen"]
PLACES = ["the marital residence on Larkspur Lane", "Greenville Memorial pediatric clinic", "the Riverside Elementary "
          "pickup loop", "a Spartanburg gas station", "the parties' joint Wells account", "the guardian ad litem's office"]
VERBS = ["testified that", "stated in a text message that", "wrote in an email that", "told the counselor that",
         "alleged in the affidavit that", "denied that", "admitted at deposition that"]
CLAIMS = ["the child was returned {n} hours late", "a payment of ${m:,} was withdrawn without notice",
          "the exchange was moved without agreement", "the child missed {n} days of school",
          "the other parent consumed alcohol before driving", "the counseling session was cancelled",
          "a new partner stayed overnight {n} times", "the medical appointment was rescheduled"]


def record_paragraph(rng, i):
    d = f"20{rng.randint(22, 25)}-{rng.randint(1, 12):02d}-{rng.randint(1, 28):02d}"
    a, b = f"{rng.choice(FIRST)} {rng.choice(LAST)}", f"{rng.choice(FIRST)} {rng.choice(LAST)}"
    c = rng.choice(CLAIMS).format(n=rng.randint(1, 9), m=rng.randint(200, 25000))
    return (f"[Exhibit {i}] On {d}, at {rng.choice(PLACES)}, {a} {rng.choice(VERBS)} {c}. {b} disputes this account "
            f"and contends the event occurred on a different date. The record reflects document control number "
            f"DCN-{rng.randint(100000, 999999)}.")


def tok_count(text):
    req = urllib.request.Request(URL + "/tokenize", data=json.dumps({"model": MODEL, "prompt": text}).encode(),
                                 headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(req, timeout=600))["count"]


_DOCS = None
def build_real(target_tokens, seed):
    """Concatenate public CourtListener opinions + RECAP filings (data/*.jsonl) to target_tokens (exact via /tokenize)."""
    global _DOCS
    if _DOCS is None:
        _DOCS = [json.loads(l) for f in ("data/opinions_cl.jsonl", "data/filings_recap.jsonl") for l in open(f)]
    docs = _DOCS[:]; random.Random(seed).shuffle(docs)
    parts, chars = [], 0
    ratio = 3.3                                    # chars/token first guess; corrected below
    for d in docs:
        parts.append(f"=== {d.get('title', '')} ({d.get('court', '')}, {d.get('date', '')}) ===\n{d['text']}")
        chars += len(parts[-1])
        if chars / ratio >= target_tokens * 1.02:
            n = tok_count("\n\n".join(parts)); ratio = chars / n
            if n >= target_tokens:
                break
    text = "\n\n".join(parts)
    while tok_count(text) > target_tokens:            # trim to the target
        text = text[: int(len(text) - (tok_count(text) - target_tokens) * ratio) - 50]
    return text


def build_record(target_tokens, seed):
    rng = random.Random(seed)
    paras, i, approx = [], 1, 0
    while approx < target_tokens:
        p = record_paragraph(rng, i)
        paras.append(p); approx += len(p) / 3.6; i += 1   # ~3.6 chars/token for this text (calibrated below)
    return "\n".join(paras)


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.stop = threading.Event()
        self.minavail = {n: 1e9 for n in NODES}; self.maxtemp = {n: 0 for n in NODES}
        self.clk = {n: [] for n in NODES}; self.util = {n: [] for n in NODES}

    def run(self):
        while not self.stop.is_set():
            for n in NODES:
                try:
                    o = subprocess.run(["ssh", "-o", "BatchMode=yes", n,
                                        "awk '/MemAvailable/{print $2}' /proc/meminfo; nvidia-smi --query-gpu=temperature.gpu,clocks.gr,utilization.gpu --format=csv,noheader,nounits"],
                                       capture_output=True, text=True, timeout=10).stdout.split("\n")
                    self.minavail[n] = min(self.minavail[n], int(o[0]) / 1048576)
                    t, c, u = [x.strip() for x in o[1].split(",")]
                    self.maxtemp[n] = max(self.maxtemp[n], int(t)); self.clk[n].append(int(c)); self.util[n].append(int(u))
                except Exception:
                    pass
            self.stop.wait(2)


def metrics():
    try:
        txt = urllib.request.urlopen(URL + "/metrics", timeout=5).read().decode()
    except Exception:
        return {}
    out = {}
    for k in ("spec_decode_num_accepted_tokens_total", "spec_decode_num_draft_tokens_total",
              "spec_decode_num_drafts_total"):
        vals = re.findall(rf"^vllm:{k}(?:{{[^}}]*}})? ([0-9.e+]+)$", txt, re.M)
        out[k] = sum(float(v) for v in vals) if vals else None
    return out


def run_one(prompt, max_tokens, thinking, temperature, top_p, seed=None):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": temperature, "top_p": top_p, "stream": True,
            "stream_options": {"include_usage": True, "continuous_usage_stats": False},
            "chat_template_kwargs": {"thinking": thinking}, "cache_salt": uuid.uuid4().hex}
    if seed is not None:
        body["seed"] = seed
    req = urllib.request.Request(URL + "/v1/chat/completions", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = t_last = None; usage = None; text = []; reasoning = []; finish = None
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            line = line.decode().strip()
            if not line.startswith("data: ") or line == "data: [DONE]":
                continue
            ev = json.loads(line[6:])
            if ev.get("usage"):
                usage = ev["usage"]
            for ch in ev.get("choices", []):
                d = ch.get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning_content") or d.get("reasoning") or "")
                if piece:
                    now = time.time(); t_first = t_first or now; t_last = now
                    (reasoning if (d.get("reasoning_content") or d.get("reasoning")) else text).append(piece)
                finish = ch.get("finish_reason") or finish
    pt, ct = usage["prompt_tokens"], usage["completion_tokens"]
    ttft = t_first - t0
    dec = (ct - 1) / (t_last - t_first) if t_last and t_last > t_first else None
    return {"prompt_tokens": pt, "completion_tokens": ct, "ttft_s": round(ttft, 2),
            "prefill_tps": round(pt / ttft, 1), "decode_tps": round(dec, 2) if dec else None,
            "total_s": round(time.time() - t0, 1), "finish": finish,
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "text": "".join(text), "reasoning": "".join(reasoning)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", type=int, nargs="+", default=[0])
    ap.add_argument("--workload", nargs="+", default=["legal"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=1500)
    ap.add_argument("--thinking", action="store_true")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--out", default="logs/bench.jsonl")
    ap.add_argument("--keep-text", action="store_true")
    ap.add_argument("--filler", choices=["synthetic", "real"], default="synthetic")
    a = ap.parse_args()
    for ctx in a.ctx:
        for wl in a.workload:
            rows = []
            for rep in range(a.reps):
                nonce = f"Session {uuid.uuid4().hex[:12]}.\n"
                if a.filler == "real" and ctx:
                    rec = build_real(ctx, seed=1000 * ctx + rep)
                    prompt = nonce + f"MATERIALS (public court documents):\n{rec}\n\nEND OF MATERIALS.\n\n" + WORKLOADS[wl]
                else:
                    rec = build_record(ctx, seed=1000 * ctx + rep) if ctx else ""
                    prompt = nonce + (f"CASE RECORD (fictional):\n{rec}\n\nEND OF RECORD.\n\n" if rec else "") + WORKLOADS[wl]
                m0 = metrics(); s = Sampler(); s.start()
                try:
                    r = run_one(prompt, a.max_tokens, a.thinking, a.temperature, a.top_p)
                    err = None
                except Exception as e:
                    r, err = {}, repr(e)
                s.stop.set(); s.join(); m1 = metrics()
                acc = None
                if m0.get("spec_decode_num_accepted_tokens_total") is not None and m1.get("spec_decode_num_drafts_total"):
                    da = m1["spec_decode_num_accepted_tokens_total"] - m0["spec_decode_num_accepted_tokens_total"]
                    dd = m1["spec_decode_num_draft_tokens_total"] - m0["spec_decode_num_draft_tokens_total"]
                    dn = m1["spec_decode_num_drafts_total"] - m0["spec_decode_num_drafts_total"]
                    acc = {"accept_rate": round(da / dd, 3) if dd else None,
                           "tokens_per_round": round(1 + da / dn, 2) if dn else None}
                row = {"ts": time.strftime("%FT%T"), "target_ctx": ctx, "workload": wl, "rep": rep, "filler": a.filler, "filler": a.filler,
                       "thinking": a.thinking, "temperature": a.temperature, "top_p": a.top_p, "error": err,
                       **{k: v for k, v in r.items() if k not in ("text", "reasoning")}, "spec": acc,
                       "min_avail_gib": {n: round(v, 2) for n, v in s.minavail.items()},
                       "max_temp_c": s.maxtemp,
                       "gpu_clock_med": {n: (statistics.median(v) if v else None) for n, v in s.clk.items()},
                       "gpu_util_med": {n: (statistics.median(v) if v else None) for n, v in s.util.items()},
                       "tail": r.get("text", "")[-300:]}
                if a.keep_text:
                    row["text"], row["reasoning"] = r.get("text"), r.get("reasoning")
                rows.append(row)
                with open(a.out, "a") as f:
                    f.write(json.dumps(row) + "\n")
                print(json.dumps({k: row[k] for k in ("target_ctx", "workload", "rep", "prompt_tokens", "completion_tokens",
                                                       "ttft_s", "prefill_tps", "decode_tps", "finish", "cached_tokens",
                                                       "spec", "min_avail_gib", "max_temp_c", "error") if k in row}), flush=True)
            d = [x["decode_tps"] for x in rows if x.get("decode_tps")]
            if d:
                print(f"## ctx {ctx} {wl}: decode median {statistics.median(d):.1f} tok/s, range {min(d):.1f}-{max(d):.1f}, n={len(d)}", flush=True)


if __name__ == "__main__":
    main()
