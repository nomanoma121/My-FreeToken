import sys, json, re, time, urllib.request
BASE = "http://127.0.0.1:1919/v1"

def chat(model, messages, max_tokens=1024, effort="low"):
    payload = {"model": model, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0.0, "reasoning_effort": effort}
    req = urllib.request.Request(BASE + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    for attempt in range(3):
        try:
            d = json.load(urllib.request.urlopen(req, timeout=600))
            m = d["choices"][0]["message"]
            if not (m.get("content") or "").strip():
                open("/tmp/mmlu_debug.txt", "a").write(
                    "finish=%s clen=%d rlen=%d\n" % (
                        d["choices"][0].get("finish_reason"),
                        len(m.get("content") or ""),
                        len(m.get("reasoning_content") or "")))
            return m.get("content") or ""
        except Exception as e:
            if attempt == 2:
                return "[ERROR %s]" % e
            time.sleep(10)

def ans_idx(a):
    return a if isinstance(a, int) else ord(a) - 65

def last_number(t):
    nums = re.findall(r"-?[\d,]+\.?\d*", t.replace("$", ""))
    return nums[-1].replace(",", "") if nums else None

def gsm_target(a):
    return a.split("####")[-1].strip().replace(",", "")

def run_gsm(model, n, out):
    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    shots = list(ds.select(range(5)))
    pre = ""
    for s in shots:
        pre += "Question: %s\nAnswer: %s\n\n" % (s["question"], s["answer"])
    ok = tot = 0
    res = []
    for i, row in enumerate(ds.select(range(5, 5 + n))):
        t0 = time.time()
        r = chat(model, [{"role": "user",
                          "content": pre + "Question: %s\nAnswer:" % row["question"]}])
        pred = last_number(r)
        tgt = gsm_target(row["answer"])
        hit = (pred == tgt)
        ok += hit
        tot += 1
        res.append({"i": i, "pred": pred, "tgt": tgt, "hit": bool(hit),
                    "dt": round(time.time() - t0, 1)})
        print("[%d/%d] pred=%s tgt=%s %s %.1fs" % (i + 1, n, pred, tgt, "OK" if hit else "NG", res[-1]["dt"]), flush=True)
    json.dump({"acc": ok / tot, "ok": ok, "tot": tot, "rows": res}, open(out, "w"), indent=1)
    print("GSM8K acc=%d/%d=%.3f" % (ok, tot, ok / tot))

def run_mmlu(model, subjects, n_each, out):
    from datasets import load_dataset
    ok = tot = 0
    res = []
    for subj in subjects:
        ds = load_dataset("cais/mmlu", subj, split="test")
        dev = load_dataset("cais/mmlu", subj, split="dev")
        pre = ""
        for s in dev.select(range(min(5, len(dev)))):
            pre += "Q: %s\n" % s["question"]
            for k, c in enumerate("ABCD"):
                pre += "%s. %s\n" % (c, s["choices"][k])
            pre += "A: %s\n\n" % s["choices"][ans_idx(s["answer"])]
        for i, row in enumerate(ds.select(range(min(n_each, len(ds))))):
            q = "Q: %s\n" % row["question"]
            for k, c in enumerate("ABCD"):
                q += "%s. %s\n" % (c, row["choices"][k])
            q += "A:"
            t0 = time.time()
            r = chat(model, [{"role": "user", "content": pre + q + " Reply with only the letter."}], max_tokens=1024)
            if r.startswith("[ERROR") or not r.strip():
                open("/tmp/mmlu_debug.txt", "a").write("Q%d raw=%r\n" % (i, r[:300]))
            m = re.search(r"\b([A-D])\b", r)
            pred = m.group(1) if m else None
            tgt = chr(ans_idx(row["answer"]) + 65)
            hit = (pred == tgt)
            ok += hit
            tot += 1
            res.append({"subj": subj, "i": i, "pred": pred, "tgt": tgt,
                        "hit": bool(hit), "dt": round(time.time() - t0, 1)})
            print("[%s %d] pred=%s tgt=%s %s" % (subj, i + 1, pred, tgt, "OK" if hit else "NG"), flush=True)
    json.dump({"acc": ok / tot, "ok": ok, "tot": tot, "rows": res}, open(out, "w"), indent=1)
    print("MMLU acc=%d/%d=%.3f" % (ok, tot, ok / tot))

if __name__ == "__main__":
    which = sys.argv[1]
    if which == "gsm":
        run_gsm(sys.argv[2], int(sys.argv[3]), sys.argv[4])
    else:
        run_mmlu(sys.argv[2], sys.argv[3].split(","), int(sys.argv[4]), sys.argv[5])
