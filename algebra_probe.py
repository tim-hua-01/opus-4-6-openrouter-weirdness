"""
Follow-up probe: is claude-opus-4.6 degraded in general under reasoning effort=low,
or only on the random-number prompt?

Same 4 providers x 2 reasoning conditions, but the task is 8 abstract-algebra
questions with unambiguous integer answers. Measures accuracy + how often the
completion is something other than a bare number (the junk-token artifact).
"""

import asyncio
import json
import os
import random
import re
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

load_dotenv("/Users/timhua/Documents/aisafety_githubs/ais_ablationsbench/.env")

OUT = Path(__file__).parent
RESULTS = OUT / "algebra_results.jsonl"

MODEL = "anthropic/claude-opus-4.6"
PROVIDERS = ["anthropic", "amazon-bedrock", "google-vertex", "azure/us-east-2"]
REASONING_CONDITIONS = [None, "low"]
N_PER_CELL = 10  # per question

QUESTIONS = [
    ("groups_order_12", "How many groups of order 12 are there up to isomorphism?", 5),
    ("gl3_f2", "What is the order of the group GL(3, F_2)?", 168),
    ("conj_classes_s7", "How many conjugacy classes does the symmetric group S_7 have?", 15),
    ("abelian_720", "How many abelian groups of order 720 are there up to isomorphism?", 10),
    ("galois_x4_minus_2", "What is the order of the Galois group of x^4 - 2 over Q?", 8),
    ("irreps_a5", "How many irreducible complex representations does A_5 have?", 5),
    ("aut_q8", "What is the order of the automorphism group of the quaternion group Q_8?", 24),
    ("irred_deg6_f2", "How many monic irreducible polynomials of degree 6 are there over F_2?", 9),
]

SUFFIX = " Provide only the number."
MAX_TOKENS = 8192
MAX_RETRIES = 5

client = AsyncOpenAI(
    api_key=os.environ["OPENROUTER_API_KEY"],
    base_url="https://openrouter.ai/api/v1",
    timeout=300.0,
    max_retries=0,
)


def extract_number(text):
    if not text:
        return None
    nums = re.findall(r"[-+]?\d+", text.replace(",", ""))
    return int(nums[-1]) if nums else None


async def run_one(provider, reasoning, qid, question, answer, idx):
    body = {"provider": {"order": [provider], "allow_fallbacks": False}, "usage": {"include": True}}
    if reasoning is not None:
        body["reasoning"] = {"effort": reasoning}

    last_err = None
    for attempt in range(MAX_RETRIES):
        try:
            r = await client.chat.completions.create(
                model=MODEL,
                max_tokens=MAX_TOKENS,
                messages=[{"role": "user", "content": question + SUFFIX}],
                extra_body=body,
            )
            if not r.choices:
                raise RuntimeError(f"no choices: {getattr(r, 'error', None) or r}")
            msg = r.choices[0].message
            content = msg.content or ""
            got = extract_number(content)
            u = r.usage
            return {
                "provider": provider, "reasoning": reasoning or "none", "qid": qid,
                "sample_idx": idx, "content": content,
                "thinking": getattr(msg, "reasoning", None),
                "extracted": got, "correct": got == answer, "expected": answer,
                "bare_number": bool(re.fullmatch(r"[-+]?\d+", content.strip())),
                "finish_reason": r.choices[0].finish_reason,
                "provider_used": getattr(r, "provider", None),
                "output_tokens": u.completion_tokens if u else None,
                "cost": getattr(u, "cost", None) if u else None,
                "error": None, "timestamp": time.time(),
            }
        except Exception as e:  # noqa: BLE001
            last_err = e
            if attempt < MAX_RETRIES - 1:
                await asyncio.sleep(min(30, 2**attempt) * (0.5 + random.random()))

    return {
        "provider": provider, "reasoning": reasoning or "none", "qid": qid,
        "sample_idx": idx, "content": None, "thinking": None, "extracted": None,
        "correct": False, "expected": answer, "bare_number": False,
        "finish_reason": None, "provider_used": None, "output_tokens": None, "cost": None,
        "error": f"{type(last_err).__name__}: {last_err}"[:500], "timestamp": time.time(),
    }


async def main():
    done = set()
    if RESULTS.exists():
        for line in open(RESULTS):
            if line.strip():
                d = json.loads(line)
                if d.get("error") is None:
                    done.add((d["provider"], d["reasoning"], d["qid"], d["sample_idx"]))

    jobs = [
        (p, rz, qid, q, a, i)
        for p in PROVIDERS
        for rz in REASONING_CONDITIONS
        for qid, q, a in QUESTIONS
        for i in range(N_PER_CELL)
        if (p, rz or "none", qid, i) not in done
    ]
    print(f"{len(jobs)} jobs", flush=True)
    random.shuffle(jobs)

    sem = asyncio.Semaphore(40)
    lock = asyncio.Lock()
    fh = open(RESULTS, "a")
    ctr = {"n": 0, "err": 0, "cost": 0.0}
    t0 = time.time()

    async def worker(job):
        async with sem:
            rec = await run_one(*job)
        async with lock:
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            ctr["n"] += 1
            ctr["err"] += bool(rec["error"])
            ctr["cost"] += rec.get("cost") or 0
            if ctr["n"] % 100 == 0:
                print(f"  {ctr['n']}/{len(jobs)} | {ctr['err']} err | ${ctr['cost']:.2f} | {time.time()-t0:.0f}s", flush=True)

    await asyncio.gather(*(worker(j) for j in jobs))
    fh.close()
    print(f"DONE: {ctr['n']} calls, {ctr['err']} errors, ${ctr['cost']:.2f}, {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    asyncio.run(main())
