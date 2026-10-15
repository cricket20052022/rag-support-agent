"""Minimal RAG: chunking + BM25 retrieval + answer composition with citations.
Standard library only. Plug in a real LLM by setting OPENAI_API_KEY (optional)."""
import math, os, re, json, urllib.request
from collections import Counter
from pathlib import Path

STOP = set("a an the of to in on for and or is are be can by with at from as it this that your you we our any how what when do does i my me".split())

SYN = {"much": ["$", "month", "price"], "cost": ["$", "month", "price"], "price": ["$", "month"], "pricing": ["$", "month", "plans"],
       "money": ["refund"], "refunded": ["refund"], "2fa": ["two", "factor", "authentication"], "mfa": ["two", "factor", "authentication"],
       "limit": ["rate", "requests"], "down": ["uptime", "status"], "outage": ["uptime", "status"], "delete": ["deleted", "closed"],
       "teammates": ["team", "members", "invite"], "users": ["members"], "ship": ["shipping", "ships"], "delivery": ["shipping"],
       "stored": ["data", "centers"], "where": [], "trial": ["free", "trial"], "pay": ["accept", "payment"], "yearly": ["annual"], "annually": ["annual"], "minimum": ["least"], "length": ["long", "character"], "log": ["access"], "login": ["access"], "back": ["refund"], "method": ["accept", "visa"], "teammate": ["team", "member"], "turn": ["enable"], "arrive": ["returned"], "phone": ["authenticator"], "damag": ["damaged"], "down": ["status", "uptime"]}

def expand(q):
    out = []
    for t in tokens(q): out += [t] + [stem(x) for x in SYN.get(t, [])]
    return out

def stem(t):
    for suf in ("ing", "ed", "es", "s"):
        if len(t) > 4 and t.endswith(suf): return t[: -len(suf)]
    return {"lost": "lose", "lose": "lose"}.get(t, t)

def tokens(text):
    return [stem(t.strip(".")) for t in re.findall(r"[a-z0-9$%.]+", text.lower().replace("-", " ")) if t not in STOP and t.strip(".")]

class Index:
    def __init__(self, kb_dir="."):
        self.chunks = []
        for p in sorted(Path(kb_dir).glob("kb_*.md")):
            lines = p.read_text().splitlines()
            title = lines[0].lstrip("# ").strip()
            for i, line in enumerate(l for l in lines[1:] if l.strip()):
                self.chunks.append({"id": f"{p.stem[3:]}#{i+1}", "doc": p.stem[3:], "title": title, "text": line.strip(), "toks": tokens(title + " " + line)})
        self.N = len(self.chunks); self.avgdl = sum(len(c["toks"]) for c in self.chunks) / self.N
        df = Counter(t for c in self.chunks for t in set(c["toks"])); self.vocab = set(df)
        self.idf = {t: math.log(1 + (self.N - n + 0.5) / (n + 0.5)) for t, n in df.items()}

    def search(self, query, k=3, k1=1.4, b=0.75):
        q = expand(query); scored = []
        for c in self.chunks:
            tf = Counter(c["toks"]); dl = len(c["toks"]); s = 0.0
            for t in q:
                if t in tf:
                    s += self.idf[t] * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * dl / self.avgdl))
            if s > 0: scored.append((s, c))
        scored.sort(key=lambda x: -x[0])
        orig = [t for t in tokens(query) if not t.startswith("$")]
        out = []
        for s, c in scored[:k]:
            known = lambda t: t in self.vocab or any(stem(x) in self.vocab for x in SYN.get(t, []))
            cov = sum(1 for t in orig if known(t)) / max(1, len(orig))
            out.append({"score": round(s, 2), "coverage": round(cov, 2), **{k2: v for k2, v in c.items() if k2 != "toks"}})
        return out

def llm(prompt):
    """Optional: OpenAI-compatible chat call. Returns None if no key is configured."""
    key = os.getenv("OPENAI_API_KEY")
    if not key: return None
    body = json.dumps({"model": os.getenv("LLM_MODEL", "gpt-4o-mini"), "messages": [{"role": "user", "content": prompt}], "temperature": 0}).encode()
    req = urllib.request.Request(os.getenv("LLM_BASE_URL", "https://api.openai.com/v1") + "/chat/completions", data=body,
                                 headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"})
    return json.loads(urllib.request.urlopen(req, timeout=30).read())["choices"][0]["message"]["content"]

def compose(question, hits):
    """Grounded answer: LLM if configured, otherwise extractive (top passages)."""
    ctx = "\n".join(f"[{h['id']}] {h['text']}" for h in hits)
    out = llm(f"Answer ONLY from the context. Cite sources like [doc#n]. If not answerable, say so.\n\nContext:\n{ctx}\n\nQuestion: {question}")
    if out: return out
    best = hits[0]; extra = [h for h in hits[1:2] if h["doc"] == best["doc"] and h["score"] > best["score"] * 0.9]
    return " ".join(f"{h['text']} [{h['id']}]" for h in [best] + extra)
