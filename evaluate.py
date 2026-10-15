"""Run the agent over a labelled question set and score routing + retrieval + citations."""
import json, time
from agent import Agent

def main(path="eval_set.json", out="eval_results.json"):
    a = Agent(); cases = json.load(open(path)); rows = []
    t0 = time.perf_counter()
    for c in cases:
        r = a.handle(c["q"]); ok_route = r["route"] == c["route"]
        hit1 = hit3 = cited = None
        if c["route"] == "answer":
            hit1 = r.get("sources", [None])[0] == c["chunk"] if r["route"] == "answer" else False
            hit3 = c["chunk"] in r.get("sources", []) if r["route"] == "answer" else False
            cited = f"[{c['chunk']}]" in r["reply"]
        rows.append({**c, "got_route": r["route"], "route_ok": ok_route, "hit@1": hit1, "hit@3": hit3, "cited": cited, "reply": r["reply"]})
    ms = (time.perf_counter() - t0) / len(cases) * 1000
    ans = [x for x in rows if x["route"] == "answer"]
    m = {"questions": len(rows), "routing_accuracy": sum(x["route_ok"] for x in rows) / len(rows),
         "retrieval_hit@1": sum(bool(x["hit@1"]) for x in ans) / len(ans), "retrieval_hit@3": sum(bool(x["hit@3"]) for x in ans) / len(ans),
         "correct_citation": sum(bool(x["cited"]) for x in ans) / len(ans),
         "handoff_recall": sum(x["got_route"] == "handoff" for x in rows if x["route"] == "handoff") / sum(x["route"] == "handoff" for x in rows),
         "avg_ms_per_question": round(ms, 2), "tickets": len(a.tickets)}
    json.dump({"metrics": m, "rows": rows}, open(out, "w"), indent=2)
    print(json.dumps(m, indent=2))
    for x in rows:
        if not x["route_ok"] or x["hit@1"] is False: print("MISS:", x["q"], "->", x["got_route"], x["reply"][:90])

if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1: main(sys.argv[1], sys.argv[2])
    else: main()
