"""Render the 3 case-study images from the eval results and demo transcript."""
import json, textwrap, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
NAVY, BLUE, LIGHT, GREY, BG, AMBER, GREEN = "#1F3A5F", "#3B82C4", "#9CC3E6", "#6B7280", "#F7F8FA", "#B7791F", "#2E7D5B"
plt.rcParams.update({"font.family": "DejaVu Sans"})
W, H = 16, 12
v1 = json.load(open("eval_results_v1.json"))["metrics"]; v2 = json.load(open("eval_results_v2.json"))["metrics"]
ho = json.load(open("eval_results_holdout.json")); hm = ho["metrics"]
tr = json.load(open("demo_transcript.json"))

def box(ax, x, y, w, h, title, sub, ec=BLUE, tc=NAVY):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.25", fc="white", ec=ec, lw=2))
    ax.text(x + w/2, y + h*0.66, title, ha="center", va="center", fontsize=18, weight="bold", color=tc)
    ax.text(x + w/2, y + h*0.3, sub, ha="center", va="center", fontsize=13, color=GREY)

def arrow(ax, x1, y1, x2, y2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1), arrowprops=dict(arrowstyle="-|>", color=BLUE, lw=2, mutation_scale=20))

# 1. HOW IT WORKS
fig, ax = plt.subplots(figsize=(W, H), facecolor=BG); fig.subplots_adjust(0.02, 0.02, 0.98, 0.98); ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis("off")
ax.text(0.8, 11, "How it works: a support agent that knows when to stop", fontsize=28, weight="bold", color=NAVY)
ax.text(0.8, 10.35, "Every message is routed: answer from the docs, call a tool, or hand off to a person.", fontsize=16, color=GREY)
box(ax, 0.8, 6.3, 3.2, 2.0, "Customer", "question")
box(ax, 5.2, 6.3, 3.6, 2.0, "Router", "rules + confidence", ec=NAVY)
arrow(ax, 4.0, 7.3, 5.2, 7.3)
box(ax, 10.3, 8.4, 4.9, 1.8, "Answer (RAG)", "BM25 search → cited answer", ec=GREEN)
box(ax, 10.3, 6.1, 4.9, 1.8, "Tool call", "order status lookup", ec=BLUE)
box(ax, 10.3, 3.8, 4.9, 1.8, "Human hand-off", "ticket + reason", ec=AMBER)
arrow(ax, 8.8, 7.6, 10.3, 9.3); arrow(ax, 8.8, 7.3, 10.3, 7.0); arrow(ax, 8.8, 7.0, 10.3, 4.7)
rules = ["Asks for a person / complaint  →  hand-off", "Shares a password or card number  →  hand-off", "Refund above $500  →  billing specialist",
         "Order number (NW-12345)  →  order tool", "Low retrieval confidence  →  hand-off, never guess"]
ax.text(0.8, 4.9, "Routing rules", fontsize=17, weight="bold", color=NAVY)
for i, r in enumerate(rules): ax.text(0.8, 4.25 - i*0.6, "•  " + r, fontsize=14, color=NAVY)
ax.text(0.8, 0.5, "Python · RAG (BM25) · citations · optional OpenAI-compatible LLM · evaluation harness", fontsize=15, color=GREY)
fig.savefig("01_how_it_works.png", dpi=100); plt.close(fig)

# 2. EVALUATION
fig = plt.figure(figsize=(W, H), facecolor=BG)
fig.text(0.05, 0.94, "Measured, not guessed: evaluation results", fontsize=29, weight="bold", color=NAVY)
fig.text(0.05, 0.905, "35 labelled questions used for tuning, plus 15 held-out questions never used during tuning.", fontsize=16, color=GREY)
ax = fig.add_axes([0.08, 0.40, 0.86, 0.43]); ax.set_facecolor("white")
keys = [("routing_accuracy", "Correct route"), ("retrieval_hit@1", "Right source\nranked #1"), ("retrieval_hit@3", "Right source\nin top 3"), ("correct_citation", "Correct\ncitation"), ("handoff_recall", "Hand-offs\ncaught")]
x = range(len(keys)); w = 0.26
for j, (m, lab, col) in enumerate([(v1, "Baseline", LIGHT), (v2, "Tuned", NAVY), (hm, "Held-out (unseen)", BLUE)]):
    vals = [m[k] * 100 for k, _ in keys]
    bars = ax.bar([i + (j - 1) * w for i in x], vals, w, color=col, label=lab)
    for b, v in zip(bars, vals): ax.text(b.get_x() + b.get_width()/2, v + 1.2, f"{v:.0f}", ha="center", fontsize=11, color=NAVY)
ax.set_xticks(list(x)); ax.set_xticklabels([l for _, l in keys], fontsize=13); ax.set_ylim(0, 112); ax.set_ylabel("%", fontsize=13)
ax.spines[["top", "right"]].set_visible(False); ax.legend(frameon=False, fontsize=13, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08))
wrong_ho = sum(1 for r in ho["rows"] if r["got_route"] == "answer" and (r["route"] != "answer" or not r["hit@1"]))
cards = [(f"{wrong_ho}", "wrong answers on held-out set\n(misses were safe hand-offs)"), (f"{hm['avg_ms_per_question']:.1f} ms", "retrieval + routing time\nper question"), ("0", "paid API calls needed\nto run the demo")]
for i, (v, l) in enumerate(cards):
    a = fig.add_axes([0.05 + i*0.31, 0.08, 0.28, 0.2]); a.axis("off")
    a.add_patch(FancyBboxPatch((0, 0), 1, 1, boxstyle="round,pad=0,rounding_size=0.06", fc="white", ec="#D1D5DB", transform=a.transAxes))
    a.text(0.5, 0.66, v, ha="center", fontsize=30, weight="bold", color=NAVY); a.text(0.5, 0.25, l, ha="center", va="center", fontsize=13, color=GREY)
fig.savefig("02_evaluation.png", dpi=100); plt.close(fig)

# 3. CONVERSATION
fig, ax = plt.subplots(figsize=(W, H), facecolor=BG); fig.subplots_adjust(0.02, 0.02, 0.98, 0.98); ax.set_xlim(0, 16); ax.set_ylim(0, 12); ax.axis("off")
ax.text(0.8, 11.2, "Sample conversation", fontsize=29, weight="bold", color=NAVY)
ax.text(0.8, 10.6, "Grounded answers with sources, a live tool call, and clean hand-offs.", fontsize=16, color=GREY)
COL = {"answer": GREEN, "tool": BLUE, "handoff": AMBER}; LAB = {"answer": "ANSWER", "tool": "TOOL", "handoff": "HAND-OFF"}
y = 9.9
for t in [x for x in tr if not x["user"].startswith("Is there a discount")][:5]:
    ax.add_patch(FancyBboxPatch((7.2, y - 0.5), 8.0, 0.55, boxstyle="round,pad=0.05,rounding_size=0.2", fc=LIGHT, ec="none"))
    ax.text(15.0, y - 0.22, t["user"].replace("$", r"\$"), ha="right", va="center", fontsize=13, color=NAVY)
    reply = textwrap.fill(t["reply"], 82).replace("$", r"\$"); n = reply.count("\n") + 1; hgt = 0.34 * n + 0.22
    ax.add_patch(FancyBboxPatch((0.8, y - 0.7 - hgt), 10.6, hgt, boxstyle="round,pad=0.05,rounding_size=0.2", fc="white", ec=COL[t["route"]], lw=1.6))
    ax.text(1.0, y - 0.7 - 0.12, LAB[t["route"]], fontsize=10, weight="bold", color=COL[t["route"]], va="top")
    ax.text(2.4, y - 0.7 - 0.1, reply, fontsize=12.5, color="#1F2937", va="top", linespacing=1.3)
    y -= 0.85 + hgt + 0.2
fig.savefig("03_conversation.png", dpi=100); plt.close(fig)
print("media done")
