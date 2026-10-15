"""Support agent: routes each message to answer (RAG), tool call (order lookup) or human hand-off."""
import re
from rag import Index, compose

ORDERS = {"NW-10234": "shipped on Oct 9 via express, arriving Oct 11", "NW-10577": "packed, ships within 2 business days"}
HUMAN = re.compile(r"\b(human|agent|person|representative|manager|complain|lawyer|cancel my account)\b", re.I)
SENSITIVE = re.compile(r"\b(password|backup code|credit card number|card number)\b.*\b(is|here|send)\b", re.I)
REFUND_AMT = re.compile(r"refund.*\$\s?(\d[\d,]*)", re.I)
MIN_SCORE = 3.0
MIN_COVERAGE = 0.5   # share of question terms the knowledge base actually knows

class Agent:
    def __init__(self, kb="."):
        self.index = Index(kb); self.tickets = []

    def escalate(self, msg, reason):
        tid = f"T-{1000 + len(self.tickets) + 1}"; self.tickets.append({"id": tid, "message": msg, "reason": reason})
        return {"route": "handoff", "reply": f"I've passed this to our support team (ticket {tid}). Reason: {reason}. We'll reply by email.", "sources": []}

    def handle(self, msg):
        if SENSITIVE.search(msg):
            return self.escalate(msg, "possible sensitive data shared")
        if HUMAN.search(msg):
            return self.escalate(msg, "customer asked for a person")
        m = REFUND_AMT.search(msg)
        if m and int(m.group(1).replace(",", "")) > 500:
            return self.escalate(msg, "refund above $500 needs a billing specialist")
        o = re.search(r"\bNW-\d{5}\b", msg, re.I)
        if o:
            oid = o.group(0).upper(); st = ORDERS.get(oid)
            if st: return {"route": "tool", "reply": f"Order {oid}: {st}.", "sources": ["tool:order_lookup"]}
            return self.escalate(msg, f"order {oid} not found")
        hits = self.index.search(msg, k=3)
        if not hits or hits[0]["score"] < MIN_SCORE or hits[0]["coverage"] < MIN_COVERAGE:
            return self.escalate(msg, "no confident answer in the knowledge base")
        return {"route": "answer", "reply": compose(msg, hits), "sources": [h["id"] for h in hits], "top_doc": hits[0]["doc"], "score": hits[0]["score"]}

if __name__ == "__main__":
    a = Agent()
    for q in ["How much is the Pro plan?", "Where is my order NW-10234?", "I want a refund of $800", "Do you support SAML single sign-on?"]:
        print(q, "->", a.handle(q))
