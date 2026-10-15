"""Short conversation showing the three routes: grounded answer, tool call, human hand-off."""
import json
from agent import Agent

DEMO = ["How much is the Pro plan?", "Is there a discount if I pay yearly?", "Where is my order NW-10234?",
        "I lost my authenticator phone, how do I log in?", "Please refund $1,200 for my annual plan", "Do you support SAML single sign-on?"]

if __name__ == "__main__":
    a = Agent(); log = []
    for q in DEMO:
        r = a.handle(q); log.append({"user": q, **r}); print(f"\nUSER : {q}\nAGENT [{r['route']}]: {r['reply']}")
    json.dump(log, open("demo_transcript.json", "w"), indent=2)
