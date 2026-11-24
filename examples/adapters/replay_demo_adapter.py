"""Deterministic backends for the offline multi-hop RAG demo."""
import json

AUTHOR = "Mara Vale"
OBSERVATORY = "Aster Ridge Observatory"
COUNTRY = "Chile"
BOOK = "The Glass Meridian"
QUESTION = f"In which country is the observatory where the author of {BOOK} first catalogued a comet?"
AUTHOR_QUERY = f"Who wrote {BOOK}?"
PUBLICATION_QUERY = "When was the first edition published?"
OBSERVATORY_QUERY = f"At which observatory did {AUTHOR} first catalogue a comet?"
COUNTRY_QUERY = f"In which country is {OBSERVATORY} located?"

EVIDENCE = {
    AUTHOR_QUERY: ("Author registry", f"{AUTHOR} wrote {BOOK}.", AUTHOR),
    PUBLICATION_QUERY: ("First edition publication date", "The first edition was published in 1987.", "1987"),
    OBSERVATORY_QUERY: ("Mara Vale comet catalogue", f"{AUTHOR} first catalogued a comet at {OBSERVATORY}.", OBSERVATORY),
    COUNTRY_QUERY: ("Aster Ridge Observatory country", f"{OBSERVATORY} is located in {COUNTRY}.", COUNTRY),
}
DOCUMENTS = [
    {"title": BOOK, "sentences": [
        f"{BOOK} is a novel. Its author first catalogued a comet at an observatory.",
        "This entry discusses the story and its reception.",
    ]},
    *[{"title":title,"sentences":[sentence]} for title,sentence,_ in EVIDENCE.values()],
    {"title":"Hilltop Observatory","sentences":["Hilltop Observatory is located in Peru."]},
]


class Generator:
    def generate(self, prompt, *, max_new_tokens, schema=None):
        state = json.loads(prompt.rsplit("\n",1)[1])
        if schema is not None:
            seen = state["previously_proposed_questions"]
            if not seen:
                questions = [AUTHOR_QUERY,PUBLICATION_QUERY]
            else:
                found = {row["answer"] for row in state["state"]["executed_followups"]}
                if OBSERVATORY in found:
                    questions = [COUNTRY_QUERY]
                elif AUTHOR in found:
                    questions = [OBSERVATORY_QUERY]
                else:
                    questions = [AUTHOR_QUERY]
            text = json.dumps({"questions":questions})
        elif "followup_question" in state:
            title,_,answer = EVIDENCE[state["followup_question"]]
            text = answer if any(d["title"]==title for d in state["retrieved_documents"]) else "insufficient evidence"
        else:
            found = {r["answer"] for r in state["executed_followups"]}
            text = next((answer for answer in (COUNTRY,OBSERVATORY,AUTHOR) if answer in found), "insufficient evidence")
        return {"text":text}


class Gate:
    def __init__(self, conservative):
        self.conservative = conservative

    def predict(self, kind, request):
        state = request["state"]
        if kind=="sufficiency":
            eligible = {COUNTRY} if self.conservative else {COUNTRY,OBSERVATORY}
            choice = "sufficient" if state["current_intermediate_answer"] in eligible else "continue"
        else:
            choice = "skip" if not self.conservative and state["proposed_followup"]==PUBLICATION_QUERY else "execute"
        return {"choice":choice,"probabilities":{
            opt["value"]:float(opt["value"]==choice) for opt in request["options"]
        }}


def build_generator(config):
    return Generator()


def build_jev_gate(config):
    return Gate(conservative=False)


def build_qwen_gate(config):
    return Gate(conservative=True)
