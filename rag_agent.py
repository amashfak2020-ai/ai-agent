"""Minimal RAG agent that tracks the total number of tokens used.

Retrieval is TF-IDF based (no external services). Tokens are counted with
tiktoken when installed, otherwise with a regex-based approximation.
Pass a custom `llm` callable (prompt -> str) to use a real model; the default
is an offline extractive generator.
"""
import math
import re
import sys
from collections import Counter

try:
    import tiktoken
    _enc = tiktoken.get_encoding("cl100k_base")

    def count_tokens(text):
        return len(_enc.encode(text))
except Exception:
    def count_tokens(text):
        return len(re.findall(r"\w+|[^\w\s]", text))


def _words(text):
    return re.findall(r"\w+", text.lower())


class RAGAgent:
    def __init__(self, documents, llm=None, top_k=2):
        self.docs = list(documents)
        self.top_k = top_k
        self.llm = llm or self._extractive_llm
        self.usage = {"prompt_tokens": 0, "completion_tokens": 0}
        n = len(self.docs)
        df = Counter(w for d in self.docs for w in set(_words(d)))
        self.idf = {w: math.log((1 + n) / (1 + c)) + 1 for w, c in df.items()}

    @property
    def total_tokens(self):
        return self.usage["prompt_tokens"] + self.usage["completion_tokens"]

    def retrieve(self, query):
        q = Counter(_words(query))
        scored = []
        for d in self.docs:
            tf = Counter(_words(d))
            scored.append((sum(q[w] * tf[w] * self.idf.get(w, 0) ** 2 for w in q), d))
        scored.sort(key=lambda x: -x[0])
        return [d for s, d in scored[: self.top_k] if s > 0]

    @staticmethod
    def _extractive_llm(prompt):
        ctx = prompt.split("Context:\n", 1)[1].split("\n\nQuestion:", 1)[0]
        return ctx.strip() or "I don't know."

    def ask(self, question):
        context = "\n".join(self.retrieve(question))
        prompt = f"Answer using the context.\nContext:\n{context}\n\nQuestion: {question}"
        answer = self.llm(prompt)
        self.usage["prompt_tokens"] += count_tokens(prompt)
        self.usage["completion_tokens"] += count_tokens(answer)
        return answer


if __name__ == "__main__":
    docs = [
        "Python is a programming language created by Guido van Rossum.",
        "RAG combines document retrieval with text generation.",
        "Tokens are the units of text that language models process.",
    ]
    agent = RAGAgent(docs)
    for q in sys.argv[1:] or ["What is RAG?", "Who created Python?"]:
        print(f"Q: {q}\nA: {agent.ask(q)}\n")
    print(f"Prompt tokens: {agent.usage['prompt_tokens']}")
    print(f"Completion tokens: {agent.usage['completion_tokens']}")
    print(f"Total tokens used: {agent.total_tokens}")
