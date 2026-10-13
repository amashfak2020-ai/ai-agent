# ai-agent
## RAG agent with token counting
`python rag_agent.py "your question"` retrieves relevant documents, answers, and reports total tokens used
(prompt + completion). Uses `tiktoken` if installed, else an approximate count. Plug in a real LLM via `RAGAgent(docs, llm=fn)`.
