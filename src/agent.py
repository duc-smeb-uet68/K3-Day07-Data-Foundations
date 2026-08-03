from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str, top_k: int = 3) -> str:
        """Retrieve context and delegate the final answer to the injected LLM."""
        results = self.store.search(question, top_k=top_k)
        if results:
            context = "\n\n".join(
                f"[Chunk {result['id']}]\n{result['content']}"
                for result in results
            )
        else:
            context = "(No relevant context was retrieved.)"

        prompt = (
            "Answer the question using only the retrieved context below. "
            "If the context does not contain enough information, say that clearly "
            "instead of guessing.\n\n"
            f"Question: {question}\n\n"
            f"Retrieved context:\n{context}"
        )
        return self.llm_fn(prompt)
