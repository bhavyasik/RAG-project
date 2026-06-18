"""RAG service — generate answers using retrieved context and an LLM."""

import ast
import logging
import os
from typing import Iterator

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq

from .config import LLM_MODEL

load_dotenv()

logger = logging.getLogger(__name__)


class RAGService:
    """Send context + question to Groq LLM and return the answer."""

    def __init__(self) -> None:
        groq_api_key = os.getenv("GROQ_API_KEY")
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY is not set. "
                "Add it to your .env file or export it."
            )

        self.llm = ChatGroq(
            model_name=LLM_MODEL,
            api_key=groq_api_key,
            temperature=0,
        )

        self.prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=(
                "You are a knowledgeable assistant. Answer the question "
                "thoroughly and accurately using ONLY the context below.\n\n"
                "Rules:\n"
                "- Provide a detailed, well-structured answer.\n"
                "- Cite specific facts, figures, and dates from the context.\n"
                "- If the context does not contain enough information, "
                "clearly state what is known and what is missing.\n"
                "- Do NOT invent or assume information beyond the context.\n\n"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )

        # History-aware prompt used for streaming (web UI)
        self.chat_prompt = PromptTemplate(
            input_variables=["history", "context", "question"],
            template=(
                "You are a knowledgeable assistant engaged in a conversation. "
                "Answer using ONLY the context provided below.\n\n"
                "Rules:\n"
                "- Provide a detailed, well-structured answer.\n"
                "- Cite specific facts, figures, and dates from the context.\n"
                "- If the context does not contain enough information, "
                "clearly state what is known and what is missing.\n"
                "- Use the Chat History only to resolve pronouns or references "
                "(e.g. 'they', 'that company') — do NOT answer from history alone.\n"
                "- Do NOT invent or assume information beyond the context.\n\n"
                "{history_section}"
                "Context:\n{context}\n\n"
                "Question: {question}\n\n"
                "Answer:"
            ),
        )

    def generate_answer(self, context: str, question: str) -> str:
        """Format the prompt and invoke the LLM.

        Raises
        ------
        RuntimeError
            If the LLM API call fails.
        """
        formatted = self.prompt.format(context=context, question=question)
        logger.debug("Prompt length: %d chars", len(formatted))
        try:
            response = self.llm.invoke(formatted)
            if not response or not response.content:
                logger.warning("LLM returned empty response")
                return "Unable to generate answer. Please try again."
            return response.content
        except Exception as e:
            logger.error("LLM API error: %s", e)
            raise RuntimeError(f"Failed to generate answer: {e}") from e

    def generate_answer_stream(
        self, context: str, question: str, history: str = ""
    ) -> Iterator[str]:
        """Stream the LLM answer token-by-token.

        Yields
        ------
        str
            Each text chunk from the LLM as it is generated.

        Raises
        ------
        RuntimeError
            If the LLM API call fails.
        """
        history_section = (
            f"Chat History:\n{history}\n\n" if history.strip() else ""
        )
        formatted = self.chat_prompt.format(
            history_section=history_section,
            context=context,
            question=question,
        )
        logger.debug("Stream prompt length: %d chars", len(formatted))
        try:
            for chunk in self.llm.stream(formatted):
                if chunk.content:
                    yield chunk.content
        except Exception as e:
            logger.error("LLM streaming error: %s", e)
            raise RuntimeError(f"Failed to stream answer: {e}") from e

    def decompose_query(self, question: str) -> list[str]:
        """Decompose a multi-entity question into sub-queries.

        Returns a single-item list for simple questions.

        Raises
        ------
        RuntimeError
            If the LLM API call fails.
        """
        try:
            response = self.llm.invoke(
                f"""
                You are a strict query decomposition system.

                Your job is to decide whether the question must be split into
                multiple independent sub-questions.

                ONLY decompose when:
                1. The question asks for separate information about multiple
                   entities independently.
                2. The question clearly requires parallel answers (e.g.,
                   revenue of A and B, founders of A and B).
                3. The question explicitly asks to compare two entities.

                DO NOT decompose when:
                - The question is about a relationship between entities.
                - The question describes a single event involving multiple entities.
                - The question is a single factual query.
                - The question requires joint reasoning across entities.
                - The question is analytical or explanatory.
                - The question can be answered as one coherent response.

                IMPORTANT:
                - If decomposition is NOT required, return the original question
                  inside a single-item Python list.
                - Return ONLY a valid Python list of strings.
                - Do NOT explain.
                - Do NOT add extra text.
                - Do NOT use markdown.

                Examples:

                Single entity:
                "What is Google's revenue?"
                -> ['What is Google\\'s revenue?']

                Relationship (NO decomposition):
                "How is SpaceX related to Tesla?"
                -> ['How is SpaceX related to Tesla?']

                Event (NO decomposition):
                "Was a Tesla car launched into space?"
                -> ['Was a Tesla car launched into space?']

                Independent multi-entity (YES):
                "Google and Microsoft revenue this year"
                -> ['What is Google\\'s revenue this year?', 'What is Microsoft\\'s revenue this year?']

                Comparison (YES):
                "Compare Tesla and SpaceX"
                -> ['What are key facts about Tesla?', 'What are key facts about SpaceX?']

                Question:
                {question}
                """
            )

            raw = response.content.strip()
            logger.debug("Decomposition raw output: %s", raw)

            try:
                result = ast.literal_eval(raw)
                if isinstance(result, list) and all(
                    isinstance(s, str) for s in result
                ):
                    return result
            except Exception:
                logger.warning(
                    "Failed to parse decomposition output — using original query."
                )

            return [question]
        except Exception as e:
            logger.error("Query decomposition error: %s", e)
            return [question]