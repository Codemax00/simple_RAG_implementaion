import re
from typing import List, Dict, Any, Optional
from langchain_core.messages import SystemMessage, HumanMessage


class ConversationHistoryMemory:
    """Memory Type 1: Tracks multi-turn conversational dialogue per session thread."""

    def __init__(self, max_turns: int = 8):
        self.max_turns = max_turns
        self._sessions: Dict[str, List[Dict[str, str]]] = {}

    def add_turn(self, thread_id: str, role: str, content: str) -> None:
        if thread_id not in self._sessions:
            self._sessions[thread_id] = []
        self._sessions[thread_id].append({"role": role, "content": content.strip()})
        # Keep window within max_turns * 2 (user + assistant pairs)
        if len(self._sessions[thread_id]) > self.max_turns * 2:
            self._sessions[thread_id] = self._sessions[thread_id][-self.max_turns * 2:]

    def get_history(self, thread_id: str, limit: int = 4) -> List[Dict[str, str]]:
        history = self._sessions.get(thread_id, [])
        return history[-limit * 2:] if history else []

    def get_formatted_history(self, thread_id: str, limit: int = 4) -> str:
        turns = self.get_history(thread_id, limit=limit)
        if not turns:
            return ""
        lines = []
        for t in turns:
            speaker = "User" if t["role"] == "user" else "Assistant"
            # Summarize long assistant turns to save context tokens
            content = t["content"]
            if len(content) > 280:
                content = content[:280] + "..."
            lines.append(f"{speaker}: {content}")
        return "\n".join(lines)

    def clear(self, thread_id: str) -> None:
        if thread_id in self._sessions:
            self._sessions[thread_id] = []


class KeywordConceptMemory:
    """Memory Type 2: Extracts, weights, and retains key entities, concepts, and technical terms across turns."""

    # Default knowledge base anchor terms
    KNOWN_ANCHORS = {
        "dopamine", "molecule of more", "reward prediction error", "mesolimbic pathway",
        "48 laws of power", "robert greene", "sun tzu", "33 strategies of war",
        "machiavelli", "law of power", "warfare", "indirect approach", "polarity",
        "blitzkrieg", "divide and conquer", "grand strategy", "neuron", "action potential",
        "synapse", "sebi", "quantitative aptitude"
    }

    def __init__(self):
        self._sessions: Dict[str, Dict[str, int]] = {}

    def extract_and_add(self, thread_id: str, text: str) -> List[str]:
        if thread_id not in self._sessions:
            self._sessions[thread_id] = {}

        mem = self._sessions[thread_id]
        extracted = []

        # 1. Match known domain anchors
        lower_text = text.lower()
        for anchor in self.KNOWN_ANCHORS:
            if anchor in lower_text:
                mem[anchor] = mem.get(anchor, 0) + 2
                if anchor not in extracted:
                    extracted.append(anchor)

        # 2. Extract capitalized title words / multi-word entities (e.g. "Law 33", "Robert Greene")
        capitalized = re.findall(r"\b[A-Z][a-zA-Z0-9_\-]{2,}(?:\s+[A-Z][a-zA-Z0-9_\-]{2,})*\b", text)
        stopwords = {"What", "Explain", "How", "Why", "Tell", "Does", "Could", "Which", "Where", "When", "There", "Their", "About", "Please", "Assistant", "User"}
        for cap in capitalized:
            if cap not in stopwords and len(cap) > 3:
                norm = cap.strip()
                mem[norm] = mem.get(norm, 0) + 1
                if norm not in extracted:
                    extracted.append(norm)

        # 3. Extract technical terms in quotes or parentheses
        quoted = re.findall(r"['\"]([a-zA-Z0-9\s\-]+)['\"]", text)
        for q in quoted:
            if len(q.strip()) > 3:
                norm = q.strip()
                mem[norm] = mem.get(norm, 0) + 1
                if norm not in extracted:
                    extracted.append(norm)

        return extracted

    def get_top_keywords(self, thread_id: str, limit: int = 6) -> List[str]:
        mem = self._sessions.get(thread_id, {})
        if not mem:
            return []
        sorted_keys = sorted(mem.items(), key=lambda x: x[1], reverse=True)
        return [k for k, _ in sorted_keys[:limit]]

    def clear(self, thread_id: str) -> None:
        if thread_id in self._sessions:
            self._sessions[thread_id] = {}


class UserInputOptimizer:
    """Synthesizes raw user prompt + History Memory + Keyword Memory into an optimized retrieval query."""

    def __init__(self):
        self.history_memory = ConversationHistoryMemory(max_turns=6)
        self.keyword_memory = KeywordConceptMemory()

    async def aoptimize_query(
        self,
        user_input: str,
        thread_id: str,
        llm: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Rewrites ambiguous or context-dependent queries into standalone dense retrieval queries."""
        raw_query = user_input.strip()

        # Update keyword memory with terms from current query
        new_keywords = self.keyword_memory.extract_and_add(thread_id, raw_query)
        top_keywords = self.keyword_memory.get_top_keywords(thread_id, limit=6)
        history_text = self.history_memory.get_formatted_history(thread_id, limit=3)

        # If there is no prior history and query is already comprehensive, return cleanly
        if not history_text and len(raw_query.split()) > 7 and not any(p in raw_query.lower().split() for p in ["it", "this", "that", "they", "its", "their"]):
            return {
                "original": raw_query,
                "optimized": raw_query,
                "keywords": top_keywords,
                "was_rewritten": False,
            }

        # If we have an LLM and prior history or relevant keywords, use LLM to synthesize standalone query
        if llm is not None and (history_text or top_keywords):
            prompt = (
                "You are an expert retrieval query optimizer for a Vector Knowledge Base containing books "
                "such as 'The 33 Strategies of War', 'The 48 Laws of Power', and 'The Molecule of More'.\n\n"
                "Your Task: Given the conversation history and active session concepts, rewrite the user's latest query "
                "into an explicit, keyword-rich standalone search query optimized for vector similarity retrieval.\n"
                "Rules:\n"
                "- Replace all ambiguous pronouns ('it', 'that', 'they', 'this rule', 'the book') with the actual subject.\n"
                "- Include specific concepts, entity names, or book titles if discussed.\n"
                "- DO NOT answer the question. Only return the rewritten query text in one concise sentence.\n\n"
                f"Conversation History:\n{history_text or '(None)'}\n\n"
                f"Active Session Keywords: {', '.join(top_keywords) if top_keywords else '(None)'}\n\n"
                f"User's Latest Query: {raw_query}\n\n"
                "Rewritten Retrieval Query:"
            )

            try:
                response = await llm.ainvoke([
                    SystemMessage(content="You are a query rewriting optimizer. Output only the rewritten query string."),
                    HumanMessage(content=prompt),
                ])
                content = response.content
                if isinstance(content, list):
                    content = "".join([p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"])
                optimized = str(content).strip().strip('"').strip("'")

                # Fallback safeguard if model generates an empty or overly verbose response
                if optimized and len(optimized) < 300 and not optimized.lower().startswith("here"):
                    return {
                        "original": raw_query,
                        "optimized": optimized,
                        "keywords": top_keywords,
                        "was_rewritten": True,
                    }
            except Exception as e:
                print(f"[WARN] LLM Query Optimization fallback: {e}")

        # Heuristic fallback if LLM is unavailable or query is short follow-up
        heuristics_keywords = [k for k in top_keywords if k.lower() not in raw_query.lower()]
        if heuristics_keywords:
            heuristic_query = f"{raw_query} ({', '.join(heuristics_keywords[:3])})"
            return {
                "original": raw_query,
                "optimized": heuristic_query,
                "keywords": top_keywords,
                "was_rewritten": True,
            }

        return {
            "original": raw_query,
            "optimized": raw_query,
            "keywords": top_keywords,
            "was_rewritten": False,
        }

    def record_turn(self, thread_id: str, role: str, content: str) -> None:
        """Stores interaction in History Memory and extracts keywords into Concept Memory."""
        self.history_memory.add_turn(thread_id, role, content)
        self.keyword_memory.extract_and_add(thread_id, content)

    def clear_session(self, thread_id: str) -> None:
        """Resets both memories for a thread."""
        self.history_memory.clear(thread_id)
        self.keyword_memory.clear(thread_id)
