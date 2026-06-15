import json
import re

import redis.asyncio as aioredis

from config import settings

MAX_TURNS    = 50         # turns stored per session
HISTORY_TTL  = 86400 * 7  # 7 days

# Long-term user-facts memory.
# Stored separately from per-session messages and survives across sessions.
# Examples: "user is a beginner", "user often confuses past tense forms".
USER_FACTS_KEY_FMT = "user:{user_id}:facts"
USER_FACTS_MAX     = 30           # cap so context doesn't explode
USER_FACTS_TTL     = 86400 * 90   # 90 days — facts age out

# Words that carry no topical signal — excluded from keyword scoring
_STOP = {
    "i", "a", "an", "the", "is", "are", "was", "were", "be", "been", "am",
    "to", "do", "did", "it", "in", "of", "and", "or", "at", "on", "for",
    "my", "me", "you", "your", "we", "they", "he", "she", "that", "this",
    "what", "how", "why", "when", "where", "who", "which",
    "can", "could", "would", "should", "have", "has", "had", "will", "may",
    "not", "no", "yes", "ok", "okay", "just", "so", "well", "but", "if",
    "then", "than", "with", "from", "up", "out", "about", "like", "get",
    "got", "its", "also", "very", "too", "some", "any", "more", "much",
}


def _keywords(text: str) -> set[str]:
    words = re.findall(r"\b[a-z']+\b", text.lower())
    return {w for w in words if w not in _STOP and len(w) > 2}


class RedisMemory:
    def __init__(self):
        self._redis = aioredis.from_url(settings.redis_url, decode_responses=True)

    def _key(self, session_id: str) -> str:
        return f"session:{session_id}:messages"

    async def get_messages(self, session_id: str) -> list[dict]:
        raw = await self._redis.get(self._key(session_id))
        return json.loads(raw) if raw else []

    async def get_relevant_messages(
        self,
        session_id: str,
        current_input: str,
        recent_turns: int = 4,    # always include the last N turns
        relevant_turns: int = 3,  # add up to N topically relevant older turns
    ) -> list[dict]:
        """Return a context window blending recency + keyword relevance."""
        all_msgs = await self.get_messages(session_id)
        if not all_msgs:
            return []

        # Pair up messages into (user, assistant) turns
        pairs: list[tuple[dict, dict]] = []
        for i in range(0, len(all_msgs) - 1, 2):
            if i + 1 < len(all_msgs):
                pairs.append((all_msgs[i], all_msgs[i + 1]))

        if len(pairs) <= recent_turns:
            return all_msgs  # small history — include everything

        recent = pairs[-recent_turns:]
        older  = pairs[:-recent_turns]

        # Score older turns by keyword overlap with the current utterance
        query_kw = _keywords(current_input)
        scored: list[tuple[int, tuple[dict, dict]]] = []
        for pair in older:
            turn_kw = _keywords(pair[0]["content"]) | _keywords(pair[1]["content"])
            score   = len(query_kw & turn_kw)
            scored.append((score, pair))

        scored.sort(key=lambda x: x[0], reverse=True)
        top = [pair for score, pair in scored[:relevant_turns] if score > 0]

        # Layout: relevant context first (older), then recent turns
        result: list[dict] = []
        for pair in top:
            result.extend(pair)
        for pair in recent:
            result.extend(pair)
        return result

    async def add_turn(self, session_id: str, user_text: str, assistant_text: str) -> None:
        messages = await self.get_messages(session_id)
        messages.append({"role": "user",      "content": user_text})
        messages.append({"role": "assistant", "content": assistant_text})

        if len(messages) > MAX_TURNS * 2:
            messages = messages[-(MAX_TURNS * 2):]
        await self._redis.set(self._key(session_id), json.dumps(messages), ex=HISTORY_TTL)

    async def clear(self, session_id: str) -> None:
        await self._redis.delete(self._key(session_id))

    # ─── Long-term user facts (cross-session) ──────────────────────────────────
    # These are persisted strings about the *user* (level, recurring mistakes,
    # topics they enjoy). The conversational LLM injects them into its system
    # prompt so it adapts across sessions without re-discovering everything.
    # Extraction strategy is left to the caller — typically a small LLM call
    # after each turn looking for stable, useful facts.

    def _facts_key(self, user_id: str) -> str:
        return USER_FACTS_KEY_FMT.format(user_id=user_id)

    async def get_user_facts(self, user_id: str) -> list[str]:
        raw = await self._redis.get(self._facts_key(user_id))
        return json.loads(raw) if raw else []

    async def add_user_fact(self, user_id: str, fact: str) -> bool:
        """Append a fact to the user's long-term memory.

        - Trims whitespace; skips empty or duplicate facts (case-insensitive)
        - Caps at USER_FACTS_MAX entries (oldest dropped first)
        - Refreshes TTL on every write
        Returns True if the fact was actually added.
        """
        fact = (fact or "").strip()
        if not fact:
            return False

        facts = await self.get_user_facts(user_id)
        lowered = {f.lower() for f in facts}
        if fact.lower() in lowered:
            return False

        facts.append(fact)
        if len(facts) > USER_FACTS_MAX:
            facts = facts[-USER_FACTS_MAX:]

        await self._redis.set(
            self._facts_key(user_id),
            json.dumps(facts),
            ex=USER_FACTS_TTL,
        )
        return True

    async def replace_user_fact(self, user_id: str, index: int, fact: str) -> bool:
        """Replace the fact at `index` (0-based) with a more specific version.
        Preserves position. Returns True iff index was valid and content changed."""
        fact = (fact or "").strip()
        if not fact:
            return False
        facts = await self.get_user_facts(user_id)
        if not 0 <= index < len(facts):
            return False
        if facts[index].lower() == fact.lower():
            return False
        facts[index] = fact
        await self._redis.set(
            self._facts_key(user_id),
            json.dumps(facts),
            ex=USER_FACTS_TTL,
        )
        return True

    async def remove_user_fact(self, user_id: str, fact: str) -> bool:
        """Remove an exact fact (case-insensitive). Returns True if removed."""
        facts = await self.get_user_facts(user_id)
        target = (fact or "").strip().lower()
        kept = [f for f in facts if f.lower() != target]
        if len(kept) == len(facts):
            return False
        await self._redis.set(
            self._facts_key(user_id),
            json.dumps(kept),
            ex=USER_FACTS_TTL,
        )
        return True

    async def clear_user_facts(self, user_id: str) -> None:
        await self._redis.delete(self._facts_key(user_id))

    async def health(self) -> bool:
        try:
            await self._redis.ping()
            return True
        except Exception:
            return False
