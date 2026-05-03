"""
Unit tests for RedisMemory — uses fakeredis so no live Redis is needed.
"""
import pytest
import fakeredis.aioredis as fake_aioredis

from services.redis_memory import MAX_TURNS, RedisMemory


@pytest.fixture
async def memory():
    mem = RedisMemory()
    mem._redis = fake_aioredis.FakeRedis(decode_responses=True)
    return mem


class TestRedisMemory:
    async def test_empty_session_returns_empty_list(self, memory):
        result = await memory.get_messages("brand-new-session")
        assert result == []

    async def test_add_turn_stores_both_messages(self, memory):
        await memory.add_turn("s1", "Hello!", "Hi there!")
        msgs = await memory.get_messages("s1")

        assert len(msgs) == 2
        assert msgs[0] == {"role": "user", "content": "Hello!"}
        assert msgs[1] == {"role": "assistant", "content": "Hi there!"}

    async def test_multiple_turns_accumulate(self, memory):
        await memory.add_turn("s2", "Turn 1 user", "Turn 1 bot")
        await memory.add_turn("s2", "Turn 2 user", "Turn 2 bot")
        msgs = await memory.get_messages("s2")

        assert len(msgs) == 4

    async def test_trims_to_max_turns(self, memory):
        for i in range(MAX_TURNS + 3):
            await memory.add_turn("s3", f"user {i}", f"bot {i}")

        msgs = await memory.get_messages("s3")

        assert len(msgs) == MAX_TURNS * 2
        # Oldest messages should have been dropped; most recent are kept
        assert msgs[-1]["content"] == f"bot {MAX_TURNS + 2}"
        assert msgs[-2]["content"] == f"user {MAX_TURNS + 2}"

    async def test_oldest_messages_dropped_first(self, memory):
        for i in range(MAX_TURNS + 2):
            await memory.add_turn("s4", f"user {i}", f"bot {i}")

        msgs = await memory.get_messages("s4")

        # First message kept should NOT be "user 0" (it was trimmed)
        assert msgs[0]["content"] != "user 0"

    async def test_clear_removes_all_messages(self, memory):
        await memory.add_turn("s5", "Hello", "Hi")
        await memory.clear("s5")
        msgs = await memory.get_messages("s5")

        assert msgs == []

    async def test_different_sessions_are_isolated(self, memory):
        await memory.add_turn("session-a", "A says hi", "Bot A response")
        await memory.add_turn("session-b", "B says hi", "Bot B response")

        a_msgs = await memory.get_messages("session-a")
        b_msgs = await memory.get_messages("session-b")

        assert len(a_msgs) == 2
        assert len(b_msgs) == 2
        assert a_msgs[0]["content"] == "A says hi"
        assert b_msgs[0]["content"] == "B says hi"

    async def test_clear_only_affects_target_session(self, memory):
        await memory.add_turn("keep", "Keep me", "Kept!")
        await memory.add_turn("drop", "Drop me", "Dropped!")
        await memory.clear("drop")

        assert await memory.get_messages("keep") != []
        assert await memory.get_messages("drop") == []

    async def test_health_returns_true(self, memory):
        result = await memory.health()
        assert result is True
