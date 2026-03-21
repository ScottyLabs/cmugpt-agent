import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


from agent.agent_hub import run_agent
from agent.schema import UserInput


async def test1():
    print("=" * 60)
    print("TEST 1: Single Query")
    print("=" * 60)

    # Example: Single query
    user_query = UserInput(
        query="Where is the Gates Hillman Center?", user_id="test_user_123"
    )
    print("QUERY:", user_query.query)

    result_1 = await run_agent(user_query)
    print("\nANSWER:")
    print(result_1.model_dump_json(indent=2))

    print("\n" + "=" * 60)
    print("TEST 2: Multi-turn Conversation")
    print("=" * 60)

    # Build conversation history from first query
    history = [
        {"role": "user", "content": user_query.query},
        {"role": "assistant", "content": result_1.response_text},
    ]

    follow_up = UserInput(
        query="What's the closest cafe to it?",
        context={"previous_location": "Gates Hillman Center"},
        user_id="test_user_123",
    )
    print("FOLLOW-UP QUERY:", follow_up.query)

    result_2 = await run_agent(follow_up, message_history=history)
    print("\nANSWER:")
    print(result_2.model_dump_json(indent=2))

    print("\n" + "=" * 60)
    print("TEST 3: Three-turn Conversation")
    print("=" * 60)

    # Extend history with second exchange
    history.append({"role": "user", "content": follow_up.query})
    history.append({"role": "assistant", "content": result_2.response_text})

    follow_up_2 = UserInput(
        query="Are there any dining hours restrictions?",
        context={
            "previous_location": "Gates Hillman Center",
            "previous_cafe": result_2.response_text,
        },
        user_id="test_user_123",
    )
    print("FOLLOW-UP QUERY 2:", follow_up_2.query)

    result_3 = await run_agent(follow_up_2, message_history=history)
    print("\nANSWER:")
    print(result_3.model_dump_json(indent=2))

    print("\n" + "=" * 60)
    print("CONVERSATION SUMMARY")
    print("=" * 60)
    print(f"Turn 1 - Confidence: {result_1.thought.confidence}")
    print(f"Turn 2 - Confidence: {result_2.thought.confidence}")
    print(f"Turn 3 - Confidence: {result_3.thought.confidence}")


if __name__ == "__main__":
    asyncio.run(test1())
