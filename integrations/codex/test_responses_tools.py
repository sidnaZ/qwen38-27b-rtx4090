#!/usr/bin/env python3
"""Regression tests for the optional Codex/Responses compatibility layer.

Run with the vLLM environment used by this repository:
    venv/bin/python integrations/codex/test_responses_tools.py
"""

from vllm.entrypoints.openai.engine.protocol import (
    DeltaFunctionCall,
    DeltaMessage,
    DeltaToolCall,
)
from vllm.entrypoints.openai.responses.streaming_events import (
    SimpleStreamingEventProcessor,
    SimpleStreamingState,
    _StateType,
    split_delta,
)


def main() -> None:
    state = SimpleStreamingState(
        current_item_id="item_open",
        tool_call_id="call_open",
        tool_call_name="exec_command",
        tool_call_index=0,
        accumulated_text='{"cmd":"echo test"',
        has_emitted_tool_call_delta=True,
        current_state=_StateType.TOOL_CALL,
    )
    processor = SimpleStreamingEventProcessor(state=state)
    compound = DeltaMessage(
        reasoning="The command is running.",
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(name=None, arguments="}"),
            )
        ],
    )

    parts = split_delta(compound, processor.state.current_state)
    assert parts[0].tool_calls, "the open tool's argument tail must be first"
    assert parts[1].reasoning == "The command is running."

    event_types: list[str] = []
    for part in parts:
        target_state, tool_call = processor.resolve_target_state(part)
        assert target_state != _StateType.NONE
        if processor.needs_transition(target_state, tool_call):
            event_types.extend(event.type for event in processor.close_current())
            event_types.extend(
                event.type for event in processor.open(target_state, tool_call)
            )
        event_types.extend(
            event.type
            for event in processor.emit_delta(
                part, output=None  # type: ignore[arg-type]
            )
        )

    tail = event_types.index("response.function_call_arguments.delta")
    done = event_types.index("response.function_call_arguments.done")
    reasoning = event_types.index("response.reasoning_text.delta")
    assert tail < done < reasoning, event_types

    single = SimpleStreamingEventProcessor(allow_parallel_tool_calls=False)
    first = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=0,
                function=DeltaFunctionCall(
                    name="exec_command", arguments='{"cmd":"true"}'
                ),
            )
        ]
    )
    assert single.filter_parallel_tool_calls(first).tool_calls
    target_state, tool_call = single.resolve_target_state(first)
    for event in single.open(target_state, tool_call):
        assert event.type
    single.emit_delta(first, output=None)  # type: ignore[arg-type]

    malformed_second = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=1,
                function=DeltaFunctionCall(
                    name="exec_command", arguments='{"cmd":"unterminated',
                ),
            )
        ]
    )
    assert not single.filter_parallel_tool_calls(malformed_second).tool_calls

    orphan = DeltaMessage(
        tool_calls=[
            DeltaToolCall(
                index=2,
                function=DeltaFunctionCall(name=None, arguments='"}'),
            )
        ]
    )
    assert not single.filter_parallel_tool_calls(orphan).tool_calls
    print("PASS: tool-tail ordering and single-call filtering")


if __name__ == "__main__":
    main()
