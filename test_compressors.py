"""Quick smoke test for compressors.py"""
import sys
sys.path.insert(0, str(__file__).rsplit("\\", 1)[0])

from src.compressors import (
    SlidingWindowCompressor,
    SummarizationCompressor,
    HybridCompressor,
    all_compressors,
)
from src.models import Conversation, Turn

# Build a minimal 6-turn conversation
turns = [
    Turn(role="user", content="What stock did you recommend?"),
    Turn(role="assistant", content="I recommended buying ACME at $120 per share."),
    Turn(role="user", content="How many shares?"),
    Turn(role="assistant", content="I suggested 50 shares, total $6000."),
    Turn(role="user", content="Any stop-loss?"),
    Turn(role="assistant", content="Set a stop-loss at $100 per share."),
]
conv = Conversation(
    id="test-001",
    domain="personal finance and scheduling",
    history=turns,
    final_question="What is my maximum loss if the stop-loss triggers?",
    ground_truth_answer="$1000",
    critical_turn_indices=[1, 5],
)

# --- SlidingWindowCompressor ---
sw2 = SlidingWindowCompressor(2)
result = sw2.compress(conv)
assert len(result) == 2, f"Expected 2 turns, got {len(result)}"
assert result[0] == turns[-2]
assert result[1] == turns[-1]
print(f"SlidingWindow(2): {len(result)} turns  [PASS]")

sw6 = SlidingWindowCompressor(6)
result = sw6.compress(conv)
assert len(result) == 6, "Window=6 should return all 6 turns"
print(f"SlidingWindow(6): {len(result)} turns  [PASS]")

sw10 = SlidingWindowCompressor(10)
result = sw10.compress(conv)
assert len(result) == 6, "Window > len should clamp to len"
print(f"SlidingWindow(10 > len=6): {len(result)} turns  [PASS]")

# --- HybridCompressor ---
hy2 = HybridCompressor(2)
result = hy2.compress(conv)
assert len(result) == 4, f"Expected 4 turns (2 anchors + 2 recent), got {len(result)}"
assert result[-2] == turns[-2]
assert result[-1] == turns[-1]
print(f"Hybrid(k=2): {len(result)} turns  [PASS]")

hy10 = HybridCompressor(10)
result = hy10.compress(conv)
assert len(result) == 6, "k > len should clamp to len"
print(f"Hybrid(k=10 > len=6): {len(result)} turns  [PASS]")

# --- Factory ---
cs = all_compressors()
assert len(cs) == 9, f"Expected 9 compressors, got {len(cs)}"
names = [c.name for c in cs]
assert names.count("sliding_window") == 3
assert names.count("summarization") == 3
assert names.count("hybrid") == 3
print(f"all_compressors(): {len(cs)} instances  [PASS]")
for c in cs:
    print(f"  {c.name}  {c.hyperparams}")

print("\nAll smoke tests passed.")
