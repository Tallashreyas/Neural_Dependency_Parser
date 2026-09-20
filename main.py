"""
main.py
=======
The demo you actually run:

    python main.py                       -> demo sentences, then a prompt
    python main.py "The boy kicked the ball."   -> parse one sentence and exit
    python main.py --retrain             -> train from scratch first

Type a sentence at the prompt and the parser prints the head/relation table,
an arrow diagram, and an indented tree.
"""

import os
import sys

from parser import note_on_real_decoders, parse_sentence, show_parse, tokenize
from train import CHECKPOINT, load_model, train

DEMOS = [
    "The dog chased the cat",          # the running example
    "The boy kicked the ball",         # the sentence from the assignment brief
    "A small dog quickly chased the angry cat in the garden",
    "Mary saw John",
    "The happy bird sang loudly",
]

# Sentences designed to BREAK the parser, so the limitations section is not
# just a claim. None of these patterns exist in the training templates.
STRESS = [
    "The dog chased the cat and the mouse",   # coordination: never seen
    "The zebra devoured the pizza",           # two unknown words
    "The dog that barked chased the cat",     # a relative clause: never seen
]


def ensure_model():
    """Train on first run; reuse the checkpoint afterwards."""
    if not os.path.exists(CHECKPOINT):
        print("No trained model found. Training now (about 30 seconds)...\n")
        train()
        print()
    return load_model(CHECKPOINT)


def parse_and_show(model, wv, lv, text: str) -> None:
    words = tokenize(text)
    if not words:
        return
    heads, labels, conf = parse_sentence(model, words, wv, lv)

    unknown = [w for w in words if wv.get(w) == 1]   # 1 == UNK_ID
    print(f'\n=== "{text}" ===')
    if unknown:
        print(f"   (out-of-vocabulary, seen as <unk>: {', '.join(unknown)})")
    show_parse(words, heads, labels, conf)


def main() -> None:
    args = [a for a in sys.argv[1:]]
    if "--retrain" in args:
        args.remove("--retrain")
        if os.path.exists(CHECKPOINT):
            os.remove(CHECKPOINT)

    model, wv, lv, ckpt = ensure_model()
    print(f"Loaded parser  |  vocabulary {len(wv)} words  |  {len(lv)} relation types")
    print(f"Held-out accuracy: UAS {ckpt['test_uas']:.1f}%   "
          f"LAS {ckpt['test_las']:.1f}%")

    # One-shot mode: python main.py "some sentence"
    if args:
        parse_and_show(model, wv, lv, " ".join(args))
        return

    print("\n" + "=" * 62)
    print("DEMO: sentences the grammar covers")
    print("=" * 62)
    for s in DEMOS:
        parse_and_show(model, wv, lv, s)

    print("=" * 62)
    print("STRESS TEST: things the training data never contained")
    print("=" * 62)
    for s in STRESS:
        parse_and_show(model, wv, lv, s)
    print("Read those three carefully -- the mistakes are the interesting part.")
    print("They are discussed in section 11 of REPORT.md.\n")

    print("=" * 62)
    print("Your turn. Type a sentence (blank line or 'quit' to exit).")
    print("Known vocabulary:", ", ".join(wv.itos[3:]))
    print("=" * 62)

    while True:
        try:
            text = input("\nsentence> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not text or text.lower() in {"quit", "exit", "q"}:
            break
        if text.lower() == "help":
            print(note_on_real_decoders())
            continue
        parse_and_show(model, wv, lv, text)

    print("bye")


if __name__ == "__main__":
    main()
