"""
parser.py
=========
STEPS 8-9: turn the model's raw scores into an actual tree, and draw it.

The model gives us a [T, T] grid of scores. Getting a TREE out of that grid is
called DECODING. We use the simplest possible decoder (greedy argmax), then
repair it if it turns out not to be a tree -- and we explain at the bottom how
real parsers do this properly.
"""

from typing import List, Optional, Tuple

import torch

from dataset import PAD_ID, ROOT_ID, Sentence, Vocab, encode, make_batch
from model import NeuralDependencyParser

# Words we silently drop from user input, because the training data has no
# punctuation and therefore no `punct` label to give them.
PUNCT = set(".,!?;:\"'()[]")


def tokenize(text: str) -> List[str]:
    """'The boy kicked the ball.' -> ['The', 'boy', 'kicked', 'the', 'ball']

    Deliberately primitive: split on whitespace, strip punctuation, keep the
    original capitalisation for display. Lower-casing happens later, only when
    looking the word up in the vocabulary, so that 'The' and 'the' share one
    embedding. With a 68-word vocabulary we cannot afford two entries for the
    same word.
    """
    raw = text.replace("﻿", "").replace("'", " '").split()
    toks = []
    for t in raw:
        t = "".join(ch for ch in t if ch not in PUNCT)
        if t:
            toks.append(t)
    return toks


# --------------------------------------------------------------------------
# DECODING
# --------------------------------------------------------------------------
def greedy_decode(arc_scores: torch.Tensor) -> List[int]:
    """Each word independently picks its highest-scoring head.

    arc_scores : [T, T] for ONE sentence (row 0 = <root>, ignored)
    returns    : heads[1..n] as a python list, e.g. [2, 3, 0, 5, 3]

    This is one line of real work:  heads = argmax over each row.

    It is also WRONG in general, and knowing why is the point of this
    function. Each word chooses in isolation, so nothing enforces the global
    property "the result must be a tree". Two failures are possible:
      (a) a CYCLE:  a -> b -> a  (nobody reaches ROOT)
      (b) MULTIPLE ROOTS: three different words all pick head 0.
    On our toy grammar this almost never happens once the model is trained,
    but it is the exact reason serious parsers use a maximum-spanning-tree
    algorithm instead (see `note_on_real_decoders` below).
    """
    return arc_scores[1:].argmax(dim=-1).tolist()


def fix_cycles(heads: List[int], arc_scores: torch.Tensor,
               max_fixes: int = 10) -> List[int]:
    """Repair greedy output until it is a valid tree.

    heads      : 1-based head list from greedy_decode, index i-1 holds head of word i
    arc_scores : [T, T] the same score grid, used to pick the replacement edge

    Strategy (deliberately simple, and defensible in a viva):
      1. Walk up from each word. If we return to a word we already visited,
         we found a cycle.
      2. Inside that cycle, find the edge the model was LEAST confident about.
      3. Re-point that one word at its next-best head OUTSIDE the cycle.
      4. Repeat.
    Breaking the weakest link is the greedy approximation of what the optimal
    algorithm (Chu-Liu/Edmonds) does exactly.
    """
    n = len(heads)
    heads = list(heads)

    for _ in range(max_fixes):
        cycle = _find_cycle(heads)
        if cycle is None:
            return heads

        # Confidence of each edge in the cycle = its score.
        weakest = min(cycle, key=lambda i: arc_scores[i, heads[i - 1]].item())

        # Best head for `weakest` that is not itself inside the cycle.
        row = arc_scores[weakest].clone()
        for j in cycle:
            row[j] = -1e9                       # forbid staying in the cycle
        row[weakest] = -1e9
        heads[weakest - 1] = int(row.argmax().item())

    return heads


def _find_cycle(heads: List[int]) -> Optional[List[int]]:
    """Return the members of some cycle, or None if the graph is acyclic.

    Standard 'walk to the root and see if you come back' check.
    """
    n = len(heads)
    for start in range(1, n + 1):
        seen, node = [], start
        while node != 0 and node not in seen:
            seen.append(node)
            node = heads[node - 1]
        if node != 0:                            # we re-entered a visited node
            return seen[seen.index(node):]
    return None


def note_on_real_decoders() -> str:
    return (
        "Greedy argmax treats each word's decision as independent, so it can\n"
        "produce cycles or several roots. Production parsers score all edges\n"
        "the same way we do, then run Chu-Liu/Edmonds' maximum spanning tree\n"
        "algorithm, which finds the highest-scoring set of edges that is\n"
        "GUARANTEED to be a tree. Transition-based parsers (arc-standard /\n"
        "arc-eager with a stack and buffer) get the same guarantee differently:\n"
        "they only ever offer the classifier actions that keep the partial\n"
        "structure well-formed."
    )


# --------------------------------------------------------------------------
# PARSING A NEW SENTENCE END TO END
# --------------------------------------------------------------------------
@torch.no_grad()
def parse_sentence(model: NeuralDependencyParser, words: List[str],
                   wv: Vocab, lv: Vocab) -> Tuple[List[int], List[str], List[float]]:
    """words -> (heads, labels, confidences)

    Example
      in : ['The', 'boy', 'kicked', 'the', 'ball']
      out: heads      [2, 3, 0, 5, 3]
           labels     ['det', 'nsubj', 'root', 'det', 'obj']
           confidence [0.99, 0.97, 0.99, 0.99, 0.96]   (P of the chosen head)

    The four steps are exactly the four boxes of the architecture diagram:
    encode ids -> BiLSTM -> arc scores -> decode -> label scores.
    """
    model.eval()

    # Note: we reuse `encode` from dataset.py with dummy heads/labels, because
    # at test time we obviously do not have the gold annotation.
    dummy = Sentence(words, [0] * len(words), [lv.itos[0]] * len(words))
    ids, _, _ = encode(dummy, wv, lv)
    word_ids, _, _, lengths = make_batch([(ids, [-100] * len(ids), [-100] * len(ids))])

    H = model.encode(word_ids, lengths)                  # [1, T, 192]
    arcs = model.arc_scores(H, lengths)[0]               # [T, T]

    heads = greedy_decode(arcs)                          # list of n ints
    heads = fix_cycles(heads, arcs)

    # Softmax over each row turns scores into probabilities -> confidence.
    probs = torch.softmax(arcs, dim=-1)
    conf = [probs[i + 1, h].item() for i, h in enumerate(heads)]

    # Now label the edges we just chose. The label head needs a [1, T] tensor
    # with position 0 (the <root> row) filled in with anything.
    head_tensor = torch.tensor([[0] + heads], dtype=torch.long)
    lab = model.label_scores(H, head_tensor)[0]          # [T, n_labels]
    label_ids = lab[1:].argmax(dim=-1).tolist()
    labels = [lv.itos[i] for i in label_ids]

    # A tiny consistency rule: whatever attaches to ROOT is called 'root'.
    # The classifier almost always gets this right on its own; forcing it
    # costs nothing and removes a silly-looking error class.
    for i, h in enumerate(heads):
        if h == 0:
            labels[i] = "root"

    return heads, labels, conf


# --------------------------------------------------------------------------
# VISUALISATION
# --------------------------------------------------------------------------
# Everything below is plain ASCII on purpose: Windows terminals often use a
# code page that cannot print box-drawing characters, and a UnicodeEncodeError
# in the middle of a demo is not a good look.
def format_table(words: List[str], heads: List[int], labels: List[str],
                 conf: Optional[List[float]] = None) -> str:
    """The CoNLL-style table the assignment asks for."""
    w = max(8, max(len(x) for x in words) + 2)
    head_texts = ["ROOT" if h == 0 else words[h - 1] for h in heads]
    hw = max(8, max(len(x) for x in head_texts) + 2)

    out = [f"{'Word':<{w}}{'Head':<{hw}}{'Relation':<10}" + ("Conf" if conf else "")]
    out.append("-" * (w + hw + 10 + (6 if conf else 0)))
    for i, word in enumerate(words):
        line = f"{word:<{w}}{head_texts[i]:<{hw}}{labels[i]:<10}"
        if conf:
            line += f"{conf[i]:.2f}"
        out.append(line)
    return "\n".join(out)


def format_arcs(words: List[str], heads: List[int], labels: List[str]) -> str:
    r"""Draw each dependency as an arrow above the sentence.

        +------------->        nsubj
        The  dog  chased  the  cat

    One arc per line (widest first) so the picture is always readable and the
    drawing code stays about fifteen lines long.
    """
    # Column where each word starts, with two spaces between words.
    col, cols = 0, []
    for wd in words:
        cols.append(col)
        col += len(wd) + 2
    width = col

    # Centre of each word, used as the anchor point of an arrow.
    centre = [cols[i] + len(words[i]) // 2 for i in range(len(words))]

    arcs = sorted(
        [(i, heads[i], labels[i]) for i in range(len(words)) if heads[i] != 0],
        key=lambda t: -abs(centre[t[0]] - centre[t[1] - 1]),
    )

    lines = []
    for dep, head, lab in arcs:
        cd, ch = centre[dep], centre[head - 1]
        row = [" "] * width
        lo, hi = min(cd, ch), max(cd, ch)
        for x in range(lo, hi + 1):
            row[x] = "-"
        row[ch] = "+"                         # tail sits on the head word
        row[cd] = "<" if cd < ch else ">"     # arrow head points at dependent
        lines.append("".join(row).rstrip() + "   " + lab)

    root_idx = [i for i, h in enumerate(heads) if h == 0]
    if root_idx:
        r = centre[root_idx[0]]
        lines.append(" " * r + "^" + "   root (ROOT -> " + words[root_idx[0]] + ")")

    lines.append("  ".join(words))
    return "\n".join(lines)


def format_tree(words: List[str], heads: List[int], labels: List[str]) -> str:
    """Indented tree, the view that makes the hierarchy obvious.

        ROOT
        `-- chased (root)
            |-- dog (nsubj)
            |   `-- The (det)
            `-- cat (obj)
                `-- the (det)
    """
    children = {i: [] for i in range(len(words) + 1)}
    for i, h in enumerate(heads, start=1):
        children[h].append(i)

    lines = ["ROOT"]

    def walk(node: int, prefix: str):
        kids = children[node]
        for k, child in enumerate(kids):
            last = k == len(kids) - 1
            lines.append(f"{prefix}{'`-- ' if last else '|-- '}"
                         f"{words[child - 1]} ({labels[child - 1]})")
            walk(child, prefix + ("    " if last else "|   "))

    walk(0, "")
    return "\n".join(lines)


def show_parse(words: List[str], heads: List[int], labels: List[str],
               conf: Optional[List[float]] = None) -> None:
    print()
    print(format_table(words, heads, labels, conf))
    print()
    print(format_arcs(words, heads, labels))
    print()
    print(format_tree(words, heads, labels))
    print()
