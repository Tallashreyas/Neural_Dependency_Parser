"""
train.py
========
STEP 7: teach the network by showing it examples and correcting its mistakes.

The loop, in words:
    1. take a batch of sentences
    2. forward pass  -> arc scores + label scores
    3. compare with the gold annotation -> a single number, the loss
    4. loss.backward() -> how should every weight change to lower that number?
    5. optimizer.step() -> actually change them, a little
    6. repeat for a few dozen epochs

Run:  python train.py
"""

import random
from typing import List, Tuple

import torch
from torch.optim import Adam

from dataset import (Vocab, build_vocabs, encode, generate_corpus, make_batch,
                     Sentence, train_test_split)
from model import NeuralDependencyParser, parser_loss
from parser import fix_cycles, greedy_decode

CHECKPOINT = "model.pt"


# --------------------------------------------------------------------------
# EVALUATION METRICS
# --------------------------------------------------------------------------
@torch.no_grad()
def evaluate(model: NeuralDependencyParser, data: List[Sentence],
             wv: Vocab, lv: Vocab) -> Tuple[float, float]:
    """Return (UAS, LAS) -- the two standard dependency-parsing metrics.

    UAS = Unlabelled Attachment Score
        fraction of words whose HEAD is correct.
        "Did you find the right parent?"

    LAS = Labelled Attachment Score
        fraction of words whose head AND label are both correct.
        "Did you find the right parent AND name the relation correctly?"

    LAS <= UAS always, because LAS demands strictly more. Reporting both tells
    you where the errors live: if UAS is 95% and LAS is 70%, attachment is
    fine and the label classifier is the problem.
    """
    model.eval()
    correct_head = correct_both = total = 0

    for sent in data:
        ids, gold_heads, gold_labels = encode(sent, wv, lv)
        word_ids, _, _, lengths = make_batch([(ids, gold_heads, gold_labels)])

        H = model.encode(word_ids, lengths)
        arcs = model.arc_scores(H, lengths)[0]

        pred_heads = fix_cycles(greedy_decode(arcs), arcs)

        head_tensor = torch.tensor([[0] + pred_heads], dtype=torch.long)
        pred_labels = model.label_scores(H, head_tensor)[0][1:].argmax(-1).tolist()

        for i in range(len(sent)):
            total += 1
            if pred_heads[i] == sent.heads[i]:
                correct_head += 1
                if lv.itos[pred_labels[i]] == sent.labels[i]:
                    correct_both += 1

    return 100.0 * correct_head / total, 100.0 * correct_both / total


def plot_loss_ascii(history, height: int = 12) -> str:
    """Draw the loss curve in the terminal -- no matplotlib needed.

    history: list of (epoch, loss, train_uas, test_uas, test_las)

    The shape you should expect: a steep drop in the first ~10 epochs (the
    model discovers the big regularities: determiners attach right, verbs are
    roots), then a long flat tail near zero (it is only polishing).
    """
    if not history:
        return ""
    losses = [h[1] for h in history]
    epochs = [h[0] for h in history]
    lo, hi = min(losses), max(losses)
    span = max(hi - lo, 1e-9)

    grid = [[" "] * len(losses) for _ in range(height)]
    for x, v in enumerate(losses):
        y = int(round((1 - (v - lo) / span) * (height - 1)))
        grid[y][x] = "*"

    lines = [f"training loss (epochs {epochs[0]}..{epochs[-1]})"]
    for y, row in enumerate(grid):
        value = hi - (y / (height - 1)) * span
        lines.append(f"{value:6.2f} |" + "".join(c + " " for c in row))
    lines.append("       +" + "-" * (2 * len(losses)))
    lines.append("        " + "".join(f"{e:<2}"[:2] for e in epochs))
    return "\n".join(lines)


# --------------------------------------------------------------------------
# THE TRAINING LOOP
# --------------------------------------------------------------------------
def train(epochs: int = 60, batch_size: int = 16, lr: float = 2e-3,
          seed: int = 1, verbose: bool = True):
    torch.manual_seed(seed)
    random.seed(seed)

    # ---- data ----------------------------------------------------------
    corpus = generate_corpus(n_sentences=200)
    train_data, test_data = train_test_split(corpus, test_ratio=0.2)
    wv, lv = build_vocabs(train_data)

    # Encode once, up front. Our corpus is tiny, so there is no reason to
    # re-encode the same strings every epoch.
    encoded = [encode(s, wv, lv) for s in train_data]

    if verbose:
        print(f"train sentences : {len(train_data)}")
        print(f"test sentences  : {len(test_data)}")
        print(f"word vocabulary : {len(wv)}")
        print(f"labels          : {lv.itos}")
        print()

    # ---- model + optimizer ---------------------------------------------
    model = NeuralDependencyParser(vocab_size=len(wv), n_labels=len(lv))

    # Adam: gradient descent with a per-parameter adaptive step size. We use
    # it rather than plain SGD because it needs almost no tuning -- important
    # when the point of the exercise is the parser, not the optimiser.
    optimizer = Adam(model.parameters(), lr=lr)

    n_params = sum(p.numel() for p in model.parameters())
    if verbose:
        print(f"trainable parameters: {n_params:,}")
        print()
        print("epoch   loss    head    label   train UAS  test UAS  test LAS")
        print("-" * 62)

    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        random.shuffle(encoded)          # different batch composition each epoch

        totals = [0.0, 0.0, 0.0]
        n_batches = 0

        for start in range(0, len(encoded), batch_size):
            batch = encoded[start:start + batch_size]
            word_ids, heads, labels, lengths = make_batch(batch)

            # ---- 1. forward ------------------------------------------
            arc_scores, label_scores = model(word_ids, lengths, heads)

            # ---- 2. how wrong were we? --------------------------------
            loss, hl, ll = parser_loss(arc_scores, label_scores, heads, labels)

            # ---- 3. backward ------------------------------------------
            # zero_grad first: PyTorch ACCUMULATES gradients, so without this
            # every batch would be polluted by the previous batch's gradient.
            optimizer.zero_grad()
            loss.backward()               # fills p.grad for every parameter

            # Clip gradients: RNNs occasionally produce one enormous gradient
            # that would throw the weights far away from anything useful.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)

            # ---- 4. update --------------------------------------------
            optimizer.step()              # p <- p - lr * (adapted gradient)

            totals[0] += loss.item()
            totals[1] += hl.item()
            totals[2] += ll.item()
            n_batches += 1

        avg = [t / n_batches for t in totals]

        if epoch % 5 == 0 or epoch == 1:
            train_uas, _ = evaluate(model, train_data, wv, lv)
            test_uas, test_las = evaluate(model, test_data, wv, lv)
            history.append((epoch, avg[0], train_uas, test_uas, test_las))
            if verbose:
                print(f"{epoch:<7} {avg[0]:<7.3f} {avg[1]:<7.3f} {avg[2]:<7.3f} "
                      f"{train_uas:<10.1f} {test_uas:<9.1f} {test_las:.1f}")

    if verbose:
        print()
        print(plot_loss_ascii(history))

    # ---- final report ---------------------------------------------------
    train_uas, train_las = evaluate(model, train_data, wv, lv)
    test_uas, test_las = evaluate(model, test_data, wv, lv)
    if verbose:
        print()
        print("FINAL")
        print(f"  train  UAS {train_uas:.1f}%   LAS {train_las:.1f}%")
        print(f"  test   UAS {test_uas:.1f}%   LAS {test_las:.1f}%")

    # ---- save -----------------------------------------------------------
    # We save the vocabularies alongside the weights. Weights alone are
    # useless: id 22 only means "dog" if you have the same word->id mapping.
    torch.save(
        {
            "state_dict": model.state_dict(),
            "word_itos": wv.itos,
            "label_itos": lv.itos,
            "test_uas": test_uas,
            "test_las": test_las,
        },
        CHECKPOINT,
    )
    if verbose:
        print(f"\nsaved -> {CHECKPOINT}")

    return model, wv, lv, history


def load_model(path: str = CHECKPOINT):
    """Rebuild the trained parser from disk."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)

    wv = Vocab([], specials=ckpt["word_itos"])
    lv = Vocab([], specials=ckpt["label_itos"])

    model = NeuralDependencyParser(vocab_size=len(wv), n_labels=len(lv))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, wv, lv, ckpt


if __name__ == "__main__":
    train()
