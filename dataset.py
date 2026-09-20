"""
dataset.py
==========
STEP 2 of the project: the data.

A dependency-parsed sentence is stored in the same shape that real treebanks
(CoNLL-U format) use, just stripped down to the three columns we actually need:

    index   word      head    label
    1       the       2       det      <- "the" depends on word #2 ("dog")
    2       dog       3       nsubj    <- "dog" depends on word #3 ("chased")
    3       chased    0       root     <- head 0 is the artificial ROOT node
    4       the       5       det
    5       cat       3       obj

WHY heads are stored as integers and not as words:
  A sentence can repeat the same word ("the" appears twice above). If we stored
  the head as the *string* "the", we could not tell which "the" we meant.
  Positions are unambiguous, so heads are always integer indices.

WHY head == 0 means ROOT:
  Every tree needs exactly one node with no parent. Instead of special-casing
  it, we pretend there is an invisible token at position 0 called <root>.
  Then EVERY word has a head, and the whole problem becomes uniform:
  "for each word, choose one of positions 0..n as its head".
"""

import random
from dataclasses import dataclass
from typing import Dict, List, Tuple

# --------------------------------------------------------------------------
# Special tokens.
# --------------------------------------------------------------------------
# <pad>  : filler used to make all sentences in a batch the same length.
# <unk>  : stands for any word the model never saw during training.
# <root> : the artificial head-of-the-sentence node described above.
PAD, UNK, ROOT = "<pad>", "<unk>", "<root>"
PAD_ID, UNK_ID, ROOT_ID = 0, 1, 2


@dataclass
class Sentence:
    """One annotated sentence.

    words  : ["the", "dog", "chased", "the", "cat"]
    heads  : [2, 3, 0, 5, 3]          (1-based word positions, 0 = ROOT)
    labels : ["det", "nsubj", "root", "det", "obj"]

    All three lists always have the same length n = number of words.
    """
    words: List[str]
    heads: List[int]
    labels: List[str]

    def __len__(self) -> int:
        return len(self.words)

    def pretty(self) -> str:
        rows = ["idx  word         head         label"]
        for i, (w, h, l) in enumerate(zip(self.words, self.heads, self.labels), start=1):
            head_word = "ROOT" if h == 0 else self.words[h - 1]
            rows.append(f"{i:<4} {w:<12} {head_word:<12} {l}")
        return "\n".join(rows)


# --------------------------------------------------------------------------
# A tiny hand-built lexicon, grouped by syntactic category.
# --------------------------------------------------------------------------
# WHY group by category: it lets us write a handful of sentence *patterns*
# (templates) and then fill the slots with different words. Writing 160
# sentences by hand is error-prone; writing 23 patterns by hand is not, and the
# annotations stay guaranteed-correct because the pattern carries them.
LEXICON: Dict[str, List[str]] = {
    "DET":  ["the", "a"],
    "ADJ":  ["small", "big", "black", "white", "happy", "angry", "old", "young", "red", "lazy"],
    "N":    ["dog", "cat", "boy", "girl", "man", "woman", "ball", "bird",
             "mouse", "teacher", "student", "book", "car", "tree", "garden", "box"],
    "VT":   ["chased", "kicked", "saw", "caught", "ate", "watched",
             "found", "liked", "pushed", "carried", "hit", "threw"],   # transitive
    "VI":   ["barked", "ran", "slept", "jumped", "laughed", "smiled", "walked", "sang"],
    "ADV":  ["quickly", "slowly", "loudly", "suddenly", "happily", "angrily"],
    "NAME": ["john", "mary", "tom", "anna", "sam"],
    "P":    ["in", "on", "under", "near", "behind", "with"],
}

# --------------------------------------------------------------------------
# The sentence templates: (categories, heads, labels)
# --------------------------------------------------------------------------
# Read the first one as: DET N VT DET N  ==  "the dog chased the cat"
#   word 1 (DET)  -> head 2  label det
#   word 2 (N)    -> head 3  label nsubj
#   word 3 (VT)   -> head 0  label root
#   word 4 (DET)  -> head 5  label det
#   word 5 (N)    -> head 3  label obj
#
# Design decisions worth knowing for the viva:
#  * No punctuation. Real treebanks attach "." to the root with label `punct`;
#    it adds a label and zero insight, so we drop it.
#  * Prepositional phrases always attach to the NOUN before them (`nmod`),
#    never to the verb. Real language is ambiguous here ("chased the cat in
#    the garden" -- who is in the garden?). We fixed one convention so the
#    training data is consistent; the ambiguity is discussed in limitations.
#  * We follow Universal Dependencies conventions: the preposition is a
#    dependent (`case`) of its noun, not the head of the phrase.
TEMPLATES: List[Tuple[List[str], List[int], List[str]]] = [
    # --- basic transitive ---
    (["DET", "N", "VT", "DET", "N"],
     [2, 3, 0, 5, 3],
     ["det", "nsubj", "root", "det", "obj"]),

    (["DET", "ADJ", "N", "VT", "DET", "N"],
     [3, 3, 4, 0, 6, 4],
     ["det", "amod", "nsubj", "root", "det", "obj"]),

    (["DET", "N", "VT", "DET", "ADJ", "N"],
     [2, 3, 0, 6, 6, 3],
     ["det", "nsubj", "root", "det", "amod", "obj"]),

    (["DET", "ADJ", "N", "VT", "DET", "ADJ", "N"],
     [3, 3, 4, 0, 7, 7, 4],
     ["det", "amod", "nsubj", "root", "det", "amod", "obj"]),

    # --- adverbs ---
    (["DET", "N", "ADV", "VT", "DET", "N"],
     [2, 4, 4, 0, 6, 4],
     ["det", "nsubj", "advmod", "root", "det", "obj"]),

    (["DET", "N", "VT", "DET", "N", "ADV"],
     [2, 3, 0, 5, 3, 3],
     ["det", "nsubj", "root", "det", "obj", "advmod"]),

    (["DET", "N", "ADV", "VT", "DET", "ADJ", "N"],
     [2, 4, 4, 0, 7, 7, 4],
     ["det", "nsubj", "advmod", "root", "det", "amod", "obj"]),

    # --- intransitive ---
    (["DET", "N", "VI"],
     [2, 3, 0],
     ["det", "nsubj", "root"]),

    (["DET", "N", "VI", "ADV"],
     [2, 3, 0, 3],
     ["det", "nsubj", "root", "advmod"]),

    (["DET", "ADJ", "N", "VI"],
     [3, 3, 4, 0],
     ["det", "amod", "nsubj", "root"]),

    (["DET", "ADJ", "N", "VI", "ADV"],
     [3, 3, 4, 0, 4],
     ["det", "amod", "nsubj", "root", "advmod"]),

    (["DET", "N", "ADV", "VI"],
     [2, 4, 4, 0],
     ["det", "nsubj", "advmod", "root"]),

    # --- proper names (a subject with no determiner) ---
    (["NAME", "VT", "DET", "N"],
     [2, 0, 4, 2],
     ["nsubj", "root", "det", "obj"]),

    (["NAME", "VT", "DET", "ADJ", "N"],
     [2, 0, 5, 5, 2],
     ["nsubj", "root", "det", "amod", "obj"]),

    (["NAME", "VI"],
     [2, 0],
     ["nsubj", "root"]),

    (["NAME", "VI", "ADV"],
     [2, 0, 2],
     ["nsubj", "root", "advmod"]),

    (["NAME", "VT", "NAME"],
     [2, 0, 2],
     ["nsubj", "root", "obj"]),

    (["DET", "N", "VT", "NAME"],
     [2, 3, 0, 3],
     ["det", "nsubj", "root", "obj"]),

    # --- prepositional phrases ---
    (["DET", "N", "VT", "DET", "N", "P", "DET", "N"],
     [2, 3, 0, 5, 3, 8, 8, 5],
     ["det", "nsubj", "root", "det", "obj", "case", "det", "nmod"]),

    (["DET", "N", "P", "DET", "N", "VI"],
     [2, 6, 5, 5, 2, 0],
     ["det", "nsubj", "case", "det", "nmod", "root"]),

    (["DET", "ADJ", "N", "VT", "DET", "N", "P", "DET", "N"],
     [3, 3, 4, 0, 6, 4, 9, 9, 6],
     ["det", "amod", "nsubj", "root", "det", "obj", "case", "det", "nmod"]),

    (["NAME", "VT", "DET", "N", "P", "DET", "N"],
     [2, 0, 4, 2, 7, 7, 4],
     ["nsubj", "root", "det", "obj", "case", "det", "nmod"]),

    (["DET", "N", "P", "DET", "N", "VT", "DET", "N"],
     [2, 6, 5, 5, 2, 0, 8, 6],
     ["det", "nsubj", "case", "det", "nmod", "root", "det", "obj"]),

    # --- the long ones: every optional slot filled at once ---
    (["DET", "N", "ADV", "VT", "DET", "N", "P", "DET", "N"],
     [2, 4, 4, 0, 6, 4, 9, 9, 6],
     ["det", "nsubj", "advmod", "root", "det", "obj", "case", "det", "nmod"]),

    (["DET", "ADJ", "N", "VT", "DET", "ADJ", "N", "P", "DET", "N"],
     [3, 3, 4, 0, 7, 7, 4, 10, 10, 7],
     ["det", "amod", "nsubj", "root", "det", "amod", "obj", "case", "det", "nmod"]),

    (["DET", "ADJ", "N", "ADV", "VT", "DET", "ADJ", "N"],
     [3, 3, 5, 5, 0, 8, 8, 5],
     ["det", "amod", "nsubj", "advmod", "root", "det", "amod", "obj"]),

    (["DET", "ADJ", "N", "ADV", "VT", "DET", "ADJ", "N", "P", "DET", "N"],
     [3, 3, 5, 5, 0, 8, 8, 5, 11, 11, 8],
     ["det", "amod", "nsubj", "advmod", "root", "det", "amod", "obj",
      "case", "det", "nmod"]),
]


def generate_corpus(n_sentences: int = 200, seed: int = 42) -> List[Sentence]:
    """Fill the templates with random words to build the toy treebank.

    Input : how many sentences we want.
    Output: a list of `Sentence` objects, e.g.
            Sentence(words=['the','dog','chased','the','cat'],
                     heads=[2,3,0,5,3],
                     labels=['det','nsubj','root','det','obj'])

    We use a fixed random seed so every run produces the SAME corpus --
    otherwise "my accuracy changed" could mean "my data changed", and we would
    never know which.
    """
    rng = random.Random(seed)
    seen = set()
    corpus: List[Sentence] = []

    # Cap the attempts so we can never loop forever if the templates run out
    # of unique combinations.
    attempts = 0
    while len(corpus) < n_sentences and attempts < n_sentences * 200:
        attempts += 1
        cats, heads, labels = rng.choice(TEMPLATES)
        words = [rng.choice(LEXICON[c]) for c in cats]

        key = " ".join(words)
        if key in seen:
            continue                      # no duplicate sentences
        seen.add(key)
        corpus.append(Sentence(words, list(heads), list(labels)))

    return corpus


# --------------------------------------------------------------------------
# STEP 3 (part 1): turning words into integers.
# --------------------------------------------------------------------------
# A neural network cannot consume the string "dog". It consumes numbers.
# So we build a dictionary  word -> integer id  and replace every word by its
# id. The embedding layer (model.py) then maps each id to a learned vector.
class Vocab:
    """Two-way mapping between strings and integer ids."""

    def __init__(self, tokens: List[str], specials: List[str]):
        self.itos: List[str] = list(specials) + sorted(set(tokens) - set(specials))
        self.stoi: Dict[str, int] = {s: i for i, s in enumerate(self.itos)}

    def __len__(self) -> int:
        return len(self.itos)

    def get(self, word: str) -> int:
        """Unknown words fall back to <unk> instead of crashing.

        This is the whole reason <unk> exists: at test time the user may type
        'rabbit', which is not in our 60-word toy vocabulary. The model then
        has to guess from context alone -- which is exactly what the BiLSTM is
        good at.
        """
        return self.stoi.get(word.lower(), UNK_ID)


def build_vocabs(train: List[Sentence]) -> Tuple[Vocab, Vocab]:
    """Build the word vocabulary and the dependency-label vocabulary.

    IMPORTANT: built from the TRAINING SET ONLY.
    If we built it from all the data, the model would already "know" the test
    words -- a subtle form of cheating called information leakage.
    """
    words = [w for s in train for w in s.words]
    labels = sorted({l for s in train for l in s.labels})
    word_vocab = Vocab(words, specials=[PAD, UNK, ROOT])
    label_vocab = Vocab(labels, specials=[])      # labels need no specials
    return word_vocab, label_vocab


def encode(sentence: Sentence, wv: Vocab, lv: Vocab) -> Tuple[List[int], List[int], List[int]]:
    """Convert one Sentence into three integer lists the model can eat.

    Input  (n = 5 words):
        words  ['the','dog','chased','the','cat']
        heads  [2, 3, 0, 5, 3]
        labels ['det','nsubj','root','det','obj']

    Output (length n + 1 = 6 -- the +1 is the prepended <root> token):
        ids    [ROOT_ID, id(the), id(dog), id(chased), id(the), id(cat)]
        heads  [  -100,      2,      3,        0,         5,       3   ]
        labels [  -100,  id(det), id(nsubj), id(root), id(det), id(obj)]

    Two things to notice:

    1) We physically PREPEND <root> to the token sequence. That means the
       BiLSTM computes a real vector for ROOT too, so "which word is the root
       of the sentence?" becomes the same operation as any other head choice:
       position 0 is just another candidate head. Because ids[0] is <root>,
       a gold head value of `h` already points at the right slot -- no
       index arithmetic needed anywhere else in the code.

    2) -100 at position 0. That is PyTorch's default `ignore_index` for
       cross-entropy: the ROOT token itself has no head and no label, so we
       tell the loss function to skip it rather than invent a fake target.
    """
    ids = [ROOT_ID] + [wv.get(w) for w in sentence.words]
    heads = [-100] + list(sentence.heads)
    labels = [-100] + [lv.get(l) for l in sentence.labels]
    return ids, heads, labels


def make_batch(examples: List[Tuple[List[int], List[int], List[int]]]):
    """Pad a list of encoded sentences into rectangular tensors.

    Input : [(ids, heads, labels), ...]  -- sentences of DIFFERENT lengths
    Output: word_ids [B, T] long
            heads    [B, T] long   (-100 where there is nothing to predict)
            labels   [B, T] long   (-100 likewise)
            lengths  [B]    long   (true length of each row, including <root>)

    WHY padding: a tensor must be rectangular, but sentences are not all the
    same length. We pad the short ones to T = max length in the batch with
    <pad> (id 0), and carry `lengths` around so every other part of the code
    knows which cells are real. Padding is a storage trick, never data:
    - the LSTM skips it (pack_padded_sequence),
    - the arc scorer masks it out of the candidate list,
    - the loss ignores it (-100).
    """
    import torch  # local import so the dataset module stays usable without torch

    B = len(examples)
    T = max(len(ids) for ids, _, _ in examples)

    word_ids = torch.zeros(B, T, dtype=torch.long)          # 0 == PAD_ID
    heads = torch.full((B, T), -100, dtype=torch.long)
    labels = torch.full((B, T), -100, dtype=torch.long)
    lengths = torch.zeros(B, dtype=torch.long)

    for b, (ids, hs, ls) in enumerate(examples):
        n = len(ids)
        word_ids[b, :n] = torch.tensor(ids, dtype=torch.long)
        heads[b, :n] = torch.tensor(hs, dtype=torch.long)
        labels[b, :n] = torch.tensor(ls, dtype=torch.long)
        lengths[b] = n

    return word_ids, heads, labels, lengths


def train_test_split(corpus: List[Sentence], test_ratio: float = 0.2,
                     seed: int = 0) -> Tuple[List[Sentence], List[Sentence]]:
    """Shuffle, then cut off the last `test_ratio` as held-out test data.

    WHY: measuring accuracy on sentences the model was trained on tells you
    how well it memorised, not how well it generalises. The test set is the
    only honest number.
    """
    data = list(corpus)
    random.Random(seed).shuffle(data)
    n_test = max(1, int(len(data) * test_ratio))
    return data[n_test:], data[:n_test]


if __name__ == "__main__":
    corpus = generate_corpus()
    train, test = train_test_split(corpus)
    wv, lv = build_vocabs(train)

    print(f"corpus        : {len(corpus)} sentences")
    print(f"train / test  : {len(train)} / {len(test)}")
    print(f"word vocab    : {len(wv)} entries")
    print(f"label vocab   : {len(lv)} -> {lv.itos}")
    print()
    print("--- example annotation ---")
    demo = Sentence(["the", "dog", "chased", "the", "cat"],
                    [2, 3, 0, 5, 3],
                    ["det", "nsubj", "root", "det", "obj"])
    print(demo.pretty())
    print()
    ids, heads, labels = encode(demo, wv, lv)
    print("encoded ids   :", ids)
    print("encoded heads :", heads)
    print("encoded labels:", labels)
