# A Neural Dependency Parser from Scratch

**Course:** Natural Language Processing
**Implementation:** Python 3.13 + PyTorch 2.9 (CPU), NumPy. No pretrained models, no BERT/GPT, no spaCy parser.

```bash
python main.py
```

---

## 1. Introduction

Dependency parsing is the task of discovering the grammatical skeleton of a
sentence: which word depends on which other word, and what the nature of that
dependency is. Unlike a phrase-structure (constituency) tree, which groups
words into nested boxes like NP and VP, a dependency tree connects words
**directly to other words**. For a sentence of *n* words the answer is always
*n* arrows, which makes the representation compact and makes the learning
problem unusually clean.

This project implements such a parser end-to-end in about 700 lines of
commented Python: a hand-designed toy treebank, a word-embedding layer, a
BiLSTM encoder, a bilinear head-scoring function, a relation classifier, a
training loop, a decoder and a tree visualiser. Every component is written
explicitly rather than imported, because the goal is understanding the
mechanism, not maximising a score.

## 2. Problem Statement

Given a sentence *w₁ … wₙ*, produce for every word *wᵢ*:

1. a **head** *h(i) ∈ {0, 1, …, n}*, where 0 denotes the artificial ROOT node, and
2. a **relation label** *l(i)* drawn from a fixed inventory (`nsubj`, `obj`,
   `det`, `amod`, `advmod`, `case`, `nmod`, `root`),

such that the resulting set of arrows forms a single rooted tree.

For `The dog chased the cat` the target is:

| idx | word | head | label |
|-----|------|------|-------|
| 1 | The | 2 (dog) | det |
| 2 | dog | 3 (chased) | nsubj |
| 3 | chased | 0 (ROOT) | root |
| 4 | the | 5 (cat) | det |
| 5 | cat | 3 (chased) | obj |

## 3. Objective

* Understand and implement head prediction as an *n+1*-way classification per word.
* Understand and implement relation prediction as a separate classification over the chosen edge.
* Show why contextual encoding (a BiLSTM) is required and what it contributes.
* Train with cross-entropy and backpropagation, and evaluate honestly with UAS/LAS on a held-out split.
* Identify precisely where this simplified design breaks, and name the standard techniques that fix it.

## 4. Dependency Parsing Intuition

**Head and dependent.** In every syntactic relation one word is the *head* —
the word that carries the core meaning and determines the grammatical type of
the phrase — and the other is the *dependent*, which modifies or completes it.
In *the dog*, `dog` is the head and `the` is the dependent: you can say *the dog
barked* or *dogs barked*, but not *the barked*. The head is the survivor.

**The tree.** Chaining these local head–dependent decisions produces a tree. In
our running example, `The → dog → chased → ROOT` and `the → cat → chased →
ROOT`. Exactly one word (the main verb `chased`) attaches to ROOT, every other
word has exactly one head, and no arrows form a loop. Those three conditions
are what makes the structure a tree rather than an arbitrary graph.

**Why not rules?** A rule such as "a determiner attaches to the noun on its
right" survives *the dog* but not *the very old dog*, and *"a noun before a verb
is the subject"* dies on *the cat the dog chased ran away*. Real grammar is a
web of soft, interacting preferences, and coverage of hand-written rules decays
faster than the rule list grows. A statistical model learns these preferences
from data as continuous weights, degrades gracefully instead of failing
outright, and produces a confidence score for every decision.

**Why embeddings?** A neural network needs numbers. The naive encoding is
one-hot: `dog` = position 22 set to 1, the other 67 zeros. But under that
encoding `dog` is exactly as similar to `cat` as it is to `quickly` — all
distinct one-hot vectors are equidistant. Anything the network learns about
`dog` transfers to nothing. A dense 64-dimensional embedding, learned by
gradient descent, lets the model place `dog`, `cat` and `bird` close together,
so a pattern learned from one applies to the others. That generalisation is the
entire point.

**What the BiLSTM learns.** An embedding is context-free: `chased` has the same
vector in every sentence. But its syntactic role is not context-free — whether
it is the root depends on what surrounds it. The forward LSTM reads
left-to-right and hands `chased` a summary of everything before it (*the dog* —
a complete noun phrase); the backward LSTM reads right-to-left and hands it
everything after (*the cat* — another complete noun phrase). Concatenating both
gives a 192-dimensional vector that effectively encodes "I am a verb with a
noun phrase on each side," which is precisely the evidence for "I am the root,
and those two phrases are my subject and object."

**How the model decides `dog → chased`.** Each word is projected into two
different small vectors, one for its role as a dependent and one for its role
as a candidate head. The score of an arrow is a compatibility measure between
the two:

```
score(dog ← chased) = d_dog · W · e_chased  +  u · e_chased
```

`d_dog` is *what kind of head am I looking for?* `e_chased` is *what kind of
head am I?* The learned matrix `W` measures how well those two match, and the
final term is a bias for words that are good heads in general (a main verb
attracts many dependents). Computing this for all pairs gives a matrix of
scores; row `dog` looks like:

```
              ROOT    The   dog   chased   the    cat
score          1.2    0.4   -inf    6.1    -0.3   0.8
softmax       0.01   0.00     0    0.97    0.00   0.02
                                    ^ argmax -> head of "dog" is "chased"
```

Two candidates are eliminated before scoring: a word cannot be its own head,
and padding positions are not words.

**Why label prediction is separate.** Knowing `dog → chased` does not tell you
*what kind* of dependency it is. The same pair of parts of speech can be
`nsubj` or `obj` depending on position and voice. So once the head is chosen,
a second classifier reads the pair `[dog ; chased]` — both endpoints, because
the label is a property of the edge, not of either word — and picks one of the
eight relation names.

**How the loss is computed.** Head prediction is an (n+1)-way classification,
so we use cross-entropy: the loss for one word is `−log P(correct head)`. If
the model assigns `chased` a probability of 0.97, the loss is 0.03 and the
gradient is small; if it assigns 0.01, the loss is 4.6 and the gradient is
large. The loss literally measures *how surprised the model was by the right
answer*. Label prediction gets an identical cross-entropy over the eight
labels, and the two are added with equal weight.

**How backpropagation improves the parser.** `loss.backward()` applies the
chain rule backwards through the network and computes, for every one of the
176,936 parameters, the derivative of the loss with respect to that parameter —
i.e. "if I nudge this weight up, does the loss go up or down, and how fast?"
`optimizer.step()` then nudges every parameter a small distance in the
direction that lowers the loss. Crucially the gradient flows *all the way back*:
an error on `dog`'s head choice adjusts the arc matrix `W`, the projection MLPs,
the LSTM gates, **and the embeddings of `dog` and `chased` themselves`. That is
why the word vectors end up syntactically organised — nobody designed them, the
head-prediction errors shaped them.

## 5. Dataset

`dataset.py` builds a 200-sentence toy treebank in the CoNLL style (word, head
index, relation label).

* **Construction.** 27 hand-written syntactic templates carry the gold
  annotation; a small categorised lexicon (2 determiners, 10 adjectives,
  16 nouns, 12 transitive and 8 intransitive verbs, 6 adverbs, 5 names,
  6 prepositions) fills the slots. Writing 27 patterns by hand is far less
  error-prone than writing 200 annotations by hand, and the annotations are
  correct by construction.
* **Coverage.** Determiners, adjectival and adverbial modification, transitive
  and intransitive clauses, proper-name subjects and objects, and prepositional
  phrases — for example `the small dog quickly chased the angry cat in the garden`.
* **Conventions.** Universal-Dependencies style: prepositions are `case`
  dependents of their noun, not heads. Punctuation is omitted (it would add a
  `punct` label and no insight). Prepositional phrases always attach to the
  preceding **noun** (`nmod`) — a fixed convention chosen because the real
  attachment is genuinely ambiguous (see §11).
* **Split.** Shuffled with a fixed seed, 80 % train (160) / 20 % test (40).
  Vocabularies are built from the **training set only**, so test-set words the
  model never saw map to `<unk>` — no information leakage.
* **Vocabulary.** 68 entries including `<pad>`, `<unk>`, `<root>`; 8 relation labels.

## 6. Model Architecture

```
word ids                  [B, T]            integers, T includes <root>
   │  nn.Embedding(68, 64)
   ▼
word vectors              [B, T, 64]        context-free meaning
   │  nn.LSTM(64 → 96, bidirectional)
   ▼
contextual vectors        [B, T, 192]       96 forward ⊕ 96 backward
   ├──────────────────────────┬──────────────────────────
   │ arc_dep / arc_head MLPs  │ lab_dep / lab_head MLPs
   ▼                          ▼
arc scores [B, T, T]      label scores [B, T, 8]
"who is my head?"         "what is this edge called?"
```

`B` = batch size, `T` = padded length including the prepended `<root>` token.
Total trainable parameters: **176,936**.

Design decisions worth defending:

* **`<root>` is a real token** prepended to the sequence, so the BiLSTM computes
  a genuine vector for it. "Which word is the sentence root?" then becomes the
  same operation as any other head choice — position 0 is simply another
  candidate — and no special-casing is needed anywhere.
* **Two separate projections per word** (`arc_dep`, `arc_head`). A word plays two
  different roles — looking for a parent, and offering itself as one — and the
  relevant information differs, so a single shared projection would conflate them.
* **A bilinear form rather than a dot product.** `W` is initialised to the
  identity matrix, so training *starts* at a plain dot product (similarity) and
  learns from there. A bare dot product is symmetric, but "nouns like verbs as
  heads" is not a symmetric relation, so the learned `W` is needed to express it.
* **Masking, not filtering.** Impossible candidates (self-loops, padding) get a
  score of −1e9, which softmax drives to zero probability without producing NaNs.
* **Teacher forcing for labels.** During training the label classifier is fed
  the *gold* heads. Early in training, head predictions are random, and naming
  edges that do not exist would be pure noise. The train/test mismatch this
  introduces is a known and accepted simplification — the standard biaffine
  parser makes the same one.

## 7. Training Procedure

| Setting | Value | Reason |
|---|---|---|
| Optimiser | Adam, lr 2e-3 | Adaptive per-parameter step size; needs almost no tuning |
| Epochs | 60 | Loss is flat well before this |
| Batch size | 16 | 10 updates per epoch on 160 sentences |
| Dropout | 0.25 | 177k parameters on 160 sentences overfits quickly |
| Gradient clipping | max-norm 5.0 | RNNs occasionally emit one huge gradient |
| Seeds | fixed (torch, random) | So "accuracy changed" never means "data changed" |

Each step: pad a batch → forward pass → cross-entropy loss → `zero_grad()`
(PyTorch *accumulates* gradients, so skipping this pollutes the batch with the
previous one) → `backward()` → clip → `step()`.

## 8. Loss Function

```
L = CE(arc_scores, gold_heads) + CE(label_scores, gold_labels)
```

* `arc_scores` is `[B, T, T]`, flattened to `[B·T, T]`: for each word, a
  distribution over `T` candidate heads.
* `label_scores` is `[B, T, 8]`, flattened to `[B·T, 8]`.
* `ignore_index = -100` marks the `<root>` row (it has no head and no label)
  and every padding row, so they contribute neither loss nor gradient.
* The two terms are weighted equally; minimising the sum minimises both.

## 9. Results

Trained on CPU in roughly 25 seconds.

```
epoch   loss    head    label   train UAS  test UAS  test LAS
--------------------------------------------------------------
1       3.703   1.790   1.913   36.0       30.2      12.2
5       0.983   0.567   0.416   91.9       87.8      85.1
10      0.157   0.113   0.044   99.8       98.9      97.3
15      0.094   0.079   0.015   100.0      99.6      99.6
25      0.014   0.010   0.004   100.0      100.0     100.0
60      0.004   0.003   0.001   100.0      100.0     100.0
```

| Metric | Train | Test |
|---|---|---|
| UAS (head correct) | 100.0 % | 100.0 % |
| LAS (head + label correct) | 100.0 % | 100.0 % |

**These numbers must be read honestly.** 100 % is not evidence of a good
parser; it is evidence that the test set was drawn from the same 27 templates
as the training set. The model has learned the toy grammar completely, which is
exactly what it was asked to do. The informative results are the failures in
§10, on sentences drawn from outside that grammar. Note also the shape of the
learning curve: the model reaches 92 % UAS within 5 epochs (it discovers the
broad regularities — determiners attach rightwards, verbs are roots) and spends
the remaining 55 polishing.

## 10. Example Predictions

```
$ python main.py "The boy kicked the ball."

Word    Head    Relation  Conf
--------------------------------
The     boy     det       1.00
boy     kicked  nsubj     1.00
kicked  ROOT    root      1.00
the     ball    det       1.00
ball    kicked  obj       1.00

             +----------->   obj
      <------+   nsubj
                   <-----+   det
 <----+   det
             ^   root (ROOT -> kicked)
The  boy  kicked  the  ball

ROOT
`-- kicked (root)
    |-- boy (nsubj)
    |   `-- The (det)
    `-- ball (obj)
        `-- the (det)
```

A longer one, fully correct including the prepositional phrase:

```
$ python main.py "A small dog quickly chased the angry cat in the garden"

ROOT
`-- chased (root)
    |-- dog (nsubj)
    |   |-- A (det)
    |   `-- small (amod)
    |-- quickly (advmod)
    `-- cat (obj)
        |-- the (det)
        |-- angry (amod)
        `-- garden (nmod)
            |-- in (case)
            `-- the (det)
```

**Failure cases** (run automatically by `python main.py` as a stress test):

*Unknown words are handled well* — `The zebra devoured the pizza` parses
perfectly even though all three content words map to `<unk>`. The BiLSTM
recognises the *shape* `DET ? ? DET ?` from context and position alone, which
is a direct demonstration of what contextual encoding buys you.

*Unseen constructions fail* — `The dog chased the cat and the mouse` produces
**two** roots (`chased` and `and`), because coordination never appears in the
training data and greedy decoding does not enforce a single root.
`The dog that barked chased the cat` mangles the relative clause for the same
reason.

## 11. Limitations

1. **Toy grammar.** 27 templates, 68 words, no coordination, relative clauses,
   auxiliaries, negation, questions or embedded clauses. The 100 % test score
   measures coverage of that grammar, nothing more.
2. **Greedy decoding gives no tree guarantee.** Each word picks its head
   independently, so the output can contain cycles or multiple roots — as the
   coordination example demonstrates. A cycle-breaking heuristic
   (`fix_cycles`) repairs the worst cases by re-pointing the least-confident
   edge in a cycle, but it is an approximation, not a guarantee.
3. **PP-attachment ambiguity is defined away.** The dataset always attaches a
   prepositional phrase to the preceding noun. In *chased the cat in the
   garden*, whether the garden modifies the cat or the chasing is genuinely
   ambiguous and requires world knowledge; our parser can only ever produce one
   of the two readings.
4. **Teacher forcing mismatch.** Labels are trained on gold heads but predicted
   from predicted heads; a wrong head makes the label wrong too.
5. **No morphology or POS features.** The model sees whole lowercased word
   forms, so `dog` and `dogs` would be unrelated symbols, and every
   out-of-vocabulary word is the single undifferentiated `<unk>`.
6. **Projective structures only, implicitly.** Nothing in the model forbids
   crossing arcs, but nothing in the data contains them either, so
   non-projective phenomena are untested.
7. **No parameter search.** Hidden sizes, dropout and learning rate were chosen
   by convention, not by a validation sweep — there is no validation set.

## 12. Future Improvements

1. **Chu-Liu/Edmonds maximum-spanning-tree decoding.** Score every edge as we
   already do, then find the highest-scoring edge set that is provably a tree.
   This is the single highest-value upgrade and removes limitation 2 entirely.
2. **A real treebank.** Train on Universal Dependencies English-EWT (~12k
   sentences, 37 relation types) instead of templates, which would turn the
   meaningless 100 % into a meaningful number around 88–92 % LAS.
3. **Full biaffine attention** (Dozat & Manning, 2017): add the dependent-side
   bias term we omitted, so the score is `dᵀWe + u₁ᵀe + u₂ᵀd + b`.
4. **Character-level or subword embeddings** so out-of-vocabulary words get
   representations from their spelling (`-ed`, `-ing`, `-s`) instead of a single
   shared `<unk>` vector.
5. **POS tags as an extra input embedding**, concatenated with the word
   embedding — cheap and historically worth several LAS points.
6. **Deeper, wider encoder**: 3 stacked BiLSTM layers with variational dropout,
   the configuration the biaffine parser actually uses.
7. **A transition-based alternative** (arc-standard with a stack and buffer),
   which guarantees well-formedness by construction and runs in O(n) rather
   than O(n²), for comparison.
8. **Sample-weighted or curriculum training**, and a proper train/dev/test split
   with early stopping on dev.

## 13. Conclusion

A dependency parser can be built from a surprisingly small set of ideas: embed
words so that similar words share statistical strength, run a BiLSTM so every
word knows its context, score every (dependent, head) pair with a learned
bilinear form, pick the best head for each word, and name the resulting edge
with a second classifier. Two cross-entropy losses and backpropagation do the
rest, and 177k parameters learn the toy grammar to 100 % held-out accuracy in
under half a minute of CPU time.

Equally valuable is what the failures teach. The parser handles three unknown
words without difficulty — evidence that the BiLSTM genuinely encodes syntactic
context and not just memorised word identities — but breaks on coordination,
emitting two roots, which pinpoints exactly why serious parsers replace greedy
argmax with maximum-spanning-tree decoding. The gap between this project and a
production parser is not a different idea; it is a bigger treebank, a better
decoder and richer input features, all of which slot into the same skeleton.

---

## Appendix: file guide

| File | Contents |
|---|---|
| `dataset.py` | Templates, lexicon, `Sentence`, vocabularies, encoding, batching, split |
| `model.py` | Embedding → BiLSTM → arc scorer → label scorer, and the loss |
| `train.py` | Training loop, UAS/LAS evaluation, ASCII loss curve, checkpointing |
| `parser.py` | Greedy decoding, cycle repair, and the three visualisations |
| `main.py` | Demo sentences, stress tests, interactive prompt |

Each file runs standalone for inspection: `python dataset.py` prints an example
annotation and its encoding, `python model.py` prints every tensor shape.
