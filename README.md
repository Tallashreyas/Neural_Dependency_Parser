# A Neural Dependency Parser from Scratch

[![Python](https://img.shields.io/badge/Python-3.13-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.9_(CPU)-ee4c2c.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Parameters-176%2C936-brightgreen.svg)]()
[![UAS](https://img.shields.io/badge/Held--out_UAS-100.0%25-success.svg)]()
[![LAS](https://img.shields.io/badge/Held--out_LAS-100.0%25-success.svg)]()
[![Dependencies](https://img.shields.io/badge/Pretrained_Models-None_(Pure_Scratch)-orange.svg)]()

An end-to-end, graph-based **neural dependency parser** implemented completely from scratch in PyTorch (CPU) and NumPy. This repository demonstrates how a modern dependency parser works under the hood without relying on black-box libraries, pre-trained language models (no HuggingFace, no BERT/GPT), or external parser packages (no spaCy).

The parser features a **BiLSTM contextual encoder**, a **bilinear head-scoring metric**, a **dual-classifier edge relation predictor**, a **multi-task cross-entropy objective**, a **greedy argmax decoder with cycle-repair heuristics**, and an interactive terminal visualizer.

---

## Table of Contents
- [Architecture Overview](#architecture-overview)
- [Repository Structure](#repository-structure)
- [Quickstart & Usage](#quickstart--usage)
  - [1. Interactive Parsing Demo](#1-interactive-parsing-demo)
  - [2. One-Shot Sentence Parsing](#2-one-shot-sentence-parsing)
  - [3. Retraining from Scratch](#3-retraining-from-scratch)
  - [4. Inspecting Individual Modules](#4-inspecting-individual-modules)
- [How It Works: Neural Mechanics](#how-it-works-neural-mechanics)
  - [1. Contextual Encoding & Sequence Packing](#1-contextual-encoding--sequence-packing)
  - [2. Dual-Role Projections (Arc MLPs)](#2-dual-role-projections-arc-mlps)
  - [3. Bilinear Scoring with Identity Initialization](#3-bilinear-scoring-with-identity-initialization)
  - [4. Structural Masking](#4-structural-masking)
  - [5. Edge-Conditioned Relation Prediction](#5-edge-conditioned-relation-prediction)
  - [6. Multi-Task Loss with `ignore_index = -100`](#6-multi-task-loss-with-ignore_index---100)
- [Dataset & Grammar Construction](#dataset--grammar-construction)
- [Training & Convergence Results](#training--convergence-results)
- [Stress Tests & Empirical Insights](#stress-tests--empirical-insights)
- [Limitations & Production Roadmap](#limitations--production-roadmap)
- [Technical Reports & Documentation](#technical-reports--documentation)

---

## Architecture Overview

```
Token Sequence [B, T] (prepended with <root> at index 0)
        │
        ▼  nn.Embedding(vocab_size=68, emb_dim=64, padding_idx=0)
Word Vectors [B, T, 64]
        │
        ▼  nn.LSTM(input_size=64, hidden_size=96, bidirectional=True)
Contextual Representations [B, T, 192] (96 forward ⊕ 96 backward)
        ├───────────────────────────────────────┬───────────────────────────────────────
        │                                       │
        ▼  Arc MLPs (192 → 64, ReLU, Drop 0.25)  ▼  Label MLPs (192 → 48, ReLU, Drop 0.25)
  arc_dep & arc_head                      lab_dep & lab_head
        │                                       │
        ▼  Bilinear Score + Candidate Bias       ▼  Gather Head Representation [dep ; head]
  Arc Scores [B, T, T]                    Label Scores [B, T, 8]
  "Who is my governor?"                   "What is the name of this dependency?"
        │                                       │
        └───────────────────┬───────────────────┘
                            ▼
        Greedy Argmax Decoding + Cycle Repair (fix_cycles)
                            ▼
        Valid Dependency Tree G = (V, E) with Heads & Relation Labels
```

### Parameter Census (Total: 176,936)
| Module | Input Tensor | Output Tensor | Layer Configuration | Parameter Count |
|---|---|---|---|---:|
| **Embedding** | `[B, T]` (IDs) | `[B, T, 64]` | `nn.Embedding(68, 64, padding_idx=0)` | 4,352 |
| **BiLSTM** | `[B, T, 64]` | `[B, T, 192]` | `nn.LSTM(64, 96, bidirectional=True, batch_first=True)` | 124,416 |
| **arc_dep** | `[B, T, 192]` | `[B, T, 64]` | `Linear(192 → 64) + ReLU + Dropout(0.25)` | 12,352 |
| **arc_head** | `[B, T, 192]` | `[B, T, 64]` | `Linear(192 → 64) + ReLU + Dropout(0.25)` | 12,352 |
| **W_arc** | `[B, T, 64]` | `[B, T, 64]` | Bilinear parameter matrix ($64 \times 64$, init $I$) | 4,096 |
| **head_bias** | `[B, T, 64]` | `[B, T, 1]` | `Linear(64 → 1, bias=False)` | 64 |
| **lab_dep** | `[B, T, 192]` | `[B, T, 48]` | `Linear(192 → 48) + ReLU + Dropout(0.25)` | 9,264 |
| **lab_head** | `[B, T, 192]` | `[B, T, 48]` | `Linear(192 → 48) + ReLU + Dropout(0.25)` | 9,264 |
| **label_out** | `[B, T, 96]` | `[B, T, 8]` | `Linear(96 → 8)` on concatenated edge pairs | 776 |
| **Total** | | | | **176,936** |

---

## Repository Structure

```
dependency_neural_parser/
├── dataset.py                          # Lexicon, 27 templates, Sentence class, vocabularies, batching
├── model.py                            # NeuralDependencyParser, bilinear scorer, loss function
├── train.py                            # Training loop, Adam optimizer, UAS/LAS evaluation, checkpointing
├── parser.py                           # Greedy decoder, cycle repair heuristic, 3 visualizations
├── main.py                             # Interactive CLI prompt, demo sentences, stress tests
├── model.pt                            # Trained PyTorch model checkpoint (~715 KB)
├── Neural_Dependency_Parser_Report.pdf  # Comprehensive 10-page technical & viva reference report
├── REPORT.md                           # Detailed architectural write-up and failure analysis
└── README.md                           # Repository documentation (this file)
```

---

## Quickstart & Usage

### Prerequisites
* Python 3.10+ (tested on Python 3.13)
* PyTorch 2.0+ (CPU is sufficient; no GPU required)
* NumPy

```bash
pip install torch numpy
```

### 1. Interactive Parsing Demo
Run the main script to parse built-in demo sentences, evaluate out-of-distribution stress tests, and open an interactive prompt:

```bash
python main.py
```

```
==============================================================
DEMO: sentences the grammar covers
==============================================================

=== "The boy kicked the ball" ===

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

### 2. One-Shot Sentence Parsing
Parse any custom sentence directly from the command line:

```bash
python main.py "A small dog quickly chased the angry cat in the garden"
```

### 3. Retraining from Scratch
To train the model from scratch on CPU (~25 seconds for 60 epochs):

```bash
python main.py --retrain
# or directly:
python train.py
```

### 4. Inspecting Individual Modules
Each file runs standalone for inspection and shape sanity checking:

```bash
# Print example treebank annotation and token/head/label encodings
python dataset.py

# Verify tensor dimensions at every layer of the forward pass
python model.py
```

---

## How It Works: Neural Mechanics

### 1. Contextual Encoding & Sequence Packing
* Word IDs are projected into 64-dimensional dense vectors: $X \in \mathbb{R}^{B \times T \times 64}$.
* `pack_padded_sequence` passes only active tokens to the BiLSTM, ensuring padding tokens never contaminate recurrent states across sentences of different lengths.
* The forward LSTM reads left-to-right; the backward LSTM reads right-to-left. Concatenation produces a 192-dimensional contextual representation $h_i = [\vec{h}_i ; \overleftarrow{h}_i]$.

### 2. Dual-Role Projections (Arc MLPs)
In syntax, words play two fundamentally distinct roles:
1. **As a dependent** searching for a parent ($d_i$).
2. **As a candidate head** offering itself as a parent ($e_j$).

Conflating these two roles into a single representation would force the network to compromise. Therefore, separate projections are used:
$$d_i = \text{ReLU}(W_{\text{arc\_dep}} h_i + b_{\text{arc\_dep}}) \in \mathbb{R}^{64}$$
$$e_j = \text{ReLU}(W_{\text{arc\_head}} h_j + b_{\text{arc\_head}}) \in \mathbb{R}^{64}$$

### 3. Bilinear Scoring with Identity Initialization
Head candidate scores are calculated using a learned bilinear metric plus a candidate head bias:
$$\text{Score}(i \leftarrow j) = d_i^T W_{\text{arc}} e_j + u^T e_j$$
* **Why Bilinear:** A bare dot product $d_i^T e_j$ is symmetric. Syntactic dependencies are strictly asymmetric (verbs govern nouns, but nouns do not govern verbs). The learned square matrix $W_{\text{arc}} \in \mathbb{R}^{64 \times 64}$ models this asymmetric relationship.
* **Identity Initialization:** Initializing $W_{\text{arc}} = I$ allows training to start with standard cosine-like similarity before learning asymmetric distortions.
* **Head Bias ($u^T e_j$):** Acts as a prior indicating how likely a token is to be a head in general (e.g., finite main verbs attract multiple dependents).

### 4. Structural Masking
Before computing softmax:
* **Padding Mask:** Padding cells ($j \ge \text{length}$) are assigned $-10^9$.
* **Self-Loop Mask:** Words cannot govern themselves ($i = j$). Diagonal entries are assigned $-10^9$.
Softmax over rows turns scores into valid probability distributions with exact zero probability on masked cells without NaN issues.

### 5. Edge-Conditioned Relation Prediction
Relation labels characterize edges, not isolated words. The classifier gathers the head's representation down to the dependent's row and classifies the concatenated pair:
$$\text{Pair}_i = [d_i^{\text{lab}} ; e_{h_i}^{\text{lab}}] \in \mathbb{R}^{96} \xrightarrow{\text{Linear}} \mathbb{R}^8$$
* **Teacher Forcing:** During training, $h_i$ is set to the **gold head** to prevent early noisy predictions from destabilizing label learning.

### 6. Multi-Task Loss with `ignore_index = -100`
$$\mathcal{L} = \text{CrossEntropy}(\text{arc\_scores}, \text{heads}^*) + \text{CrossEntropy}(\text{label\_scores}, \text{labels}^*)$$
* The sentinel `<root>` node at index 0 and all padding positions carry target $-100$, directing PyTorch to mask them from loss calculation and gradient updates.

---

## Dataset & Grammar Construction

Instead of using a noisy external corpus, [dataset.py](file:///c:/Users/TALLA%20SHREYAS/OneDrive/Desktop/nlp/dependency_neural_parser/dataset.py) programmatically synthesizes a mathematically clean 200-sentence treebank based on 27 templates across 8 lexical categories:

* **Lexicon:** Determiners (`the, a`), Adjectives (`small, big, angry...`), Nouns (`dog, cat, boy, garden...`), Transitive Verbs (`chased, kicked...`), Intransitive Verbs (`barked, sang...`), Adverbs (`quickly, loudly...`), Proper Names (`John, Mary...`), and Prepositions (`in, on, under...`).
* **Zero Leakage:** Shuffled with a fixed seed and split into **160 train (80%)** and **40 test (20%)**. The vocabulary is built **exclusively from the training set**; unseen words in the test set map to `<unk>` (ID 1).
* **Physical `<root>` Token Prepending:** `<root>` (ID 2) is explicitly prepended at index 0. This makes predicting the root verb of a sentence uniform with all other word attachments (the main verb simply selects head 0).

---

## Training & Convergence Results

Trained on CPU with Adam ($\text{lr} = 2 \times 10^{-3}$), batch size 16, dropout 0.25, and gradient clipping norm 5.0.

```
epoch   loss    head    label   train UAS  test UAS  test LAS
--------------------------------------------------------------
1       3.703   1.790   1.913   36.0%      30.2%     12.2%
5       0.983   0.567   0.416   91.9%      87.8%     85.1%
10      0.157   0.113   0.044   99.8%      98.9%     97.3%
15      0.094   0.079   0.015   100.0%     99.6%     99.6%
25      0.014   0.010   0.004   100.0%     100.0%    100.0%
60      0.004   0.003   0.001   100.0%     100.0%    100.0%
```

```
training loss curve (epochs 1..60)
  3.70 |*                         
  3.37 |                          
  3.03 |                          
  2.69 |                          
  2.36 |                          
  2.02 |                          
  1.69 |                          
  1.35 |                          
  1.01 |  *                       
  0.68 |                          
  0.34 |                          
  0.00 |    * * * * * * * * * * * 
       +--------------------------
        1 5 1015202530354045505560
```

> [!NOTE]
> **Interpreting 100% Accuracy:** The 100% UAS/LAS score confirms that the model has completely converged and generalized over the underlying 27 templates. The most informative evaluation comes from out-of-distribution stress tests.

---

## Stress Tests & Empirical Insights

### 1. Out-of-Vocabulary Generalization (Success)
* **Sentence:** *"The zebra devoured the pizza"*
* **Behavior:** All three content words (*zebra*, *devoured*, *pizza*) are out-of-vocabulary and map to `<unk>`. The model sees `[The, <unk>, <unk>, the, <unk>]`.
* **Result:** **100% correct parse!**
  * `devoured` $\rightarrow$ `ROOT` (`root`)
  * `zebra` $\rightarrow$ `devoured` (`nsubj`)
  * `pizza` $\rightarrow$ `devoured` (`obj`)
* **Insight:** The BiLSTM encodes positional syntactic shape (`DET ? ? DET ?`), deducing that position 3 acts as a transitive verb flanked by subject and object noun phrases.

### 2. Coordination Stress Test (Failure)
* **Sentence:** *"The dog chased the cat and the mouse"*
* **Failure:** Outputs **two roots**: `chased (root)` and `and (root)`.
* **Reason:** Coordination never appeared in training templates. Greedy argmax decoding chooses heads independently, with no global constraint enforcing a single root.

### 3. Relative Clause Stress Test (Failure)
* **Sentence:** *"The dog that barked chased the cat"*
* **Failure:** Flattens the embedded clause by misattaching `that` and `barked` directly to `chased`.
* **Reason:** Subordinate clauses are outside the grammar; the BiLSTM attaches verbs to the primary predicate.

---

## Limitations & Production Roadmap

| Current Prototype Limitation | How Production Parsers (e.g., Stanza, spaCy) Solve It |
|---|---|
| **Multiple roots & cycles from greedy decoding** | **Chu-Liu/Edmonds Maximum Spanning Tree (MST):** Finds the optimal tree in $O(V^2)$ with a single root and zero cycles. |
| **Small synthetic grammar (27 templates)** | **Universal Dependencies (UD) English-EWT Treebank:** 12,543 real sentences across 37 relation labels (realistic 88–92% LAS). |
| **Unseen word inflections map to `<unk>`** | **Subword Tokenization (BPE/WordPiece) / Char-CNN:** Captures morphological affixes (`-ed`, `-ing`, `-ly`). |
| **No POS knowledge** | **POS Embeddings:** Concatenating learned Part-of-Speech tags with word vectors (+2 to +4 LAS points). |
| **Bilinear score asymmetry only** | **Full Biaffine Attention (Dozat & Manning, 2017):** Includes dependent bias and scalar bias: $d_i^T W e_j + u_1^T e_j + u_2^T d_i + b$. |

---

## Technical Reports & Documentation

For in-depth analysis, mathematical proofs, viva defense questions, and extended case studies, refer to:
* 📄 **[Neural_Dependency_Parser_Report.pdf](Neural_Dependency_Parser_Report.pdf):** 10-page comprehensive technical reference document.
* 📝 **[REPORT.md](REPORT.md):** Complete markdown report covering architecture, design decisions, and failure analysis.
