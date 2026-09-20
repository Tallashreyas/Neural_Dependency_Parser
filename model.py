r"""
model.py
========
STEPS 3-6: embeddings, BiLSTM, head scoring, relation scoring.

The whole network in one picture, for the sentence "the dog chased the cat"
(n = 5 words, so T = 6 positions once we prepend <root>):

    word ids                [B, T]            integers
        |  nn.Embedding
        v
    word vectors            [B, T, 64]        each word -> a 64-dim vector
        |  nn.LSTM(bidirectional=True)
        v
    contextual vectors      [B, T, 192]       96 forward + 96 backward
        |                          \
        | arc MLPs                  \ label MLPs
        v                            v
    arc scores [B, T, T]        label scores [B, T, n_labels]
    "who is my head?"           "what is the name of that link?"

B = batch size, T = padded sequence length (including <root>).
"""

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class NeuralDependencyParser(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_labels: int,
        emb_dim: int = 64,
        lstm_hidden: int = 96,
        arc_dim: int = 64,
        label_dim: int = 48,
        dropout: float = 0.25,
        pad_id: int = 0,
    ):
        super().__init__()
        self.pad_id = pad_id

        # ------------------------------------------------------------------
        # 1. EMBEDDINGS  --  "what does this word mean, out of context?"
        # ------------------------------------------------------------------
        # A lookup table of shape [vocab_size, emb_dim]. Row 17 is the vector
        # for word id 17. These numbers start random and are LEARNED by
        # backpropagation like any other weight.
        #
        # WHY not one-hot vectors? A one-hot vector for "dog" is 68 numbers,
        # 67 of which are zero, and it is exactly as far from "cat" as it is
        # from "quickly". The network would have to learn every word from
        # scratch. A dense 64-dim vector lets the model place "dog" and "cat"
        # near each other, so anything it learns about "dog" transfers to
        # "cat" for free. That is the single biggest reason embeddings exist.
        #
        # padding_idx=pad_id keeps the <pad> vector pinned at all zeros and
        # frozen (it receives no gradient) -- padding must never influence
        # anything.
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_id)

        # ------------------------------------------------------------------
        # 2. BiLSTM  --  "what does this word mean, IN this sentence?"
        # ------------------------------------------------------------------
        # The forward LSTM reads left-to-right and gives word i a summary of
        # everything to its left; the backward LSTM reads right-to-left and
        # gives it everything to its right. Concatenating both means the
        # vector for "chased" knows a noun phrase sits on its left and another
        # sits on its right -- which is precisely the evidence needed to
        # decide "I am the root, and those two are my subject and object".
        #
        # Shapes:  in  [B, T, emb_dim=64]
        #          out [B, T, 2*lstm_hidden=192]
        self.lstm = nn.LSTM(
            input_size=emb_dim,
            hidden_size=lstm_hidden,
            num_layers=1,
            bidirectional=True,
            batch_first=True,
        )
        lstm_out_dim = 2 * lstm_hidden

        self.emb_dropout = nn.Dropout(dropout)
        self.lstm_dropout = nn.Dropout(dropout)

        # ------------------------------------------------------------------
        # 3. ARC (head-prediction) MLPs
        # ------------------------------------------------------------------
        # Every word plays TWO different roles in a tree:
        #   - as a dependent, looking for a parent   ("dog" looking for "chased")
        #   - as a head, offering itself as a parent ("chased" offering itself)
        # The information needed for the two roles is different, so we project
        # the same BiLSTM vector through two different small networks.
        # This is why we have arc_dep and arc_head instead of one projection.
        #
        # Shapes: [B, T, 192] -> [B, T, arc_dim=64]
        self.arc_dep = self._mlp(lstm_out_dim, arc_dim, dropout)
        self.arc_head = self._mlp(lstm_out_dim, arc_dim, dropout)

        # The bilinear scoring matrix W:  score(i<-j) = dep_i . W . head_j
        # Initialised to the identity matrix, so that at the very start the
        # score is a plain dot product dep_i . head_j (similarity). Training
        # then bends W into whatever asymmetric compatibility function the
        # data actually needs -- "nouns like verbs as heads" is not a
        # symmetric relation, and a bare dot product cannot express it.
        self.W_arc = nn.Parameter(torch.eye(arc_dim))

        # A per-candidate bias: some words are good heads no matter who is
        # asking (a main verb attracts many dependents). This term depends
        # only on j, not on i.
        self.head_bias = nn.Linear(arc_dim, 1, bias=False)

        # ------------------------------------------------------------------
        # 4. LABEL (relation-prediction) MLPs
        # ------------------------------------------------------------------
        # Once we know dog -> chased, we still must name the edge: nsubj.
        # We feed the classifier BOTH endpoints, because the label is a
        # property of the pair, not of either word alone. ("dog" is nsubj in
        # one sentence and obj in another; only the pairing tells you which.)
        self.lab_dep = self._mlp(lstm_out_dim, label_dim, dropout)
        self.lab_head = self._mlp(lstm_out_dim, label_dim, dropout)
        self.label_out = nn.Linear(2 * label_dim, n_labels)

    @staticmethod
    def _mlp(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
        """A one-hidden-layer projection: Linear -> ReLU -> Dropout.

        WHY not use the BiLSTM output directly? The BiLSTM vector carries
        everything about the word (its identity, its neighbours, its position).
        These small nets let the model squeeze out just the part relevant to
        one question, and they add a non-linearity so the head score is not a
        purely linear function of the LSTM states.
        """
        return nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout))

    # ----------------------------------------------------------------------
    # Forward pass, split into three readable pieces.
    # ----------------------------------------------------------------------
    def encode(self, word_ids: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """ids -> contextual word representations.

        word_ids : [B, T] long
        lengths  : [B]    long, real length of each sentence (incl. <root>)
        returns  : [B, T, 2*lstm_hidden]
        """
        x = self.embedding(word_ids)          # [B, T, emb_dim]
        x = self.emb_dropout(x)

        # pack/unpack tells the LSTM where each sentence really ends, so the
        # padding positions never update the recurrent state. Without this,
        # a short sentence in a long batch would have its final states
        # overwritten by a run of <pad> tokens.
        packed = pack_padded_sequence(
            x, lengths.cpu(), batch_first=True, enforce_sorted=False
        )
        out, _ = self.lstm(packed)
        H, _ = pad_packed_sequence(out, batch_first=True, total_length=word_ids.size(1))
        return self.lstm_dropout(H)           # [B, T, 192]

    def arc_scores(self, H: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        r"""The heart of head prediction.

        H       : [B, T, 192]
        returns : [B, T, T]   where  scores[b, i, j] = "how good is word j
                              as the head of word i, in sentence b?"

        The maths, for one sentence and one pair (i = dog, j = chased):

            d_i = ReLU(W_dep  . h_dog    + b_dep)      64 numbers
            e_j = ReLU(W_head . h_chased + b_head)     64 numbers
            score(dog <- chased) = d_i^T  W  e_j  +  u^T e_j
                                   \_________/     \______/
                                   compatibility    "is chased
                                   of this pair     a good head
                                                    in general?"

        Row i of the output is therefore a vector of n+1 scores -- one per
        candidate head, including ROOT at index 0. Softmax over that row turns
        it into a probability distribution, and argmax picks the head.
        """
        B, T, _ = H.shape
        dep = self.arc_dep(H)                          # [B, T, arc_dim]
        head = self.arc_head(H)                        # [B, T, arc_dim]

        # (dep @ W) : [B, T, a] ; then batched matmul with head^T : [B, a, T]
        scores = torch.bmm(dep @ self.W_arc, head.transpose(1, 2))   # [B, T, T]
        scores = scores + self.head_bias(head).transpose(1, 2)       # + [B, 1, T]

        # ---- masking: kill impossible candidate heads -------------------
        # (a) padding positions are not words, so they can never be heads.
        # (b) a word cannot be its own head (that would be a 1-node cycle).
        # -1e9 instead of -inf: after softmax it is effectively zero
        # probability, but it never produces NaNs.
        idx = torch.arange(T, device=H.device)
        is_real = idx.unsqueeze(0) < lengths.unsqueeze(1)            # [B, T]
        scores = scores.masked_fill(~is_real.unsqueeze(1), -1e9)
        self_loop = torch.eye(T, dtype=torch.bool, device=H.device)
        scores = scores.masked_fill(self_loop.unsqueeze(0), -1e9)
        return scores

    def label_scores(self, H: torch.Tensor, heads: torch.Tensor) -> torch.Tensor:
        """Name each edge, given who the head is.

        H       : [B, T, 192]
        heads   : [B, T] long -- the head index chosen for each word
                  (GOLD heads while training, PREDICTED heads at test time)
        returns : [B, T, n_labels]

        We gather the head's representation down to the dependent's row, so
        row i holds [dependent_i ; head_of_i] and the classifier sees the pair.
        """
        dep = self.lab_dep(H)                          # [B, T, label_dim]
        hd = self.lab_head(H)                          # [B, T, label_dim]

        # clamp(min=0): ignored rows carry -100, which would crash gather.
        # Their output is discarded by the loss anyway.
        idx = heads.clamp(min=0).unsqueeze(-1).expand(-1, -1, hd.size(-1))
        head_repr = hd.gather(1, idx)                  # [B, T, label_dim]

        pair = torch.cat([dep, head_repr], dim=-1)     # [B, T, 2*label_dim]
        return self.label_out(pair)                    # [B, T, n_labels]

    def forward(self, word_ids: torch.Tensor, lengths: torch.Tensor,
                heads: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Training-time convenience wrapper. `heads` are the GOLD heads.

        Feeding gold heads to the label classifier during training is called
        TEACHER FORCING. We do it because otherwise, early in training when
        head prediction is random, the label classifier would be learning to
        name edges that do not exist -- pure noise. The mismatch with test
        time (where we must use predicted heads) is a known, accepted
        simplification, and the same one the standard biaffine parser makes.
        """
        H = self.encode(word_ids, lengths)
        return self.arc_scores(H, lengths), self.label_scores(H, heads)


def parser_loss(arc_scores: torch.Tensor, label_scores: torch.Tensor,
                gold_heads: torch.Tensor, gold_labels: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """The training objective: two classification problems added together.

    arc_scores   [B, T, T]         gold_heads  [B, T]   (-100 = ignore)
    label_scores [B, T, n_labels]  gold_labels [B, T]   (-100 = ignore)

    HEAD LOSS. For each word we have T scores, one per candidate head. Softmax
    turns them into probabilities; cross-entropy is  -log P(correct head).
    If the model gives "chased" probability 0.9 as dog's head, the loss for
    that word is -log(0.9) = 0.105 (small, little to fix). If it gives it
    0.01, the loss is 4.6 (large, big gradient). So the loss literally
    measures "how surprised was the model by the right answer".

    LABEL LOSS. Identical idea over the 8 relation names.

    We just add them (equal weight). Minimising the sum minimises both.

    `ignore_index=-100` skips the <root> row and all padding rows, so they
    contribute neither loss nor gradient.
    """
    n_cand = arc_scores.size(-1)
    n_lab = label_scores.size(-1)
    head_loss = F.cross_entropy(
        arc_scores.reshape(-1, n_cand), gold_heads.reshape(-1), ignore_index=-100
    )
    label_loss = F.cross_entropy(
        label_scores.reshape(-1, n_lab), gold_labels.reshape(-1), ignore_index=-100
    )
    return head_loss + label_loss, head_loss, label_loss


if __name__ == "__main__":
    # A shape sanity-check you can run on its own: python model.py
    torch.manual_seed(0)
    B, T, V, L = 2, 6, 68, 8
    model = NeuralDependencyParser(vocab_size=V, n_labels=L)

    word_ids = torch.randint(3, V, (B, T))
    lengths = torch.tensor([6, 4])
    word_ids[1, 4:] = 0                       # pad the shorter sentence
    heads = torch.full((B, T), -100)
    heads[0, 1:] = torch.tensor([2, 3, 0, 5, 3])      # the dog chased the cat
    heads[1, 1:4] = torch.tensor([2, 3, 0])           # the dog barked

    H = model.encode(word_ids, lengths)
    arcs = model.arc_scores(H, lengths)
    labs = model.label_scores(H, heads)

    print("word_ids    ", tuple(word_ids.shape), "[B, T]")
    print("embeddings  ", tuple(model.embedding(word_ids).shape), "[B, T, emb_dim]")
    print("BiLSTM out  ", tuple(H.shape), "[B, T, 2*hidden]")
    print("arc scores  ", tuple(arcs.shape), "[B, T, T]  (row i = scores over candidate heads)")
    print("label scores", tuple(labs.shape), "[B, T, n_labels]")
    print()
    print("padded candidates are masked out for sentence 1:")
    print(arcs[1, 1].tolist())
