"""
For Better Embedding

[TOKEN CLASS]
NOTE SEQ
0 : PAD
1 : EOS
2 : BAR
3 : PICTH
4 : VELOCITY
5 : CHORD
6 : DURATION
7 : POSITION
META
8 : BPM
9 : KEY
10 : TIME SIGNATURE
11 : PTICH RANGE
12 : NUM OF MEASURE
13 : INSTRUMENT
14 : GENRE
15 : META VELOCITY
16 : TRACK ROLE
17 : RHYTHM
"""

import torch.nn as nn


class Classifier(nn.Module):

    def __init__(self, hidden_dim, num_cls=18):
        super().__init__()
        self.classifier_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_cls)
        )

    def forward(self, x):
        return self.classifier_layer(x)
