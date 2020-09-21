# Simulator of single-cell profiles with two modalities

We design a simulation pipeline to generate cell profiles where ground truth subpupulations are known. We simulated the situation for which only a proportion of true cluster identities could be observed from each modality, but all clusters could be discriminated using both modalities. To accomplish this, we divided the true clusters into two non-overlapping groups that were each assigned to one of the two modalities. Then, in each group, clusters were merged with a probability `p` providing observed cluster labels `l_x`,`l_y` for the two modalities. For detailed information, refer to the `Methods` section in manuscript.

To run the simulator, 

```python
from __future__ import print_function
import numpy as np
import random
from copy import deepcopy

data = simulation.multi_modal_simulator(n_clusters, n,
                                        d_1, d_2,
                                        k,
                                        sigma_1, sigma_2,
                                        decay_coef_1, decay_coef_2,
                                        merge_prob)

data_a = data['data_a_dropout']
data_b = data['data_b_dropout']
label_a = data['data_a_label']
label_b = data['data_b_label']
label_true = data['true_cluster']

```
where

```
Parameters:

  n_clusters:       number of ground truth clusters.
  n:                number of cells to simulate.
  d_1:              dimension of features for transcript modality.
  d_2:              dimension of features for morphological modality.
  k:                dimension of latent code to generate simulate data (for both modality)
  sigma_1:          variance of gaussian noise for transcript modality.
  sigma_2:          variance of gaussian noise for morphological modality.
  decay_coef_1:     decay coefficient of dropout rate for transcript modality.
  decay_coef_2:     decay coefficient of dropout rate for morphological modality.
  merge_prob:       probability to merge neighbor clusters for the generation of modality-specific
                    clusters (same for both modalities)


Outputs:

  a dataframe with keys as follows

  'true_cluster':   true cell clusters, a vector of length n

  'data_a_full':    feature matrix of transcript without dropouts
  'data_a_dropout': feature matrix of transcript with dropouts
  'data_a_label':   cluster labels to generate transcript features after merging

  'data_b_full':    feature matrix of morphology without dropouts
  'data_b_dropout': feature matrix of morphology with dropouts
  'data_b_label':   cluster labels to generate morphological features after merging
```



## Copyright
Software provided as is under **MIT License**.

Copyright (c) 2020 Altschuler and Wu Lab

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

