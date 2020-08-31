# MUSE: multi-modality structured embedding for single-cell spatial transcriptomics analysis

MUSE is a deep learning approach characterizing tissue composition through combined analysis of morphologies and transcriptional states for single-cell spatial transcriptomics data.

> Citation: 
> Characterizing tissue composition through combined analysis of single-cell morphologies and transcriptional states. Feng Bao<sup>\*</sup>, Yue Deng<sup>\*</sup>, Sen Wan, Bo Wang, Qionghai Dai<sup>\#</sup>, Steven J. Altschuler<sup>\#</sup>, Lani F. Wu<sup>\#</sup>

## Overview

![avatar](./overview.png)
<center>**Fig. 1 | An overview of the study.** (**a**) Single-cell spatial transcriptomics technologies provide cell infromation from both transcriptional and morphological modalities, which reveal cell heterogeneity from different angles. (**b**) Cell differences in the tissue can be characterized based on each single modalties. (**c**) By combining both heterogeneities from two modalities, fine-grained cell subpopulations can be revealed. </center>

Decomposing cell heterogeniety of complex biological systems is an important step to the comprehensive understanding of their organizations and mechanisms.  Morphologies are the most direct and distinguishable  features for  cell differences.  Single-cell profiling from morphologies has been a powerful and widely used tool to characterize cell identities and quantify cellular/subcellular dynamics at high temporal/spatial resolution and large scale. Complementarily, transcriptional profiles represent  cellular activities. The transcriptional differences among cells can reveal different cell states, linages and subpopulations. With the development of single-cell spatial transcriptomics, we can profile morpholigiclal and transcriptonal properties from the same cell simutaneously. We developed the multi-modality structural embedding (MUSE), a deep learning approach that aggregates the heterogeneity from morphologies and transcripts and dissects cell subpopulations at finer resolution.

![avatar](./method.png)
<center>**Fig. 2 | The model architecture of MUSE**</center>

MUSE combines features from transcripts (x) and morphology (y) into a joint latent representation z.The self-reconstruction loss encourages the learned joint feature representation (z) to faithfully retain information from the original individual input feature modalities (x and y). The self-supervised learning exploits triple-loss functions to encourage cells with the same cluster label (i.e. with the same pseudo label in either l<sub>x</sub> or l<sub>y</sub>) to remain close—and cells with different cluster labels to remain far apart—in the joint latent space.

## MUSE software package

DAK requires the following packages for installation:

- Python >= 3.6
- TensorFlow-GPU >= 1.4.1
- (TensorFlow >= 1.4.1 if only use CPU) 
- Numpy >= 1.13.0
- Scipy >= 1.0.0
- phenograph >=


## Copyright
Software provided as is under MIT License.

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

