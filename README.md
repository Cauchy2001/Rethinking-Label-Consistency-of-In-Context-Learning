#  Data Synthesis 

![chart](https://github.com/Cauchy2001/TopK-SD_for_ICL/blob/main/chart.png)

## Overview

This code is for the paper _Rethinking Label Consistency of In-Context Learning: An Implicit Transductive Label Propagation Perspective_. Our code is based on the <a href="https://github.com/Shark-NLP/OpenICL/tree/main">OpenICL repository</a> and <a href="https://github.com/Romainpkq/revisit_demon_selection_in_ICL">revisit_demon_selection_in_ICL repository</a>.

The reference works and related projects are as follows:

[OpenICL: An Open-Source Framework for In-context Learning](https://arxiv.org/abs/2303.02913)

## Installation
Note: OpenICL requires Python 3.8+
**Installation for local development:**
```
git clone https://github.com/Cauchy2001/TopK-SD_for_ICL.git

cd TopK-SD_for_ICL
pip install -e .
```

## Examples
Following example shows you how to perform ICL on sentiment classification dataset.  More examples and tutorials can be found at [examples](https://github.com/Shark-NLP/OpenICL/tree/main/examples)
Our code is placed under the "exp" folder, and "run_classfication.py" is the code for the classification experiment.
```python

cd exp

CUDA_VISIBLE_DEVICES=0 python run_classification.py
```

