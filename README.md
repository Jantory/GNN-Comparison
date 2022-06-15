# GNN-Comparison
We provide code to reproduce results shown in the paper "<em>A Comparative Study on Robust Graph Neural Networks to Structural Noises</em>". The paper can be viewed [here](https://arxiv.org/pdf/2112.06070.pdf).

Our original paper compares the model performance of eight mainstream robust GNNs under consistent noise settings. The main contribution is the design of three structural noises under different granularity: local, community, and global. The noise generation methods can be found in the <em>noise.py</em> file.

For simplicity, we only give an example of how to use the noise generation module based on the GraphSAGE. The basic workflow is: perturbing a clean graph by calling the implemented noise functions; afterward, feeding the poisoned graph to a robust model.

