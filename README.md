![bsfda](https://github.com/user-attachments/assets/28ee48f2-2e53-4e6f-92c9-d0d2cdbc367b)

This repository contains the implementation of BSFDA (Bayesian Scalable Functional Data Analysis) proposed in the paper "Integrated Model Selection and Scalability in Functional Data Analysis through Bayesian Learning". It handles irregularly sampled curves and 4-D spatio-temporal fields while automatically choosing the number of principal components and kernel bases.

* **Active-subspace variational inference** — scales linearly in number of observations.
* **Automatic rank & basis selection** via sparsity-promoting priors.
* Handles **1-D to 4-D functional domains** without gridding.

## Installation

The packages can be installed using conda and pip through the following commands:

```conda create -n {a_new_environment_name} python=3.9```

```conda activate {a_new_environment_name}```

```pip install -r requirements.txt```


## Example
****
The following code shows how to run the proposed method on the simulated data. It typically takes about 10 minutes to run on a laptop.

```python -u ./code/example.py```

## Reproduce the results in the paper

The commands to reproduce the results in the paper are in the file ```./code/cmd.sh```.

## Cite

If you use BSFDA in academic work, please cite the paper:

Tao, Wenzheng, Sarang Joshi, and Ross Whitaker. 2025. "Integrated Model Selection and Scalability in Functional Data Analysis Through Bayesian Learning" Algorithms 18, no. 5: 254. https://doi.org/10.3390/a18050254

BibTeX:

@article{tao2025integrated,
   author = {Tao, Wenzheng and Joshi, Sarang and Whitaker, Ross},
   title = {Integrated Model Selection and Scalability in Functional Data Analysis Through Bayesian Learning},
   journal = {Algorithms},
   volume = {18},
   number = {5},
   ISSN = {1999-4893},
   DOI = {10.3390/a18050254},
   url = {https://www.mdpi.com/1999-4893/18/5/254/pdf},
   year = {2025},
   publisher={MDPI},
   type = {Journal Article}
}
