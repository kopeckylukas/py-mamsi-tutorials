![MAMSI_tutorials_logo](MAMSI_tutorials_logo.png)

[![License](https://img.shields.io/badge/License-BSD_3--Clause-blue.svg)](LICENSE)
[![Docs](https://img.shields.io/badge/docs-available-brightgreen.svg)](https://kopeckylukas.github.io/py-mamsi/)
[![PyPI version](https://img.shields.io/pypi/v/mamsi.svg)](https://pypi.org/project/mamsi/)
[![DOI](https://zenodo.org/badge/823594568.svg)](https://zenodo.org/doi/10.5281/zenodo.13619607)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1wITfnDgGnJt6yJNur8E_ti6vqSQxpWb6#scrollTo=yMIY5stCq4JN)



The **MAMSI Tutorials** is a tutorial repository for the [MAMSI project](https://github.com/kopeckylukas/py-mamsi/tree/main). It provides a quickstart guide for integrating multiple multi-assay liquid chromatography – mass spectrometry (LC-MS) metabolomics datasets. 

# Content

| File                                                               | Description                                          |
| ----------------------------------------------------------------   | ---------------------------------------------------- | 
| [tutorials/classification.ipynb](tutorials/classification.ipynb)   | An easy dive to integrative classification analysis  |
| [tutorials/regression.ipynb](tutorials/regression.ipynb)           | An easy dive to integrative regression analysis      |
| [sample_code/permtest_hpc/](sample_code/permtest_hpc/)             | A guide to perform permutation testing on a computer cluster </br> (the configuration provided is relevant for Imperial HPC) |

# Quickstart
To start learning from the **MAMSI Tutorials** repository, you need to install the MAMSI package first. To do so, you can visit the MAMSI project [repository](https://github.com/kopeckylukas/py-mamsi/tree/main) or [documentation](https://kopeckylukas.github.io/py-mamsi/) for more information or install it using following commands: 

## Installation
### Installing with Pip

You can install MAMSI from PyPI using pip: 
```bash
pip install mamsi
```

For more information visit [MAMSI docs](https://kopeckylukas.github.io/py-mamsi/).

##

You can clone this repository and use the tutorials provided in the form of Jupyter notebooks (listed above), or you can follow this quickstart guide.

**Load Packages**
```python 
from mamsi.mamsi_pls import MamsiPls
from mamsi.mamsi_struct_search import MamsiStructSearch
import pandas as pd
import numpy as np
```

**Load Sample Data** 
<br> Data used within this quickstart guide originate from the AddNeuroMed [[1](#references)] cohort - dataset of Alzheimer's disease patients. 

```python
metadata = pd.read_csv('../sample_data/alz_metadata.csv')
# The PLS algorithm requires the response variable to be numeric. 
# We will encode the outcome "Gender" (Biological Sex) as 1 for female and 0 for male subjects. 
y = metadata["Gender"].apply(lambda x: 1 if x == 'Female' else 0)

# Import LC-MS data
# Add prefix to the columns names. This will be crucial for interpreting the results later on.
hpos = pd.read_csv('./sample_data/alz_hpos.csv').add_prefix('HPOS_')
lpos = pd.read_csv('./sample_data/alz_lpos.csv').add_prefix('LPOS_')
lneg = pd.read_csv('./sample_data/alz_lneg.csv').add_prefix('LNEG_')
```

**Fit MB-PLS Model and Estimate LVs**
```python 
mamsipls = MamsiPls(n_components=1)
mamsipls.fit([hpos, lpos, lneg], y_train)

mamsipls.estimate_lv([hpos, lpos, lneg], y_train, metric='auc')
```

**Estimate Feature Importance**
<br> You can visualise the MB-VIP:
```python
mb_vip = mamsipls.mb_vip(plot=True)
```
or estimate empirical p-values for all features: 

```python
p_vals, null_vip = mamsipls.mb_vip_permtest([hpos, lpos, lneg], y, n_permutations=10000, return_scores=True)
```

**Interpret Statistically Significant Features**
```python
x = pd.concat([hpos, lpos, lneg], axis=1)

mask = np.where(p_vals < 0.01)
selected = x.iloc[:, mask[0]]
```
Use `MamsiStrustSearch` to search for structural links within the statistically significant features. <br>
Firstly, all features are split into retention time (*RT*) windows of 5 seconds intervals, then each RT window is searched for isotopologue signature’s by searching mass differences of 1.00335 *Da* between mass-to-charge ratios (*m/z*) of the features; if two or more features resemble mass isotopologue signature then they are grouped together. This is followed by a search for common adduct signatures. This is achieved by calculating hypothetical neutral masses based on common adducts in electrospray ionisation. If hypothetical neutral masses match for two or more features within a pre-defined tolerance (15 *ppm*) then these features are grouped together. Overlapping adduct clusters and isotopologues clusters are then merged to form structural clusters. Further, we search cross-assay clusters using [M+H]<sup>+</sup>/[M-H]<sup>-</sup> as link references. Additionally, our structural search tool, that utilises region of interest [(ROI) files](https://github.com/phenomecentre/npc-open-lcms) from peakPantheR [[2](#references)], allows for automated annotation of  some features based on the *RT* for a given chromatography and m/z.
   
```python
struct = MamsiStructSearch(rt_win=5, ppm=10)
struct.load_lcms(selected)
struct.get_structural_clusters(annotate=True)
```
Further, use can find correlation clusters
```python
struct.get_correlation_clusters(flat_method='silhouette', max_clusters=11)
```
Finally, we visualise the structural relationships using a network plot. The different node colours represent different flattened hierarchical correlation clusters, while the edges between nodes identify their structural links. You can also save the network as an NX object and review in Cytoscape to get better insight on what these the structural relationship between individual features are (e.g. adduct links, isotopologues, cross-assay links).
```python
network = struct.get_structural_network(include_all=True, interactive=False, labels=True, return_nx_object=True)
```

# Issues and Collaboration
Thank you for supporting the MAMSI project. MAMSI is an open-source software and welcome any forms of contribution and support.

## Issues
Please submit any bugs or issues via the MAMSI project's GitHub [issue page](https://github.com/kopeckylukas/py-mamsi/issues) and any include details about the (```mamsi.__version__```) together with any relevant input data/metadata. 

## Collaboration
### Pull requests
You can actively collaborate on MAMSI package by submitting any changes via a pull request. All pull requests will be reviewed by the MAMSI team and merged in due course. 

### Contributions
If you would like to become a contributor on the MAMSI project please contact [Lukas Kopecky](https://profiles.imperial.ac.uk/l.kopecky22).

# Acknowledgement
This MAMSI and MAMSI Tutorials repositories were developed as part of Lukas Kopecky's PhD project at [Imperial College London](https://www.imperial.ac.uk/metabolism-digestion-reproduction/research/systems-medicine/), funded by [Waters UK](https://www.waters.com/nextgen/gb/en.html). It is free to use published under BSD 3-Clause [licence](./LICENSE).

The authors of the MAMSI package would like to acknowledge the authors of the [mbpls](https://pypi.org/project/mbpls/) [[3](#references)] package which became the backbone of MAMSI. Further, we would like thank to Prof Simon Lovestone for allowing us to use their AddNeuroMed [[1](#references)] data for the development of this package and for use in these tutorials. 

# Citing us
If you use MAMSI in a scientific publication, we would appreciate citations. 

## Release
```
@misc{MAMSI2024,
  author       = {Lukas Kopecky, Elizabeth J Want, Timothy MD Ebbels},
  title        = {MAMSI: Multi-Assay Mass Spectrometry Integration},
  year         = 2024,
  url          = {https://doi.org/10.5281/zenodo.13619607},
  note         = {Zenodo. Version 1.0.0},
  doi          = {10.5281/zenodo.13619607}
}
```

## Publication
The MAMSI publication is currently under the review process. 

# References
[1] S. Lovestone *et al.*, "AddNeuroMed—The European Collaboration for the Discovery of Novel Biomarkers for Alzheimer's Disease," *Ann. N. Y. Acad. Sci*, vol. 1180, no. 1, pp. 36-46, 2009, doi: [10.1111/j.1749-6632.2009.05064.x](https://nyaspubs.onlinelibrary.wiley.com/doi/10.1111/j.1749-6632.2009.05064.x).

[2] A. M. Wolfer *et al.*, "peakPantheR, an R package for large-scale targeted extraction and integration of annotated metabolic features in LC–MS profiling datasets," *Bioinformatics*, vol. 37, no. 24, pp. 4886-4888, 2021, doi: [10.1093/bioinformatics/btab433](https://academic.oup.com/bioinformatics/article/37/24/4886/6298587).

[3] A. Baum and L. Vermue, "Multiblock PLS: Block dependent prediction modeling for Python," *J. Open Source Softw.*, vol. 4, no. 34, 2019, doi: [10.21105/joss.01190](https://joss.theoj.org/papers/10.21105/joss.01190).