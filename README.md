# Image Embedding Association Test (iEAT)

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[Ryan Steed](https://rbsteed.com), [Aylin Caliskan](https://www2.seas.gwu.edu/~aylin/)

Read the paper [here](https://arxiv.org/abs/2010.15052). View code and data [here](https://github.com/ryansteed/ieat).

.. info:: iEAT @ [FAccT 2021](https://facctconference.org/2021/acceptedpapers.html)
    - [Slides](https://github.com/ryansteed/ieat/blob/master/docs/docs/slides.pdf)
    - [Paper](https://dl.acm.org/doi/10.1145/3442188.3445932)

## Installation

```bash
# gather the dependencies for running scripts in this repo
conda env update environment.yml
conda activate ieat
# install the weat package locally
pip install -e weat
# install the ieat package locally
pip install -e .
```

## Usage
The `ieat` package does not have a CLI. Use it programmatically by accessing the API module (`ieat.api`). 

To run a basic test on a set of images, use the `test` function in `ieat.api`. 
SimCLR is downloaded automatically - but you must download a pre-trained version of iGPT yourself. 

For an example of how to use the API programmatically, see the [documentation](#documentation) and [tutorials](#tutorials-and-replications). 

### Tutorials and Replications

This repo uses Colab scripts in the `notebooks/` directory. Check out `notebooks/README.md` for a full description.

To open a `.ipynb` file in Colab, navigate to Colab's [Github Interface](http://colab.research.google.com/github) and search for this repo.


### Documentation
Documentation for the `ieat` API is published at [rbsteed.com/ieat](https://rbsteed.com/ieat).

To generate the documentation, use `pdoc3`:
```
pdoc3 --html --output-dir docs --force ieat --template-dir docs/templates
git subtree push --prefix docs/ieat origin gh-pages
```

## Contents
- `data/` - images and other data used for bias tests in the paper
- `embeddings/` - location for caching computed embeddings - includes pre-computed embeddings for convenience; 
to generate your own, use the `from_cache=False` option
- `ieat/` - software package for generating image embeddings and testing for bias
- `notebooks/` - Colab notebooks containing tutorials and data exploration
- `output/` - location for storing results tables
- `environment.yml` - Conda environment file with dependencies for Jupyter, etc.
- `docs/ieat` - source for [documentation](https://rbsteed.com/ieat)
