# Image Embedding Association Test

[![License: CC BY-NC-SA 4.0](https://img.shields.io/badge/License-CC%20BY--NC--SA%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

[Ryan Steed](https://rbsteed.com), [Aylin Caliskan](https://www2.seas.gwu.edu/~aylin/)

Forthcoming in [FAccT 2021](https://facctconference.org/2021/acceptedpapers.html). Read the paper [here](https://arxiv.org/abs/2010.15052). View code and data [here](https://github.com/ryansteed/ieat).

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
To run a basic test on a set of images, use the `test` function in `ieat.api`. 
SimCLR is downloaded automatically - but you must download a pre-trained version of iGPT yourself. 
For an example, see the [documentation](#documentation) and [tutorials](#tutorials-and-replications). 

### Documentation
The `ieat` package does not have a CLI. Use it programmatically by accessing the API module (`ieat.api`). 
Documentation for the `ieat` API is published at [rbsteed.com/ieat](https://rbsteed.com/ieat).

To generate the documentation, use `pdoc3`:
```
pdoc3 --html --output-dir docs --force ieat --template-dir docs/templates
git subtree push --prefix docs/ieat origin gh-pages
```

### Tutorials and Replications

This repo uses Colab scripts.

To open a `.ipynb` file in Colab, navigate to Colab's [Github Interface](http://colab.research.google.com/github) and search for this repo.

To save changes, choose Save -> Save to Github.

## Contents
- `data/` - images and other data used for bias tests in the paper
- `embeddings/` - location for caching computed embeddings - includes pre-computed embeddings for convenience; 
to generate your own, use the `from_cache=False` option
- `ieat/` - software package for generating image embeddings and testing for bias
- `notebooks/` - Colab notebooks containing tutorials and data exploration
- `output/` - location for storing results tables
- `environment.yml` - Conda environment file with dependencies for Jupyter, etc.
- `docs/ieat` - source for [documentation](https://rbsteed.com/ieat)
