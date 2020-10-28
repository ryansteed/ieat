# Image Embedding Association Test

Ryan Steed

The Image Embedding Association Test (iEAT) is a statistical test for bias in unsupervised image representations. Read the paper [here](arxiv).

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
pdoc3 --html --output-dir docs --force ieat
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