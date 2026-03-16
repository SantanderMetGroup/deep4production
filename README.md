# deep4production

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)

## Description

`deep4production` is a Python library designed for:
- Create AI-ready datasets tailored for downscaling tasks.
- Store state-of-the-art downscaling models
- Tools to compute and generate projections (ensuring compatibility with standard climate data formats like NetCDF).
....

**Ease of Research**: 
Researchers can focus on innovative tasks such as novel architecture development, rather than re-implementing standard routines from scratch.

**Established Deep Learning Models**:
`deep4production` also supplies established deep learning models that can be used to downscale global climate model outputs. Beyond the models themselves, the library provides:



In addition to these main goals, `deep4production` includes:


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Documentation](#documentation)
- [Contributing](#contributing)

## Installation

### 1. Clone the Repository
```bash
git clone https://github.com/SantanderMetGroup/deep4production/
cd deep4production
```

### 2. Create a conda environment from the provided .yml
```bash
conda env create -f requirements/deep4production-gpu.yml
```
If you prefer a CPU-only setup, use the `requirements/deep4production-cpu.yml` file instead.
(NOT READY YET)

## Usage

We provide a set of Jupyter notebooks in the `notebooks` directory that demonstrate the basic functionality of the `deep4downscaling` library. These notebooks cover topics such as:

As new features are developed and added to `deep4production`, additional example notebooks will be included to help you stay up-to-date with the latest capabilities.

(NOT READY YET)

## Documentation

While `deep4production` does not currently offer a formal documentation website, all library functions include comprehensive `docstrings` describing their purpose, parameters, and return values. This ensures that the code is self-explanatory for developers who want to use or extend the library.

For further guidance on how to use `deep4production`, please refer to:
- The notebooks in `notebooks`, which provide example workflows.  
- The `docstrings` in the source code, which offer detailed explanations of functions and classes.  

Should you have any questions or need clarifications, feel free to open an issue or contribute to improving the documentation.

## Contributing

We welcome contributions of all kinds to `deep4production`—from reporting bugs and suggesting improvements to submitting pull requests for new features or fixes. Here’s how you can get involved:

1. **Report Bugs:**  
   If you find an issue, please [open a new GitHub issue](https://github.com/SantanderMetGroup/deep4production/issues) with details on how to reproduce it.

2. **Suggest Enhancements or New Features:**  
   We value user feedback! Feel free to create an issue describing your idea or improvement.

3. **Submit Pull Requests:**  
   - Fork the repository and create a new branch for your changes.  
   - Make your changes or add your new feature.  
   - Open a pull request (PR) to the main branch of `deep4production`.   