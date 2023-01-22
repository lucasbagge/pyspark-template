# Data Science Cookie Cutter for Prefect

## Why Should You Use This Template?

## Tools used in this project

* [pyspark](https://python-poetry.org/): Dependency management

## Project structure
```bash
.
├── 01-data-raw            
│   ├── mtcars.csv                       # data after training the model
│   ├── nasdaq_data.csv                   # data after processing
│   ├── nasdaq_index.csv                         # raw data
├── .flake8                         # configuration for flake8 - a Python formatter tool
├── .gitignore                      # ignore files that cannot commit to Git
├── Makefile                        # store useful commands to set up the environment
├── 01_spark_training.py                             # store source code
└── tests                           # store tests
```

## How to use this project

Install Cookiecutter:
```bash
pip install cookiecutter
```

Create a project based on the template:
```bash
cookiecutter https://github.com/lucasbagge/pyspark-template
```