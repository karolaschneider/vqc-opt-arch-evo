### Program Overview

This program generates a population of Variational Quantum Circuits and optimizes them using an Evolutionary Algorithm. Each circuit is evaluated in the Coin Game environment, where agents play a round of 50 steps against themselves. The agents' fitness is assessed based on their score in the Coin Game and the size of their circuits.

**Output**: The program generates a .csv file in the data directory, containing information from all generations, including performance metrics from the Coin Game and circuit sizes.

### Optional:

- create virtual environment (in root of project): `python3 -m venv venv`
- activate virtual environment: `source venv/bin/activate`

### Running the project:

- install dependecies: `pip install -r requirements.txt`
- run main.py with arguments : `python src/main.py`
- mandatory arguments:
  - seed (`--seed` / `-s`)
    - with some numerical seed value
  - architecture concept (`--type` / `-t`)
    - Layer-Based: `layer`,
    - Gate-Based: `gate`,
    - Prototype-Based: `prototype`
      INFO: the size specifications for all concepts are in the parse_args function
  - evolution type (`--evolution` / `-e`)
    - mutation only: `mut-only`
    - recombination and mutation: `mut-recomb`
  - number for architecture mutations (`--num-arch-mut` / `-a`)
    - integer value
    - set this to specify a number of architecture mutations
- optional arguments:
  - dynamic mutation power (`--dynamic-mut-pow` / `-d`)
    - float between 0 and 1
    - if set dynamic mut pow in main.py is used, else the static mut pow in main.py is used
  - increased exploitation in last quarter of the generations (`--exploit` / `-x`)
    - no value
    - if set selection changes to Truncation Selection with top 5 agents in last quarter

---
