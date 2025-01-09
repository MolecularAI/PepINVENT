# PepINVENT: Generative Peptide Design Beyond Natural Amino Acids

## Overview

**PepINVENT** is a generative reinforcement learning (RL) framework designed for generative peptide design, including peptides in the peptidic space that go beyond the standard 20 natural amino acids. This project enables the discovery and optimization of novel peptides with desired properties through multi-parameter optimization (MPO), making it applicable to peptide-based drug design and development projects. PepINVENT proposes amino acids from the peptide landscape unlocked by atomic representation of the peptide molecules to replace the query positions of an input peptide. 

Some features include:
- **Novel peptide generation** by generating peptides with natural and non-natural amino acids.
- **User-defined objectives** where the user can specify a set of objectives or integrate new objectives to steer the generative model to peptide ideas with desired properties.
- **Explorative sampling**: used to leverage the generative models to explore diverse designs from the extended peptide space.
- **Integration with RDKit**: allows common physicochemical features to be incorporated to the objectives of generative runs.

## Getting Started

### Prerequisites

- **Python**: Version 3.8 or higher
- Cuda-enabled GPU

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/MolecularAI/PepINVENT.git
   cd PepINVENT
   ```

2. Create the Conda environment:

   ```bash
    conda env create -f pepinvent_env.yml
   ```

3. Activate the environment:

   ```bash
   conda activate pepinvent
   ```

Now you can use the tool!

### Repository Structure

- **`test_data/`**: Test datasets used in the sampling experiments of the manuscript. In case of supplying a new csv file to sampling, a `Source_Mol` column with the masked peptides is needed.  
- **`models/`**: Pre-trained generative models and the predictive model for macrocyclic peptide permeability.
- **`pepinvent/`**: Core code for peptide generation and optimization.
- **`reinvent_models/`**: reinvent_models codebase extracted from https://github.com/MolecularAI/transformer_rl/
- **`experiment_configurations/`**: The configuration files to sample peptides from the pretrained generative model or run RL-based generative runs. The json files indicate the run setups for the experiments conducted in the maunuscript.

## Usage

The only thing that needs modification for a standard run are the file and folder paths. The generative runs produce logs that can be monitored by tensorboard.

**1. Sample Peptides from the generative model**: Sample peptides with new amino acids in the query positions using the pre-trained generative model.

   ```bash
   python input_to_sampling.py your_sampling_parameters.json
   ```

**2. Generate Peptides with RL**: Design peptides in the RL loop, optimized for user-defined criteria.

   ```bash
   python input_to_reinforcement_learning.py your_generative_run_setup.json
   ```

## Configuration

Model parameters and settings for sampling and RL-based generation are managed through JSON configuration files in the `experiment_configurations/` directory. Each file contains adjustable options to tailor the workflow to different peptide design goals.


## Citation

If you use PepINVENT in your research, please cite the preprint:

```bibtex
@misc{geylan2024pepinventgenerativepeptidedesign,
      title={PepINVENT: Generative peptide design beyond the natural amino acids}, 
      author={Gökçe Geylan and Jon Paul Janet and Alessandro Tibo and Jiazhen He and Atanas Patronov and Mikhail Kabeshov and Florian David and Werngard Czechtizky and Ola Engkvist and Leonardo De Maria},
      year={2024},
      eprint={2409.14040},
      archivePrefix={arXiv},
      primaryClass={q-bio.BM},
      url={https://arxiv.org/abs/2409.14040}, 
}
```

## License

This project is licensed under the Apache 2.0 License. See the `LICENSE` file for more details.
