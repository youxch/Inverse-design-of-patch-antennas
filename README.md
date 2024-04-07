# Inverse-Design-of-Patch-Antennas

This repository hosts a **simple demonstration** of a deep learning approach for the inverse design of patch antennas. The goal is to explore energy-efficient designs and to significantly reduce simulation costs compared to conventional methods. 

## Overview

This repository presents a novel inverse design methodology for patch antennas using deep learning techniques. The project encompasses several key components:

- **Data Generation**: Generation of a comprehensive dataset for training the deep learning model.
- **Data Preprocessing**: Preprocessing of the dataset to ensure high-quality input for the network.
- **Network Training & Testing**: Implementation and training of the deep learning model, followed by rigorous testing to validate its performance.
- **Prediction**: Utilization of the trained model to predict optimal patch antenna designs.
- **Inverse Design**: Process of designing patch antennas based on the predictions from the deep learning model.

## Achievements

- **Proposed Method**: A deep learning method is proposed for energy-efficient design of patch antennas.
- **Simulation Cost Reduction**: Significantly reducing the simulation cost compared to conventional methods.
- **Designed Antennas**: Three different antennas are designed through the deep learning network.
- **Experimental Verification**: The machine-designed antennas are experimentally verified showing good agreement with full-wave simulations.

## Getting Started

To get started with this simple demonstration of inverse design for patch antennas, follow these steps:

1. **Clone the Repository**:
   Clone this repository to your local machine using the following command:
   ```bash
   git clone https://github.com/username/Inverse-Design-of-Patch-Antennas.git

## Code Files

- **requirements.txt**: Lists all the Python libraries required for the project. Use the command `pip install -r requirements.txt` to install all necessary dependencies.

- **train_data.txt**: Represents the training dataset, which includes the structural parameters of patch antennas and the reflection coefficient and gain data obtained from full-wave simulations. This file is the foundation for training the deep learning model.

- **train.py**: Contains the preprocessing steps for the training data and the training process for the Multilayer Perceptron (MLP) network. Running this script is the entry point for training the model.

- **saved-model-5000.h5**: Stores the weights of the trained model. This file is generated after running the `train.py` script, where 5000 represents the number of iterations or another possible training metric.

- **predict.py**: Contains the steps to use the trained model for predicting large samples and performing inverse design. This script allows users to design new patch antennas based on the model's predictions.


## Contact

For any questions or suggestions, please open an issue or directly contact the maintainers.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.
