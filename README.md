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

To get started with the inverse design of patch antennas using this repository, please follow these steps:

1. **Clone the Repository**: Clone this repository to your local machine.
2. **Install Dependencies**: Install all necessary dependencies as specified in the `requirements.txt` file.
3. **Data Generation**: Use the provided scripts to generate or obtain the dataset required for training.
4. **Data Preprocessing**: Preprocess the dataset as outlined in the preprocessing section.
5. **Training**: Train the deep learning model using the training dataset.
6. **Testing**: Test the model's performance using the testing dataset.
7. **Prediction**: Use the trained model to predict new patch antenna designs.
8. **Inverse Design**: Implement the inverse design process to create new patch antennas based on the predictions.

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
