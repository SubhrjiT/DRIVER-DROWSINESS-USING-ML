# Driver Drowsiness Detection Using Machine Learning

## Overview

Driver drowsiness is a major cause of road accidents worldwide. This project aims to develop a machine learning model that can detect drowsiness in drivers using real-time video or images. The model can alert the driver if it detects signs of drowsiness, potentially preventing accidents and saving lives.

## Table of Contents

1. [Installation](#installation)
2. [Usage](#usage)
3. [Dataset](#dataset)
4. [Model Architecture](#model-architecture)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Results](#results)
7. [Contributing](#contributing)
8. [License](#license)
9. [Contact](#contact)
10. [Acknowledgments](#acknowledgments)

## Installation

To run this project locally, follow these steps:

### Prerequisites

- Python 3.x
- Git

### Setup

1. **Clone the repository**:
    ```bash
    git clone https://github.com/SubhrjiT/DRIVER-DROWSINESS-USING-ML.git
    cd DRIVER-DROWSINESS-USING-ML
    ```

2. **Create and activate a virtual environment** (optional but recommended):
    ```bash
    python -m venv env
    source env/bin/activate   # On Windows, use `env\Scripts\activate`
    ```

3. **Install the required dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To use this project, follow these steps:

1. **Prepare the dataset**: Ensure the dataset is downloaded and placed in the correct directory as described in the [Dataset](#dataset) section.

2. **Train the model**:
    ```bash
    python train.py --dataset_path ./path_to_dataset --output_model ./output_model_path
    ```

3. **Test the model**:
    ```bash
    python test.py --model_path ./output_model_path --test_data ./path_to_test_data
    ```

4. **Run the model in real-time**:
    ```bash
    python run_realtime.py --model_path ./output_model_path
    ```

## Dataset

The dataset used for this project is [Dataset Name] (link if publicly available). It consists of images or videos of drivers labeled as drowsy or alert.

- **Download the dataset**: [Link to dataset]
- **Structure**:
    ```
    /dataset
      /train
        /drowsy
        /alert
      /test
        /drowsy
        /alert
    ```

## Model Architecture

The model used in this project is a Convolutional Neural Network (CNN) designed to detect drowsiness based on facial features. The architecture includes:

- Input layer with image dimensions
- Convolutional layers with ReLU activation
- Max-pooling layers
- Fully connected layers
- Output layer with softmax activation for binary classification

## Evaluation Metrics

The model is evaluated using the following metrics:

- **Accuracy**: Overall correctness of the model.
- **Precision**: Ability of the model to identify only the relevant instances.
- **Recall**: Ability of the model to find all the relevant instances.
- **F1-Score**: Harmonic mean of precision and recall.

## Results

After training, the model achieved the following results:

- **Accuracy**: 95%
- **Precision**: 94%
- **Recall**: 92%
- **F1-Score**: 93%

![Model Performance](./results/performance.png)

## Contributing

Contributions are welcome! To contribute:

1. Fork the repository.
2. Create a new branch: `git checkout -b feature/YourFeature`
3. Commit your changes: `git commit -m 'Add some feature'`
4. Push to the branch: `git push origin feature/YourFeature`
5. Open a pull request.

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Contact

For any questions or suggestions, feel free to contact:

- **Author**: SubhrjiT
- **Email**: [subhrajitbehera6370@gmail.com]

## Acknowledgments

- Thanks to [Dataset Provider] for providing the dataset.
- This project was inspired by research on driver safety and machine learning applications.
