# FishNet: Invasive Fish Species Detection

FishNet is a deep learning project designed to identify invasive fish species in the Mediterranean Sea using video data. This project utilizes a YOLO model trained on a custom dataset created from video footage.

![Detection Demo](Images/DetectionGif.gif)

## Table of Contents
- [Project Overview](#project-overview)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Model Training](#model-training)
- [Real-Time Detection](#real-time-detection)
- [Running the Inference Script](#running-the-inference-script)
- [Contributing](#contributing)
- [Acknowledgements](#acknowledgements)

## Project Overview
FishNet aims to assist in the early detection and monitoring of invasive fish species. The application processes video frames in real-time, identifies different fish species, and alerts relevant stakeholders when invasive species are detected.

## Features
- Real-time fish species detection
- Alerts and notifications for invasive species
- Data logging for further analysis
- User-friendly interface for fishermen and researchers

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/FishNet.git
   cd FishNet
   
2. Create a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

4. Install the required dependencies:
   ```bash
   pip install -r requirements.txt

6. Download the pre-trained YOLO model weights and place them in the weights directory. Download link

## Usage
Processing Video Data
1. Place your video files in the data/videos directory.

2. Run the video processing script:
   ```bash
   python main.py


## Dataset
The dataset used for training the model consists of frames extracted from video footage, annotated with bounding boxes for different fish species. The dataset is divided into training, validation, and test sets. Dataset link


## Model Training
To train the YOLO model, follow these steps:

1. Prepare your dataset and ensure it is correctly formatted for YOLO.

2. Modify the configuration files as needed.

3. Train the model: An example run file is attached.

4. Monitor the training process and adjust hyperparameters if necessary.

## Real-Time Detection
To enable real-time detection, the application processes video streams frame by frame, using the trained YOLO model to identify fish species. Alerts are generated when invasive species are detected.

## Running the Inference Script
1. Ensure you have the necessary libraries installed:
   ```bash
   pip install ultralytics openpyxl pandas


### Example
1. Here is an example of how to run the script with a specific \`videos_base_dir\`:
   ```bash
   python infrance_run_example.py --videos_base_dir "/media/ronm/Crucial X6/chunk_4/israchz091121A/videos"


## Contributing
Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (git checkout -b feature/your-feature-name).
3. Make your changes.
4. Commit your changes (git commit -m 'Add your feature').
5. Push to the branch (git push origin feature/your-feature-name).
6. Create a new Pull Request.

## Acknowledgements
1. Special thanks to the School of Zoology for providing the video data.
2. Thanks to the open-source community for tools and resources.







