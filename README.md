# Poisoned-Data-Detector

This script detects poisoned data samples in a dataset, specifically aimed at identifying label-flipping attacks or manipulated samples. It utilizes the GroQ API for natural language processing and applies anomaly detection using the TF-IDF feature extraction method followed by an Isolation Forest model. The goal is to identify anomalous or manipulated data points that may negatively affect machine learning models.

## Features

- Loads a sample dataset (IMDB movie reviews).
- Injects poisoned (manipulated) samples into the dataset to simulate attacks.
- Uses TF-IDF to extract features from the dataset's text.
- Applies the Isolation Forest algorithm to detect anomalous or poisoned data points.
- Generates a histogram to visualize anomaly scores.
- Prints out the top suspected poisoned samples.

## Requirements

The following libraries are required to run the script:

- `groq`
- `datasets` (Hugging Face dataset library)
- `scikit-learn`
- `matplotlib`
- `numpy`

You can install the necessary libraries using pip:
pip install groq datasets scikit-learn matplotlib numpy

## How It Works
Dataset Loading and Poisoning: The script loads the IMDB dataset and injects manipulated samples (label-flipping attacks) to simulate poisoned data.
Feature Extraction: TF-IDF vectorization is used to convert the text data into numerical features.
Anomaly Detection: The Isolation Forest algorithm is applied to identify anomalous (poisoned) samples based on their feature vectors.
Visualization: A histogram of anomaly scores is generated to visualize the detection process.
Output: The script prints the top suspected poisoned samples, showing a snippet of the text for review.

## Functions
load_poisoned_data(): Loads the IMDB dataset and injects poisoned samples by flipping the labels of some reviews.
detect_poisoned_samples(texts): Extracts features using TF-IDF and applies the Isolation Forest model to detect anomalies (poisoned data). It also generates and saves a histogram plot of anomaly scores.
main(): Runs the full workflow, loading the poisoned data, detecting anomalies, and printing the results.

## Running the Script
To run the script, execute the following command in your terminal:
python groq_poisoned_data_detector.py

This will load the dataset, detect the poisoned samples, and display the results. A histogram image (poison_detection.png) will also be saved to visualize the detection process.

## Customization
Poisoning Strategy: You can modify the load_poisoned_data() function to inject different types of poisoned samples or use other datasets.
Anomaly Detection: Adjust the contamination parameter in the IsolationForest model to change the sensitivity of anomaly detection.
Visualization: You can modify how the anomaly scores are visualized or customize the histogram plot.
