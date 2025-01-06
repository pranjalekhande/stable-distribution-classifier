# Stable Naive Bayes Classifier

This repository contains an implementation of the Stable Naive Bayes (SNB) classifier in Python, which integrates an external R package using `rpy2`. The SNB model addresses limitations of traditional Gaussian Naive Bayes by introducing stable distributions to handle datasets with skewness, heavy tails, and other non-Gaussian characteristics.

## Overview
In many real-world scenarios, datasets deviate from normal or exponential distributions, exhibiting significant skewness or kurtosis. Such deviations are common in domains like finance, sensor data analysis, and environmental monitoring. These characteristics can degrade the performance of traditional classifiers. The SNB classifier builds on the Naive Bayes framework by incorporating stable distributions, a generalization of the Gaussian distribution, to improve robustness and accuracy.

Additionally, the repository supports Beta and Student's t-distributions as alternative assumptions for feature modeling, enabling comparisons among various distributional frameworks.

## Features
- Extends the Naive Bayes algorithm with stable, Beta, and Student's t-distributions.
- Seamlessly integrates R functionality into Python using `rpy2`.
- Handles datasets with skewed, heavy-tailed, or bounded feature distributions.
- Includes comparative evaluations of Stable Naive Bayes (SNB), Gaussian Naive Bayes (GNB), Beta Naive Bayes (BNB), and Student's t Naive Bayes (TNB).

## Installation

### Step 1: Install Required R Package
The implementation relies on the `stable` R package for parameter estimation of stable distributions. Ensure that you have R or RStudio installed on your system.

#### Instructions:
1. **Download the `stable` R package**  
   Visit the official site or repository for the `stable` package and download it.
   
2. **Install the Package**  
   Follow these steps to install the package in R:
   - Open R or RStudio.
   - Run the following command:
     ```R
     install.packages("path_to_downloaded_package", repos = NULL, type = "source")
     ```

3. **Verify Installation**  
   Ensure the package is installed correctly by running:
   ```R
   library(stable)
