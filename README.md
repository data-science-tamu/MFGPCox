# MFGPCox

Code for the paper **Bayesian Joint Model of Multi-Sensor and Failure Event Data for Multi-Mode Failure Prediction**.  
Published paper: [Technometrics](https://doi.org/10.1080/00401706.2026.2653564)  
Preprint: [arXiv:2506.17036](https://arxiv.org/abs/2506.17036)

## 📄 Overview

This repository contains the implementation of **MFGPCox**, a unified Bayesian framework for jointly modeling:

1. **Time-to-event data** (failure times)  
2. **Condition-monitoring signals** from multiple sensors  
3. **Multiple Failure modes** (categorical outcomes)  

The model integrates:

- **Convolved Multi-output Gaussian Process (CMGP)** for modeling sensor signals  
- **Cox proportional hazards model** for survival analysis  
- **Multinomial distribution** for failure mode modeling  

within a **hierarchical Bayesian framework**, enabling accurate prediction and uncertainty quantification.

---

## 📄 Data Sources

![Data Types](./Data.png)


---

## 📄 Model Framework

![Model Framework](./Model.png)

---

## 📄 Prediction Pipeline

![Prediction Framework](./Prediction.png)

---

## 📄 Repository Structure

1. `case_study/`  
Files for the case study, including CMGP hyperparameter optimization code and outputs, ELBO optimization code, prediction code, evaluation code, benchmark comparison files, and associated data/results.

2. `numerical_study/`  
Files for the numerical study, including data generation, CMGP hyperparameter optimization code and outputs, ELBO optimization code, prediction code, evaluation code, benchmark comparison files, and associated data/results.

3. `utils/`  
Shared Python utility modules used across the case study and numerical study workflows.

4. `requirements.txt`  
Python dependency list for the main code environment.

---

## 📄 Data Availability

### Case Study Data

The dataset used in the case study is publicly available and was introduced in:

Z. Li, Y. Li, X. Yue, E. Zio, and J. Wu, "A Deep Branched Network for Failure Mode Diagnostics and Remaining Useful Life Prediction," *IEEE Transactions on Instrumentation and Measurement*, vol. 71, pp. 1–11, 2022, Art no. 3519111.  
DOI: [10.1109/TIM.2022.3195280](https://doi.org/10.1109/TIM.2022.3195280)

This dataset can be accessed at:

📌 https://github.com/kernelLZ/Turbofan-Engine-Degradation-Dataset-with-Multiple-Failure-Modes

### Numerical Study Data

For the **numerical study**, the data are synthetically generated, and the corresponding data and data generation code is available in the `numerical_study/Data_Generation/` folder.

---

## 📄 Paper

This repository accompanies the following paper:

📌 https://doi.org/10.1080/00401706.2026.2653564  
(*Published in Technometrics, 2026*)

📌 https://arxiv.org/abs/2506.17036  
(*This is a preprint version. The final, peer-reviewed version is available in Technometrics and should be considered the authoritative version.*)

---

## 📄 Citation

If you use this code or find it helpful, please cite:

```bibtex
author = {Sina Aghaee Dabaghan Fard and Minhee Kim and Akash Deep and Jaesung Lee},
title = {Bayesian Joint Model of Multi-Sensor and Failure Event Data for Multi-Mode Failure Prediction},
journal = {Technometrics},
volume = {0},
number = {ja},
pages = {1--24},
year = {2026},
publisher = {Taylor \& Francis},
doi = {10.1080/00401706.2026.2653564},
}

