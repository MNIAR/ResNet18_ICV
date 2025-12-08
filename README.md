# Introduction to Computer Vision Final Project — ResNet-18 for POC Dataset Classification
This project implements and evaluates a ResNet-18 model for image classification on the POC dataset as part of the Introduction to Computer Vision final project.

## Experiment summary
| Experiment          | Best Epoch | Validation Acc (%) | Validation Loss | Test Acc (%) | Test Loss |
| ------------------- | ---------- | ------------------ | --------------- | ------------ | --------- |
| **adam_1e-3_step**  | 16         | 97.35              | 0.0954          | **63.44**    | **1.6870**    |
| **adam_5e-4_step**  | 15         | 97.35              | 0.0868          | 62.77        | 1.8036    |
| **sgd_1e-2_cosine** | 15         | **97.68**          | **0.0784**      | 62.33        | 2.6863    |
