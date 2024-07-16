# XAI-ResUNet: Analysing the Impact of Pre-training in ResUNet Architectures for Multiple Sclerosis Lesion Segmentation using EigenGradCAM

## Overview
This project is an implementation of the paper "XAI-ResUNet: Analysing the Impact of Pre-training in ResUNet Architectures for Multiple Sclerosis Lesion Segmentation using EigenGradCAM," which has been accepted for the 26th Irish Machine Vision and Image Processing Conference at the University of Limerick, Ireland, from Wednesday, August 21st, 2024, to Friday, August 23rd, 2024.

## Dataset
The dataset used for this paper, MSSEG-2016, can be downloaded from https://shanoir.irisa.fr/shanoir-ng/welcome.

## Proposed Architecture

## Results
Qualitative results so far: <br>
1. Rows 1 & 2 are ImageNet XAI-ResUNet trained using ImageNet-trained ResNet-50 weights for the encoder (EX1) (red) <br>
2. Rows 3 & 4 are RadImageNet XAI-Res-UNet trained using RadImageNet-trained ResNet-50 weights for the encoder (EX2) (green) <br>
3. Rows 5 & 6 are Vanilla XAI-ResUNet trained using no pre-trained weights for the encoder (EX3) (blue). <br>

![qualitative_2](https://github.com/user-attachments/assets/085a1dac-b627-44f7-924e-50c5b8515534)

Sample qualitative results including EigenGradCAM localization heatmaps overlaid on the original images where red indicates higher pixel importance and ground truth lesion contours are shown in black. 

 

## Setup

## Contact
Please feel free to raise an issue or contact me at vayangi.ganepola2@mail.dcu.ie with any queries or for discussions.

## Acknowledgement

Vayangi Ganepola, Prateek Mathur, Oluwabukola Adegboro, Julia Dietlmeier, Aonghus Lawlor, Noel E. O'Connor, Claudia Mazo. "XAI-ResUNet: Analysing the Impact of Pre-training in ResUNet Architectures for Multiple Sclerosis Lesion Segmentation using EigenGradCAM". 26th Irish Machine Vision and Image Processing Conference, University of Limerick, Ireland. August 21st, 2024 - August 23rd, 2024.
