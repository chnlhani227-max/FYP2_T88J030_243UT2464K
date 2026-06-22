# Intrusion Detection in WiFi 7 IoT Networks using Edge AI

## Project Description

This repository contains the source code and supporting files for the FYP2 project titled Intrusion Detection in WiFi 7 IoT Networks using Edge AI.

The project develops an Edge AI-based Intrusion Detection System using physical traffic collection, NS-3 attack simulation, machine learning model training, Raspberry Pi deployment, and Streamlit dashboard visualisation.

## Traffic Classes

The system classifies network traffic into four categories:

1. Normal Traffic
2. Flooding / Port Scanning Attack
3. Blackhole Attack
4. Grayhole Attack

## Hardware Environment

### Training Environment
- Device: ASUS VivoBook Laptop
- Operating System: Linux Mint
- Python Environment: `fyp-env`

### Edge Deployment Environment
- Device: Raspberry Pi 4 Model B
- Deployment Model: XGBoost
- Connectivity: LAN connection used during deployment testing for stable network access

## Dataset Repository
The dataset files are included in this repository and are also available through Google Drive:
- https://drive.google.com/drive/folders/1ePmJjc3adkhEeJSmYP8RFXMPKm_vuFF-?usp=sharing

## Software Requirements

Install the required libraries using:

```bash
pip install -r requirements.txt# Intrusion Detection in WiFi 7 IoT Networks using Edge AI

## Instructions
1. Activate environment: source venv/bin/activate
2. Run Dashboard: streamlit run app.py
