# bim-bot v1.0
Author:
Jayzen Micah Balagao
Christine Insigne
Paula Sophia Macalindong

**bim-bot** is an interactive robot-assisted hazard detection and mapping system built on top of **pySLAM**, ESP32 camera streaming, and a custom dashboard interface. This repository provides the full pipeline for running the bim-bot app — including environment setup, dashboard execution, video recording, and automatic area map generation.

## Table of Contents
- Overview
- Installation
- Requirements
- Setup Instructions
- Running the Dashboard
- Robot Connection
- Hazard Notification & Map Generation
- Saving & Uploading Exploration Logs
- Generating the Area Map

## Overview
The bim-bot system integrates:
- A local dashboard for robot control and monitoring
- Live video and sensor data streaming from an ESP32-CAM robot
- Hazard notifications
- Recording and saving exploration logs
- Area map generation from recorded video
- A unified python environment using `pySLAM`

```
├── main_dashboard/
├── pyslam/
├── scripts/
├── data/
```
## Arduino BIM-BOT Setup
Before running the dashboard, upload the BIM‑BOT firmware to the robot’s microcontroller.

### 1. Install the Arduino IDE
Download and install the Arduino IDE from the official page:
🔗 https://www.arduino.cc/en/software

### 2. Open the Firmware in Arduino IDE
1. Launch the Arduino IDE
2. Click File → Open…
3. Navigate to and open the folder below under the downloaded BIM-BOT repository:

```bash
BIM-BOT_arduino_setup/
```
4. Select the ESP32.ino file inside the folder

### 3.  Connect the BIM‑BOT Robot 
1. Plug the robot's microcontroller (e.g., Arduino UNO, ESP32‑CAM, or ESP32 board) into your laptop
2. In the Arduino IDE:
- Tools → Board → Choose the correct board
- Tools → Port → Select the COM port of your robot

### 4. Install Required Arduino Libraries
If any required libraries are missing, install them through:
**Sketch → Include Library → Manage Libraries…**

Common required libraries include:
- WiFi.h
- ESPAsyncWebServer
- ArduinoJson
- DHT sensor library

(The IDE will usually prompt you automatically.)

### 5. Upload the Firmware
Click the Upload (→) button in the Arduino IDE.
After uploading, your robot will:

1. Start broadcasting its Wi‑Fi network (e.g., ESP32)
2. Begin camera streaming (ESP32‑CAM)
3. Begin sensor data transmission

You can now proceed to the Python environment setup.


## Installation
### 1. Clone This Repository and Thirdparties Needed
```bash

# 1) Clone bim-bot with submodules
git clone --recursive https://github.com/pineappleapi/bimbot.git
cd bimbot

# 2) Download upstream pySLAM as ZIP (master branch)
curl -L -o pyslam.zip https://github.com/luigifreda/pyslam/archive/refs/heads/master.zip
unzip pyslam.zip

# 3) Copy ONLY the thirdparty folder into bimbot/
cp -r pyslam-master/thirdparty ./thirdparty

# 4) Clean up extracted files and zip
rm -rf pyslam.zip pyslam-master

```

## Requirements
The installation process follows the standard pySLAM setup.
Refer to the Installation section in `README_pySLAM.md for OS‑specific procedures.

Supported OS:
- Ubuntu
- MacOS
- Windows (via WSL2 recommended)



## Setup Instructions
### 2. Install Dependencies
See Installation in `README_pySLAM.md`.

Once installation is completed, a Python environment named pyslam will be created.

### 3. Activate the pyslam Environment
```bash
. pyenv-activate.sh
```

## Running the Dashboard
### 4. Navigate to the Dashboard Folder
```bash
cd "main_dashboard"
```

### 5. Run the Application
```bash
./app.py
```
The terminal will show initialization updates, including SLAM loading, dashboard modules, and server activation.

## Robot Connection
### 6. Connect to the bim-bot Robot
On your device (laptop/phone):
1. Open Wi‑Fi settings
2. Connect to the robot network:
```
ESP32
```
Once connected, the terminal will display messages confirming video and sensor data transmission.

## Dashboard Access
### 7.  Open the Dashboard in a Browser
```
http://127.0.0.1:4000
```
You should now see the bim-bot control interface.

## Hazard Notification & Map Generation
### 8. Start Exploration
Hazard notifications update automatically once the robot is connected.

To start gathering data for mapping:

1. Ensure the camera stream is visible on the left panel by clicking the **Connect** button.
2. Click **Record** when you begin the exploration
Begin navigating the robot
3. Begin navigating the robot

The live stream updates continuously during the exploration.

### 9. Stop Recording
When the exploration is complete:
- Press the **Stop** button.
This will save a video/log file to your local file system.

## Saving & Uploading Exploration Logs
### 10. Upload Log for Map Generation
In the Area Map Generation section of the dashboard:
1. Under **Upload Log**, select the newly saved recording file
2. Click **Upload Video**

A confirmation message will appear in the terminal.

## Generating the Area Map
### 11. Generate Map
Once the file is uploaded successfully:
1. Click **Generate Map**
The system will process the video and generate a 2D map of the explored area. Output will be available in the maps folder or displayed directly in the dashboard (depending on your build).

Your bim-bot system is now fully operational — from robot control and hazard monitoring to area map generation.
