# 🌱 Krushi Mithra — IoT Smart Farming Using Machine Learning

> **Project:** Landform Analysis and Smart Farming Using Machine Learning Techniques  
> **Platform:** ESP32 + environmental sensors + Python backend + Machine Learning  
> **Purpose:** Combine field measurements, weather information, agricultural ML predictions, and irrigation control in one smart-farming system.

Krushi Mithra is an IoT-based smart-farming project designed to monitor environmental conditions, support crop recommendation, provide irrigation decision support, and explore soil-moisture and rainfall forecasting. The project contains ESP32/Arduino firmware, data-cleaning and preprocessing scripts, ML training/testing scripts, trained-model folders, and a Python application that connects the device, weather information, predictions, and control functions.

**Important:** This README is based on the project structure shown in the supplied folder screenshots. Exact GPIO pin numbers, firmware library names, environment-variable names, model filenames, API provider, and command-line arguments must be confirmed from the source code before use. Do not guess these values.

---

## Contents

- [1. Features](#1-features)
- [2. System architecture](#2-system-architecture)
- [3. Hardware](#3-hardware)
- [4. Repository structure](#4-repository-structure)
- [5. How the system works](#5-how-the-system-works)
- [6. Machine-learning modules](#6-machine-learning-modules)
- [7. Software requirements](#7-software-requirements)
- [8. Installation](#8-installation)
- [9. Dataset preparation and training](#9-dataset-preparation-and-training)
- [10. Connecting and programming the ESP32](#10-connecting-and-programming-the-esp32)
- [11. Running the system](#11-running-the-system)
- [12. Internet and weather API](#12-internet-and-weather-api)
- [13. Irrigation modes and safety](#13-irrigation-modes-and-safety)
- [14. Backup and offline behaviour](#14-backup-and-offline-behaviour)
- [15. Troubleshooting](#15-troubleshooting)
- [16. Security](#16-security)
- [17. Reproducibility and limitations](#17-reproducibility-and-limitations)
- [18. Research references](#18-research-references)

---

## 1. Features

- **Environmental monitoring:** Reads supported measurements from the connected sensors.
- **ESP32 connectivity:** Sends device readings to the application using the communication method configured in the firmware.
- **Wi-Fi setup interface:** The project includes a Wi-Fi setup/captive-portal workflow where implemented by the firmware.
- **Web dashboard:** Displays supported readings, status, predictions, and controls exposed by the application.
- **Manual and automatic irrigation:** Provides pump control through the configured application and device-side logic.
- **Timed watering:** Available where implemented by the firmware and dashboard.
- **Weather API integration:** Retrieves external weather information when internet access and valid API configuration are available.
- **Four ML tasks:** Crop recommendation, context-aware irrigation, future soil-moisture prediction, and rainfall prediction.
- **Data pipeline:** Includes dataset cleaning, preprocessing, model training, testing, and result directories.
- **Local device operation:** Essential sensor and pump functions may continue without the PC-based ML backend only to the extent implemented in the ESP32 firmware.

Feature availability depends on the actual code and configuration. A feature should be described as operational only after it has been tested on the target hardware.

---

## 2. System architecture

```text
Agricultural environment
        |
        v
Sensors connected to ESP32
        |
        v
ESP32 firmware and Wi-Fi
        |
        v
Python application / backend
        |------------------------|
        |                        |
        v                        v
Weather API                 ML inference
        |                        |
        |               M1 / M2 / M3 / M4
        |                        |
        |------------------------|
                     |
                     v
          Dashboard and control logic
                     |
                     v
             ESP32 / relay / pump
```

### Main layers

1. **Sensing layer:** Measures local environmental conditions.
2. **Embedded layer:** ESP32 reads sensor signals, manages supported local control, and communicates over the configured network.
3. **Application layer:** Python modules receive device data, expose application functionality, and coordinate API and ML operations.
4. **ML layer:** Trained models produce predictions from inputs compatible with their training pipelines.
5. **Control layer:** Applies irrigation rules and safety checks before supported pump actions.
6. **Presentation layer:** The web interface displays available readings, statuses, controls, and predictions.

Sensor readings, external weather data, and ML predictions are different types of information. The dashboard should not label an API forecast as a physical sensor measurement.

---

## 3. Hardware

The project documentation identifies the following components:

| Component | Purpose |
|---|---|
| ESP32 development board | Sensor acquisition, Wi-Fi connectivity, and supported device-side control |
| Capacitive soil-moisture sensor | Measures a signal related to soil moisture |
| DHT11 | Measures ambient air temperature and relative humidity |
| DS18B20 | Measures temperature at the probe location in soil |
| FC-37 / raindrop module | Detects wetness on its sensing plate; it does not directly measure rainfall in millimetres |
| LDR | Detects changes in ambient light |
| Relay module | Switches the pump circuit when correctly rated and wired |
| Water pump and suitable power supply | Delivers water for irrigation |

### Wiring precautions

- Use the **actual GPIO assignments defined in the firmware**. Do not use guessed pin numbers.
- Check the voltage and output type of each sensor before connecting it to the ESP32.
- ESP32 GPIO inputs are **not 5 V tolerant**. Do not apply a 5 V signal directly to an ESP32 input.
- Connect grounds appropriately for low-voltage sensor circuits where required.
- Power the pump from a supply rated for the pump, not from an ESP32 GPIO pin.
- Use a suitable relay/driver and follow electrical isolation and load-current ratings.
- Disconnect power before changing wiring.
- Keep water away from exposed electronics and mains wiring.

Add a wiring diagram to `docs/` once the final, verified pin mapping is documented.

---

## 4. Repository structure

The following structure is based on the supplied screenshots. File extensions and names should be checked against the actual files in the repository.

```text
IOT-FARM/
├── CODE/
│   ├── CODE.ino
│   ├── crop.py
│   ├── pump.py
│   ├── rain.py
│   └── soil.py
│
├── DATASET/
│   ├── CLEANED/
│   ├── PREPROCESSED/
│   │   ├── M1_Crop_Recommendation/
│   │   ├── M2_Context_Aware_Irrigation/
│   │   ├── M3_Future_Soil_Moisture/
│   │   └── M4_Rainfall_Forecasting/
│   ├── Crop_Recommendation.csv
│   ├── Rain_Prediction.csv
│   ├── Smart_Irrigation.csv
│   ├── Soil_Moisture_Prediction.csv
│   ├── clean.py
│   └── preprocess.py
│
├── MODELS/
│   ├── TEST_RESULTS/
│   ├── TRAINED_MODELS/
│   ├── TRAINING_RESULTS/
│   ├── test.py
│   └── train.py
│
├── PAPERS/
│   └── Research papers (PDF files)
│
├── SYSTEM/
│   ├── data/
│   ├── backend.py
│   ├── config.py
│   ├── control.py
│   ├── esp32.py
│   ├── main.py
│   ├── ml.py
│   └── weather.py
│
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

### What each file/folder is for

#### `CODE/`

- **`CODE.ino`** — expected to contain the Arduino/ESP32 firmware entry point. Confirm the actual sketch name and its sensor pins, Wi-Fi setup, communication protocol, and relay logic from the file.
- **`crop.py`** — a crop-related Python script, based on its name. Confirm whether it performs inference, preprocessing, or another task.
- **`pump.py`** — a pump-related Python script. Confirm whether it controls, tests, or models pump behaviour.
- **`rain.py`** — a rain-related Python script. Confirm its actual role and whether it works with the sensor, weather data, or ML model.
- **`soil.py`** — a soil-related Python script. Confirm whether it processes soil data, predicts moisture, or performs another task.

The Python scripts inside `CODE/` should not automatically be assumed to be part of the firmware; Python and Arduino code run in different environments.

#### `DATASET/`

- **`Crop_Recommendation.csv`** — source data for the crop-recommendation task, subject to dataset provenance and licensing.
- **`Rain_Prediction.csv`** — source data for the rainfall-prediction task.
- **`Smart_Irrigation.csv`** — source data for irrigation-related training.
- **`Soil_Moisture_Prediction.csv`** — source data for soil-moisture forecasting.
- **`clean.py`** — dataset cleaning script.
- **`preprocess.py`** — data preparation script for the training pipeline.
- **`CLEANED/`** — generated cleaned datasets.
- **`PREPROCESSED/`** — model-ready datasets, organised by M1–M4.

The precise transformations, target columns, and feature lists are defined by the scripts and should be documented from the actual code.

#### `MODELS/`

- **`train.py`** — training entry point for the configured algorithms and modules.
- **`test.py`** — testing/evaluation entry point.
- **`TRAINED_MODELS/`** — generated model artifacts, organised by module.
- **`TRAINING_RESULTS/`** — training metrics and other outputs, depending on the implementation.
- **`TEST_RESULTS/`** — testing/evaluation outputs, depending on the implementation.

Large model binaries and generated outputs can be kept outside Git and regenerated using the training scripts. If model files are required to run inference immediately, document how to obtain them.

#### `SYSTEM/`

- **`main.py`** — likely the main application entry point; verify its imports and run configuration before using it.
- **`backend.py`** — backend functionality; verify its routes and responsibilities.
- **`config.py`** — application configuration; do not commit secrets.
- **`control.py`** — irrigation/device control functionality, subject to the actual implementation.
- **`esp32.py`** — ESP32 communication functionality, subject to the actual protocol used.
- **`ml.py`** — model-loading and prediction integration, subject to the actual implementation.
- **`weather.py`** — weather API integration, subject to the actual provider and endpoints.
- **`data/`** — application data or generated files; inspect contents before publishing.

#### `PAPERS/`

Contains research papers consulted during project development. Respect each paper's copyright and sharing terms. When redistribution is not permitted, publish citations/DOIs and a literature summary instead of the PDF itself.

---

## 5. How the system works

### 5.1 Sensor acquisition

The ESP32 reads the connected sensor channels according to the firmware. Analog sensors such as the capacitive soil-moisture sensor and LDR use analog inputs; digital sensors use their corresponding communication interfaces.

Sensor readings should be validated and calibrated. Raw ADC values are not automatically percentages, and the rain/wetness module does not produce a rainfall forecast.

### 5.2 Device-to-application communication

The firmware communicates with the Python application using the protocol and endpoints configured in the code. Confirm whether the project uses HTTP, sockets, MQTT, or another method before documenting the exact ports or URLs.

The application should check received values and timestamps before using them for display, inference, or control.

### 5.3 Weather information

The weather module requests data from the configured external API when internet access and valid configuration are available. The returned response should be checked for errors, missing fields, units, location, and timestamp.

### 5.4 Machine-learning inference

The ML integration loads trained artifacts and passes appropriately prepared features to the relevant model. Inference requires the same feature order, encoding, units, and preprocessing conventions used during training.

### 5.5 Irrigation decision and pump control

The control layer evaluates the information available to it and applies the configured logic. The ESP32/relay should retain appropriate hardware-side safety protections. A model prediction should not bypass safety checks.

### 5.6 Dashboard

The browser-based interface presents only the readings, status fields, predictions, and controls exposed by the application. Actual update frequency depends on sensor sampling, network communication, backend processing, and dashboard refresh behaviour.

---

## 6. Machine-learning modules

| Module | Task | Planned model | Output |
|---|---|---|---|
| M1 | Crop recommendation | Random Forest | Crop category |
| M2 | Context-aware irrigation | XGBoost | Pump state/category |
| M3 | Future soil-moisture prediction | XGBoost | Moisture estimate for the trained forecast target |
| M4 | Rainfall prediction | XGBoost | The rainfall target defined by the dataset |

The model names above reflect the project's planned configuration. Confirm the actual algorithms, features, target columns, and saved artifacts in `MODELS/train.py` and the relevant dataset/preprocessing scripts.

### Training versus prediction

- **Training** learns model parameters from prepared historical examples and writes model artifacts/results.
- **Prediction/inference** loads a trained model and runs it on new, compatible inputs.
- Training is not normally required every time a sensor value arrives.
- A model cannot make a valid prediction if required features are missing or are represented differently from the training data.

Do not report accuracy or forecast horizon unless it comes from the project's actual evaluation results.

---

## 7. Software requirements

- A supported Python 3 version compatible with the packages in `requirements.txt`.
- Arduino IDE or PlatformIO for the ESP32 firmware.
- USB data cable and appropriate ESP32 board configuration.
- Access to the permitted datasets.
- Wi-Fi network for network-based features.
- Valid weather API credentials if required by the chosen provider.
- A computer capable of running the backend and training scripts.

Python packages are separate from Arduino/ESP32 libraries. Installing `requirements.txt` does not install the firmware dependencies.

---

## 8. Installation

Open a terminal in the project root directory.

### Windows

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

If PowerShell prevents activation, use Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### Linux/macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Use the Python version supported by the project's dependencies. If installation fails, record the exact error and check package/Python compatibility rather than randomly upgrading or downgrading every package.

---

## 9. Dataset preparation and training

Run commands from the repository root unless the script specifies otherwise.

### Step 1: Inspect the scripts

Before execution, open `DATASET/clean.py`, `DATASET/preprocess.py`, and `MODELS/train.py` and confirm:
- Input dataset paths.
- Output folder paths.
- Expected column names.
- Required command-line arguments.
- Module and algorithm configuration.
- Where trained artifacts and results are written.

### Step 2: Clean the datasets

The expected entry point is:

```bash
python DATASET/clean.py
```

Run this only if the script is configured to run directly and its paths match the current repository.

### Step 3: Preprocess the datasets

The expected entry point is:

```bash
python DATASET/preprocess.py
```

Check that the output folders for M1, M2, M3, and M4 contain the files expected by the training script.

### Step 4: Train the models

The expected entry point is:

```bash
python MODELS/train.py
```

Use any arguments required by the actual script. Training may take different amounts of time depending on dataset size, algorithms, CPU/GPU availability, and configuration.

### Step 5: Evaluate the models

The expected entry point is:

```bash
python MODELS/test.py
```

Check the script's required arguments and confirm that it evaluates the intended saved model and test split. Review the generated files in `TRAINING_RESULTS/` and `TEST_RESULTS/`.

**Do not run these commands blindly if the source code uses different paths or requires arguments.** The folder layout alone does not reveal the exact command-line interface.

---

## 10. Connecting and programming the ESP32

### Step 1: Inspect the firmware

Open `CODE/CODE.ino` in Arduino IDE or the corresponding sketch file. Check:
- Board type and ESP32 board package.
- GPIO pin assignments.
- Sensor libraries and versions.
- Wi-Fi setup/captive portal implementation.
- Backend address, protocol, and port.
- Relay active level and pump-control logic.
- Any required compile-time settings.

### Step 2: Install the board package and libraries

In Arduino IDE, install the ESP32 board support package and the exact libraries imported by the firmware. Select the correct ESP32 board and the connected serial/COM port.

Do not install libraries based only on a guessed list; inspect the `#include` statements and firmware configuration.

### Step 3: Connect the hardware

Wire each sensor to the GPIO and power pins specified by the firmware and verified wiring diagram. Verify supply voltage and common-ground requirements. Never connect a 5 V sensor output directly to an ESP32 GPIO.

Keep the pump power circuit separate from the ESP32's GPIO drive capability and use a correctly rated relay/driver and power supply.

### Step 4: Upload and inspect serial output

Compile and upload the firmware. Open the serial monitor at the baud rate specified by the sketch. Check for startup messages, sensor readings, Wi-Fi state, and backend connection errors.

### Step 5: Configure Wi-Fi

Follow the actual setup/captive-portal instructions implemented in the firmware. The exact access-point name, address, setup URL, and credential workflow must be taken from the code or device instructions; they are not assumed here.

### Step 6: Verify telemetry

Confirm that the device is connected and that the backend receives fresh readings. Compare displayed values with the serial output and verify timestamps where available.

### Step 7: Test the pump safely

Initially test the relay without the pump load if practical. Then test pump operation with suitable power and water precautions. Verify manual mode, automatic mode, OFF behaviour, timed operation, and recovery behaviour before leaving the system unattended.

---

## 11. Running the application

The likely main entry point is `SYSTEM/main.py`, based on the current folder layout. Confirm that it imports the correct modules and includes a direct-run entry point before starting it.

From the repository root, the candidate command is:

```bash
python SYSTEM/main.py
```

If the code is designed to be launched from inside `SYSTEM/`, the appropriate command may instead be:

```bash
cd SYSTEM
python main.py
```

Use the method supported by the actual imports and configuration. Do not run both approaches interchangeably without checking how relative paths are resolved.

After starting the application:

1. Read the terminal output for the actual local URL/port and startup errors.
2. Open the reported URL in a browser.
3. Connect the ESP32 to the configured network.
4. Verify that readings are updating.
5. Check weather API status separately from local sensor status.
6. Test each ML module with valid inputs.
7. Test pump controls under safe conditions.

If the application uses a Flask CLI instead of a direct-run entry point, follow the run instructions defined in the source code.

---

## 12. Internet and weather API

The weather integration depends on the configured provider, endpoint, location, credentials, network availability, and response format.

### Configuration

1. Inspect `SYSTEM/weather.py` and `SYSTEM/config.py`.
2. Identify the exact API provider and required environment-variable names.
3. Create a local `.env` file if the application uses `python-dotenv`.
4. Add valid credentials and location configuration locally.
5. Test a successful response and an error/timeout response.

Use `.env.example` to document variable names without including real secrets.

### API behaviour

- A successful API request can provide weather data for the configured location.
- A forecast is an estimate supplied by the external provider, not a direct sensor measurement.
- API refresh frequency depends on the implementation and provider.
- API data should be checked for timestamps, units, missing values, and errors.
- API data may be unavailable without internet access.

Do not publish API keys, account identifiers, or private location information.

---

## 13. Irrigation modes and safety

### Manual mode

The user sends a pump command through the available interface. The command must be received and validated by the control path before the relay is switched.

### Automatic mode

The system uses its configured rules and/or ML output to support pump decisions. Verify the actual priority between the model prediction, local sensor rules, wetness checks, manual override, and safety conditions in the code.

### Safety checks to verify

- A reliable OFF command is available.
- Invalid or out-of-range sensor readings are handled.
- Repeated ON/OFF switching is controlled using appropriate filtering/hysteresis where implemented.
- Timed watering has a defined stop behaviour.
- Pump state is safe after startup, reboot, or communication loss.
- Rain/wetness restrictions behave as intended.
- The pump supply, relay, and wiring are correctly rated.

The FC-37/raindrop module detects wetness at its plate; it should not be described as measuring rainfall in millimetres. The system should not be left unattended until hardware and software safeguards have been tested.

---

## 14. Backup and offline behaviour

Different backup mechanisms solve different problems:

- **Source-code backup:** GitHub preserves versioned code.
- **Dataset backup:** Retain original datasets separately and keep track of their source and license.
- **Model backup:** Save trained model artifacts and document how to regenerate them.
- **API fallback:** A cached previous response is possible only if caching is implemented; cached data should be marked with its timestamp.
- **Offline sensor operation:** The ESP32 can sample sensors without internet if the firmware supports it.
- **Offline irrigation:** Local pump control can continue without the PC backend only if the required control logic resides on the ESP32.
- **Offline buffering:** Recovery of missed telemetry requires an implemented storage queue; reconnecting alone does not recover lost readings.

Test these behaviours individually. Do not assume that excluding model files from GitHub creates a backup, or that a device can run all ML predictions offline.

---

## 15. Troubleshooting

| Problem | Checks |
|---|---|
| Python package installation fails | Check Python version, virtual-environment activation, and the full pip error |
| Application does not start | Check imports, working directory, configuration, and the terminal traceback |
| ESP32 does not connect | Check Wi-Fi credentials, network band/support, signal, and serial output |
| Dashboard shows no readings | Check device IP/host, backend URL, protocol/port, firewall, and telemetry logs |
| Weather data unavailable | Check internet access, API credentials, endpoint, rate limits, and response errors |
| Model cannot be loaded | Check artifact path, filename, library versions, and whether training completed |
| Prediction fails | Check feature order, missing values, units, encoding, and preprocessing consistency |
| Moisture reading fluctuates | Check sensor power, wiring, ADC configuration, calibration, placement, and filtering |
| Pump does not switch | Check relay supply, active-high/active-low configuration, wiring, command path, and pump power |
| Pump switches repeatedly | Check sensor noise, calibration, hysteresis, soak delay, and control priorities |

When reporting an issue, include the exact error message, relevant terminal output, the command used, and the affected file—never credentials.

---

## 16. Security

- Keep `.env` out of Git.
- Never commit Wi-Fi passwords, API keys, or private tokens.
- Use `.env.example` with placeholder values only.
- Protect remote pump-control endpoints with authentication and appropriate authorization.
- Use secure communication for remote access and avoid exposing unauthenticated control ports directly to the public internet.
- Validate commands and incoming data.
- Check research and dataset redistribution licenses.
- Inspect Git history before publishing to ensure secrets were not committed previously.

Removing a secret from the current file does not remove it from previous commits. If a credential was exposed, revoke or rotate it.

---

## 17. Reproducibility and limitations

For reproducible results, record:
- Python and package versions.
- Dataset source, version, and license.
- Random seeds and train/validation/test split method.
- Feature and target definitions for every module.
- Model parameters and evaluation metrics.
- Model artifact filenames and generation date.
- ESP32 board settings, sensor pin mapping, and firmware dependencies.

Forecast accuracy, water savings, crop-yield improvements, remote-access availability, and offline recovery should be reported only when supported by tests or measured results.

---

## 18. Research references

The `PAPERS/` folder contains research material used during project development. Add a reference list here with each paper's title, authors, publication year, DOI or official URL, and a brief note describing its relevance.

Only redistribute full-text PDFs when their license or permission allows it. Otherwise, link to the publisher or repository record.

---

## 🌱 Project statement

**Krushi Mithra connects environmental sensing, agricultural machine learning, weather information, and irrigation control to explore practical approaches to data-driven farming.**

Built as an engineering project to learn and demonstrate IoT, embedded systems, data preparation, machine learning, API integration, and hardware–software integration.
