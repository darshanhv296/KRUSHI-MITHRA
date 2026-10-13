# Krushi Mithra — Smart Farming & Intelligent Irrigation System

> **Project:** IoT Farming — Landform Analysis and Smart Farming Using Machine Learning Techniques  
> **Platform:** ESP32 + Python + Flask + machine learning + Open-Meteo weather data  
> **Timezone used by the project:** `Asia/Kolkata`

Krushi Mithra combines field sensors, an ESP32 controller, a Python runtime, weather data and four machine-learning modules. The ESP32 reads the physical sensors and controls the pump relay. The Python application coordinates the models, weather service, control decisions and backend APIs.

This README describes the repository layout, each source file, dataset-generation scripts, ML pipeline, hardware wiring, first-time Wi-Fi provisioning portal, normal startup, API checks and troubleshooting.

> **Important source/configuration note:** File names, routes and data schemas below are based on the supplied project files. The exact Wi-Fi access-point SSID/password, firmware-specific captive-portal behaviour, and any local pin differences must be confirmed against the `CODE/CODE.ino` currently flashed to your ESP32. Do not assume a default AP password if the firmware does not display or define one.

---

## Contents

1. [System overview](#1-system-overview)
2. [Repository structure](#2-repository-structure)
3. [What each source file does](#3-what-each-source-file-does)
4. [Four ML modules](#4-four-ml-modules)
5. [Dataset generation scripts](#5-dataset-generation-scripts)
6. [Cleaning, preprocessing, training and testing](#6-cleaning-preprocessing-training-and-testing)
7. [Hardware and wiring](#7-hardware-and-wiring)
8. [ESP32 Wi-Fi setup portal: first-time login](#8-esp32-wi-fi-setup-portal-first-time-login)
9. [Start the complete Python system](#9-start-the-complete-python-system)
10. [Dashboard and backend endpoints](#10-dashboard-and-backend-endpoints)
11. [Irrigation logic and safety](#11-irrigation-logic-and-safety)
12. [Troubleshooting](#12-troubleshooting)
13. [Data quality and model limitations](#13-data-quality-and-model-limitations)
14. [Quick checklist](#14-quick-checklist)

---

## 1. System overview

### Main data flow

```text
Capacitive soil-moisture sensor ─┐
DHT11 (air temperature/humidity) ├──> ESP32 ──> Wi-Fi / HTTP API
DS18B20 (soil temperature)       ┤                 │
FC-37 / rain module (wetness)     ┤                 ▼
LDR (light level)                ┘          Python runtime
                                               │
                         ┌─────────────────────┼────────────────────┐
                         ▼                     ▼                    ▼
                    Flask backend        Open-Meteo service     ML engine
                         │                                          │
                         └─────────────────────┬────────────────────┘
                                               ▼
                                      Irrigation controller
                                               │
                                               ▼
                                         ESP32 relay
                                               │
                                               ▼
                                            Water pump
```

### Responsibilities

- **ESP32 firmware (`CODE/CODE.ino`)**: reads connected sensors, provides device HTTP endpoints, handles Wi-Fi provisioning, and executes hardware commands through the relay.
- **Python runtime (`SYSTEM/main.py`)**: starts the Flask backend, connects to the ESP32, initializes ML and weather services, and runs background polling/control/display loops.
- **ML engine (`SYSTEM/ml.py`)**: loads trained model artifacts and prepares predictions for M1–M4.
- **Irrigation controller (`SYSTEM/control.py`)**: combines current state, configured thresholds, rain protection and model outputs to decide whether to request pump operation.
- **ESP32 client (`SYSTEM/esp32.py`)**: sends HTTP requests to the ESP32 and can discover it on the local network using UDP port `4210`.
- **Flask backend (`SYSTEM/backend.py`)**: exposes status, telemetry, Wi-Fi provisioning proxy and ML API endpoints on port `5000`.
- **Weather service (`SYSTEM/weather.py`)**: retrieves current, forecast and historical weather information from Open-Meteo.
- **Dataset scripts**: fetch/build historical training data. They are offline data-preparation tools, not the live pump controller.

The normal dashboard is served by the ESP32 at `http://<ESP32-IP>/`. The Python backend runs on the computer at `http://127.0.0.1:5000/` for local API access.

---

## 2. Repository structure

The following is the expected high-level layout. Generated files and model-artifact folders appear after running the relevant scripts.

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
│   ├── Crop_Recommendation.csv
│   ├── Rain_Prediction.csv
│   ├── Smart_Irrigation.csv
│   ├── Soil_Moisture_Prediction.csv
│   ├── clean.py
│   ├── preprocess.py
│   ├── CLEANED/
│   └── PREPROCESSED/
│       ├── M1_Crop_Recommendation/
│       ├── M2_Context_Aware_Irrigation/
│       ├── M3_Future_Soil_Moisture/
│       └── M4_Rainfall_Forecasting/
│
├── MODELS/
│   ├── train.py
│   ├── test.py
│   ├── TRAINED_MODELS/
│   ├── TRAINING_RESULTS/
│   └── TEST_RESULTS/
│
├── PAPERS/
│   └── Research papers and reference material
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

**Paths matter.** Run the commands from the repository root unless a step explicitly says otherwise. Dataset scripts generally write outputs relative to their own script location; model and runtime artifact locations are determined by the path settings in the project's configuration and training scripts. If your local folder names differ, follow the paths in your actual files.

---

## 3. What each source file does

### `CODE/` — firmware and dataset-building scripts

#### `CODE/CODE.ino`
The ESP32 firmware. It is responsible for the physical-device side of the project:
- initialises the sensors and relay pin;
- reads and reports telemetry;
- handles Wi-Fi setup/provisioning;
- exposes HTTP endpoints for sensor data, device status and control;
- accepts commands for system enable/disable, AUTO/MANUAL mode and pump control;
- enforces device-side rules before operating the relay.

Upload this file to the ESP32 using Arduino IDE or another compatible ESP32 build workflow. Select the correct ESP32 board and serial port. Install the libraries included by the sketch's `#include` statements before compiling. The sketch is the authoritative source for exact AP SSID/password, sensor libraries, route behaviour and pin definitions.

#### `CODE/crop.py`
Builds a crop-recommendation dataset using crop/location profiles and historical daily weather summaries fetched from the Open-Meteo archive. It calculates or assembles fields such as location, year, soil type, annual rainfall, temperature, humidity and soil moisture, then creates crop-labelled records.

- **Purpose:** data generation for **M1 — Crop Recommendation**.
- **Network:** internet access is needed when uncached weather data must be fetched.
- **Output:** the generated crop CSV is written according to `OUTPUT_FILE` in the script. Check that variable in your local copy before running; do not assume it always writes to `DATASET/Crop_Recommendation.csv`.
- **Important:** the script uses crop profiles and generated/balanced examples as part of dataset construction. This is not the same as collecting every training label directly from a field experiment.

#### `CODE/pump.py`
Builds an hourly irrigation dataset for the Tumakuru/Tiptur area from Open-Meteo historical weather/archive data. It converts volumetric soil moisture to a percentage, calculates rolling six-hour rainfall, extracts time features and applies hysteresis thresholds to create the pump-status label.

The script's shown thresholds are:
- `MOISTURE_ON_THRESH = 25.0`: pump-status label changes to ON when moisture falls below this threshold;
- `MOISTURE_OFF_THRESH = 30.0`: pump-status label changes to OFF after moisture recovers above this threshold.

The gap prevents rapid ON/OFF switching around one threshold. These are **dataset-label rules**, not automatically proof that the physical pump will use these exact thresholds at runtime.

- **Purpose:** data generation for **M2 — Context-Aware Irrigation Control**.
- **Output:** `tumakuru_irrigation_clean_hysteresis.csv` in the script's working/output location.
- **Cache:** `tumakuru_hourly_raw_cache.csv` avoids fetching the same raw data again.
- **Network:** internet access is required on the first uncached run.

#### `CODE/soil.py`
Builds an hourly soil-moisture dataset from Open-Meteo historical/archive observations. It gathers air temperature, air humidity, soil temperature, soil moisture and precipitation for configured regions, converts volumetric soil moisture to percent, and adds time features such as hour, date, month and day of year.

- **Purpose:** data generation for **M3 — Future Soil Moisture Prediction**.
- **Output:** check `OUTPUT_FILE` in the script; the final schema shown in the supplied file uses `time`, `date`, `hour`, `day_of_year`, `month`, `air_tempe`, `air_humidi`, `soil_tempe`, `soil_moist` and `rainfall_m`.
- **Cache:** the script uses a cache CSV to avoid refetching already retrieved regions.
- **Important:** this is historical/modelled gridded weather-service soil-moisture data, not a direct log from the project's capacitive probe. The live model pipeline needs the column schema expected by `DATASET/preprocess.py`; verify that the generated output is the intended input before replacing the project's locked source CSV.

#### `CODE/rain.py`
Downloads hourly historical weather for all 31 Karnataka districts over **2023–2025** and combines it into one district-wise CSV.

Variables include rainfall, air temperature, relative humidity, mean sea-level pressure, cloud cover and wind speed. It also adds date/time/year/month/day-of-year/hour features.

- **Purpose:** data generation for **M4 — Rainfall Forecasting**.
- **Output:** `karnataka_3years_hourly_all_districts.csv` in the script directory.
- **Cache:** per-district/per-year CSVs are stored in `raw_cache/`.
- **Network/time:** the first run can take a while because it fetches many district-years. Cached runs are faster.
- **Important:** the model-preprocessing pipeline expects the column names defined in `DATASET/clean.py` and `DATASET/preprocess.py`. Confirm that the output filename/schema is copied or mapped to `DATASET/Rain_Prediction.csv` before running the cleaning step.

### `DATASET/` — data preparation

#### `DATASET/clean.py`
Checks that the four locked input CSVs exist and match expected column names/order, reports duplicate rows and cleans the data into `DATASET/CLEANED/`.

Expected input files:
- `Crop_Recommendation.csv`
- `Rain_Prediction.csv`
- `Smart_Irrigation.csv` (the cleaner's current expected spelling for this file)
- `Soil_Moisture_Prediction.csv`

Expected schemas include:
- M1: `Location`, `Year`, `Soil_Type`, `Annual_Rainfall_mm`, `Temperature`, `Humidity`, `Soil_Moisture`, `Crop`
- M2: `timestamp`, `air_temperature_c`, `air_humidity_pct`, `soil_temperature_c`, `soil_moisture_pct`, `water_pump_status`
- M3: `site`, `timestamp`, `air_temperature_c`, `air_humidity_pct`, `soil_temperature_c`, `soil_moisture_pct`, `moisture_lag_1h`, `moisture_lag_3h`, `rain_now_mm`, `rain_sum_6h_mm`, `rain_next_1h_mm`, `hour_sin`, `hour_cos`, `target_moisture_1h`, `split`
- M4: `district`, `timestamp`, `date`, `time`, `year`, `month`, `day_of_year`, `hour`, `rainfall_mm`, `temperature_c`, `humidity_percent`, `pressure_msl_hpa`, `cloud_cover_percent`, `wind_speed_kmh`

**Filename warning:** the cleaner expects `Smart_Irigation.csv` (single “r” in “Irigation”) in its current configuration, while the repository listing or other files may show `Smart_Irrigation.csv` (double “r”). Keep the actual filename and `clean.py` configuration consistent. Do not rename a file blindly.

#### `DATASET/preprocess.py`
Reads the cleaned datasets, validates required fields, parses/sorts timestamps, constructs model features and creates train/test CSVs in `DATASET/PREPROCESSED/`.

Output folders:
- `M1_Crop_Recommendation/`
- `M2_Context_Aware_Irrigation/`
- `M3_Future_Soil_Moisture/`
- `M4_Rainfall_Forecasting/`

Each folder contains `train.csv` and `test.csv`. The script uses a chronological split for time-series modules so future timestamps are held out rather than randomly mixed with earlier observations. M3 builds a one-hour-ahead soil-moisture target; M4 holds out later timestamps for rainfall testing.

### `MODELS/` — model training and evaluation

#### `MODELS/train.py`
Loads the preprocessed training files and trains the supported classification/regression candidates. The script's model suite includes Random Forest, XGBoost and LightGBM for the relevant tasks. It saves model artifacts and training metrics in the output folders configured in the script.

Training cannot complete successfully if the expected `train.csv` files are missing or if required Python packages are not installed. The exact artifact filenames used by the live system must match those expected by `SYSTEM/ml.py`.

#### `MODELS/test.py`
Loads preprocessed test sets and available trained artifacts, performs predictions in batches, shows progress and computes task-appropriate metrics. For classification it reports classification metrics; for regression it reports MAE, RMSE and R². It saves consolidated CSV/JSON results under the configured test-results directory.

A test may be skipped when a test set or model file is missing. Read the console output rather than assuming every module/model was tested.

### `SYSTEM/` — live runtime

#### `SYSTEM/config.py`
Central settings for the system: ESP32 connection settings, polling intervals, location/weather configuration, runtime timing and ML defaults. Review this file before deployment. Keep Wi-Fi passwords out of source code and do not commit secrets.

#### `SYSTEM/esp32.py`
HTTP client for the ESP32. It communicates with the device's data/status/configuration and control endpoints, and provides Wi-Fi scan/connect/status helpers. It also supports UDP discovery on port `4210`; discovery generally requires the computer and ESP32 to be on the same local network and local firewall rules to permit the traffic.

#### `SYSTEM/weather.py`
Open-Meteo integration for current conditions, forecasts and historical weather references. It provides location/district helpers and rainfall aggregation used by ML and control. The supplied code uses Open-Meteo endpoints and does not show a required API key. Internet access is still required for fresh remote weather requests.

#### `SYSTEM/ml.py`
Loads trained artifacts and runs M1–M4 predictions. The supplied runtime expects these model-artifact families:
- M1: Random Forest under `M1_Crop_Recommendation/`
- M2: LightGBM under `M2_Context_Aware_Irrigation/`
- M3: LightGBM under `M3_Future_Soil_Moisture/`
- M4: XGBoost under `M4_Rainfall_Forecasting/`, with a compatibility fallback for `M4_Hourly_Rainfall/`

The loader also expects supporting preprocessing pipelines, feature definitions and (where relevant) label encoders. A model file alone may not be enough; the complete artifact format and folder name must match what `ml.py` expects.

M3 stores recent sensor history in `SYSTEM/data/m3_sensor_history.jsonl` so it can form lag features for future-moisture prediction. This history is runtime state, not the source training dataset.

**Soil moisture calibration caution:** the supplied ML helper currently maps raw ADC values using `dry=4095` and `wet=1500`. Your physical capacitive sensor readings previously observed around raw 2708–2728 in open air and 1123–1134 in water. Those endpoints are not equivalent to a calibrated dry-soil/wet-soil pair, but they show why the conversion must be checked against the actual sensor and ESP32 ADC range before trusting displayed percentages or pump thresholds.

#### `SYSTEM/control.py`
Central irrigation-decision layer. It evaluates AUTO/MANUAL mode, system enablement, moisture thresholds, rain protection and available M2/M3/M4 results, then sends the final requested pump command to the ESP32. It does not itself read GPIO or directly power the pump.

#### `SYSTEM/backend.py`
Flask API server. It serves backend status, device status/data, discovery, ML options/predictions and Wi-Fi setup proxy endpoints. It listens on port `5000` when started by the runtime.

#### `SYSTEM/main.py`
The main entry point. It starts the Flask backend on `0.0.0.0:5000` in a background thread (unless a backend is already responding), initializes the ESP32 client, loads ML artifacts, initializes the weather service, and starts background loops for sensor polling, weather updates, control and console status display.

Stop the runtime with **Ctrl+C** in the terminal.

---

## 4. Four ML modules

| Module | Task | Main inputs | Output |
|---|---|---|---|
| **M1** | Crop recommendation | Location, soil type, year, annual rainfall, temperature, humidity, soil moisture | Recommended crop/class |
| **M2** | Context-aware irrigation | Air temperature, air humidity, soil temperature, soil moisture, hour, day of week | Pump-status class |
| **M3** | Future soil moisture | Site, current environmental readings, recent moisture lags, current/recent rainfall, cyclic time features | Predicted soil moisture about one hour ahead |
| **M4** | Rainfall forecasting | District, year/month/day-of-year/hour, current rainfall and weather measurements | Rainfall prediction for the next hour through the live prediction flow |

**Live artifact selection is not identical to the training candidate list.** The live `SYSTEM/ml.py` expects the specific model families listed above. If training produces different names or artifact structures, either the training output or runtime loader must be aligned before deployment.

---

## 5. Dataset generation scripts

Run these only when you want to rebuild or refresh the source datasets. They require internet access for uncached Open-Meteo requests.

From the repository root, use the script paths below **if those files are located exactly as shown**:

```powershell
python CODE/crop.py
python CODE/pump.py
python CODE/soil.py
python CODE/rain.py
```

The generated filenames and output locations are controlled by each script. After running, inspect the printed `Saved To` path and copy/map the result to the exact CSV filename and schema expected by `DATASET/clean.py`.

### Do not confuse the data sources

- `CODE/crop.py`, `pump.py`, `soil.py`, and `rain.py` create datasets.
- `DATASET/clean.py` validates and cleans the four main CSVs.
- `DATASET/preprocess.py` prepares ML-ready splits.
- `MODELS/train.py` trains artifacts.
- `MODELS/test.py` evaluates artifacts.
- `SYSTEM/ml.py` uses the trained artifacts during live operation.

These scripts are not necessarily a one-click pipeline: their output filenames and schemas must line up with the locked input filenames and columns expected by the next step.

---

## 6. Cleaning, preprocessing, training and testing

### Install Python

Use a currently supported 64-bit Python version compatible with the packages in `requirements.txt`. Open PowerShell in the repository root.

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

If PowerShell blocks virtual-environment activation, use Command Prompt instead:

```bat
.venv\Scripts\activate.bat
```

If the project does not include a complete `requirements.txt`, install the dependencies imported by the scripts, including the relevant packages such as `pandas`, `numpy`, `requests`, `scikit-learn`, `joblib`, `xgboost`, `lightgbm`, `flask`, and the Open-Meteo client packages used by `weather.py`/`rain.py`. Prefer the repository's actual requirements file where available.

### Recommended data/model order

1. Ensure the four canonical source CSVs are present in `DATASET/` and have the expected names/columns.
2. If rebuilding data, run the four `CODE/*.py` dataset scripts and map their outputs to the canonical inputs after checking their schema.
3. Clean the canonical CSVs:

   ```powershell
   python DATASET/clean.py
   ```

4. Prepare train/test splits:

   ```powershell
   python DATASET/preprocess.py
   ```

5. Train the candidate models:

   ```powershell
   python MODELS/train.py
   ```

6. Test the available trained models:

   ```powershell
   python MODELS/test.py
   ```

7. Confirm that the model artifact names/folders produced by training are the same ones expected by `SYSTEM/ml.py`.

**Do not run training blindly over valuable existing artifacts.** Keep a backup of existing datasets and models before regenerating them. The weather/dataset generation scripts can take time and may produce new rows or distributions.

---

## 7. Hardware and wiring

The pin mapping below is the project's documented ESP32 arrangement. Verify it against the exact firmware being flashed.

| Component | Signal | ESP32 connection | Purpose |
|---|---|---:|---|
| DHT11 | DATA | GPIO 25 | Air temperature and relative humidity |
| Capacitive soil-moisture sensor | Analog output (AOUT) | GPIO 35 | Soil-moisture ADC reading |
| FC-37 / rain module | AO | GPIO 34 | Analog wetness reading, if used by firmware |
| FC-37 / rain module | DO | GPIO 27 | Digital wetness state, if used by firmware |
| LDR module | DO | GPIO 26 | Light/day-night state |
| DS18B20 | DATA | GPIO 4 | Soil temperature; use the required pull-up resistor for the sensor/bus |
| Relay module | Control input | GPIO 23 | Pump ON/OFF command |

### Wiring and power safety

1. **Turn off all power before wiring.**
2. Connect sensor grounds to ESP32 GND so signals share a common reference.
3. ESP32 GPIOs are **not 5 V tolerant**. Do not feed 5 V into an ESP32 input or ADC pin. Power sensor modules at a suitable voltage and ensure their output signal never exceeds the ESP32 input limit.
4. Check the sensor module's pin labels (`VCC`, `GND`, `AO`, `DO`, `DATA`) rather than relying only on wire colours.
5. For DS18B20, use the appropriate pull-up resistor between DATA and its supply as required by the module/sensor wiring.
6. Do not power a water pump from an ESP32 GPIO. Use a relay/driver rated for the pump's voltage and current, and a separate pump supply rated for the load.
7. Confirm the relay module is compatible with the ESP32's logic level and the firmware's active level before connecting the pump.
8. Keep water away from the ESP32, relay contacts, mains wiring and exposed connections. Use a protected enclosure and proper insulation.
9. Test the controller with the pump supply disconnected first. Verify sensor readings and relay status before a wet test.
10. Never leave a pump unattended during initial testing. Provide a safe way to cut power.

### Relay behaviour

The documented firmware arrangement uses GPIO23. The previously reviewed firmware logic treats relay activation as **active LOW** and releases the relay by changing the pin to a high-impedance/input state. This is unusual compared with a simple always-output HIGH/LOW implementation, so inspect the exact `CODE.ino` before modifying it. Do not change relay polarity without testing the actual module.

---

## 8. ESP32 Wi-Fi setup portal: first-time login

The ESP32 setup portal is used to give the device the credentials for the Wi-Fi network that the computer/phone and ESP32 should share. The ESP32's setup access point is normally reachable at **`192.168.4.1`**. The setup page route exposed by the device is `/wifi-setup`.

### Before starting

- Flash the correct `CODE/CODE.ino` firmware.
- Power the ESP32 from a stable USB or regulated supply.
- Keep your phone or laptop close to the ESP32.
- Know the Wi-Fi network name (SSID) and password you want the ESP32 to join.
- For initial provisioning, use a compatible 2.4 GHz Wi-Fi network if the firmware/board configuration requires it; ESP32 Wi-Fi is 2.4 GHz, not 5 GHz.
- Avoid entering credentials for a captive network that requires a browser login, unless the firmware specifically supports it.

### Method A — Connect directly to the ESP32 setup access point

1. **Power on the ESP32.**
2. Open Wi-Fi settings on your phone or laptop.
3. Look for the setup network broadcast by the ESP32. Its exact SSID is defined by the firmware currently flashed to the board; inspect the Wi-Fi/AP name in `CODE/CODE.ino` or the serial startup log. Do not assume an SSID or password that is not defined there.
4. Select that network and connect. If the network is password-protected, enter the password configured by the firmware. If it is open, your device may show a security warning; connect only to your own ESP32 setup network.
5. Your phone may say **“No internet”**. That is expected while connected directly to the ESP32 setup AP. Choose to remain connected to it for setup.
6. Open a browser and type this address into the address bar:

   **`http://192.168.4.1/`**

   If the root page does not show the setup interface, try:

   **`http://192.168.4.1/wifi-setup`**

7. If the page includes a Wi-Fi scan button, scan for available networks and choose your router/hotspot SSID.
8. Enter the Wi-Fi password carefully. Passwords are case-sensitive. Submit/connect.
9. Wait for the device to attempt connection. Do not repeatedly submit the form or power-cycle the ESP32 while it is connecting.
10. If the portal reports success or displays the new IP address, note that IP address.
11. Your phone/laptop may disconnect from the ESP32 AP once the ESP32 joins the router. This is normal.
12. Connect your computer/phone back to the **same router or hotspot** that the ESP32 joined.
13. Open the ESP32 dashboard using the IP address shown by the portal/serial log:

   `http://<ESP32-IP>/`

   Example only: if the device reports `192.168.1.50`, open `http://192.168.1.50/`. The example IP is not a guaranteed address.
14. The runtime also prints an ESP32 setup URL in the form `http://<ESP32-IP>/wifi-setup`.

### Method B — Configure Wi-Fi through the Python backend

The Flask backend includes Wi-Fi provisioning proxy endpoints and uses `192.168.4.1` as the setup AP address. When the computer is connected to the ESP32 setup network, start the Python runtime and use the backend's Wi-Fi setup UI/endpoints if that interface is present in your project.

- Backend base URL: `http://127.0.0.1:5000`
- Setup AP address: `192.168.4.1`
- The backend looks for optional setup HTML files named `44.html` or `wifi_setup.html` in the project root or `SYSTEM/`.
- If those files are not present, the ESP32's own setup page is the direct route to use.

Do not confuse the ESP32 setup page with the Python dashboard/API. The ESP32 hosts the device dashboard; the Python backend provides the separate ML/control API.

### After Wi-Fi setup

For normal operation, the computer running Python and the ESP32 must be on the same local network. If the router assigns a new IP after reboot, use the new IP shown in the router/serial log or allow the project's UDP discovery to find the device. The runtime's discovery implementation listens for the device response on UDP port `4210`; host firewall rules can block discovery.

### If the setup page will not open

1. Confirm the phone/laptop is still connected to the ESP32 AP, not silently back on home Wi-Fi/mobile data.
2. Type the address exactly as `http://192.168.4.1/`; HTTPS may fail because the local setup portal is HTTP.
3. Disable mobile data temporarily if the phone keeps switching away from a Wi-Fi network with no internet.
4. Try the alternate path `http://192.168.4.1/wifi-setup`.
5. Forget the ESP32 setup network and reconnect.
6. Restart the ESP32 and watch the serial monitor for AP startup messages and the exact SSID.
7. Confirm that the firmware actually starts a setup AP and serves the route. If the AP never appears, the issue is firmware/power/startup—not the Python ML models.
8. If the portal loads but cannot find the home network, check 2.4 GHz compatibility, SSID visibility, password spelling and router security settings.
9. If it connects but the dashboard stops responding, reconnect the computer to the router and find the ESP32's new LAN IP.
10. Do not factory-reset or erase saved configuration unless necessary; the exact reset procedure depends on the firmware.

### Wi-Fi and privacy

The supplied backend code says it does not store Wi-Fi passwords. Enter credentials only into your own device's setup portal. Do not publish router passwords, tokens or other secrets in screenshots, logs, Git commits or this README.

---

## 9. Start the complete Python system

After Wi-Fi provisioning, connect the computer and ESP32 to the same local network.

### 1. Configure the project

Review `SYSTEM/config.py`. Check:
- ESP32 IP/hostname behaviour;
- polling intervals;
- location and weather settings;
- runtime paths and model-artifact paths;
- soil-moisture thresholds and rain protection settings.

Do not put router credentials into Python source. Keep local secrets out of version control.

### 2. Check trained model artifacts

Before starting the live runtime, make sure the artifact folders expected by `SYSTEM/ml.py` exist and contain the pipeline/model/encoder files required by the loader. If not, complete preprocessing and training first.

### 3. Start the runtime

From the repository root, with the virtual environment active:

```powershell
python SYSTEM/main.py
```

During initialization the program starts the backend, attempts to connect to the ESP32, loads ML models, initializes weather services and starts background loops. It prints the ESP32 dashboard and setup URLs when an ESP32 client is configured.

The backend is started on `0.0.0.0:5000`, meaning it listens on available network interfaces. On a trusted private network, the local health endpoint is:

`http://127.0.0.1:5000/api/health`

Use **Ctrl+C** in the runtime terminal to stop the application cleanly.

### 4. Access the device dashboard

Open the IP printed by the application or reported by the ESP32:

`http://<ESP32-IP>/`

To open the device setup page after it has joined Wi-Fi:

`http://<ESP32-IP>/wifi-setup`

The actual address depends on the router's DHCP assignment.

---

## 10. Dashboard and backend endpoints

The exact set of routes is defined by the firmware and backend source. The following routes are present in the supplied Python code or documented by the ESP32 client.

### Python backend (port 5000)

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/health` | Basic backend health check |
| GET | `/api/backend/status` | Backend/runtime status |
| GET | `/api/device/discover` | Try to discover ESP32 on the LAN |
| GET | `/api/device/status` | Device status |
| GET | `/api/device/ip` | Current configured/discovered device IP |
| GET | `/api/data` | Latest device telemetry |
| GET | `/api/ml/status` | Model availability/status |
| GET | `/api/ml/options` | Available ML prediction options |
| POST | `/api/ml/live` | Run the combined live M1–M4 prediction flow |
| GET | `/api/wifi/scan` | Proxy Wi-Fi scan to the ESP32 |
| POST | `/api/wifi/connect` | Proxy Wi-Fi connection request to the ESP32 |
| GET | `/api/wifi/status` | Proxy Wi-Fi connection status |

Health-check example:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/health
```

Model-status example:

```powershell
Invoke-RestMethod http://127.0.0.1:5000/api/ml/status
```

The live prediction endpoint expects a JSON POST body. Example payload:

```json
{
  "location": "Chikkamagaluru",
  "soil_type": "Red Soil",
  "irrigation": "Drip",
  "crop_type": null,
  "best_time": "Any",
  "region": "Chikkamagaluru"
}
```

This example demonstrates the request shape; it is not a guarantee that predictions succeed if sensor telemetry or model artifacts are missing.

### ESP32 device endpoints

The Python client communicates with device routes including:
- `GET /api/data`
- `GET /api/status`
- `GET /api/config`
- `POST /api/system`
- `POST /api/mode`
- `POST /api/pump`
- `POST /api/config`
- `GET /api/wifi/scan`
- `POST /api/wifi/connect`
- `GET /api/wifi/status`

These are device endpoints served by the ESP32, not by the Flask server. Open them using the ESP32 IP, not `127.0.0.1:5000`.

---

## 11. Irrigation logic and safety

### Operating modes

- **AUTO:** the control layer evaluates sensor telemetry, configured moisture thresholds, rain protection and available model outputs before requesting pump operation.
- **MANUAL:** an operator can request pump control through the supported interface, subject to the device's safety rules.
- **System disabled:** should prevent normal automatic operation. Confirm the actual firmware's handling before unattended use.

### Control ownership

The Python controller chooses the requested action, but the ESP32 owns the physical relay and pump execution. The Python process cannot switch a physical pump if the ESP32 is offline or the device endpoint cannot be reached.

### Safe commissioning sequence

1. Test the ESP32 and each sensor with the pump power disconnected.
2. Check readings in open air, in representative soil and after watering.
3. Verify the displayed moisture percentage against a calibration made for the actual probe and soil.
4. Verify rain/wetness polarity by testing both dry and wet conditions.
5. Test AUTO/MANUAL and system enable/disable behaviour without a live pump load first.
6. Test the relay with a safe indicator/load if available.
7. Connect the pump supply only after relay polarity, current rating and wiring are verified.
8. Observe a complete watering pulse and soak period; make sure the sensor is not directly next to the water outlet, which can create a misleading local reading.

### Soil-moisture calibration

The model helper in the supplied `SYSTEM/ml.py` currently converts raw ADC values with `dry=4095` and `wet=1500`. Previously observed readings for this project's sensor were roughly `2708–2728` in open air and `1123–1134` in water. Open air is not the same as dry soil, and immersion in water is not the same as saturated soil. Calibrate using the actual ESP32 ADC, probe, soil type and installation depth before relying on a displayed percentage for automatic irrigation.

---

## 12. Troubleshooting

| Symptom | Likely checks |
|---|---|
| ESP32 setup Wi-Fi does not appear | Check USB/power, firmware upload, serial startup logs and AP setup code in `CODE/CODE.ino`. |
| `192.168.4.1` does not load | Reconnect to the ESP32 AP, use HTTP rather than HTTPS, try `/wifi-setup`, and check that the firmware serves the route. |
| ESP32 joins Wi-Fi but Python cannot find it | Put PC and ESP32 on the same LAN; check the device IP; inspect `SYSTEM/config.py`; allow local UDP/HTTP traffic through the firewall. |
| Dashboard works but ML fails | Check `DATASET/PREPROCESSED/`, trained artifact folders, exact artifact names, `SYSTEM/ml.py` logs and `/api/ml/status`. |
| Backend health works but ESP32 is offline | Test `http://<ESP32-IP>/api/status`; verify Wi-Fi and LAN IP; try device discovery. |
| `FileNotFoundError` for CSV | Confirm current working directory, expected filename and exact spelling; check the paths printed by the scripts. |
| Cleaner reports missing columns | Compare the CSV header to `DATASET/clean.py`'s `EXPECTED_COLUMNS`; do not silently rename or reorder columns without checking downstream code. |
| Preprocessing fails on timestamp or missing data | Inspect invalid timestamps/NaNs reported by the script; check that the right source dataset was used. |
| Training skips a module | Confirm that its `train.csv` exists in the expected M1–M4 folder. |
| Runtime says model/artifact missing | Confirm training completed and that the artifact names, directory names and formats match `SYSTEM/ml.py`. |
| Moisture reading jumps or seems wrong | Check ADC wiring, shared ground, power noise, filtering, calibration and sensor placement. Keep the probe away from the direct water stream. |
| Pump turns on/off unexpectedly | Verify AUTO/MANUAL mode, system enable flag, soil threshold configuration, rain-protection state, sensor calibration and relay active level. |
| Weather requests fail | Confirm internet connectivity, Open-Meteo availability, system date/time and request logs. Cached historical data may still exist. |
| Wi-Fi scan/connect works on ESP32 but not through backend | Ensure the PC can reach the setup AP while provisioning and check the backend logs and `/api/wifi/*` endpoints. |

---

## 13. Data quality and model limitations

1. **Historical weather is not identical to local sensor data.** Open-Meteo soil moisture is a gridded/modelled weather variable, not a measurement from this project's exact capacitive probe.
2. **Dataset labels influence model behaviour.** M2 pump labels are derived from the script's threshold/hysteresis rules. A model trained on those labels learns that rule pattern; it is not independent proof of agronomic optimality.
3. **Synthetic/balanced crop rows are not additional independent field experiments.** Evaluate on appropriately separated data and report the data-generation method honestly.
4. **Use chronological test splits for time series.** Randomly mixing nearby hours across train and test can produce over-optimistic evaluation.
5. **Feature schema must match exactly.** Column spelling, capitalization, units, category labels and feature order/meaning must be consistent between dataset preparation, training and live inference.
6. **A successful model load does not guarantee reliable field performance.** Validate model predictions in the intended region, soil, crop and irrigation setup.
7. **Automatic irrigation is a physical action.** Start with conservative thresholds, supervise the pump, and retain a manual power cut-off.
8. **Network loss must be treated as a control fault.** If the Python controller cannot reach the ESP32, do not assume a new command has been applied.

---

## 14. Quick checklist

### Before the first run
- [ ] Upload the intended `CODE/CODE.ino`.
- [ ] Confirm the firmware's actual setup AP SSID/password from the sketch or serial output.
- [ ] Connect to the ESP32 AP and open `http://192.168.4.1/`.
- [ ] Join the ESP32 to the target 2.4 GHz Wi-Fi network.
- [ ] Reconnect the PC to the same network.
- [ ] Open `http://<ESP32-IP>/` and verify the dashboard.
- [ ] Verify sensor readings and relay polarity with the pump supply disconnected.
- [ ] Check canonical dataset filenames and columns.
- [ ] Run clean → preprocess → train → test if artifacts need to be built.
- [ ] Confirm that `SYSTEM/ml.py` can load all required live artifacts.
- [ ] Start `python SYSTEM/main.py`.
- [ ] Check `http://127.0.0.1:5000/api/health` and `/api/ml/status`.
- [ ] Test AUTO/MANUAL and the pump under supervision.

### Normal operation
1. Power the ESP32 and hardware safely.
2. Ensure the ESP32 and computer are connected to the same LAN.
3. Activate the Python virtual environment.
4. Run `python SYSTEM/main.py`.
5. Open the ESP32 dashboard.
6. Check telemetry, weather freshness, model status and current control mode.
7. Supervise automatic irrigation and stop the pump if readings or behaviour look unsafe.

---

**Project name:** Krushi Mithra  
**Project category:** IoT Smart Farming, Machine Learning and Intelligent Irrigation  
**Primary runtime:** ESP32 firmware + Python/Flask  
**Weather source:** Open-Meteo  
**ML tasks:** M1 Crop Recommendation · M2 Context-Aware Irrigation · M3 Future Soil Moisture Prediction · M4 Rainfall Forecasting
