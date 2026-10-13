# 🌾 Krushi Mithra (ಕೃಷಿ ಮಿತ್ರ) — Autonomous Smart Farming & Edge Irrigation

[![Platform: ESP32](https://img.shields.io/badge/Hardware-ESP32-green.svg?logo=espressif)](https://www.espressif.com/)
[![Runtime: Python 3.10+](https://img.shields.io/badge/Runtime-Python%203.10%2B-blue.svg?logo=python)](https://www.python.org/)
[![Framework: Flask](https://img.shields.io/badge/Backend-Flask-lightgrey.svg?logo=flask)](https://flask.palletsprojects.com/)
[![Weather: Open--Meteo](https://img.shields.io/badge/Weather-Open--Meteo-orange.svg)](https://open-meteo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

**Krushi Mithra** is a closed-loop IoT precision agriculture ecosystem that bridges soil physics, edge firmware safety, and predictive machine learning. 

Instead of treating irrigation as a blunt timer or an isolated manual switch, Krushi Mithra unifies **real-time field sensor readings**, **microclimate weather predictions**, and **four localized machine learning models** to calculate whether a crop genuinely needs irrigation—while enforcing autonomous electrical and mechanical failsafes on the ESP32 microcontroller even if the central computer goes completely offline.

---

## 📑 Table of Contents

1. [What the System Does](#1-what-the-system-does)
2. [How the System Works (Under the Hood)](#2-how-the-system-works-under-the-hood)
3. [End-to-End Hardware & Sensor Wiring Guide](#3-end-to-end-hardware--sensor-wiring-guide)
4. [Captive Wi-Fi Setup Portal & Network Lifecycle](#4-captive-wi-fi-setup-portal--network-lifecycle)
5. [In-Depth Machine Learning Suite (M1–M4)](#5-in-depth-machine-learning-suite-m1m4)
6. [Dual-Layer Safety & Fail-Safe Irrigation Engine](#6-dual-layer-safety--fail-safe-irrigation-engine)
7. [Repository File Map & Code Execution Flow](#7-repository-file-map--code-execution-flow)
8. [Step-by-Step Deployment & Operating Guide](#8-step-by-step-deployment--operating-guide)
9. [Complete REST API & Discovery Specification](#9-complete-rest-api--discovery-specification)
10. [Troubleshooting & Diagnostics](#10-troubleshooting--diagnostics)
11. [Project Demonstration Video](#11-project-demonstration-video)
12. [License](#12-license)

---

## 1. What the System Does

Modern irrigation systems suffer from two major extremes:
- **Dumb timers / simple thresholds:** Pumps run on fixed schedules or single raw sensor pins, frequently watering right before heavy rain, wasting power and causing root rot.
- **Overly complex cloud architectures:** When farm Wi-Fi drops or the internet disconnects, the cloud cannot toggle the relay, leaving pumps stuck either running continuously or completely locked out.

**Krushi Mithra solves this with a cooperative edge architecture:**

- **Continuous In-Field Telemetry:** Continuously polls 5 environmental sensors (capacitive soil moisture, digital 1-Wire soil temperature, analog wetness, digital rain interrupt, day/night light level, and ambient air temperature/humidity).
- **Embedded Web Engine:** Hosts a futuristic, responsive bio-cybernetic dashboard on the ESP32 itself (`http://krushi-mithra/`), displaying live telemetry, manual overrides, and ML prediction interfaces.
- **Predictive Irrigation Modeling:** An offline/online machine learning backend evaluates historical regional trends and live weather to predict if rain is coming in the next 60 minutes, if soil moisture will drop in the next hour, and whether irrigation is contextually justified.
- **Hardware Failsafes:** Even if your PC burns out or the Python backend crashes, the ESP32 microcontroller falls back to its own onboard closed-loop emergency state machine within 30 seconds.

---

## 2. How the System Works (Under the Hood)

### Master Architecture Diagram

```text
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                                FIELD LEVEL (EDGE HARDWARE)                             │
 │                                                                                        │
 │  [Capacitive Probe]  [DHT11 Air]  [DS18B20 Soil]  [FC-37 Wetness]  [LDR Light]         │
 │          │                 │             │                │             │              │
 │       GPIO35            GPIO25        GPIO04          GPIO34/27      GPIO26            │
 │       (Analog)         (Digital)     (1-Wire)       (Analog/DO)     (DO)               │
 │          │                 │             │                │             │              │
 │          └─────────────────┴──────┬──────┴────────────────┴─────────────┘              │
 │                                   ▼                                                    │
 │                   ┌───────────────────────────────┐                                    │
 │                   │     ESP32 MICROCONTROLLER     │                                    │
 │                   │ ───────────────────────────── │                                    │
 │                   │ • Multi-sample ADC Filtering  │                                    │
 │                   │ • SoftAP + Captive DNS Server │                                    │
 │                   │ • Dual WebServer (Port 80)    │                                    │
 │                   │ • UDP Discovery (Port 4210)   │                                    │
 │                   │ • Local Backup Safety Engine  │                                    │
 │                   └───────────────┬───────────────┘                                    │
 │                                   │ GPIO23 (Pseudo Open-Drain)                         │
 │                                   ▼                                                    │
 │                      ┌─────────────────────────┐                                       │
 │                      │   Optocoupled Relay     │──► [High-Power Water Pump]            │
 │                      └─────────────────────────┘                                       │
 └───────────────────────────────────┬────────────────────────────────────────────────────┘
                                     │ Local Wi-Fi (HTTP REST + UDP Broadcast)
                                     ▼
 ┌────────────────────────────────────────────────────────────────────────────────────────┐
 │                              GATEWAY / PC COMPUTE CLUSTER                              │
 │                                                                                        │
 │  ┌──────────────────────────────────────────────────────────────────────────────────┐  │
 │  │                         Flask Gateway Server (Port 5000)                         │  │
 │  │      • Automated UDP Handshake Listener       • Telemetry Ingestion Pipeline     │  │
 │  │      • Reverse ML Proxying                    • Real-Time Control Arbitration    │  │
 │  └───────┬──────────────────────────┬───────────────────────────────┬───────────────┘  │
 │          │                          │                               │                  │
 │          ▼                          ▼                               ▼                  │
 │  ┌───────────────┐        ┌──────────────────┐            ┌───────────────────┐        │
 │  │   ML Engine   │        │ Weather Pipeline │            │ Control Engine    │        │
 │  │ (SYSTEM/ml.py)│        │(SYSTEM/weather.py│            │(SYSTEM/control.py)│        │
 │  │ ───────────── │        │ ──────────────── │            │ ───────────────── │        │
 │  │ • M1: Crop    │        │ • Open-Meteo API │            │ • Auto / Manual   │        │
 │  │ • M2: Irrig   │        │ • 6-Hour Rolling │            │ • Rain Interlocks │        │
 │  │ • M3: Moisture│        │   Precipitation  │            │ • Timed Run Watch │        │
 │  │ • M4: Rainfall│        │ • Microclimate   │            │ • Relays Commands │        │
 │  └───────────────┘        └──────────────────┘            └───────────────────┘        │
 └────────────────────────────────────────────────────────────────────────────────────────┘
```

### Communication Flow: Step-by-Step

1. **Boot & Auto-Discovery:**
   - The ESP32 boots up, mounts its flash storage (`Preferences`), checks for saved Wi-Fi credentials, and joins the local LAN.
   - It begins listening on UDP port `4210`.
   - When `SYSTEM/main.py` starts on the host machine, it broadcasts a UDP packet with `KRUSHI_MITHRA_DISCOVER`.
   - The ESP32 replies instantly with its IP address and hostname. The Python runtime automatically binds to the ESP32 without any manual IP configuration.

2. **Telemetry Ingestion & Filtering:**
   - The ESP32 takes 5 analog samples with a $50\,\mu\text{s}$ settling delay to cancel high-frequency electrical noise from the capacitive probe and FC-37 rain board.
   - Telemetry is exposed via `GET /api/data`. The Python gateway and dashboard poll this endpoint periodically to sync field state.

3. **Inference & Control Cycle:**
   - In **AUTO Mode**, the Python engine evaluates:
     - Real-time soil moisture percentage against configured lower and upper thresholds.
     - Physical surface wetness via FC-37.
     - Open-Meteo's 6-hour rolling forecast and current rainfall.
     - Predictions from the M2 Context-Aware Irrigation model and M4 Rainfall model.
   - If irrigation is required and no rain is imminent, Python issues a command to `POST /api/pump` on the ESP32.

4. **Heartbeat & Watchdog Protection:**
   - Every telemetry polling request doubles as a heartbeat.
   - If the Python process terminates, the computer freezes, or the Wi-Fi signal drops for more than 30 seconds, the ESP32 logs `[AUTO][BACKUP] Backend unavailable - ESP32 local backup active` and takes over relay switching directly using its internal failsafe thresholds.

---

## 3. End-to-End Hardware & Sensor Wiring Guide

### Component Pin Configuration

| Component | Sensor Pin | ESP32 Pin | Signal Type | Operating VCC | Functional Description |
|---|---|:---:|---|---|---|
| **Capacitive Soil Moisture v1.2** | AOUT | `GPIO 35` | Analog Input | 3.3V | Voltage inversely proportional to soil water content; immune to probe corrosion. |
| **DHT11 Sensor** | DATA / OUT | `GPIO 25` | Digital Bi-directional | 3.3V or 5V | Measures ambient air temperature ($0\text{--}50\,^\circ\text{C}$) and relative humidity ($20\text{--}90\%$). |
| **DS18B20 Soil Probe** | DATA | `GPIO 04` | 1-Wire Digital | 3.3V or 5V | Waterproof digital probe embedded in soil; requires a $4.7\text{ k}\Omega$ pull-up resistor. |
| **FC-37 Rain Module** | AO (Analog) | `GPIO 34` | Analog Input | 3.3V | Measures rainwater density across surface traces (Raw: $4095$ dry down to $1800$ saturated). |
| **FC-37 Rain Module** | DO (Digital) | `GPIO 27` | Digital Input | 3.3V | Hardware comparator trigger; drops `LOW` on first drop of rain for rapid interrupts. |
| **LDR Sensor Module** | DO (Digital) | `GPIO 26` | Digital Input | 3.3V | Digital light comparator; drives `LOW` during daylight and `HIGH` at night. |
| **Relay Module (Optocoupled)** | IN / SIG | `GPIO 23` | Pseudo Open-Drain | 5V (Relay VCC) | Controls the submersible pump circuit; active `LOW` trigger. |

### Complete Schematic & Wiring Diagram

```text
                                  ESP32 DEVKIT V1
                                 ┌─────────────────┐
                    3V3 (REG) ───┤ 3V3         GND ├─── GROUND BUS (COMMON GROUND)
                                 │ EN           23 ├─── RELAY [IN] (Pump Control)
      DS18B20 [DATA] ────────────┤ 04           22 │
         │                       │ 36 (VP)      01 │
         │   [ 4.7k Resistor ]   │ 34           03 │
         └───[■■■■■■■■■■■■■■■]───┤ 35           21 │
                   ▲             │ 26           19 │
                   │ (to 3.3V)   │ 27           18 │
    FC-37 RAIN [AO] ─────────────┤ 34 (ADC1)    05 │
    SOIL PROBE [AOUT] ───────────┤ 35 (ADC1)    TX │
    LDR MODULE [DO] ─────────────┤ 26           RX │
    FC-37 RAIN [DO] ─────────────┤ 27           12 │
    DHT11 [DATA] ────────────────┤ 25           14 │
                                 └─────────────────┘
                                   │            │
                         GND ──────┘            └───── VIN (5V Supply In)

   RELAY & HIGH VOLTAGE WIRING (ISOLATED PUMP CIRCUIT):
   
         ESP32 GPIO 23 ────────────► RELAY [IN]
         ESP32 GND     ────────────► RELAY [GND]
         5V External   ────────────► RELAY [VCC]
         
         Pump Power Supply (+) ────► RELAY [COM] (Common)
         Pump Motor (+)        ────► RELAY [NO]  (Normally Open)
         Pump Motor (-)        ────► Pump Power Supply (-)
```

### Essential Electrical Principles

1. **ESP32 ADC Safety:**
   - The ESP32's ADC pins (`GPIO 34`, `GPIO 35`) can handle a maximum of **3.3V**. Powering the sensor board with 5V will result in analog voltages exceeding 3.3V, permanently damaging the microcontroller's internal ADC multiplexer. Always power sensor modules with **3.3V**.
2. **Pseudo Open-Drain Relay Driving:**
   - Many 5V relay boards pull their `IN` pin up to 5V internally. If you configure `GPIO 23` as `OUTPUT HIGH`, the ESP32 tries to drive 3.3V while the relay attempts to feed back 5V.
   - Krushi Mithra eliminates this using **Pseudo Open-Drain logic**:
     - **PUMP ON:** `pinMode(23, OUTPUT); digitalWrite(23, LOW);` (Pulls to GND, energizing the optocoupler).
     - **PUMP OFF:** `pinMode(23, INPUT);` (Leaves pin in high-impedance tri-state; relay pulls itself up safely to 5V without touching the ESP32).
3. **1-Wire Soil Temperature Pull-Up:**
   - The DS18B20 is a Dallas 1-Wire device and cannot transmit data over a floating line. You **must** solder a $4.7\text{ k}\Omega$ resistor between its `DATA` wire (`GPIO 4`) and the `3.3V` rail. Without this pull-up, the sensor will consistently read `-127.00 °C`.
4. **Inductive Spike Protection:**
   - **Never power your water pump from the ESP32's 3.3V or VIN pins.** Submersible DC pumps generate high inductive back-EMF spikes when cycling off, which will trigger the ESP32's brownout detector or damage its voltage regulator. Use a completely independent power adapter for the pump.

---

## 4. Captive Wi-Fi Setup Portal & Network Lifecycle

The firmware does not hardcode any router names or Wi-Fi passwords. It uses an embedded **DNS Captive Portal**:

```text
 ┌───────────────────────┐
 │ ESP32 Powered On      │
 └──────────┬────────────┘
            │
            ▼
 ┌───────────────────────┐       NO      ┌──────────────────────────────────┐
 │ Stored Wi-Fi in NVS?  ├──────────────►│ Start SoftAP: KRUSHI-MITHRA-SETUP│
 └──────────┬────────────┘               │ Start DNS Server on Port 53      │
            │ YES                        └─────────────────┬────────────────┘
            ▼                                              │
 ┌───────────────────────┐                                 ▼
 │ Attempt LAN Connection│                     ┌───────────────────────┐
 └──────────┬────────────┘                     │ Captive Portal Active │
            │                                  │ (http://192.168.4.1)  │
      ┌─────┴────────┐                         └───────────┬───────────┘
      │              │                                     │
      ▼ SUCCESS      ▼ TIMEOUT (15s)                       │ User Enters SSID & Password
 ┌──────────────┐ ┌──────────────────────┐                 │
 │ Join Farm LAN│ │ Launch Fallback      │◄────────────────┘
 │ Start mDNS   │ │ Hotspot & Portal     │
 └──────────────┘ └──────────────────────┘
```

### First-Time Provisioning Walkthrough

1. Power on the ESP32.
2. On your smartphone or laptop, open your Wi-Fi settings and look for the network:
   - **SSID:** `KRUSHI-MITHRA-SETUP`
   - **Password:** `Krushi1234`
3. Connect to the network. Your OS will automatically detect the captive portal and prompt you to "Sign in to network". If it does not, open your browser and navigate to:
   ```text
   http://192.168.4.1/wifi-setup
   ```
4. Click **Scan Networks**. The ESP32 scans 2.4 GHz channels and lists all visible SSIDs along with signal strength (RSSI).
5. Select your home or farm Wi-Fi, type the password, and click **CONNECT**.
6. The ESP32 will verify the connection. Once connected, it saves the credentials to non-volatile flash memory (`Preferences`), displays its assigned IP address on screen, and disables the setup access point.
7. Connect your device back to the same home or farm Wi-Fi router.
8. Access the dashboard at `http://krushi-mithra/` or `http://<ASSIGNED-IP>/`.
9. **Need to change routers?** Click the **WI-FI LOGOUT** button on the top-right of the dashboard. This wipes the stored credentials from NVS and re-opens the captive portal.

---

## 5. In-Depth Machine Learning Suite (M1–M4)

Krushi Mithra runs four distinct machine learning pipelines:

```text
┌────────────────────────────────────────────────────────────────────────────────────────┐
│                               MACHINE LEARNING PIPELINE                                │
├────────────┬─────────────────────────────┬──────────────────────────┬──────────────────┤
│ Module     │ Algorithm                   │ Input Features           │ Target Output    │
├────────────┼─────────────────────────────┼──────────────────────────┼──────────────────┤
│ M1: Crop   │ Random Forest Classifier    │ • District / Location    │ Recommended Crop │
│            │ (100 estimators)            │ • Soil Type              │ (e.g., Maize,    │
│            │                             │ • Irrigation Style       │ Cotton, Ragi)    │
│            │                             │ • Annual Rain, Temp, Hum │                  │
├────────────┼─────────────────────────────┼──────────────────────────┼──────────────────┤
│ M2: Irrig  │ LightGBM Classifier         │ • Air Temp & Humidity    │ Pump Decision    │
│            │ (Context-Aware)             │ • Soil Temp & Moisture   │ (0: OFF, 1: ON)  │
│            │                             │ • Hour, Day of Week      │                  │
├────────────┼─────────────────────────────┼──────────────────────────┼──────────────────┤
│ M3: Future │ LightGBM Regressor          │ • 1h & 3h Moisture Lags  │ Moisture % at    │
│     Moist  │                             │ • Cyclic Time (sin/cos)  │ $t + 1\text{ h}$ │
│            │                             │ • Real-time Soil Temp    │                  │
├────────────┼─────────────────────────────┼──────────────────────────┼──────────────────┤
│ M4: Rain   │ XGBoost Regressor           │ • Barometric MSL Press.  │ Expected Rain    │
│            │                             │ • Cloud Cover %          │ (mm) in Next 1h  │
│            │                             │ • Wind Speed, Humidity   │                  │
└────────────┴─────────────────────────────┴──────────────────────────┴──────────────────┘
```

### Detailed Breakdown of Each Module

#### M1: Crop Recommendation Engine
- **Purpose:** Identifies optimal crops based on soil composition and regional climate patterns.
- **How it Works:** Uses regional agricultural profiles across Karnataka districts. It compares seasonal rainfall, local soil characteristics (such as Red Sandy Loam, Deep Black Clay, or Laterite), and irrigation infrastructure (like Drip, Sprinkler, or Rainfed) using a Random Forest ensemble to recommend the highest-yielding crop.
- **Execution:** Users can run this directly from the web dashboard modal.

#### M2: Context-Aware Irrigation Model
- **Purpose:** Decides whether to irrigate based on contextual factors rather than just a simple moisture threshold.
- **How it Works:** A plant's water uptake varies throughout the day. Evaporation is significantly higher under direct midday sun compared to midnight. M2 incorporates ambient temperature, humidity, soil temperature, and current moisture levels to predict whether irrigation is contextually justified, helping prevent water waste.

#### M3: Next-Hour Soil Moisture Predictor
- **Purpose:** Forecasts soil moisture 60 minutes into the future to prevent unnecessary watering cycles.
- **How it Works:** Soil moisture dynamics depend heavily on recent moisture trends and cyclic temperature variations. The model uses:
  - **Lag features:** Soil moisture from 1 hour ago ($t-1$) and 3 hours ago ($t-3$), stored in `SYSTEM/data/m3_sensor_history.jsonl`.
  - **Cyclic encodings:**
    $$
    \mathrm{hour\_sin} = \sin\left(\frac{2\pi \cdot \mathrm{hour}}{24}\right), \quad
    \mathrm{hour\_cos} = \cos\left(\frac{2\pi \cdot \mathrm{hour}}{24}\right)
    $$
  - **Result:** If soil moisture is currently low ($24\%$) but the model predicts it will increase due to recent rain, irrigation is withheld.

#### M4: Microclimate Rainfall Forecaster
- **Purpose:** Predicts rainfall volume for the upcoming hour.
- **How it Works:** Uses historical meteorological observations across Karnataka alongside real-time parameters from Open-Meteo (including Mean Sea-Level Pressure, Cloud Cover percentage, Wind Speed, and Humidity). If heavy precipitation is expected in the next hour, irrigation is preemptively skipped.

---

## 6. Dual-Layer Safety & Fail-Safe Irrigation Engine

Krushi Mithra uses two layers of protection to ensure the system remains safe and reliable under all conditions:

```text
                                  PUMP COMMAND ISSUED
                                           │
                                           ▼
                       ┌───────────────────────────────────────┐
                       │      LAYER 1: SOFTWARE CHECKS         │
                       │ ───────────────────────────────────── │
                       │ • Is Open-Meteo 6h rain > threshold?  │
                       │ • Does M4 predict rain next hour?     │
                       │ • Is user manual duration valid?      │
                       └───────────────────┬───────────────────┘
                                           │ PASSED
                                           ▼
                       ┌───────────────────────────────────────┐
                       │      LAYER 2: HARDWARE WATCHDOG       │
                       │ ───────────────────────────────────── │
                       │ • Is FC-37 detecting rain?            │
                       │ • Has pump run > 10 min continuous?   │
                       │ • Is pump in 5 min cooldown window?   │
                       │ • Is sensor ADC reading valid?        │
                       └───────────────────┬───────────────────┘
                                           │ PASSED
                                           ▼
                                 RELAY SWITCHES ON
```

### Safety Rules Breakdown

1. **Physical Rain Lockout:**
   - If raindrops fall onto the FC-37 sensor, the digital output trips `LOW` and the analog wetness value drops below `3835`.
   - The ESP32 immediately turns off the relay within milliseconds, regardless of software or manual overrides.
2. **Auto Maximum Runtime & Cooldown:**
   - In automatic mode, the pump cannot run continuously for more than **10 minutes** (`AUTO_PUMP_MAX_RUN_MS`).
   - Once shut off, the system enforces a mandatory **5-minute cooldown** (`PUMP_COOLDOWN_MS`) before the relay can turn on again, preventing motor burnout and pump damage.
3. **Manual Session Limits:**
   - Manual activations require an explicit duration (in seconds, minutes, or hours, up to a maximum of 24 hours).
   - If manual mode is selected but no activation command is sent, the system automatically reverts to **AUTO** after **60 seconds** of inactivity (`MANUAL_IDLE_TIMEOUT_MS`).
4. **Onboard Emergency Backup Mode:**
   - If the Python server disconnects or the host machine goes offline for more than **30 seconds** (`BACKEND_TIMEOUT_MS`), the ESP32 automatically takes over using its internal failsafe thresholds:
     - **Soil Moisture** $\le 20\%$**:** Pump turns **ON**.
     - **Soil Moisture** $\ge 50\%$**:** Pump turns **OFF**.
5. **Soil Moisture Calibration:**
   - Converts raw ADC values ($0\text{--}4095$) into calibrated moisture percentages:
     $$
     \text{Moisture \%} = \mathrm{clamp}\left(
     \frac{2800 - \text{Raw ADC}}{2800 - 2112} \times 100,
     0, 100
     \right)
     $$
   - Raw ADC $\ge 2800$: Dry soil ($0\%$).
   - Raw ADC $\le 2112$: Saturated soil ($100\%$).

---

## 7. Repository File Map & Code Execution Flow

```text
IOT-FARM/
│
├── CODE/
│   ├── CODE.ino               # Master ESP32 production firmware (C++)
│   ├── crop.py                # Dataset builder for M1 (pulls crop profiles & weather)
│   ├── pump.py                # Dataset builder for M2 (hysteresis irrigation labels)
│   ├── soil.py                # Dataset builder for M3 (hourly soil moisture profiles)
│   └── rain.py                # Dataset builder for M4 (Karnataka district archive data)
│
├── DATASET/
│   ├── clean.py               # Checks column schemas, removes NaNs, and cleans CSVs
│   ├── preprocess.py          # Builds time splits, lag features, and train/test sets
│   ├── CLEANED/               # Sanitized source CSV outputs
│   └── PREPROCESSED/          # train.csv and test.csv splits for M1, M2, M3, and M4
│
├── MODELS/
│   ├── train.py               # Model training script (Random Forest, LightGBM, XGBoost)
│   ├── test.py                # Evaluation script (outputs MAE, RMSE, R2, and F1 metrics)
│   └── TRAINED_MODELS/        # Saved model binaries (.joblib, .json)
│
├── SYSTEM/
│   ├── backend.py             # Flask API server (port 5000)
│   ├── config.py              # Configuration settings, pin definitions, and timeouts
│   ├── control.py             # Irrigation decision engine and mode controller
│   ├── esp32.py               # UDP auto-discovery client (port 4210) & HTTP driver
│   ├── main.py                # Main orchestration entry point and background loops
│   ├── ml.py                  # Model loader, lag tracker, and inference handler
│   └── weather.py             # Open-Meteo weather client
│
├── requirements.txt           # Python dependencies
└── README.md                  # Project documentation
```

---

## 8. Step-by-Step Deployment & Operating Guide

### Step 1: Flashing the Firmware to the ESP32

1. Install the latest [Arduino IDE](https://www.arduino.cc/en/software).
2. Add ESP32 board support:
   - Go to **File** → **Preferences** → **Additional Boards Manager URLs**.
   - Add: `https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json`
   - Open **Boards Manager**, search for `esp32`, and install **esp32 by Espressif Systems**.
3. Install required libraries via **Sketch** → **Include Library** → **Manage Libraries**:
   - `ArduinoJson` (v7.x)
   - `DHT sensor library` by Adafruit
   - `OneWire` by Paul Stoffregen
   - `DallasTemperature` by Miles Burton
4. Open `CODE/CODE.ino`.
5. Under **Tools**, configure:
   - **Board:** `ESP32 Dev Module`
   - **Upload Speed:** `921600`
   - **CPU Frequency:** `240MHz`
   - **Flash Frequency:** `80MHz`
   - **Port:** Select the COM port corresponding to your connected ESP32.
6. Click **Upload**. Open the Serial Monitor at **115200 baud** to verify initial boot messages.

### Step 2: Setting Up the Python Environment

Open PowerShell (Windows) or Terminal (Linux/macOS) in the project directory:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/krushi-mithra.git
cd krushi-mithra

# 2. Create a virtual environment
python -m venv .venv

# 3. Activate the virtual environment
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On Linux / macOS:
source .venv/bin/activate

# 4. Install all dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Dataset Preparation & Model Training (Optional)

If pre-trained model files are not already present in `MODELS/TRAINED_MODELS/`:

```bash
# Clean raw datasets
python DATASET/clean.py

# Preprocess features and create train/test splits
python DATASET/preprocess.py

# Train M1 through M4 models
python MODELS/train.py

# Evaluate model performance
python MODELS/test.py
```

### Step 4: Starting the System

Ensure your computer is connected to the same local network as the ESP32:

```bash
python SYSTEM/main.py
```

What happens on startup:
1. Starts the background Flask server on `0.0.0.0:5000`.
2. Sends UDP discovery broadcasts across the local subnet on port `4210` to automatically locate the ESP32.
3. Loads all machine learning models into memory.
4. Starts background threads for weather updates, sensor polling, and irrigation control.
5. Displays the active dashboard link:
   - **Local Web Dashboard:** `http://krushi-mithra/` or `http://<ESP32-LOCAL-IP>/`
   - **Local Flask API:** `http://127.0.0.1:5000/api/health`

---

## 9. Complete REST API & Discovery Specification

### ESP32 Embedded HTTP Server (Port 80)

| Endpoint | Method | Expected Request Body | Response Format | Purpose |
|---|:---:|---|---|---|
| `/api/data` | `GET` | *None* | `JSON` | Returns real-time sensor readings, relay state, and mode. |
| `/api/status` | `GET` | *None* | `JSON` | Returns device health, IP, Wi-Fi RSSI, and backend status. |
| `/api/config` | `GET/POST` | `{"soil_trigger": 35.0, "critical_soil": 20.0, "rain_protection": true}` | `JSON` | Updates irrigation trigger thresholds. |
| `/api/mode` | `POST` | `{"mode": "AUTO"}` or `{"mode": "MANUAL"}` | `JSON` | Switches between automatic and manual operation modes. |
| `/api/pump` | `POST` | `{"state": true, "source": "manual", "duration_value": 10, "duration_unit": "MIN"}` | `JSON` | Toggles the relay with an optional runtime limit. |
| `/api/wifi/scan` | `GET` | *None* | `JSON` | Scans and lists visible 2.4 GHz Wi-Fi networks. |
| `/api/wifi/connect`| `POST` | `{"ssid": "FarmLAN", "password": "pass"}` | `JSON` | Connects the ESP32 to a local Wi-Fi router. |
| `/api/wifi/logout` | `POST` | `{}` | `JSON` | Clears saved Wi-Fi settings and re-enables the captive portal. |
| `/api/ml/crop` | `POST` | `{"location": "Tumakuru", "soil_type": "Red Loam", "irrigation": "Drip"}` | `JSON` | Proxies M1 crop recommendation requests to the Python runtime. |

### Gateway Server Endpoints (Port 5000)

| Endpoint | Method | Payload / Arguments | Purpose |
|---|:---:|---|---|
| `/api/health` | `GET` | *None* | Verifies the Flask gateway and background threads are running. |
| `/api/device/discover` | `GET` | *None* | Triggers a local UDP broadcast to find the ESP32. |
| `/api/ml/status` | `GET` | *None* | Reports which ML models (M1–M4) are currently loaded. |
| `/api/ml/live` | `POST` | Telemetry override payload | Runs inference across all models using current field telemetry. |

---

## 10. Troubleshooting & Diagnostics

| Symptom | Primary Root Cause | Verification & Fix |
|---|---|---|
| **Captive portal does not appear (`192.168.4.1`)** | Device is already connected to an existing Wi-Fi network, or power supply is unstable. | Disconnect power, hold the boot button, or open the dashboard and click **WI-FI LOGOUT** to clear saved credentials. |
| **DS18B20 reads `-127.0 °C`** | Missing 1-Wire pull-up resistor or loose wiring. | Solder a $4.7\text{ k}\Omega$ resistor between `GPIO 4` and `3.3V`. Check continuity on the ground wire. |
| **Soil moisture is stuck at $0\%$ or $100\%$** | Sensor wired to wrong ADC channel or calibration values do not match probe. | Ensure the analog pin is wired to `GPIO 35` (ADC1). Read raw ADC values in open air and water, then adjust `SOIL_DRY_RAW` and `SOIL_WET_RAW` in `CODE.ino`. |
| **Relay switches on and off repeatedly** | Soil moisture reading is hovering right around the threshold without hysteresis. | Verify hysteresis settings. The system is designed to turn on at $\le 20\%$ and off at $\ge 50\%$. Ensure the probe is not placed right next to the water outlet. |
| **Python discovery cannot find the ESP32** | Firewall is blocking UDP packets, or devices are on different network subnets. | Ensure both devices are on the same 2.4 GHz network. Allow UDP port `4210` in your firewall, or set the IP address manually in `SYSTEM/config.py`. |
| **Pump will not start** | Active rain detection, cooldown window, or manual timeout. | Check the FC-37 sensor for water droplets. If in AUTO mode, check if the 5-minute cooldown timer is currently active. |

---

## 11. Project Demonstration Video

Watch the complete hardware demonstration, captive portal setup, live dashboard controls, and real-time machine learning inference in action:

🎥 **[Watch the Working Model Demonstration on Google Drive](https://drive.google.com/file/d/1Vly6Cw_1zK1wI8aonXSZuhrR-C6hATH-/view?usp=sharing)**

*(Click the link above to view the recorded video demonstration. Replace the link placeholder with your actual Google Drive sharing link before publishing).*

---

## 12. License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for complete details. Free for academic, agricultural, and commercial open-source implementations.
