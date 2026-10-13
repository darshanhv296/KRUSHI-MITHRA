/*
  ================================================================
  KRUSHI MITHRA - ESP32 MASTER FIRMWARE (FINAL PRODUCTION)
  ================================================================
  Hardware Pin Map (LOCKED):
    DHT11       GPIO25
    Soil AOUT   GPIO35 (ADC1)
    FC-37 AO    GPIO34 (ADC1)
    FC-37 DO    GPIO27
    LDR DO      GPIO26
    DS18B20     GPIO4
    Relay       GPIO23
*/

#include <WiFi.h>
#include <WebServer.h>
#include <Preferences.h>
#include <ArduinoJson.h>
#include <HTTPClient.h>
#include <DHT.h>
#include <OneWire.h>
#include <DallasTemperature.h>
#include <DNSServer.h>
#include <ESPmDNS.h>
#include <WiFiUdp.h>
#include <stdlib.h>
#include <math.h>

// ----------------------------------------------------------------
// PIN DEFINITIONS & HARDWARE CONSTANTS
// ----------------------------------------------------------------

#define DHT_PIN        25
#define DHT_TYPE       DHT11
#define SOIL_PIN       35
#define RAIN_AO_PIN    34
#define RAIN_DO_PIN    27
#define LDR_DO_PIN     26
#define DS18B20_PIN    4
#define RELAY_PIN      23

// Soil Moisture Calibration (LOCKED)
const int SOIL_BASELINE_RAW = 3000; // dry-air reference
const int SOIL_DRY_RAW      = 2800; // operational dry threshold (0% moisture)
const int SOIL_WET_RAW      = 2112; // wet-soil reference (100% moisture)

// ESP32 Local Irrigation Backup Thresholds (LOCKED)
const float BACKUP_PUMP_ON_SOIL_PERCENT  = 20.0f; // <= 20% -> Pump ON
const float BACKUP_PUMP_OFF_SOIL_PERCENT = 70.0f; // >= 70% -> Pump OFF

// Rain Sensor (FC-37) ADC Thresholds & Digital Logic
const int RAIN_DRY_RAW      = 4095; // 0% wetness
const int RAIN_DETECT_RAW   = 3835; // AO trigger threshold
const int RAIN_MAX_WET_RAW  = 1800; // 100% wetness
const int RAIN_ACTIVE_LEVEL = LOW;  // DO active level

// Light Sensor (LDR) Digital Logic
const int LIGHT_ACTIVE_LEVEL = LOW; // LOW on DO pin indicates daylight

// AUTO mode safety limits
const unsigned long AUTO_PUMP_MAX_RUN_MS = 10UL * 60UL * 1000UL;
const unsigned long PUMP_COOLDOWN_MS = 5UL * 60UL * 1000UL;

// MANUAL timed-pump limits
const unsigned long MANUAL_IDLE_TIMEOUT_MS = 60UL * 1000UL;
const unsigned long MANUAL_MAX_DURATION_MS = 24UL * 60UL * 60UL * 1000UL;

const unsigned long WIFI_CONNECT_TIMEOUT_MS = 15000UL;
const unsigned long AP_SHUTDOWN_GRACE_MS = 2500UL;

// Backend availability watchdog (30 seconds)
const unsigned long BACKEND_TIMEOUT_MS = 30000UL;

// UDP discovery used by the Python client so no manual ESP32 IP is required.
const uint16_t DISCOVERY_PORT = 4210;
const char* DISCOVERY_MESSAGE = "KRUSHI_MITHRA_DISCOVER";

// ----------------------------------------------------------------
// NETWORK & SYSTEM STATE
// ----------------------------------------------------------------

const char* AP_SSID = "KRUSHI-MITHRA-SETUP";
const char* AP_PASSWORD = "Krushi1234";
IPAddress AP_IP(192, 168, 4, 1);
IPAddress AP_GATEWAY(192, 168, 4, 1);
IPAddress AP_SUBNET(255, 255, 255, 0);

WebServer server(80);
DNSServer dnsServer;
Preferences preferences;

const byte DNS_PORT = 53;
const char* MDNS_HOSTNAME = "krushi-mithra";
bool mdnsStarted = false;

WiFiUDP discoveryUdp;
bool discoveryStarted = false;

bool setupAPActive = false;
unsigned long apShutdownAtMs = 0;

String backendHost = "";
uint16_t backendPort = 5000;
unsigned long lastBackendContactMs = 0;

DHT dht(DHT_PIN, DHT_TYPE);
OneWire oneWire(DS18B20_PIN);
DallasTemperature ds18b20(&oneWire);

bool systemEnabled = true;
String currentMode = "AUTO";
bool relayState = false;

String savedSSID = "";
String savedPassword = "";

enum WifiState { WIFI_IDLE, WIFI_CONNECTING, WIFI_FAILED };
WifiState wifiState = WIFI_IDLE;
bool wifiConnectRequested = false;
String pendingSSID = "";
String pendingPassword = "";
unsigned long wifiAttemptStart = 0;
unsigned long lastReconnectMs = 0;

unsigned long lastSensorReadMs = 0;
const unsigned long SENSOR_CACHE_INTERVAL_MS = 500;

float cachedTemperature = NAN;
float cachedHumidity = NAN;
float cachedSoilTemperature = NAN;
int cachedSoilRaw = 4095;
float cachedSoilPercent = 0.0f;
int cachedRainRaw = 4095;
float cachedRainPercent = 0.0f;
int cachedRainDigital = HIGH;
int cachedLightDigital = HIGH;

float soilTrigger = 40.0f;
float criticalSoil = 25.0f;
bool rainProtection = true;

unsigned long pumpOnSinceMs = 0;
unsigned long pumpLockoutUntilMs = 0;

// Manual timed-pump state
bool manualTimerActive = false;
unsigned long manualPumpUntilMs = 0;
unsigned long manualModeLastActivityMs = 0;

// ----------------------------------------------------------------
// FORWARD DECLARATIONS
// ----------------------------------------------------------------
void startMDNS();
void stopMDNS();
void startSetupAP();
void stopSetupAP();
void safePumpOff();
void setRelay(bool turnOn);
void processDiscovery();
void startDiscovery();
void stopDiscovery();
void resetManualTimer();
void beginManualTimer(unsigned long durationMs);
void stopManualSessionAndReturnAuto(const char* reason);
bool isRainDetected();
bool isBackendAvailable();
void readSensors(bool force = false);
float soilRawToPercent(int raw);
float rainRawToPercent(int raw);
void loadPreferences();
void saveSystemConfig();
void sendJsonError(int code, const char* msg);
bool parseBody(JsonDocument& doc);
bool isPrivateLanIP(const IPAddress& ip);
void runAutomation();
void wifiTask();
void learnBackendFromClient();
bool requestFromSetupAP();
void captiveRedirect();
void handleConfig();
void handleData();
void handleStatus();
void handleMode();
void handlePump();
void handleWiFiScan();
void handleWiFiConnect();
void handleWiFiStatus();
void handleWiFiLogout();
void handleBackendHeartbeat();
void proxyML(const String& endpoint);
void handleCaptiveProbe();
void handleNotFound();

// ----------------------------------------------------------------
// RELAY PSEUDO OPEN-DRAIN SWITCHING LOGIC (LOCKED)
// ----------------------------------------------------------------

void setRelay(bool turnOn) {
  if (turnOn && !relayState) pumpOnSinceMs = millis();
  relayState = turnOn;
  if (turnOn) {
    pinMode(RELAY_PIN, OUTPUT);
    digitalWrite(RELAY_PIN, LOW); // Active LOW
  } else {
    pinMode(RELAY_PIN, INPUT);    // High-Z OFF
  }
}

void safePumpOff() {
  if (relayState) {
    setRelay(false);
  } else {
    pinMode(RELAY_PIN, INPUT);
  }
}

// ----------------------------------------------------------------
// MANUAL TIMED PUMP HELPERS
// ----------------------------------------------------------------

void resetManualTimer() {
  manualTimerActive = false;
  manualPumpUntilMs = 0;
}

void beginManualTimer(unsigned long durationMs) {
  manualTimerActive = true;
  manualPumpUntilMs = millis() + durationMs;
  manualModeLastActivityMs = millis();
}

void stopManualSessionAndReturnAuto(const char* reason) {
  safePumpOff();
  resetManualTimer();
  currentMode = "AUTO";
  manualModeLastActivityMs = millis();
  if (reason && reason[0] != '\0') {
    Serial.println(String("[MANUAL] ") + reason + " -> AUTO");
  }
}

// ----------------------------------------------------------------
// EMBEDDED DASHBOARD HTML
// ----------------------------------------------------------------

const char DASH_HTML[] PROGMEM = R"=====(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Krushi Mithra</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Teko:wght@500;700&family=Zen+Dots&display=swap" rel="stylesheet">
<style>
:root{--void-black:#010403;--chitin-hull:rgba(3,14,10,0.78);--plasma-green:#00ffaa;--plasma-core:#7dffc9;--plasma-dim:rgba(0,255,170,0.15);--plasma-violet:#d400ff;--plasma-violet-glow:rgba(212,0,255,0.45);--plasma-amber:#ff9d00;--plasma-crimson:#ff0055;--font-glyph:'Zen Dots',cursive;--font-runic:'Teko',sans-serif;--font-telemetry:'Share Tech Mono',monospace;--ease-biomorphic:cubic-bezier(0.23,1,0.32,1);}
*{box-sizing:border-box;margin:0;padding:0;user-select:none;-webkit-font-smoothing:antialiased;}
body{min-height:100vh;background-color:var(--void-black);color:#ccffea;font-family:var(--font-telemetry);position:relative;overflow-x:hidden;background-image:radial-gradient(circle at 50% 50%,rgba(212,0,255,0.05) 0%,transparent 60%),radial-gradient(circle at 80% 20%,rgba(0,255,170,0.08) 0%,transparent 50%),repeating-radial-gradient(circle at 50% 50%,transparent 0,transparent 40px,rgba(0,255,170,0.015) 41px,transparent 42px);background-attachment:fixed;}
.alien-abyss{position:fixed;inset:-50%;width:200%;height:200%;background:radial-gradient(circle at 50% 50%,rgba(0,255,170,0.06),transparent 45%);animation:membraneMorph 12s infinite alternate ease-in-out;pointer-events:none;z-index:1;}
@keyframes membraneMorph{0%{transform:scale(0.9) rotate(0deg);filter:blur(50px);}50%{transform:scale(1.15) rotate(180deg);filter:blur(75px);}100%{transform:scale(0.95) rotate(360deg);filter:blur(50px);}}
body::after{content:"";position:fixed;inset:0;background:repeating-linear-gradient(0deg,rgba(0,0,0,0.6) 0px,rgba(0,0,0,0.6) 1px,transparent 1px,transparent 3px);pointer-events:none;z-index:99;opacity:0.55;}
header{background:rgba(3,14,10,0.85);border-bottom:1px solid rgba(0,255,170,0.25);padding:18px 36px;display:flex;justify-content:space-between;align-items:center;backdrop-filter:blur(20px);-webkit-backdrop-filter:blur(20px);position:sticky;top:0;z-index:90;box-shadow:0 10px 30px rgba(0,0,0,0.8),inset 0 -1px 0 rgba(0,255,170,0.15);}
.brand{display:flex;align-items:center;gap:16px;}
.monolith-wrapper{position:relative;width:50px;height:50px;display:flex;align-items:center;justify-content:center;flex-shrink:0;}
.plasma-containment-ring{position:absolute;inset:-3px;border-radius:50%;border:2px dashed var(--plasma-green);animation:ringOrbit 12s linear infinite;opacity:0.7;filter:drop-shadow(0 0 6px var(--plasma-green));}
.plasma-counter-ring{position:absolute;inset:3px;border-radius:50%;border:1px solid var(--plasma-violet);animation:ringOrbitRev 7s linear infinite;opacity:0.6;}
@keyframes ringOrbit{100%{transform:rotate(360deg);}}@keyframes ringOrbitRev{100%{transform:rotate(-360deg);}}
.monolith-core{width:38px;height:38px;background:radial-gradient(circle at 35% 30%,#00ffaa,#033d26 80%);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:flex;align-items:center;justify-content:center;font-size:20px;box-shadow:0 0 20px var(--plasma-green),inset 0 0 8px rgba(255,255,255,0.7);z-index:2;}
.brand h1{font-family:var(--font-glyph);font-size:20px;font-weight:700;letter-spacing:2px;color:#ffffff;text-shadow:0 0 10px var(--plasma-green),0 0 20px var(--plasma-violet-glow);}
.brand p{font-size:11px;color:var(--plasma-core);letter-spacing:1.5px;margin-top:2px;}
.connection{display:flex;align-items:center;gap:10px;padding:8px 16px;border-radius:20px;background:rgba(0,255,170,0.08);border:1px solid rgba(0,255,170,0.35);box-shadow:0 0 15px rgba(0,255,170,0.2),inset 0 0 8px rgba(0,255,170,0.1);color:var(--plasma-green);font-family:var(--font-glyph);font-size:11px;letter-spacing:1.5px;}
.connection-dot{width:8px;height:8px;border-radius:50%;background:var(--plasma-green);box-shadow:0 0 10px var(--plasma-green),0 0 18px var(--plasma-green);animation:pulseDot 2s infinite ease-out;}
@keyframes pulseDot{0%,100%{opacity:1;transform:scale(1);}50%{opacity:0.4;transform:scale(0.75);}}
main{max-width:1280px;margin:auto;padding:34px 24px 60px;position:relative;z-index:10;}
.section-title{margin-bottom:16px;display:flex;align-items:baseline;justify-content:space-between;}
.section-title h2{font-family:var(--font-runic);font-size:26px;letter-spacing:2px;color:#ffffff;text-transform:uppercase;display:flex;align-items:center;gap:10px;}
.section-title h2::before{content:"◈";color:var(--plasma-green);font-size:16px;text-shadow:0 0 8px var(--plasma-green);}
.section-title p{font-size:12px;color:var(--plasma-core);letter-spacing:1px;}
.hull-panel{background:linear-gradient(165deg,var(--chitin-hull) 0%,rgba(1,8,5,0.94) 100%);border:1px solid rgba(0,255,170,0.24);border-radius:20px 4px 20px 4px;position:relative;backdrop-filter:blur(24px);-webkit-backdrop-filter:blur(24px);box-shadow:0 20px 50px -10px rgba(0,0,0,0.95),0 0 35px rgba(0,255,170,0.08),inset 0 1px 1px rgba(125,255,201,0.35);transition:var(--ease-biomorphic) 0.3s;clip-path:polygon(0 16px,16px 0,calc(100% - 16px) 0,100% 16px,100% calc(100% - 16px),calc(100% - 16px) 100%,16px 100%,0 calc(100% - 16px));}
.hull-panel:hover{border-color:rgba(0,255,170,0.45);box-shadow:0 24px 60px -8px rgba(0,0,0,0.95),0 0 45px rgba(0,255,170,0.18),inset 0 1px 2px rgba(125,255,201,0.55);transform:translateY(-2px);}
.status-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:36px;}
.status-card{padding:22px 24px;}
.status-label{color:#8dafa0;font-family:var(--font-runic);font-size:18px;text-transform:uppercase;letter-spacing:1.5px;}
.status-value{margin-top:6px;font-family:var(--font-glyph);font-size:26px;font-weight:700;letter-spacing:1px;color:#ffffff;}
.status-sub{font-size:11px;color:#55796a;margin-top:6px;letter-spacing:0.5px;}
.status-on{color:var(--plasma-green);text-shadow:0 0 16px var(--plasma-green);}
.status-off{color:#4b6658;text-shadow:none;}

.sensor-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;margin-bottom:36px;}
.sensor-card{padding:22px;}
.sensor-top{display:flex;justify-content:space-between;align-items:center;}
.sensor-name{color:#8dafa0;font-family:var(--font-runic);font-size:19px;letter-spacing:1.5px;text-transform:uppercase;}
.sensor-icon{font-size:20px;filter:drop-shadow(0 0 6px rgba(0,255,170,0.4));}
.sensor-value{font-family:var(--font-glyph);font-size:28px;font-weight:700;margin-top:10px;color:#ffffff;text-shadow:0 0 15px rgba(255,255,255,0.25);}
.sensor-unit{font-family:var(--font-telemetry);font-size:16px;font-weight:400;color:var(--plasma-core);margin-left:2px;}
.sensor-status{font-size:11px;color:var(--plasma-green);margin-top:10px;letter-spacing:1px;display:flex;align-items:center;gap:6px;}
.sensor-status::before{content:"◈";font-size:8px;color:var(--plasma-violet);text-shadow:0 0 6px var(--plasma-violet);}
.control-panel{padding:26px;margin-bottom:36px;}
.control-header{display:flex;justify-content:space-between;align-items:center;margin-bottom:22px;padding-bottom:12px;border-bottom:1px solid rgba(0,255,170,0.12);}
.control-header h2{font-family:var(--font-runic);font-size:24px;letter-spacing:1.5px;color:#ffffff;text-transform:uppercase;}
.control-header span{font-size:11px;color:var(--plasma-green);letter-spacing:1px;}
.control-row{display:grid;grid-template-columns:repeat(2,1fr);gap:16px;}
.control-box{background:rgba(1,6,3,0.85);border:1px solid rgba(0,255,170,0.16);border-radius:12px 2px 12px 2px;padding:18px;display:flex;flex-direction:column;justify-content:space-between;box-shadow:inset 0 2px 8px rgba(0,0,0,0.9);}
.control-box label{display:block;font-family:var(--font-runic);font-size:16px;letter-spacing:1.5px;text-transform:uppercase;color:#8dafa0;margin-bottom:12px;}
.button-group{display:flex;gap:8px;background:rgba(0,0,0,0.4);padding:4px;border-radius:8px;border:1px solid rgba(255,255,255,0.05);}
button{border:none;cursor:pointer;transition:var(--ease-biomorphic) 0.25s;}
.toggle-btn{flex:1;padding:11px;border-radius:6px;background:transparent;color:#8dafa0;font-family:var(--font-glyph);font-size:11px;letter-spacing:1.5px;border:1px solid transparent;}
.toggle-btn.active{background:linear-gradient(135deg,var(--plasma-green) 0%,#00965e 100%) !important;color:#021a0d !important;font-weight:700;}
.pump-button{width:100%;padding:12px;border-radius:8px 2px 8px 2px;background:rgba(0,255,170,0.08);border:1px solid rgba(0,255,170,0.3);color:var(--plasma-green);font-family:var(--font-glyph);font-size:11px;letter-spacing:1.5px;box-shadow:0 0 15px rgba(0,255,170,0.15);}
.pump-button:hover{background:rgba(0,255,170,0.18);box-shadow:0 0 25px rgba(0,255,170,0.35);}
.pump-button:disabled{opacity:0.5;cursor:wait;}
.pump-button.active{background:linear-gradient(135deg,var(--plasma-violet) 0%,#770099 100%) !important;color:#ffffff !important;border-color:#ffffff !important;box-shadow:0 0 25px var(--plasma-violet-glow) !important;}
.setting-value{font-family:var(--font-glyph);font-size:24px;font-weight:700;color:#ffffff;}
.setting-small{color:#55796a;font-size:11px;margin-top:4px;letter-spacing:0.5px;}
.analysis-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:18px;}
.analysis-card{padding:24px;cursor:pointer;}
.analysis-card:hover{border-color:var(--plasma-green);box-shadow:0 16px 40px -10px rgba(0,0,0,0.95),0 0 35px rgba(0,255,170,0.25);transform:translateY(-4px);}
.analysis-icon{width:48px;height:48px;border-radius:12px 2px 12px 2px;background:rgba(0,255,170,0.08);border:1px solid rgba(0,255,170,0.35);box-shadow:0 0 20px rgba(0,255,170,0.2);display:flex;align-items:center;justify-content:center;font-size:22px;margin-bottom:16px;}
.analysis-card h3{font-family:var(--font-runic);font-size:22px;letter-spacing:1.5px;color:#ffffff;margin-bottom:6px;text-transform:uppercase;}
.analysis-card p{color:#8dafa0;font-size:12px;line-height:1.5;letter-spacing:0.3px;}
.open-analysis{margin-top:18px;color:var(--plasma-green);font-family:var(--font-glyph);font-size:11px;letter-spacing:1.5px;font-weight:700;}
.modal-overlay{position:fixed;inset:0;background:rgba(1,4,3,0.85);backdrop-filter:blur(16px);-webkit-backdrop-filter:blur(16px);display:none;align-items:center;justify-content:center;padding:20px;z-index:1000;}
.modal-overlay.show{display:flex;}
.modal{width:100%;max-width:530px;background:linear-gradient(165deg,var(--chitin-hull) 0%,rgba(1,8,5,0.96) 100%);border:1px solid rgba(0,255,170,0.35);box-shadow:0 30px 80px -10px rgba(0,0,0,0.98),0 0 50px rgba(0,255,170,0.2);border-radius:28px 6px 28px 6px;padding:30px;max-height:90vh;overflow-y:auto;position:relative;clip-path:polygon(0 20px,20px 0,calc(100% - 20px) 0,100% 20px,100% calc(100% - 20px),calc(100% - 20px) 100%,20px 100%,0 calc(100% - 20px));}
.modal-header{display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:24px;padding-bottom:14px;border-bottom:1px solid rgba(0,255,170,0.15);}
.modal-header h2{font-family:var(--font-runic);font-size:26px;letter-spacing:2px;color:#ffffff;text-transform:uppercase;}
.modal-header p{font-size:12px;color:#8dafa0;margin-top:2px;}
.close-btn{width:32px;height:32px;border-radius:6px;background:rgba(0,255,170,0.08);border:1px solid rgba(0,255,170,0.25);font-size:18px;color:var(--plasma-green);display:flex;align-items:center;justify-content:center;}
.close-btn:hover{color:var(--plasma-crimson);border-color:var(--plasma-crimson);box-shadow:0 0 15px rgba(255,0,85,0.4);}
.form-group{margin-bottom:16px;}
.form-group label{display:block;font-family:var(--font-runic);font-size:16px;letter-spacing:1.5px;text-transform:uppercase;color:#8dafa0;margin-bottom:6px;}
input,select{width:100%;padding:12px 14px;border:1px solid rgba(0,255,170,0.2);border-radius:8px;background:#010603;color:#ffffff;font-family:var(--font-telemetry);font-size:13px;outline:none;box-shadow:inset 0 2px 6px rgba(0,0,0,0.9);transition:var(--ease-biomorphic) 0.25s;}
input:focus,select:focus{border-color:var(--plasma-green);box-shadow:0 0 16px rgba(0,255,170,0.3),inset 0 2px 6px rgba(0,0,0,0.9);}
select option{background:#020905;color:#ffffff;}
.readonly{background:rgba(0,0,0,0.6);color:#55796a;border-style:dashed;cursor:not-allowed;}
.form-row{display:grid;grid-template-columns:1fr 1fr;gap:12px;}
.analyze-btn{width:100%;padding:15px;background:linear-gradient(135deg,var(--plasma-green) 0%,#00965e 100%);color:#021a0d;border:none;border-radius:10px 2px 10px 2px;font-family:var(--font-glyph);font-size:12px;font-weight:700;letter-spacing:2px;text-transform:uppercase;cursor:pointer;box-shadow:0 8px 24px rgba(0,255,170,0.4),inset 0 1px 2px rgba(255,255,255,0.8);margin-top:8px;transition:var(--ease-biomorphic) 0.3s;}
.analyze-btn:hover{transform:translateY(-2px);box-shadow:0 12px 35px rgba(0,255,170,0.6);}
.result{margin-top:20px;border:1px solid rgba(0,255,170,0.3);background:rgba(0,255,170,0.05);border-radius:12px;padding:18px;display:none;box-shadow:0 0 25px rgba(0,255,170,0.1);}
.result.show{display:block;animation:cipherAppear 0.3s ease;}
@keyframes cipherAppear{from{opacity:0;transform:translateY(-8px);}to{opacity:1;transform:translateY(0);}}
.result-title{color:var(--plasma-green);font-family:var(--font-runic);font-size:16px;text-transform:uppercase;letter-spacing:1.5px;}
.result-main{font-family:var(--font-glyph);font-size:26px;font-weight:700;color:#ffffff;margin-top:6px;text-shadow:0 0 15px rgba(0,255,170,0.5);}
.result-grid{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:14px;}
.result-item{background:#010603;border:1px solid rgba(0,255,170,0.15);border-radius:8px;padding:10px;}
.result-item span{display:block;font-size:10px;color:#55796a;font-family:var(--font-telemetry);letter-spacing:0.5px;text-transform:uppercase;}
.result-item strong{display:block;margin-top:4px;font-size:12px;font-family:var(--font-glyph);color:var(--plasma-core);}
footer{text-align:center;color:#55796a;font-size:11px;letter-spacing:1px;padding:24px;position:relative;z-index:10;}
.wifi-logout{margin-left:8px;padding:8px 12px;border-radius:8px;background:rgba(212,0,255,0.08);border:1px solid rgba(212,0,255,0.3);color:var(--plasma-core);font-family:var(--font-glyph);font-size:9px;letter-spacing:1px;cursor:pointer;transition:var(--ease-biomorphic) 0.25s;}
.wifi-logout:hover{background:rgba(212,0,255,0.18);color:#fff;border-color:var(--plasma-violet);box-shadow:0 0 18px var(--plasma-violet-glow);}
.duration-unit{display:flex;gap:10px;}
.duration-unit input{flex:1;}
.duration-unit select{width:120px;}
.manual-duration-note{font-size:11px;color:#55796a;margin-top:6px;line-height:1.4;}
.modal-actions{display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:12px;}
.cancel-btn{width:100%;padding:14px;background:rgba(255,255,255,0.03);color:#8dafa0;border:1px solid rgba(255,255,255,0.1);border-radius:10px 2px 10px 2px;font-family:var(--font-glyph);font-size:11px;letter-spacing:1.5px;cursor:pointer;}
.cancel-btn:hover{border-color:var(--plasma-crimson);color:#fff;}
@media (max-width:850px){.sensor-grid,.analysis-grid,.control-row{grid-template-columns:1fr 1fr;}.status-grid{grid-template-columns:1fr 1fr 1fr;}}
@media (max-width:600px){header{padding:15px;}main{padding:20px 14px 35px;}.brand p{display:none;}.status-grid,.sensor-grid,.analysis-grid,.control-row{grid-template-columns:1fr;}.connection{padding:6px 12px;}.form-row{grid-template-columns:1fr;}}
</style>
</head>
<body>
<div class="alien-abyss"></div>
<header>
    <div class="brand">
        <div class="monolith-wrapper">
            <div class="plasma-containment-ring"></div>
            <div class="plasma-counter-ring"></div>
            <div class="monolith-core">🌱</div>
        </div>
        <div>
            <h1>Krushi Mithra</h1>
            <p>Smart Farming & Intelligent Irrigation</p>
        </div>
    </div>
    <div style="display:flex;align-items:center;">
        <div class="connection">
            <span class="connection-dot"></span>
            <span id="deviceConnectionLabel">CONNECTING...</span>
        </div>
        <button class="wifi-logout" onclick="logoutWiFi()">WI-FI LOGOUT</button>
    </div>
</header>

<main>
    <div class="section-title">
        <h2>System Status</h2>
        <p>Current Krushi Mithra operating status</p>
    </div>
    <section class="status-grid">
        <div class="hull-panel status-card">
            <div class="status-label">Device Power</div>
            <div class="status-value status-on" id="devicePowerStatus">ON</div>
            <div class="status-sub">System starts automatically with ESP32 power</div>
        </div>
        <div class="hull-panel status-card">
            <div class="status-label">Mode</div>
            <div class="status-value" id="modeStatus">AUTO</div>
            <div class="status-sub">Irrigation control mode</div>
        </div>
        <div class="hull-panel status-card">
            <div class="status-label">Pump</div>
            <div class="status-value status-off" id="pumpStatus">OFF</div>
            <div class="status-sub">Current pump state</div>
        </div>
    </section>

    <div class="section-title">
        <h2>Live Field Data</h2>
        <p>Real-time readings from the ESP32 sensor network</p>
    </div>
    <section class="sensor-grid">
        <div class="hull-panel sensor-card">
            <div class="sensor-top"><span class="sensor-name">Air Temp</span><span class="sensor-icon">🌡️</span></div>
            <div class="sensor-value"><span id="temperature">--</span><span class="sensor-unit">°C</span></div>
            <div class="sensor-status">DHT11</div>
        </div>
        <div class="hull-panel sensor-card">
            <div class="sensor-top"><span class="sensor-name">Air Humidity</span><span class="sensor-icon">💧</span></div>
            <div class="sensor-value"><span id="humidity">--</span><span class="sensor-unit">%</span></div>
            <div class="sensor-status">DHT11</div>
        </div>
        <div class="hull-panel sensor-card">
            <div class="sensor-top"><span class="sensor-name">Soil Moisture</span><span class="sensor-icon">🌱</span></div>
            <div class="sensor-value"><span id="soilMoisture">--</span><span class="sensor-unit">%</span></div>
            <div class="sensor-status">Capacitive Sensor</div>
        </div>
        <div class="hull-panel sensor-card">
            <div class="sensor-top"><span class="sensor-name">Soil Temp</span><span class="sensor-icon">🌡️</span></div>
            <div class="sensor-value"><span id="soilTemperature">--</span><span class="sensor-unit">°C</span></div>
            <div class="sensor-status">DS18B20</div>
        </div>
        <div class="hull-panel sensor-card">
            <div class="sensor-top"><span class="sensor-name">Rain</span><span class="sensor-icon">🌧</span></div>
            <div class="sensor-value">
                <span id="rainStatus">--</span>
                <span class="sensor-unit" style="font-size:16px;">(<span id="rainPercent">--</span>%)</span>
            </div>
            <div class="sensor-status">FC-37 wetness detection</div>
        </div>
        <div class="hull-panel sensor-card">
            <div class="sensor-top"><span class="sensor-name">Light</span><span class="sensor-icon">☀</span></div>
            <div class="sensor-value" id="lightStatus">--</div>
            <div class="sensor-status">LDR</div>
        </div>
    </section>

    <div class="section-title">
        <h2>Irrigation Control</h2>
        <p>Mode and pump control. System starts automatically with ESP32 power.</p>
    </div>
    <section class="hull-panel control-panel">
        <div class="control-header">
            <h2>System Controls</h2>
            <span id="ipDisplay">IP: ---</span>
        </div>
        <div class="control-row">
            <div class="control-box">
                <label>MODE</label>
                <div class="button-group">
                    <button class="toggle-btn active" id="autoBtn" onclick="setMode('AUTO')">AUTO</button>
                    <button class="toggle-btn" id="manualBtn" onclick="setMode('MANUAL')">MANUAL</button>
                </div>
            </div>
            <div class="control-box">
                <label>PUMP CONTROL</label>
                <button class="pump-button" id="pumpButton" onclick="togglePump()">TURN ON</button>
            </div>
        </div>
        <br>
        <div class="control-row">
            <div class="control-box">
                <label>SOIL TRIGGER</label>
                <div class="setting-value"><span id="soilTrigger">--</span>%</div>
                <div class="setting-small">Irrigation threshold</div>
            </div>
            <div class="control-box">
                <label>CRITICAL SOIL</label>
                <div class="setting-value" style="color: var(--plasma-crimson);"><span id="criticalSoil">--</span>%</div>
                <div class="setting-small">Critical moisture level</div>
            </div>
            <div class="control-box">
                <label>RAIN PROTECTION</label>
                <div class="setting-value status-on" id="rainProtection">--</div>
                <div class="setting-small">Prevent irrigation during rain</div>
            </div>
        </div>
    </section>

    <div class="section-title">
        <h2>Intelligent Analysis</h2>
        <p>Run the trained machine-learning models</p>
    </div>
    <section class="analysis-grid">
        <div class="hull-panel analysis-card" onclick="openModal('crop')">
            <div class="analysis-icon">🌾</div>
            <h3>Crop Recommendation</h3>
            <p>Recommend a suitable crop using soil and farming information.</p>
            <div class="open-analysis">OPEN →</div>
        </div>
        <div class="hull-panel analysis-card" onclick="openModal('soil')">
            <div class="analysis-icon">💧</div>
            <h3>Future Soil Moisture</h3>
            <p>Predict future soil moisture using current sensor and weather conditions.</p>
            <div class="open-analysis">OPEN →</div>
        </div>
        <div class="hull-panel analysis-card" onclick="openModal('rain')">
            <div class="analysis-icon">🌧</div>
            <h3>Future Rainfall</h3>
            <p>Predict next-hour rainfall using the trained weather model.</p>
            <div class="open-analysis">OPEN →</div>
        </div>
    </section>
</main>
<footer>Krushi Mithra • Smart Farming & Intelligent Irrigation System</footer>

<div class="modal-overlay" id="manualDurationOverlay" onclick="closeManualDurationOutside(event)">
    <div class="modal" style="max-width:430px;">
        <div class="modal-header">
            <div><h2>Manual Pump Duration</h2><p>Set how long the pump should run.</p></div>
            <button class="close-btn" onclick="closeManualDuration()">×</button>
        </div>
        <div class="form-group">
            <label>Duration</label>
            <div class="duration-unit">
                <input id="manualDurationValue" type="number" min="1" max="1440" step="1" value="10" placeholder="Enter duration">
                <select id="manualDurationUnit">
                    <option value="SEC">SEC</option>
                    <option value="MIN" selected>MIN</option>
                    <option value="HR">HR</option>
                </select>
            </div>
            <div class="manual-duration-note">Maximum manual run time: 24 hours. The ESP32 remains the final safety authority.</div>
        </div>
        <div class="modal-actions">
            <button type="button" class="cancel-btn" onclick="closeManualDuration()">CANCEL</button>
            <button type="button" class="analyze-btn" onclick="startTimedManualPump()">START PUMP</button>
        </div>
    </div>
</div>

<div class="modal-overlay" id="modalOverlay" onclick="closeModalOutside(event)">
    <div class="modal">
        <div class="modal-header">
            <div><h2 id="modalTitle">Analysis</h2><p id="modalDescription">Enter required information</p></div>
            <button class="close-btn" onclick="closeModal()">×</button>
        </div>
        <form id="cropForm" style="display:none;" onsubmit="runCropPrediction(event)">
            <div class="form-group">
                <label>Location</label>
                <select id="cropLocation" required>
                    <option value="" disabled selected>Select Location</option>
                    <option value="Bagalkot">Bagalkot</option>
                    <option value="Ballari">Ballari</option>
                    <option value="Belagavi">Belagavi</option>
                    <option value="Bengaluru Rural">Bengaluru Rural</option>
                    <option value="Bengaluru Urban">Bengaluru Urban</option>
                    <option value="Bidar">Bidar</option>
                    <option value="Chamarajanagar">Chamarajanagar</option>
                    <option value="Chikkaballapura">Chikkaballapura</option>
                    <option value="Chikkamagaluru">Chikkamagaluru</option>
                    <option value="Chitradurga">Chitradurga</option>
                    <option value="Dakshina Kannada">Dakshina Kannada</option>
                    <option value="Davanagere">Davanagere</option>
                    <option value="Dharwad">Dharwad</option>
                    <option value="Gadag">Gadag</option>
                    <option value="Hassan">Hassan</option>
                    <option value="Haveri">Haveri</option>
                    <option value="Kalaburagi">Kalaburagi</option>
                    <option value="Kodagu">Kodagu</option>
                    <option value="Kolar">Kolar</option>
                    <option value="Koppal">Koppal</option>
                    <option value="Mandya">Mandya</option>
                    <option value="Mysuru">Mysuru</option>
                    <option value="Raichur">Raichur</option>
                    <option value="Ramanagara">Ramanagara</option>
                    <option value="Shivamogga">Shivamogga</option>
                    <option value="Tumakuru">Tumakuru</option>
                    <option value="Udupi">Udupi</option>
                    <option value="Uttara Kannada">Uttara Kannada</option>
                    <option value="Vijayanagara">Vijayanagara</option>
                    <option value="Vijayapura">Vijayapura</option>
                    <option value="Yadgir">Yadgir</option>
                </select>
            </div>
            <div class="form-group">
                <label>Soil Type</label>
                <select id="soilType" required>
                    <option value="" disabled selected>Select Soil Type</option>
                    <option value="Coastal Alluvial & Laterite">Coastal Alluvial & Laterite</option>
                    <option value="Deep Black Clay">Deep Black Clay</option>
                    <option value="Laterite">Laterite</option>
                    <option value="Medium Black Clay">Medium Black Clay</option>
                    <option value="Red Clay Loam">Red Clay Loam</option>
                    <option value="Red Laterite">Red Laterite</option>
                    <option value="Red Loam">Red Loam</option>
                    <option value="Red Sandy Loam">Red Sandy Loam</option>
                </select>
            </div>
            <div class="form-group">
                <label>Irrigation Type</label>
                <select id="irrigationType" required>
                    <option value="" disabled selected>Select Irrigation Type</option>
                    <option value="Drip">Drip</option>
                    <option value="Sprinkler">Sprinkler</option>
                    <option value="Rainfed">Rainfed</option>
                    <option value="Canal">Canal</option>
                    <option value="Borewell / Tube Well">Borewell / Tube Well</option>
                    <option value="Flood / Furrow">Flood / Furrow</option>
                </select>
            </div>
            <button class="analyze-btn" type="submit">ANALYZE CROP</button>
            <div class="result" id="cropResult">
                <div class="result-title">Recommended Crop</div>
                <div class="result-main" id="cropOutput">—</div>
                <div class="result-grid">
                    <div class="result-item"><span>Model</span><strong>Random Forest</strong></div>
                </div>
            </div>
        </form>
        <form id="soilForm" style="display:none;" onsubmit="runSoilPrediction(event)">
            <div class="form-group">
                <label>Current Moisture</label>
                <input class="readonly" id="soilCurrent" value="-- %" readonly>
            </div>
            <div class="form-row">
                <div class="form-group">
                    <label>Air Temperature</label>
                    <input class="readonly" id="soilAirTemperature" value="-- °C" readonly>
                </div>
                <div class="form-group">
                    <label>Air Humidity</label>
                    <input class="readonly" id="soilAirHumidity" value="-- %" readonly>
                </div>
            </div>
            <div class="form-group">
                <label>Soil Temperature</label>
                <input class="readonly" id="soilTemperatureInput" value="-- °C" readonly>
            </div>
            <button class="analyze-btn" type="submit">PREDICT SOIL MOISTURE</button>
            <div class="result" id="soilResult">
                <div class="result-title">Future Soil Moisture</div>
                <div class="result-main" id="soilOutput">—</div>
                <div class="result-grid">
                    <div class="result-item"><span>Model</span><strong>LightGBM</strong></div>
                  <div class="result-item"><span>Lag Inputs</span><strong id="soilHistoryStatus">—</strong></div>
                </div>
            </div>
        </form>
        <form id="rainForm" style="display:none;" onsubmit="runRainPrediction(event)">
            <div class="form-group">
                <label>Region</label>
                <select id="rainRegion">
                    <option value="Bagalkot">Bagalkot</option>
                    <option value="Ballari">Ballari</option>
                    <option value="Belagavi">Belagavi</option>
                    <option value="Bengaluru Rural">Bengaluru Rural</option>
                    <option value="Bengaluru Urban">Bengaluru Urban</option>
                    <option value="Bidar">Bidar</option>
                    <option value="Chamarajanagar">Chamarajanagar</option>
                    <option value="Chikkaballapura">Chikkaballapura</option>
                    <option value="Chikkamagaluru" selected>Chikkamagaluru</option>
                    <option value="Chitradurga">Chitradurga</option>
                    <option value="Dakshina Kannada">Dakshina Kannada</option>
                    <option value="Davanagere">Davanagere</option>
                    <option value="Dharwad">Dharwad</option>
                    <option value="Gadag">Gadag</option>
                    <option value="Hassan">Hassan</option>
                    <option value="Haveri">Haveri</option>
                    <option value="Kalaburagi">Kalaburagi</option>
                    <option value="Kodagu">Kodagu</option>
                    <option value="Kolar">Kolar</option>
                    <option value="Koppal">Koppal</option>
                    <option value="Mandya">Mandya</option>
                    <option value="Mysuru">Mysuru</option>
                    <option value="Raichur">Raichur</option>
                    <option value="Ramanagara">Ramanagara</option>
                    <option value="Shivamogga">Shivamogga</option>
                    <option value="Tumakuru">Tumakuru</option>
                    <option value="Udupi">Udupi</option>
                    <option value="Uttara Kannada">Uttara Kannada</option>
                    <option value="Vijayanagara">Vijayanagara</option>
                    <option value="Vijayapura">Vijayapura</option>
                    <option value="Yadgir">Yadgir</option>
                </select>
            </div>
            <button class="analyze-btn" type="submit">PREDICT RAINFALL</button>
            <div class="result" id="rainResult">
                <div class="result-title" id="rainResultTitle">Next-Hour Rainfall</div>
                <div class="result-main" id="rainOutput">—</div>
                <div class="result-grid">
                    <div class="result-item"><span>Region</span><strong id="rainResultRegion">—</strong></div>
                    <div class="result-item"><span>Model</span><strong>XGBoost</strong></div>
                </div>
            </div>
        </form>
    </div>
</div>

<script>
    let currentMode = "AUTO";
    let pumpState = false;
    let manualRemainingMs = 0;
    let manualPumpDeadlineMs = 0;

    const setManualRemaining = (remainingMs) => {
      manualRemainingMs = Math.max(0, Number(remainingMs) || 0);
      manualPumpDeadlineMs = manualRemainingMs > 0
        ? Date.now() + manualRemainingMs
        : 0;
    };

    const openModal = (type) => {
        document.getElementById("modalOverlay").classList.add("show");
        document.getElementById("cropForm").style.display = "none";
        document.getElementById("soilForm").style.display = "none";
        document.getElementById("rainForm").style.display = "none";
        document.getElementById("cropResult").classList.remove("show");
        document.getElementById("soilResult").classList.remove("show");
        document.getElementById("rainResult").classList.remove("show");

        if (type === "crop") {
            document.getElementById("modalTitle").textContent = "Crop Recommendation";
            document.getElementById("modalDescription").textContent = "Provide the required farming information.";
            document.getElementById("cropForm").style.display = "block";
        } else if (type === "soil") {
            document.getElementById("modalTitle").textContent = "Future Soil Moisture";
            document.getElementById("modalDescription").textContent = "Predict future soil moisture from current conditions.";
            document.getElementById("soilForm").style.display = "block";
        } else if (type === "rain") {
            document.getElementById("modalTitle").textContent = "Future Rainfall";
            document.getElementById("modalDescription").textContent = "Predict next-hour rainfall for the selected region.";
            document.getElementById("rainForm").style.display = "block";
        }
    };

    const closeModal = () => { document.getElementById("modalOverlay").classList.remove("show"); };
    const closeModalOutside = (event) => { if (event.target === document.getElementById("modalOverlay")) closeModal(); };

    const showToast = (message, error = false) => {
        let toast = document.getElementById('toast');
        if (!toast) {
            toast = document.createElement('div');
            toast.id = 'toast';
            toast.style.cssText = `position:fixed; right:20px; bottom:20px; z-index:2000; padding:14px 18px; border-radius:10px; font-family:var(--font-telemetry); font-size:12px; background:rgba(1,8,5,.96); border:1px solid rgba(0,255,170,.4); color:${error ? '#ff668f' : 'var(--plasma-green)'}; box-shadow:0 0 25px rgba(0,255,170,.18);`;
            document.body.appendChild(toast);
        }
        toast.textContent = message;
        toast.style.display = 'block';
        clearTimeout(window.__toastTimer);
        window.__toastTimer = setTimeout(() => { toast.style.display = 'none'; }, 3500);
    };

    const updateModeUI = () => {
        document.getElementById('modeStatus').textContent = currentMode;
        document.getElementById('autoBtn').classList.toggle('active', currentMode === 'AUTO');
        document.getElementById('manualBtn').classList.toggle('active', currentMode === 'MANUAL');
    };

    const updatePumpUI = () => {
        const status = document.getElementById("pumpStatus");
        const button = document.getElementById("pumpButton");
        if (manualPumpDeadlineMs > 0) {
            manualRemainingMs = Math.max(0, manualPumpDeadlineMs - Date.now());
            if (manualRemainingMs === 0) manualPumpDeadlineMs = 0;
        }
        status.textContent = pumpState ? "ON" : "OFF";
        status.className = pumpState ? "status-value status-on" : "status-value status-off";
        if (pumpState && manualRemainingMs > 0) {
            const totalSeconds = Math.ceil(manualRemainingMs / 1000);
            if (totalSeconds >= 3600) {
                const h = Math.floor(totalSeconds / 3600);
                const m = Math.floor((totalSeconds % 3600) / 60);
                button.textContent = `TURN OFF (${h}H ${m}M)`;
            } else if (totalSeconds >= 60) {
                const m = Math.floor(totalSeconds / 60);
                const s = totalSeconds % 60;
                button.textContent = `TURN OFF (${m}M ${s}S)`;
            } else {
                button.textContent = `TURN OFF (${totalSeconds}S)`;
            }
        } else {
            button.textContent = pumpState ? "TURN OFF" : "TURN ON";
        }
        button.classList.toggle("active", pumpState);
    };

    const setMode = async (mode) => {
        mode = String(mode).toUpperCase();
        if (!['AUTO', 'MANUAL'].includes(mode)) return;
        try {
            const response = await fetch('/api/mode', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ mode })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Mode change failed');
            currentMode = data.mode || mode;
            updateModeUI();
        } catch (error) {
            showToast(error.message, true);
        }
    };

    const closeManualDuration = () => {
        document.getElementById('manualDurationOverlay').classList.remove('show');
    };

    const closeManualDurationOutside = (event) => {
        if (event.target === document.getElementById('manualDurationOverlay')) closeManualDuration();
    };

    const openManualDuration = () => {
        const overlay = document.getElementById('manualDurationOverlay');
        overlay.classList.add('show');
        const input = document.getElementById('manualDurationValue');
        input.focus();
        input.select();
    };

    const startTimedManualPump = async () => {
        const value = Number(document.getElementById('manualDurationValue').value);
        const unit = document.getElementById('manualDurationUnit').value;

        if (!Number.isFinite(value) || value <= 0 || !Number.isInteger(value)) {
            showToast('Enter a whole-number duration greater than 0.', true);
            return;
        }

        const unitMs = { SEC: 1000, MIN: 60000, HR: 3600000 };
        const durationMs = value * unitMs[unit];
        if (durationMs > 24 * 60 * 60 * 1000) {
            showToast('Maximum manual duration is 24 hours.', true);
            return;
        }

        try {
            const response = await fetch('/api/pump', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    state: true,
                    source: 'manual',
                    duration_value: value,
                    duration_unit: unit
                })
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Pump command failed');

            pumpState = Boolean(data.pump);
            currentMode = data.mode || 'MANUAL';
            setManualRemaining(data.manual_remaining_ms);
            updateModeUI();
            updatePumpUI();
            closeManualDuration();
            showToast(`Pump started for ${value} ${unit}.`);
        } catch (error) {
            showToast(error.message, true);
        }
    };

    const togglePump = async () => {
        if (currentMode !== 'MANUAL') {
            showToast('Switch to MANUAL mode to control the pump.', true);
            return;
        }

        if (pumpState) {
            try {
                const response = await fetch('/api/pump', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ state: false, source: 'manual' })
                });
                const data = await response.json();
                if (!response.ok) throw new Error(data.error || 'Pump command failed');
                pumpState = Boolean(data.pump);
                currentMode = data.mode || currentMode;
                setManualRemaining(data.manual_remaining_ms);
                updateModeUI();
                updatePumpUI();
            } catch (error) {
                showToast(error.message, true);
            }
            return;
        }

        openManualDuration();
    };

    const logoutWiFi = async () => {
        if (!confirm('Log out Wi-Fi and return the ESP32 to setup mode?')) return;
        try {
            const response = await fetch('/api/wifi/logout', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: '{}'
            });
            const data = await response.json();
            if (!response.ok) throw new Error(data.error || 'Wi-Fi logout failed');
            showToast('Wi-Fi credentials cleared. Opening setup...');
            setTimeout(() => {
                window.location.href = 'http://192.168.4.1/wifi-setup';
            }, 800);
        } catch (error) {
            showToast(error.message, true);
        }
    };

    const requestMlPrediction = async (endpoint, payload, waitForHistory = false) => {
      let attempt = 0;
      while (true) {
        attempt += 1;
        try {
          const response = await fetch(endpoint, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(payload)
          });
          const data = await response.json().catch(() => ({}));
          if (waitForHistory && response.status === 409 && data.error_code === 'insufficient_history') {
            await new Promise(resolve => setTimeout(resolve, 60000));
            continue;
          }
          if (!response.ok || data.success === false) {
            const error = new Error(data.error || `Prediction failed (${response.status}).`);
            error.retryable = [502, 503, 504].includes(response.status);
            throw error;
          }
          return data;
        } catch (error) {
          if (!error.retryable && !(error instanceof TypeError)) throw error;
          await new Promise(resolve => setTimeout(resolve, Math.min(attempt * 2000, 15000)));
        }
      }
    };

    const runCropPrediction = async (event) => {
        event.preventDefault();
        const selectedLocation = document.getElementById('cropLocation').value;
        const selectedSoil = document.getElementById('soilType').value;
        const selectedIrrigation = document.getElementById('irrigationType').value;
        if (!selectedLocation || !selectedSoil || !selectedIrrigation) {
            showToast('Please select location, soil type, and irrigation.', true);
            return;
        }
        const output = document.getElementById('cropOutput');
        document.getElementById('cropResult').classList.add('show');
        output.textContent = 'PROCESSING...';
        try {
            const data = await requestMlPrediction('/api/ml/crop', {
                location: selectedLocation,
                soil_type: selectedSoil,
                irrigation: selectedIrrigation
            });
            output.textContent = data.crop || '—';
            
        } catch (error) {
            console.error('[M1 Crop Recommendation Error]', error);
            output.textContent = error.message || 'CROP ANALYSIS FAILED';
            
        }
    };

    const runSoilPrediction = async (event) => {
        event.preventDefault();
        const output = document.getElementById('soilOutput');
        const resBox = document.getElementById('soilResult');
        if (resBox) resBox.classList.add('show');
        if (output) output.textContent = 'PROCESSING...';
        try {
          const data = await requestMlPrediction('/api/ml/soil-moisture', {}, true);
            
            const predictedVal = data.future_soil_moisture_percent != null 
                ? Number(data.future_soil_moisture_percent).toFixed(2) 
                : (data.target_moisture_1h != null ? Number(data.target_moisture_1h).toFixed(2) : '—');
                
            if (output) output.textContent = `${predictedVal} %`;

            const historyStatusEl = document.getElementById('soilHistoryStatus');
            if (historyStatusEl) {
              historyStatusEl.textContent = data.history_fallback === true
                ? 'CURRENT READING USED'
                : data.history_fallback === false ? 'MEASURED HISTORY' : 'STATUS UNKNOWN';
            }
        } catch (error) {
            console.error('[M3 Soil Moisture Error]', error);
          if (output) output.textContent = error.message || 'SOIL PREDICTION FAILED';
            const historyStatusEl = document.getElementById('soilHistoryStatus');
            if (historyStatusEl) historyStatusEl.textContent = '—';
        }
    };

    const runRainPrediction = async (event) => {
        event.preventDefault();
        const region = document.getElementById('rainRegion').value;
        const output = document.getElementById('rainOutput');
        const titleEl = document.getElementById('rainResultTitle');
        document.getElementById('rainResult').classList.add('show');
        output.textContent = 'PROCESSING...';
        try {
          const data = await requestMlPrediction('/api/ml/rainfall', { region });
            output.textContent = data.predicted_next_hour_rainfall_mm != null ? `${Number(data.predicted_next_hour_rainfall_mm).toFixed(2)} mm` : '—';
            document.getElementById('rainResultRegion').textContent = region;
            if (titleEl) titleEl.textContent = `Next-Hour Rainfall (${region})`;
        } catch (error) {
            console.error('[M4 Rainfall Prediction Error]', error);
            output.textContent = error.message || 'RAINFALL PREDICTION FAILED';
            document.getElementById('rainResultRegion').textContent = '—';
        }
    };

    const formatNumber = (value, digits = 1) => { const n = Number(value); return Number.isFinite(n) ? n.toFixed(digits) : '--'; };
    const setText = (id, value) => { const el = document.getElementById(id); if (el) el.textContent = value !== null && value !== undefined ? value : '—'; };
    const setValue = (id, value, suffix = '') => { const el = document.getElementById(id); if (el) el.value = value == null || Number.isNaN(Number(value)) ? '--' : `${formatNumber(value, 1)}${suffix}`; };

    let deviceRefreshInFlight = false;
    const refreshData = async () => {
        if (deviceRefreshInFlight) return;
        deviceRefreshInFlight = true;
        try {
            const response = await fetch('/api/data', { cache: 'no-store' });
            const data = await response.json();
            setText('temperature', formatNumber(data.temperature, 1));
            setText('humidity', formatNumber(data.humidity, 1));
            setText('soilMoisture', formatNumber(data.soil_moisture_percent, 1));
            setText('soilTemperature', formatNumber(data.soil_temperature, 1));
            setText('rainStatus', data.rain_detected ? 'YES' : 'NO');
            setText('rainPercent', formatNumber(data.rain_percent, 1));
            setText('lightStatus', data.light_detected ? 'DAY' : 'NIGHT');

            setValue('soilCurrent', data.soil_moisture_percent, ' %');
            setValue('soilAirTemperature', data.temperature, ' °C');
            setValue('soilAirHumidity', data.humidity, ' %');
            setValue('soilTemperatureInput', data.soil_temperature, ' °C');
            
            currentMode = data.mode || "AUTO";
            pumpState = Boolean(data.pump);
            setManualRemaining(data.manual_remaining_ms);
            updateModeUI();
            updatePumpUI();
            
            setText('deviceConnectionLabel', data.wifi_connected ? 'ESP32 ONLINE' : 'ESP32 SETUP AP');
        } catch (error) {
            setText('deviceConnectionLabel', 'ESP32 OFFLINE');
        } finally {
            deviceRefreshInFlight = false;
        }
    };

    const refreshStatus = async () => {
        if (deviceRefreshInFlight) return;
        deviceRefreshInFlight = true;
        try {
            const response = await fetch('/api/status', { cache: 'no-store' });
            const data = await response.json();
            setText('devicePowerStatus', data.online ? 'ON' : 'OFF');
            document.getElementById('devicePowerStatus').className = data.online ? 'status-value status-on' : 'status-value status-off';
            setText('ipDisplay', 'IP: ' + data.ip);
        } catch (error) {
        } finally {
            deviceRefreshInFlight = false;
        }
    };
    
    const refreshConfig = async () => {
        if (deviceRefreshInFlight) return;
        deviceRefreshInFlight = true;
        try {
            const response = await fetch('/api/config', { cache: 'no-store' });
            const data = await response.json();
            if (response.ok) {
                setText('soilTrigger', data.soil_trigger);
                setText('criticalSoil', data.critical_soil);
                setText('rainProtection', data.rain_protection ? 'ON' : 'OFF');
                document.getElementById('rainProtection').className = data.rain_protection ? 'status-value status-on' : 'status-value status-off';
            }
        } catch (error) {
        } finally {
            deviceRefreshInFlight = false;
        }
    };

    refreshStatus().then(refreshData).then(refreshConfig);
    setInterval(refreshStatus, 2000);
    setInterval(refreshData, 1000);
    setInterval(refreshConfig, 5000);
    setInterval(() => {
      if (manualPumpDeadlineMs > 0) updatePumpUI();
    }, 1000);
</script>
</body>
</html>
)=====";

const char SETUP_HTML[] PROGMEM = R"=====(
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Krushi Mithra - Wi-Fi Setup</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Share+Tech+Mono&family=Teko:wght@500;700&family=Zen+Dots&display=swap" rel="stylesheet">
<style>
:root{--void-black:#010403;--chitin-hull:rgba(3,14,10,0.78);--plasma-green:#00ffaa;--plasma-core:#7dffc9;--plasma-dim:rgba(0,255,170,0.15);--plasma-violet:#d400ff;--plasma-violet-glow:rgba(212,0,255,0.45);--plasma-amber:#ff9d00;--plasma-crimson:#ff0055;--font-glyph:'Zen Dots',cursive;--font-runic:'Teko',sans-serif;--font-telemetry:'Share Tech Mono',monospace;}
*{margin:0;padding:0;box-sizing:border-box;user-select:none;-webkit-font-smoothing:antialiased;}
body{min-height:100vh;background-color:var(--void-black);color:#ccffea;font-family:var(--font-telemetry);display:flex;justify-content:center;align-items:center;padding:28px 16px;perspective:1500px;overflow-x:hidden;position:relative;background-image:radial-gradient(circle at 50% 50%,rgba(212,0,255,0.05) 0%,transparent 60%),radial-gradient(circle at 80% 20%,rgba(0,255,170,0.08) 0%,transparent 50%),repeating-radial-gradient(circle at 50% 50%,transparent 0,transparent 40px,rgba(0,255,170,0.015) 41px,transparent 42px);background-attachment:fixed;}
.alien-abyss{position:fixed;inset:-50%;width:200%;height:200%;background:radial-gradient(circle at 50% 50%,rgba(0,255,170,0.06),transparent 45%);animation:membraneMorph 12s infinite alternate ease-in-out;pointer-events:none;z-index:1;}
@keyframes membraneMorph{0%{transform:scale(0.9) rotate(0deg);filter:blur(50px);}50%{transform:scale(1.15) rotate(180deg);filter:blur(75px);}100%{transform:scale(0.95) rotate(360deg);filter:blur(50px);}}
body::after{content:"";position:fixed;inset:0;background:repeating-linear-gradient(0deg,rgba(0,0,0,0.6) 0px,rgba(0,0,0,0.6) 1px,transparent 1px,transparent 3px);pointer-events:none;z-index:99;opacity:0.55;}
.xenotech-pod{width:100%;max-width:520px;position:relative;z-index:10;transform-style:preserve-3d;animation:levitatePod 8s infinite ease-in-out;}
@keyframes levitatePod{0%{transform:translateY(0px) rotateX(2deg) rotateY(0deg);}50%{transform:translateY(-16px) rotateX(-2deg) rotateY(-1.5deg);}100%{transform:translateY(0px) rotateX(2deg) rotateY(0deg);}}
.hull-frame{position:relative;background:linear-gradient(170deg,var(--chitin-hull) 0%,rgba(1,8,5,0.94) 100%);border:1px solid rgba(0,255,170,0.28);border-radius:36px 6px 36px 6px;padding:38px 32px;backdrop-filter:blur(30px);-webkit-backdrop-filter:blur(30px);box-shadow:0 30px 80px -10px rgba(0,0,0,0.98),0 0 50px rgba(0,255,170,0.15),inset 0 1px 2px rgba(125,255,201,0.4),inset 0 0 35px rgba(212,0,255,0.08);clip-path:polygon(0 24px,24px 0,calc(100% - 24px) 0,100% 24px,100% calc(100% - 24px),calc(100% - 24px) 100%,24px 100%,0 calc(100% - 24px));}
.rune-node{position:absolute;width:16px;height:16px;pointer-events:none;border-color:var(--plasma-violet);}
.rune-tl{top:8px;left:8px;border-left:2px solid;border-top:2px solid;}
.rune-tr{top:8px;right:8px;border-right:2px solid;border-top:2px solid;}
.rune-bl{bottom:8px;left:8px;border-left:2px solid;border-bottom:2px solid;}
.rune-br{bottom:8px;right:8px;border-right:2px solid;border-bottom:2px solid;}
.energy-artery{position:absolute;top:0;left:0;width:100%;height:100%;background:radial-gradient(circle at 50% 0%,rgba(0,255,170,0.08),transparent 60%);pointer-events:none;}
.artifact-brand{text-align:center;margin-bottom:26px;position:relative;}
.monolith-wrapper{position:relative;width:76px;height:76px;margin:0 auto 14px;display:flex;align-items:center;justify-content:center;}
.plasma-containment-ring{position:absolute;inset:-4px;border-radius:50%;border:2px dashed var(--plasma-green);animation:ringOrbit 12s linear infinite;opacity:0.65;filter:drop-shadow(0 0 6px var(--plasma-green));}
.plasma-counter-ring{position:absolute;inset:4px;border-radius:50%;border:1px solid var(--plasma-violet);animation:ringOrbitRev 7s linear infinite;opacity:0.55;}
@keyframes ringOrbit{100%{transform:rotate(360deg);}}@keyframes ringOrbitRev{100%{transform:rotate(-360deg);}}
.monolith-core{width:54px;height:54px;background:radial-gradient(circle at 35% 30%,#00ffaa,#033d26 80%);clip-path:polygon(50% 0%,100% 25%,100% 75%,50% 100%,0% 75%,0% 25%);display:flex;align-items:center;justify-content:center;font-size:26px;box-shadow:0 0 25px var(--plasma-green),inset 0 0 10px rgba(255,255,255,0.7);z-index:2;}
.artifact-brand h1{font-family:var(--font-glyph);font-size:22px;font-weight:700;letter-spacing:3px;color:#ffffff;text-shadow:0 0 10px var(--plasma-green),0 0 25px var(--plasma-violet-glow);}
.setup-title{text-align:center;margin-bottom:22px;}
.setup-title h2{font-family:var(--font-runic);font-size:30px;letter-spacing:2px;color:#ffffff;text-transform:uppercase;}
.setup-title p{font-size:13px;color:#8dafa0;letter-spacing:0.5px;}
.scan-button{width:100%;padding:15px;background:linear-gradient(90deg,rgba(0,255,170,0.15) 0%,rgba(212,0,255,0.12) 100%);border:1px solid var(--plasma-green);border-radius:12px 2px 12px 2px;color:var(--plasma-green);font-family:var(--font-glyph);font-size:12px;letter-spacing:2px;text-transform:uppercase;cursor:pointer;box-shadow:0 0 20px rgba(0,255,170,0.25),inset 0 1px 1px rgba(255,255,255,0.4);transition:0.3s;}
.scan-button:hover:not(:disabled){background:linear-gradient(90deg,rgba(0,255,170,0.3) 0%,rgba(212,0,255,0.25) 100%);color:#ffffff;box-shadow:0 0 35px rgba(0,255,170,0.5);}
.scan-button:disabled{opacity:0.35;cursor:not-allowed;}
.section-label{font-family:var(--font-runic);font-size:20px;letter-spacing:1.5px;color:var(--plasma-core);margin:24px 0 10px 0;text-transform:uppercase;}
.network-list{border:1px solid rgba(0,255,170,0.15);border-radius:14px;background:rgba(2,7,4,0.95);max-height:230px;overflow-y:auto;}
.network-item{display:flex;align-items:center;padding:14px 18px;border-bottom:1px solid rgba(0,255,170,0.06);cursor:pointer;}
.network-item:hover{background:rgba(0,255,170,0.08);}
.network-item.selected{background:rgba(0,255,170,0.22);border-left:3px solid var(--plasma-green);}
.network-info{flex:1;margin-left:14px;}
.network-name{font-family:var(--font-glyph);font-size:12px;color:#fff;}
.network-security{font-size:11px;color:#8dafa0;}
.signal{font-family:var(--font-telemetry);font-size:14px;color:var(--plasma-green);font-weight:700;}
.empty-networks{padding:34px 16px;text-align:center;color:#55796a;font-size:12px;}
.password-section{display:none;margin-top:14px;}
.password-section.visible{display:block;}
.password-wrapper{position:relative;}
.password-input{width:100%;padding:14px 48px 14px 16px;background:#010603;border:1px solid rgba(0,255,170,0.2);border-radius:10px;color:#fff;font-size:13px;outline:none;}
.password-input:focus{border-color:var(--plasma-green);box-shadow:0 0 18px rgba(0,255,170,0.3);}
.show-password{position:absolute;right:14px;top:50%;transform:translateY(-50%);background:none;border:none;color:#8dafa0;cursor:pointer;}
.open-network-message{display:none;margin-top:16px;padding:12px 16px;background:rgba(0,255,170,0.08);border:1px solid rgba(0,255,170,0.3);border-radius:10px;color:var(--plasma-green);font-size:12px;}
.connect-button{width:100%;margin-top:24px;padding:17px;background:linear-gradient(135deg,var(--plasma-green) 0%,#00965e 100%);border:none;border-radius:14px 4px 14px 4px;color:#021a0d;font-family:var(--font-glyph);font-size:13px;font-weight:700;letter-spacing:2px;cursor:pointer;box-shadow:0 8px 30px rgba(0,255,170,0.45);}
.connect-button:hover:not(:disabled){transform:translateY(-3px);box-shadow:0 12px 42px rgba(0,255,170,0.7);}
.connect-button:disabled{background:#09140e;color:#2b4536;}
.status-box{margin-top:18px;padding:12px;border-radius:8px;background:#010603;border:1px solid rgba(255,255,255,0.08);text-align:center;font-size:12px;color:#8dafa0;}
.status-box.loading{color:var(--plasma-amber);}
.status-box.error{color:var(--plasma-crimson);border-color:rgba(255,0,85,0.4);}
.success-screen{display:none;text-align:center;}
.success-icon{width:80px;height:80px;margin:0 auto 20px;border-radius:50%;background:radial-gradient(circle at 35% 30%,var(--plasma-green),#004d2a 80%);display:flex;align-items:center;justify-content:center;font-size:34px;color:#021a0c;}
.connection-info{margin-top:22px;padding:16px 20px;background:#010603;border:1px solid rgba(0,255,170,0.2);border-radius:12px;text-align:left;}
.info-row{display:flex;justify-content:space-between;padding:8px 0;border-bottom:1px solid rgba(0,255,170,0.05);font-size:12px;}
.info-label{color:#55796a;}
.info-value{color:var(--plasma-core);font-weight:700;}
.spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(0,255,170,0.2);border-top-color:var(--plasma-green);border-radius:50%;animation:spinRotor 0.75s linear infinite;vertical-align:middle;margin-right:8px;}
@keyframes spinRotor{to{transform:rotate(360deg);}}
</style>
</head>
<body>
<div class="alien-abyss"></div>
<div class="xenotech-pod">
    <div class="hull-frame">
        <div class="energy-artery"></div>
        <div class="rune-node rune-tl"></div><div class="rune-node rune-tr"></div>
        <div class="rune-node rune-bl"></div><div class="rune-node rune-br"></div>
        
        <div id="setupScreen">
            <div class="artifact-brand">
                <div class="monolith-wrapper">
                    <div class="plasma-containment-ring"></div>
                    <div class="plasma-counter-ring"></div>
                    <div class="monolith-core">🌱</div>
                </div>
                <h1>KRUSHI MITHRA</h1>
            </div>
            
            <div class="setup-title">
                <h2>Wi-Fi Setup</h2>
                <p>Connect your Krushi Mithra device to your farm network.</p>
            </div>
            
            <button id="scanButton" class="scan-button" onclick="scanNetworks()">🔄 Scan Networks</button>
            
            <div class="section-label">Available Networks</div>
            <div id="networkList" class="network-list">
                <div class="empty-networks">Click "Scan Networks" to search.</div>
            </div>
            
            <div id="passwordSection" class="password-section">
                <div class="section-label">Wi-Fi Password</div>
                <div class="password-wrapper">
                    <input id="passwordInput" class="password-input" type="password" placeholder="Enter Wi-Fi password">
                    <button class="show-password" onclick="togglePassword()">👁</button>
                </div>
            </div>
            
            <div id="openNetworkMessage" class="open-network-message">🔓 This network does not require a password.</div>
            
            <button id="connectButton" class="connect-button" onclick="connectWiFi()" disabled>CONNECT</button>
            <div id="statusBox" class="status-box">● Select a Wi-Fi network</div>
        </div>

        <div id="successScreen" class="success-screen">
            <div class="success-icon">✓</div>
            <h2 style="font-family:var(--font-glyph);font-size:20px;color:#fff;">Wi-Fi Connected</h2>
            <div class="connection-info">
                <div class="info-row"><span class="info-label">Network</span><span id="connectedSSID" class="info-value">—</span></div>
                <div class="info-row"><span class="info-label">IP Address</span><span id="esp32IP" class="info-value">—</span></div>
                <div class="info-row"><span class="info-label">Dashboard</span><span class="info-value">http://krushi-mithra</span></div>
            </div>
            <button class="connect-button" onclick="window.location.href='http://krushi-mithra/'" style="margin-top:20px;">OPEN DASHBOARD</button>
        </div>
    </div>
</div>

<script>
let selectedNetwork = null;
let pollTimer = null;

const scanNetworks = async () => {
    const scanButton = document.getElementById("scanButton");
    const networkList = document.getElementById("networkList");
    const statusBox = document.getElementById("statusBox");

    scanButton.disabled = true;
    scanButton.innerHTML = '<span class="spinner"></span> Scanning...';
    statusBox.className = "status-box loading";
    statusBox.innerHTML = '<span class="spinner"></span> Searching...';

    try {
        const response = await fetch("/api/wifi/scan");
        const data = await response.json();
        networkList.innerHTML = "";
        if(!data.networks || data.networks.length === 0) {
            networkList.innerHTML = '<div class="empty-networks">No networks discovered. Click Scan Networks to retry.</div>';
            statusBox.className = "status-box";
            statusBox.textContent = "● No networks found";
            return;
        }
        data.networks.forEach(n => {
            const div = document.createElement("div");
            div.className = "network-item";
            div.innerHTML = `
                <input type="radio" name="net" class="network-radio">
                <div class="network-info">
                    <div class="network-name">${n.ssid}</div>
                    <div class="network-security">${n.security}</div>
                </div>
                <div class="signal">${n.rssi} dBm</div>`;
            div.onclick = () => {
                document.querySelectorAll(".network-item").forEach(i => i.classList.remove("selected"));
                div.classList.add("selected");
                div.querySelector("input").checked = true;
                selectedNetwork = n;
                document.getElementById("passwordSection").style.display = (n.security === "OPEN") ? "none" : "block";
                document.getElementById("openNetworkMessage").style.display = (n.security === "OPEN") ? "block" : "none";
                document.getElementById("connectButton").disabled = false;
                statusBox.textContent = "● Selected: " + n.ssid;
            };
            networkList.appendChild(div);
        });
        statusBox.className = "status-box";
        statusBox.textContent = "● Select a network";
    } catch(e) {
        statusBox.className = "status-box error";
        statusBox.textContent = "Scan failed.";
    } finally {
        scanButton.disabled = false;
        scanButton.innerHTML = "🔄 Scan Networks";
    }
};

const connectWiFi = async () => {
    if(!selectedNetwork) return;
    const pwd = document.getElementById("passwordInput").value;
    const btn = document.getElementById("connectButton");
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> CONNECTING...';

    try {
        const res = await fetch("/api/wifi/connect", {
            method: "POST",
            headers: {"Content-Type":"application/json"},
            body: JSON.stringify({ ssid: selectedNetwork.ssid, password: (selectedNetwork.security === "OPEN") ? "" : pwd })
        });
        const d = await res.json();
        if(d.status === "CONNECTING") {
            pollTimer = setInterval(pollStatus, 1500);
        } else {
            alert(d.error || "Rejected");
            btn.disabled = false;
            btn.innerHTML = "CONNECT";
        }
    } catch(e) {
        alert("Communication failed");
        btn.disabled = false;
        btn.innerHTML = "CONNECT";
    }
};

const pollStatus = async () => {
    try {
        const res = await fetch("/api/wifi/status");
        const s = await res.json();
        if(s.state === "CONNECTED") {
            clearInterval(pollTimer);
            document.getElementById("setupScreen").style.display = "none";
            document.getElementById("successScreen").style.display = "block";
            document.getElementById("connectedSSID").textContent = s.ssid;
            document.getElementById("esp32IP").textContent = s.ip;
            setTimeout(() => { window.location.href = 'http://krushi-mithra/'; }, 2500);
        } else if(s.state === "FAILED") {
            clearInterval(pollTimer);
            alert("Connection failed. Check your password.");
            document.getElementById("connectButton").disabled = false;
            document.getElementById("connectButton").innerHTML = "CONNECT";
        }
    } catch(e) {}
};

const togglePassword = () => {
    const p = document.getElementById("passwordInput");
    p.type = (p.type === "password") ? "text" : "password";
};

window.addEventListener("load", () => {
    setTimeout(scanNetworks, 500);
});
</script>
</body>
</html>
)=====";

// ----------------------------------------------------------------
// SENSOR PIPELINE
// ----------------------------------------------------------------

float soilRawToPercent(int raw) {
  if (raw >= SOIL_DRY_RAW) return 0.0f;
  if (raw <= SOIL_WET_RAW) return 100.0f;
  float pct = ((float)(SOIL_DRY_RAW - raw) / (float)(SOIL_DRY_RAW - SOIL_WET_RAW)) * 100.0f;
  return constrain(pct, 0.0f, 100.0f);
}

float rainRawToPercent(int raw) {
  if (raw >= RAIN_DRY_RAW) return 0.0f;
  if (raw <= RAIN_MAX_WET_RAW) return 100.0f;
  float pct = ((float)(RAIN_DRY_RAW - raw) / (float)(RAIN_DRY_RAW - RAIN_MAX_WET_RAW)) * 100.0f;
  return constrain(pct, 0.0f, 100.0f);
}

bool isRainDetected() {
  bool rainDetectedAO = (cachedRainRaw <= RAIN_DETECT_RAW);
  bool rainDetectedDO = (cachedRainDigital == RAIN_ACTIVE_LEVEL);
  return (rainDetectedAO || rainDetectedDO);
}

void readSensors(bool force) {
  unsigned long now = millis();
  if (!force && (now - lastSensorReadMs) < SENSOR_CACHE_INTERVAL_MS) return;
  lastSensorReadMs = now;

  long soilSum = 0;
  long rainSum = 0;
  for (int i = 0; i < 5; i++) {
    soilSum += analogRead(SOIL_PIN);
    rainSum += analogRead(RAIN_AO_PIN);
    delayMicroseconds(50);
  }
  cachedSoilRaw = (int)(soilSum / 5);
  cachedSoilPercent = soilRawToPercent(cachedSoilRaw);

  cachedRainRaw = (int)(rainSum / 5);
  cachedRainPercent = rainRawToPercent(cachedRainRaw);
  cachedRainDigital = digitalRead(RAIN_DO_PIN);
  cachedLightDigital = digitalRead(LDR_DO_PIN);

  static bool dhtFirst = true;
  static unsigned long lastDhtMs = 0;
  if (dhtFirst || (now - lastDhtMs) >= 2000) {
    dhtFirst = false;
    lastDhtMs = now;
    float t = dht.readTemperature();
    float h = dht.readHumidity();
    if (!isnan(t)) cachedTemperature = t;
    if (!isnan(h)) cachedHumidity = h;
  }

  static bool dsPending = false;
  static unsigned long dsRequestMs = 0;
  if (dsPending) {
    if ((now - dsRequestMs) >= 800) {
      float soilT = ds18b20.getTempCByIndex(0);
      if (soilT != DEVICE_DISCONNECTED_C && !isnan(soilT)) cachedSoilTemperature = soilT;
      dsPending = false;
    }
  } else if ((now - dsRequestMs) >= 2000) {
    ds18b20.requestTemperatures();
    dsRequestMs = now;
    dsPending = true;
  }
}

// ----------------------------------------------------------------
// PREFERENCES
// ----------------------------------------------------------------

void loadPreferences() {
  preferences.begin("krushi", false);
  savedSSID = preferences.getString("ssid", "");
  savedPassword = preferences.getString("password", "");
  systemEnabled = true;

  currentMode = "AUTO";
  resetManualTimer();

  soilTrigger = preferences.getFloat("trigger", 40.0f);
  criticalSoil = preferences.getFloat("critical", 25.0f);
  rainProtection = preferences.getBool("rainprotect", true);
  backendHost = preferences.getString("bhost", "");
  backendPort = preferences.getUShort("bport", 5000);
}

void saveSystemConfig() {
  preferences.begin("krushi", false);
  preferences.putFloat("trigger", soilTrigger);
  preferences.putFloat("critical", criticalSoil);
  preferences.putBool("rainprotect", rainProtection);
}

// ----------------------------------------------------------------
// HELPERS
// ----------------------------------------------------------------

void sendJsonError(int code, const char* msg) {
  String s = String("{\"error\":\"") + msg + "\"}";
  server.send(code, "application/json", s);
}

bool parseBody(JsonDocument& doc) {
  if (!server.hasArg("plain")) { sendJsonError(400, "Missing body"); return false; }
  DeserializationError err = deserializeJson(doc, server.arg("plain"));
  if (err) { sendJsonError(400, "Invalid JSON"); return false; }
  return true;
}

bool isPrivateLanIP(const IPAddress& ip) {
  uint8_t a = ip[0], b = ip[1];
  if (a == 10) return true;
  if (a == 172 && b >= 16 && b <= 31) return true;
  if (a == 192 && b == 168) return true;
  return false;
}

bool isBackendAvailable() {
  if (WiFi.status() != WL_CONNECTED) return false;
  if (backendHost.length() == 0) return false;
  if (lastBackendContactMs == 0) return false;
  if (millis() - lastBackendContactMs > BACKEND_TIMEOUT_MS) return false;
  return true;
}

// ----------------------------------------------------------------
// AUTOMATION & SERIAL LOGGING
// ----------------------------------------------------------------

void runAutomation() {
  unsigned long now = millis();

  static unsigned long lastSerialPrintMs = 0;
  if (now - lastSerialPrintMs >= 2000) {
    lastSerialPrintMs = now;
    Serial.printf("[TELEMETRY] Air: %.1fC | Soil: %.1f%% (Raw: %d), %.1fC | Rain: %s (Raw: %d, %.1f%%) | Light: %s | Pump: %s | Mode: %s",
      isnan(cachedTemperature) ? 0.0f : cachedTemperature,
      cachedSoilPercent,
      cachedSoilRaw,
      isnan(cachedSoilTemperature) ? 0.0f : cachedSoilTemperature,
      isRainDetected() ? "YES" : "NO",
      cachedRainRaw,
      cachedRainPercent,
      (cachedLightDigital == LIGHT_ACTIVE_LEVEL) ? "DAY" : "NIGHT",
      relayState ? "ON" : "OFF",
      currentMode.c_str()
    );

    if (manualTimerActive) {
      unsigned long remaining = (manualPumpUntilMs > now) ? (manualPumpUntilMs - now) : 0;
      Serial.printf(" | ManualRemaining: %lus", remaining / 1000UL);
    }
    Serial.println();
  }

  // 1. TIMED MANUAL SESSION
  if (currentMode == "MANUAL" && manualTimerActive) {
    if (now >= manualPumpUntilMs) {
      stopManualSessionAndReturnAuto("Timed run completed");
      return;
    }

    if (relayState && rainProtection && isRainDetected()) {
      stopManualSessionAndReturnAuto("Rain detected during manual run");
      return;
    }

    return;
  }

  // 2. MANUAL INACTIVITY TIMEOUT
  if (currentMode == "MANUAL" && !relayState) {
    if ((now - manualModeLastActivityMs) >= MANUAL_IDLE_TIMEOUT_MS) {
      stopManualSessionAndReturnAuto("Manual inactivity timeout");
    }
    return;
  }

  if (currentMode != "AUTO") return;

  // 3. AUTO MODE MAXIMUM RUNTIME LIMIT
  if (relayState && (now - pumpOnSinceMs) > AUTO_PUMP_MAX_RUN_MS) {
    safePumpOff();
    pumpLockoutUntilMs = now + PUMP_COOLDOWN_MS;
    Serial.println("[SAFETY] AUTO pump max run time reached - OFF");
    return;
  }

  // 4. PHYSICAL RAIN PROTECTION
  if (rainProtection && isRainDetected()) {
    if (relayState) {
      safePumpOff();
      Serial.println("[SAFETY] Rain detected - pump OFF");
    }
    return;
  }

  // 5. BACKEND AVAILABILITY TRACKING
  bool backendActive = isBackendAvailable();

  static bool lastBackendState = false;
  static bool initialLogDone = false;
  if (!initialLogDone || backendActive != lastBackendState) {
    initialLogDone = true;
    lastBackendState = backendActive;
    if (backendActive) {
      Serial.println("[AUTO] Backend available - Python control restored");
    } else {
      Serial.println("[AUTO][BACKUP] Backend unavailable - ESP32 local backup active");
    }
  }

  if (backendActive) return;

  // 6. ESP32 LOCAL AUTO BACKUP
  static unsigned long lastBackupEvalMs = 0;
  if (now - lastBackupEvalMs >= 2000) {
    lastBackupEvalMs = now;

    if (rainProtection && isRainDetected()) {
      if (relayState) {
        safePumpOff();
        Serial.println("[SAFETY] Rain detected - backup irrigation blocked");
      }
      return;
    }

    if (isnan(cachedSoilPercent) || cachedSoilRaw <= 0 || cachedSoilRaw > 4095) {
      if (relayState) {
        safePumpOff();
        Serial.println("[SAFETY] Sensor reading invalid - pump OFF");
      }
      return;
    }

    if (cachedSoilPercent <= BACKUP_PUMP_ON_SOIL_PERCENT) {
      if (!relayState) {
        if (now >= pumpLockoutUntilMs) {
          setRelay(true);
          Serial.println("[AUTO][BACKUP] Soil <= 20% - Pump ON");
        }
      }
    } else if (cachedSoilPercent >= BACKUP_PUMP_OFF_SOIL_PERCENT) {
      if (relayState) {
        safePumpOff();
        Serial.println("[AUTO][BACKUP] Soil >= 70% - Pump OFF");
      }
    }
  }
}

// ----------------------------------------------------------------
// WIFI TASK
// ----------------------------------------------------------------

void wifiTask() {
  unsigned long now = millis();

  if (wifiConnectRequested) {
    wifiConnectRequested = false;
    if (!setupAPActive) startSetupAP();

    WiFi.disconnect(false);
    WiFi.begin(pendingSSID.c_str(), pendingPassword.c_str());
    wifiState = WIFI_CONNECTING;
    wifiAttemptStart = now;
    Serial.println("[WIFI] Connecting to " + pendingSSID);
    return;
  }

  if (wifiState == WIFI_CONNECTING) {
    if (WiFi.status() == WL_CONNECTED) {
      savedSSID = pendingSSID.length() ? pendingSSID : WiFi.SSID();
      savedPassword = pendingPassword;

      preferences.begin("krushi", false);
      preferences.putString("ssid", savedSSID);
      preferences.putString("password", savedPassword);

      wifiState = WIFI_IDLE;
      WiFi.setSleep(false);
      startMDNS();
      startDiscovery();

      if (setupAPActive && apShutdownAtMs == 0) {
        apShutdownAtMs = now + AP_SHUTDOWN_GRACE_MS;
      }

      Serial.println("[WIFI] Connected, IP: " + WiFi.localIP().toString());
      Serial.println("[WIFI] Dashboard: http://krushi-mithra");
    } else if ((now - wifiAttemptStart) > WIFI_CONNECT_TIMEOUT_MS) {
      wifiState = WIFI_FAILED;
      WiFi.disconnect(false);
      Serial.println("[WIFI] Connection timed out. Setup hotspot ON.");
      startSetupAP();
      lastReconnectMs = now;
    }
  }

  if (WiFi.status() != WL_CONNECTED) {
    apShutdownAtMs = 0;
    if (mdnsStarted) stopMDNS();
    if (discoveryStarted) stopDiscovery();

    if (!setupAPActive && wifiState != WIFI_CONNECTING) {
      startSetupAP();
    }

    if (savedSSID.length() > 0 &&
        wifiState != WIFI_CONNECTING &&
        (now - lastReconnectMs) > 30000) {
      lastReconnectMs = now;
      pendingSSID = savedSSID;
      pendingPassword = savedPassword;
      WiFi.begin(pendingSSID.c_str(), pendingPassword.c_str());
      wifiState = WIFI_CONNECTING;
      wifiAttemptStart = now;
      Serial.println("[WIFI] Retrying saved network...");
    }
    return;
  }

  if (setupAPActive && apShutdownAtMs != 0 && now >= apShutdownAtMs) {
    stopSetupAP();
    apShutdownAtMs = 0;
  }
}

// ----------------------------------------------------------------
// API ROUTE HANDLERS
// ----------------------------------------------------------------

void learnBackendFromClient() {
  if (WiFi.status() != WL_CONNECTED) return;

  IPAddress clientIP = server.client().remoteIP();
  if (!isPrivateLanIP(clientIP)) return;
  if (clientIP == WiFi.localIP()) return;
  if (clientIP == AP_IP) return;
  if (clientIP[0] == 192 && clientIP[1] == 168 && clientIP[2] == 4) return;

  String discoveredHost = clientIP.toString();
  if (backendHost != discoveredHost || backendPort != 5000) {
    backendHost = discoveredHost;
    backendPort = 5000;
    preferences.begin("krushi", false);
    preferences.putString("bhost", backendHost);
    preferences.putUShort("bport", backendPort);
    Serial.println("[BACKEND] Discovered PC backend at: " + backendHost + ":5000");
  }
}

void startSetupAP() {
  if (WiFi.status() == WL_CONNECTED) return;
  if (setupAPActive) return;

  WiFi.mode(WIFI_AP_STA);
  WiFi.softAPConfig(AP_IP, AP_GATEWAY, AP_SUBNET);
  WiFi.softAP(AP_SSID, AP_PASSWORD);
  dnsServer.start(DNS_PORT, "*", AP_IP);

  setupAPActive = true;
  Serial.println("[AP] Setup hotspot ON: http://192.168.4.1/wifi-setup");
}

void stopSetupAP() {
  if (!setupAPActive) return;

  dnsServer.stop();
  WiFi.softAPdisconnect(true);
  WiFi.mode(WIFI_STA);

  setupAPActive = false;
  Serial.println("[AP] Setup hotspot OFF");
}

void startMDNS() {
  if (mdnsStarted || WiFi.status() != WL_CONNECTED) return;
  if (MDNS.begin(MDNS_HOSTNAME)) {
    MDNS.addService("http", "tcp", 80);
    mdnsStarted = true;
  }
}

void stopMDNS() {
  if (!mdnsStarted) return;
  MDNS.end();
  mdnsStarted = false;
}

void startDiscovery() {
  if (discoveryStarted || WiFi.status() != WL_CONNECTED) return;
  if (discoveryUdp.begin(DISCOVERY_PORT)) {
    discoveryStarted = true;
  }
}

void stopDiscovery() {
  if (!discoveryStarted) return;
  discoveryUdp.stop();
  discoveryStarted = false;
}

void processDiscovery() {
  if (!discoveryStarted || WiFi.status() != WL_CONNECTED) return;

  int packetSize = discoveryUdp.parsePacket();
  if (packetSize <= 0) return;

  char packet[96];
  int len = discoveryUdp.read(packet, sizeof(packet) - 1);
  if (len < 0) return;
  packet[len] = '\0';

  if (String(packet) != DISCOVERY_MESSAGE) return;

  JsonDocument doc;
  doc["device"] = "KRUSHI_MITHRA";
  doc["ip"] = WiFi.localIP().toString();
  doc["port"] = 80;
  doc["hostname"] = "krushi-mithra";

  String response;
  serializeJson(doc, response);

  discoveryUdp.beginPacket(discoveryUdp.remoteIP(), discoveryUdp.remotePort());
  discoveryUdp.print(response);
  discoveryUdp.endPacket();
}

bool requestFromSetupAP() {
  IPAddress clientIP = server.client().remoteIP();
  return clientIP[0] == 192 && clientIP[1] == 168 && clientIP[2] == 4;
}

void captiveRedirect() {
  if (requestFromSetupAP() || WiFi.status() != WL_CONNECTED) {
    server.sendHeader("Location", "http://192.168.4.1/wifi-setup", true);
    server.send(302, "text/html", "<meta http-equiv='refresh' content='0;url=http://192.168.4.1/wifi-setup'>");
  } else {
    server.sendHeader("Location", "http://" + WiFi.localIP().toString() + "/", true);
    server.send(302, "text/plain", "");
  }
}

void handleConfig() {
  if (server.method() == HTTP_POST) {
    JsonDocument in;
    if (!parseBody(in)) return;

    if (in["soil_trigger"].is<float>() || in["soil_trigger"].is<int>()) {
      float v = in["soil_trigger"].as<float>();
      if (v >= 0.0f && v <= 100.0f) soilTrigger = v;
    }
    if (in["critical_soil"].is<float>() || in["critical_soil"].is<int>()) {
      float v = in["critical_soil"].as<float>();
      if (v >= 0.0f && v <= 100.0f) criticalSoil = v;
    }
    if (in["rain_protection"].is<bool>()) rainProtection = in["rain_protection"].as<bool>();
    
    saveSystemConfig();
  }

  JsonDocument out;
  out["system_enabled"] = true;
  out["mode"] = currentMode;
  out["soil_trigger"] = soilTrigger;
  out["critical_soil"] = criticalSoil;
  out["rain_protection"] = rainProtection;

  String body;
  serializeJson(out, body);
  server.send(200, "application/json", body);
}

void handleData() {
  learnBackendFromClient();

  String ua = server.header("User-Agent");
  if (ua.indexOf("python") >= 0 || ua.indexOf("Krushi") >= 0) {
    lastBackendContactMs = millis();
  }

  readSensors();
  JsonDocument doc;

  if (isnan(cachedTemperature)) doc["temperature"] = nullptr; else doc["temperature"] = cachedTemperature;
  if (isnan(cachedHumidity)) doc["humidity"] = nullptr; else doc["humidity"] = cachedHumidity;
  
  doc["soil_moisture_raw"] = cachedSoilRaw;
  doc["soil_moisture_percent"] = cachedSoilPercent;
  
  if (isnan(cachedSoilTemperature)) doc["soil_temperature"] = nullptr; else doc["soil_temperature"] = cachedSoilTemperature;

  doc["rain_detected"] = isRainDetected();
  doc["rain_raw"] = cachedRainRaw;
  doc["rain_percent"] = cachedRainPercent;
  doc["light_detected"] = (cachedLightDigital == LIGHT_ACTIVE_LEVEL);
  doc["pump"] = relayState;
  doc["system_enabled"] = systemEnabled;
  doc["mode"] = currentMode;
  doc["backend_available"] = isBackendAvailable();
  doc["manual_timer_active"] = manualTimerActive;
  doc["manual_remaining_ms"] = manualTimerActive
      ? (manualPumpUntilMs > millis() ? manualPumpUntilMs - millis() : 0)
      : 0;

  bool connected = (WiFi.status() == WL_CONNECTED);
  doc["wifi_connected"] = connected;

  String s;
  serializeJson(doc, s);
  server.send(200, "application/json", s);
}

void handleStatus() {
  JsonDocument doc;
  bool connected = (WiFi.status() == WL_CONNECTED);
  doc["online"] = true;
  doc["system_enabled"] = true;
  doc["mode"] = currentMode;
  doc["pump"] = relayState;
  doc["backend_available"] = isBackendAvailable();
  doc["manual_timer_active"] = manualTimerActive;
  doc["manual_remaining_ms"] = manualTimerActive
      ? (manualPumpUntilMs > millis() ? manualPumpUntilMs - millis() : 0)
      : 0;
  doc["wifi_connected"] = connected;
  doc["setup_ap_active"] = setupAPActive;
  doc["hostname"] = "krushi-mithra";
  doc["wifi_rssi"] = connected ? WiFi.RSSI() : 0;
  doc["ip"] = connected ? WiFi.localIP().toString() : WiFi.softAPIP().toString();

  String s;
  serializeJson(doc, s);
  server.send(200, "application/json", s);
}

void handleMode() {
  if (server.method() == HTTP_POST) {
    JsonDocument doc;
    if (!parseBody(doc)) return;

    String m = doc["mode"].as<String>();
    m.toUpperCase();
    if (m != "AUTO" && m != "MANUAL") {
      sendJsonError(400, "Mode must be AUTO or MANUAL");
      return;
    }

    if (m == "AUTO") {
      stopManualSessionAndReturnAuto("Mode set to AUTO");
    } else {
      safePumpOff();
      resetManualTimer();
      currentMode = "MANUAL";
      manualModeLastActivityMs = millis();
    }
  }
  handleStatus();
}

void handlePump() {
  JsonDocument doc;
  if (!parseBody(doc)) return;

  if (!doc["state"].is<bool>()) {
    sendJsonError(400, "state must be boolean");
    return;
  }

  bool state = doc["state"].as<bool>();
  String source = doc["source"] | "manual";
  source.toLowerCase();

  if (source != "manual" && source != "pc") {
    sendJsonError(400, "source must be 'manual' or 'pc'");
    return;
  }

  if (source == "pc") {
    learnBackendFromClient();
    lastBackendContactMs = millis();
  }

  if (!state) {
    safePumpOff();
    resetManualTimer();
    if (source == "manual") manualModeLastActivityMs = millis();
    handleStatus();
    return;
  }

  if (source == "manual" && currentMode != "MANUAL") {
    sendJsonError(409, "Switch to MANUAL mode to control the pump");
    return;
  }

  if (source == "pc" && currentMode != "AUTO") {
    sendJsonError(409, "PC pump control is only allowed in AUTO mode");
    return;
  }

  if (source == "pc" && WiFi.status() != WL_CONNECTED) {
    sendJsonError(409, "ESP32 is not connected to farm Wi-Fi");
    return;
  }

  readSensors(true);

  if (rainProtection && isRainDetected()) {
    safePumpOff();
    resetManualTimer();
    sendJsonError(409, "Pump blocked by rain protection");
    return;
  }

  if (source == "manual") {
    float value = 10.0f;
    String unit = "MIN";

    if (doc["duration_value"].is<int>() || doc["duration_value"].is<float>()) {
      value = doc["duration_value"].as<float>();
      unit = doc["duration_unit"] | "MIN";
      unit.toUpperCase();
    }

    if (!isfinite(value) || value <= 0.0f || value > 1440.0f) {
      sendJsonError(400, "duration_value must be between 1 and 1440");
      return;
    }

    unsigned long multiplier = 60000UL;
    if (unit == "SEC") multiplier = 1000UL;
    else if (unit == "MIN") multiplier = 60000UL;
    else if (unit == "HR") multiplier = 3600000UL;
    else {
      sendJsonError(400, "duration_unit must be SEC, MIN, or HR");
      return;
    }

    unsigned long durationMs = (unsigned long)(value * (float)multiplier);
    if (durationMs == 0 || durationMs > MANUAL_MAX_DURATION_MS) {
      sendJsonError(400, "Manual duration exceeds 24 hour maximum");
      return;
    }

    currentMode = "MANUAL";
    beginManualTimer(durationMs);
    setRelay(true);
    handleStatus();
    return;
  }

  setRelay(true);
  handleStatus();
}

void handleWiFiScan() {
  if (setupAPActive) {
    if (WiFi.getMode() != WIFI_AP_STA) WiFi.mode(WIFI_AP_STA);
  } else {
    if (WiFi.getMode() != WIFI_STA) WiFi.mode(WIFI_STA);
  }
  delay(50);
  WiFi.disconnect(false);
  delay(100);

  int count = WiFi.scanNetworks(false, true, false, 300);
  if (count < 0) {
    delay(150);
    count = WiFi.scanNetworks(false, true, false, 300);
  }

  JsonDocument doc;
  JsonArray nets = doc["networks"].to<JsonArray>();

  if (count > 0) {
    for (int i = 0; i < count; i++) {
      String ssid = WiFi.SSID(i);
      if (ssid.length() == 0) continue;

      bool duplicate = false;
      for (JsonObject existing : nets) {
        if (existing["ssid"].as<String>() == ssid) {
          duplicate = true;
          break;
        }
      }
      if (duplicate) continue;

      JsonObject n = nets.add<JsonObject>();
      n["ssid"] = ssid;
      n["rssi"] = WiFi.RSSI(i);
      n["security"] = (WiFi.encryptionType(i) == WIFI_AUTH_OPEN) ? "OPEN" : "WPA/WPA2";
    }
  }

  WiFi.scanDelete();
  String s;
  serializeJson(doc, s);
  server.send(200, "application/json", s);
}

void handleWiFiConnect() {
  JsonDocument doc;
  if (!parseBody(doc)) return;

  String ssid = doc["ssid"].as<String>();
  String pwd = doc["password"].as<String>();

  if (ssid.length() == 0) {
    sendJsonError(400, "SSID is required");
    return;
  }

  pendingSSID = ssid;
  pendingPassword = pwd;
  wifiConnectRequested = true;
  wifiState = WIFI_CONNECTING;
  wifiAttemptStart = millis();

  server.send(200, "application/json", "{\"status\":\"CONNECTING\"}");
}

void handleWiFiStatus() {
  JsonDocument doc;
  bool connected = (WiFi.status() == WL_CONNECTED);

  const char* state = "idle";
  if (connected) state = "CONNECTED";
  else if (wifiState == WIFI_CONNECTING) state = "connecting";
  else if (wifiState == WIFI_FAILED) state = "FAILED";

  doc["state"] = state;
  doc["ssid"] = connected ? WiFi.SSID() : String("");
  doc["ip"] = connected ? WiFi.localIP().toString() : String("");
  doc["hostname"] = "krushi-mithra";
  doc["setup_ap_active"] = setupAPActive;
  doc["dashboard_url"] = "http://krushi-mithra/";

  String s;
  serializeJson(doc, s);
  server.send(200, "application/json", s);
}

void handleWiFiLogout() {
  safePumpOff();
  stopManualSessionAndReturnAuto("Wi-Fi logout");

  savedSSID = "";
  savedPassword = "";
  pendingSSID = "";
  pendingPassword = "";
  backendHost = "";
  backendPort = 5000;

  preferences.begin("krushi", false);
  preferences.putString("ssid", "");
  preferences.putString("password", "");
  preferences.putString("bhost", "");
  preferences.putUShort("bport", 5000);

  wifiConnectRequested = false;
  wifiState = WIFI_IDLE;
  apShutdownAtMs = 0;

  if (mdnsStarted) stopMDNS();
  if (discoveryStarted) stopDiscovery();

  WiFi.disconnect(true, false);
  delay(50);
  startSetupAP();

  server.send(200, "application/json", "{\"status\":\"LOGGED_OUT\",\"setup_url\":\"http://192.168.4.1/wifi-setup\"}");
}

void handleBackendHeartbeat() {
  learnBackendFromClient();
  lastBackendContactMs = millis();
  server.send(200, "application/json", "{\"status\":\"OK\",\"backend_available\":true}");
}

void proxyML(const String& endpoint) {
  learnBackendFromClient();
  if (backendHost.length() == 0 || WiFi.status() != WL_CONNECTED) {
    sendJsonError(503, "Python Backend offline");
    return;
  }

  String body = server.hasArg("plain") ? server.arg("plain") : String("{}");

  if (endpoint == "/api/ml/soil-moisture") {
    JsonDocument in;
    if (deserializeJson(in, body)) in.clear();
    readSensors(true);
    if (!isnan(cachedTemperature)) in["temperature"] = cachedTemperature;
    if (!isnan(cachedHumidity)) in["humidity"] = cachedHumidity;
    in["soil_moisture_percent"] = cachedSoilPercent;
    if (!isnan(cachedSoilTemperature)) in["soil_temperature"] = cachedSoilTemperature;
    in["rain_detected"] = isRainDetected();
    body = "";
    serializeJson(in, body);
  }

  WiFiClient client;
  HTTPClient http;
  String url = String("http://") + backendHost + ":" + String(backendPort) + endpoint;

  http.setTimeout(10000);
  if (!http.begin(client, url)) {
    sendJsonError(503, "Invalid backend URL");
    return;
  }

  int code;
  if (server.method() == HTTP_GET) {
    code = http.GET();
  } else {
    http.addHeader("Content-Type", "application/json");
    code = http.POST(body);
  }

  if (code > 0) {
    String resp = http.getString();
    lastBackendContactMs = millis();
    server.send(code, "application/json", resp);
  } else {
    sendJsonError(502, "Backend unreachable");
  }
  http.end();
}

void handleCaptiveProbe() {
  captiveRedirect();
}

void handleNotFound() {
  String uri = server.uri();
  if (uri.startsWith("/api/")) {
    sendJsonError(404, "Endpoint Not found");
    return;
  }
  captiveRedirect();
}

void setup() {
  safePumpOff();

  Serial.begin(115200);
  Serial.println("\n[BOOT] Initializing Krushi Mithra Firmware...");

  analogReadResolution(12);
#if defined(ADC_12db)
  analogSetAttenuation(ADC_12db);
#elif defined(ADC_ATTEN_DB_12)
  analogSetAttenuation(ADC_ATTEN_DB_12);
#else
  analogSetAttenuation(ADC_11db);
#endif

  pinMode(SOIL_PIN, INPUT);
  pinMode(RAIN_AO_PIN, INPUT);
  pinMode(RAIN_DO_PIN, INPUT);
  pinMode(LDR_DO_PIN, INPUT);

  dht.begin();
  ds18b20.begin();
  ds18b20.setWaitForConversion(false);

  loadPreferences();

  WiFi.setAutoReconnect(true);

  if (savedSSID.length() > 0) {
    pendingSSID = savedSSID;
    pendingPassword = savedPassword;
    WiFi.mode(WIFI_STA);
    WiFi.begin(pendingSSID.c_str(), pendingPassword.c_str());
    WiFi.setSleep(false);
    wifiState = WIFI_CONNECTING;
    wifiAttemptStart = millis();
  } else {
    startSetupAP();
    wifiState = WIFI_FAILED;
  }

  static const char* headerkeys[] = {"User-Agent"};
  server.collectHeaders(headerkeys, 1);

  server.on("/", HTTP_GET, []() {
    if (requestFromSetupAP() || WiFi.status() != WL_CONNECTED) {
      server.send_P(200, "text/html", SETUP_HTML);
    } else {
      learnBackendFromClient();
      server.send_P(200, "text/html", DASH_HTML);
    }
  });
  server.on("/wifi-setup", HTTP_GET, []() { server.send_P(200, "text/html", SETUP_HTML); });

  server.on("/generate_204", HTTP_GET, handleCaptiveProbe);
  server.on("/gen_204", HTTP_GET, handleCaptiveProbe);
  server.on("/hotspot-detect.html", HTTP_GET, handleCaptiveProbe);
  server.on("/canonical.html", HTTP_GET, handleCaptiveProbe);
  server.on("/connecttest.txt", HTTP_GET, handleCaptiveProbe);
  server.on("/ncsi.txt", HTTP_GET, handleCaptiveProbe);
  server.on("/fwlink", HTTP_GET, handleCaptiveProbe);
  server.on("/redirect", HTTP_GET, handleCaptiveProbe);

  server.on("/api/data", HTTP_GET, handleData);
  server.on("/api/status", HTTP_GET, handleStatus);
  server.on("/api/config", HTTP_GET, handleConfig);
  server.on("/api/config", HTTP_POST, handleConfig);
  server.on("/api/mode", HTTP_POST, handleMode);
  server.on("/api/pump", HTTP_POST, handlePump);
  server.on("/api/backend/heartbeat", HTTP_POST, handleBackendHeartbeat);
  server.on("/api/backend/heartbeat", HTTP_GET, handleBackendHeartbeat);
  
  server.on("/api/wifi/scan", HTTP_GET, handleWiFiScan);
  server.on("/api/wifi/connect", HTTP_POST, handleWiFiConnect);
  server.on("/api/wifi/status", HTTP_GET, handleWiFiStatus);
  server.on("/api/wifi/logout", HTTP_POST, handleWiFiLogout);

  server.on("/api/ml/crop", HTTP_POST, []() { proxyML("/api/ml/crop"); });
  server.on("/api/ml/soil-moisture", HTTP_POST, []() { proxyML("/api/ml/soil-moisture"); });
  server.on("/api/ml/rainfall", HTTP_POST, []() { proxyML("/api/ml/rainfall"); });

  server.onNotFound(handleNotFound);
  server.begin();

  Serial.println("[BOOT] Krushi Mithra Master Firmware Active.");
}

void loop() {
  if (setupAPActive) dnsServer.processNextRequest();

  server.handleClient();
  readSensors();
  runAutomation();
  wifiTask();
  processDiscovery();

  yield();
}