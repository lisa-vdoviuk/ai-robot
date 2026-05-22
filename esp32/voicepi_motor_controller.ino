#include <Arduino.h>
#include <WiFi.h>
#include <WebServer.h>
#include <ArduinoJson.h>

// ===================== USER CONFIG =====================
// Change these before flashing.
const char* WIFI_SSID = "YOUR_WIFI_SSID";
const char* WIFI_PASSWORD = "YOUR_WIFI_PASSWORD";
const char* API_TOKEN = "change-me-robot-token";

// Generic two-channel motor driver wiring.
// Wire the two left-side motors to the left channel and the two right-side
// motors to the right channel. For a 4-channel driver, duplicate the same
// direction/speed signals per side or adapt setSide().
const int LEFT_IN1 = 26;
const int LEFT_IN2 = 27;
const int LEFT_PWM = 25;

const int RIGHT_IN1 = 32;
const int RIGHT_IN2 = 33;
const int RIGHT_PWM = 14;

// Tune these after verifying wheel direction.
const bool INVERT_LEFT = false;
const bool INVERT_RIGHT = false;

const int PWM_MAX = 255;
const int MIN_DURATION_MS = 150;
const int MAX_DURATION_MS = 1800;
// =======================================================

WebServer server(80);
const char* HEADER_KEYS[] = {"X-VoicePi-Token"};
unsigned long stopAtMs = 0;
String lastCommand = "stop";
float lastSpeed = 0.0;

bool authorized() {
  String token = server.header("X-VoicePi-Token");
  return token == API_TOKEN;
}

void sendJson(int status, const String& body) {
  server.sendHeader("Access-Control-Allow-Origin", "*");
  server.send(status, "application/json", body);
}

void stopMotors() {
  analogWrite(LEFT_PWM, 0);
  analogWrite(RIGHT_PWM, 0);
  digitalWrite(LEFT_IN1, LOW);
  digitalWrite(LEFT_IN2, LOW);
  digitalWrite(RIGHT_IN1, LOW);
  digitalWrite(RIGHT_IN2, LOW);
  stopAtMs = 0;
  lastCommand = "stop";
  lastSpeed = 0.0;
}

void setSide(int in1, int in2, int pwmPin, int signedPwm, bool invert) {
  if (invert) signedPwm = -signedPwm;
  signedPwm = constrain(signedPwm, -PWM_MAX, PWM_MAX);

  if (signedPwm > 0) {
    digitalWrite(in1, HIGH);
    digitalWrite(in2, LOW);
    analogWrite(pwmPin, signedPwm);
  } else if (signedPwm < 0) {
    digitalWrite(in1, LOW);
    digitalWrite(in2, HIGH);
    analogWrite(pwmPin, -signedPwm);
  } else {
    digitalWrite(in1, LOW);
    digitalWrite(in2, LOW);
    analogWrite(pwmPin, 0);
  }
}

void drive(int leftPwm, int rightPwm) {
  setSide(LEFT_IN1, LEFT_IN2, LEFT_PWM, leftPwm, INVERT_LEFT);
  setSide(RIGHT_IN1, RIGHT_IN2, RIGHT_PWM, rightPwm, INVERT_RIGHT);
}

void requireAuthOrFail() {
  if (!authorized()) {
    sendJson(401, "{\"ok\":false,\"error\":\"unauthorized\"}");
  }
}

void handleStatus() {
  if (!authorized()) {
    sendJson(401, "{\"ok\":false,\"error\":\"unauthorized\"}");
    return;
  }
  String body = "{\"ok\":true";
  body += ",\"ip\":\"" + WiFi.localIP().toString() + "\"";
  body += ",\"last_command\":\"" + lastCommand + "\"";
  body += ",\"last_speed\":" + String(lastSpeed, 3);
  body += ",\"moving\":" + String(stopAtMs > millis() ? "true" : "false");
  body += ",\"stop_in_ms\":" + String(stopAtMs > millis() ? stopAtMs - millis() : 0);
  body += "}";
  sendJson(200, body);
}

void handleStop() {
  if (!authorized()) {
    sendJson(401, "{\"ok\":false,\"error\":\"unauthorized\"}");
    return;
  }
  stopMotors();
  sendJson(200, "{\"ok\":true,\"action\":\"stop\"}");
}

void handleMove() {
  if (!authorized()) {
    sendJson(401, "{\"ok\":false,\"error\":\"unauthorized\"}");
    return;
  }
  if (server.method() != HTTP_POST) {
    sendJson(405, "{\"ok\":false,\"error\":\"method_not_allowed\"}");
    return;
  }

  String body = server.arg("plain");
  StaticJsonDocument<256> doc;
  DeserializationError err = deserializeJson(doc, body);
  if (err) {
    sendJson(400, "{\"ok\":false,\"error\":\"invalid_json\"}");
    return;
  }

  String direction = doc["direction"] | "";
  float speed = doc["speed"] | 0.45;
  int durationMs = doc["duration_ms"] | 700;
  speed = constrain(speed, 0.0, 0.85);
  durationMs = constrain(durationMs, MIN_DURATION_MS, MAX_DURATION_MS);
  int pwm = constrain((int)(speed * PWM_MAX), 0, PWM_MAX);

  if (direction == "forward") {
    drive(pwm, pwm);
  } else if (direction == "backward") {
    drive(-pwm, -pwm);
  } else if (direction == "left") {
    drive(-pwm, pwm);
  } else if (direction == "right") {
    drive(pwm, -pwm);
  } else {
    sendJson(400, "{\"ok\":false,\"error\":\"bad_direction\"}");
    return;
  }

  stopAtMs = millis() + durationMs;
  lastCommand = direction;
  lastSpeed = speed;

  String response = "{\"ok\":true";
  response += ",\"action\":\"move\"";
  response += ",\"direction\":\"" + direction + "\"";
  response += ",\"speed\":" + String(speed, 3);
  response += ",\"duration_ms\":" + String(durationMs);
  response += "}";
  sendJson(200, response);
}

void setupPins() {
  pinMode(LEFT_IN1, OUTPUT);
  pinMode(LEFT_IN2, OUTPUT);
  pinMode(LEFT_PWM, OUTPUT);
  pinMode(RIGHT_IN1, OUTPUT);
  pinMode(RIGHT_IN2, OUTPUT);
  pinMode(RIGHT_PWM, OUTPUT);
  stopMotors();
}

void setup() {
  Serial.begin(115200);
  delay(300);
  setupPins();

  WiFi.mode(WIFI_STA);
  WiFi.begin(WIFI_SSID, WIFI_PASSWORD);
  Serial.print("Connecting to Wi-Fi");
  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }
  Serial.println();
  Serial.print("ESP32 motor API: http://");
  Serial.println(WiFi.localIP());

  server.collectHeaders(HEADER_KEYS, 1);
  server.on("/api/status", HTTP_GET, handleStatus);
  server.on("/api/move", HTTP_POST, handleMove);
  server.on("/api/stop", HTTP_POST, handleStop);
  server.onNotFound([]() {
    sendJson(404, "{\"ok\":false,\"error\":\"not_found\"}");
  });
  server.begin();
}

void loop() {
  server.handleClient();
  if (stopAtMs > 0 && millis() >= stopAtMs) {
    stopMotors();
  }
}
