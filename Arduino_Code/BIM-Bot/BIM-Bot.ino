/*READ ME - Zen
MGA NEED NA LIB/BOARD KINEME
!FOR DHT11!
- Sketch > Iclude Library > Manage Libraries > Search "DHT Sensor Library" by adafruit, install all
!FOR ESP32!
- File > Preferences > Additional Board Manager URLS input the url (para mag dl)
https://raw.githubusercontent.com/espressif/arduino-esp32/gh-pages/package_esp32_index.json
Tools > Board > Boards MAnager > Search "esp32" by Espressif Systems
*/

//DHT TEMP AND HUMIDITY
#include <DHT.h>
#define DHTPIN 9
#define DHTTYPE DHT11

#include "IR_remote.h"
#include "keymap.h"

DHT dht(DHTPIN, DHTTYPE); 

// IR REMOTE OBJECT
IRremote ir(3);

//ULTRASONIC
const int trigPin = 12;
const int echoPin = 13; 
long duration;
int distanceCm;

//IR MODULE (OBSTACLE)
int rightIR = A1;
int leftIR = A2;

// MOTOR STOP FUNCTION
void stopMotors() {
  digitalWrite(2, LOW);
  analogWrite(5, 0);
  digitalWrite(4, LOW);
  analogWrite(6, 0);
}

// LAST COMMAND MEMORY (ANTI-GLITCH)
unsigned long lastSignalTime = 0;
const unsigned long stopTimeout = 250;  // ms before auto-stop

void setup() {

  Serial.begin(9600);

  pinMode(rightIR, INPUT_PULLUP); 
  pinMode(leftIR, INPUT_PULLUP);

  pinMode(trigPin, OUTPUT);
  pinMode(echoPin, INPUT);

  Serial.println("DHT11 Test!");
  dht.begin();

  ir.begin();

  pinMode(2, OUTPUT);
  pinMode(5, OUTPUT);
  pinMode(4, OUTPUT);
  pinMode(6, OUTPUT); 
}

void loop() {

  // OBSTACLE IR
  if (digitalRead(rightIR) == LOW) {
    Serial.println("Object detected on the right!");
  }

  if (digitalRead(leftIR) == LOW) {
    Serial.println("Obstacle detected on the left!");
  }

  // ULTRASONIC
  digitalWrite(trigPin, LOW);
  delayMicroseconds(2);
  digitalWrite(trigPin, HIGH);
  delayMicroseconds(10);
  digitalWrite(trigPin, LOW);

  duration = pulseIn(echoPin, HIGH, 25000);
  distanceCm = duration * 0.034 / 2;

  Serial.print("Distance: ");
  Serial.print(distanceCm);
  Serial.println(" cm");

  // DHT (NON-BLOCKING)
  static unsigned long lastDHT = 0;
  if (millis() - lastDHT >= 500) {
    lastDHT = millis();

    float h = dht.readHumidity();
    float t = dht.readTemperature();

    if (!isnan(h) && !isnan(t)) {
      Serial.print("Humidity: ");
      Serial.print(h);
      Serial.print("%  Temp: ");
      Serial.print(t);
      Serial.println("°C");
    }
  }

  //IR REMOTE WITH SIGNAL HOLDING
  int key = ir.getIrKey(ir.getCode(), 1);

  if (key != 0) {                // only when a REAL signal is received
    lastSignalTime = millis();   // refresh timer

    if (key == IR_KEYCODE_UP) {
      digitalWrite(2,HIGH);
      analogWrite(5,150);
      digitalWrite(4,LOW);
      analogWrite(6,150);

    } 
    else if (key == IR_KEYCODE_DOWN) {
      digitalWrite(2,LOW);
      analogWrite(5,150);
      digitalWrite(4,HIGH);
      analogWrite(6,150);

    } 
    else if (key == IR_KEYCODE_LEFT) {
      digitalWrite(2,LOW);
      analogWrite(5,50);
      digitalWrite(4,LOW);
      analogWrite(6,50);

    } 
    else if (key == IR_KEYCODE_RIGHT) {
      digitalWrite(2,HIGH);
      analogWrite(5,50);
      digitalWrite(4,HIGH);
      analogWrite(6,50);

    } 
    else if (key == IR_KEYCODE_OK) {
      stopMotors();
    }
    else if (key == IR_KEYCODE_1) {
      digitalWrite(2,LOW);
      analogWrite(5,130);
      digitalWrite(4,LOW);
      analogWrite(6,130);
    }
    else if (key == IR_KEYCODE_3) {
      digitalWrite(2,HIGH);
      analogWrite(5,130);
      digitalWrite(4,HIGH);
      analogWrite(6,130);
    }
  }

  // AUTO STOP ONLY IF SIGNAL IS REALLY GONE
  if (millis() - lastSignalTime > stopTimeout) {
    stopMotors();
  }

  delay(15);
}
