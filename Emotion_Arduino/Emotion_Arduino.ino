// Arduino Emotion Controller with Serial Feedback and Buzzer
const int statusLed = 13;  // Built-in LED for status
const int buzzerPin = 2;   // Buzzer connected to pin 9

// Define output pins for each emotion
const int happyPin = 3;
const int sadPin = A0;
const int neutralPin = A1;
const int angryPin = A2;
const int surprisedPin = A3;
const int fearPin = A4;
const int disgustPin = A5;

void setup() {
  // Initialize serial communication
  Serial.begin(9600);

  // Initialize pins
  pinMode(statusLed, OUTPUT);
  pinMode(buzzerPin, OUTPUT);
  digitalWrite(buzzerPin,HIGH);

  pinMode(happyPin, OUTPUT);
  pinMode(sadPin, OUTPUT);
  pinMode(neutralPin, OUTPUT);
  pinMode(angryPin, OUTPUT);
  pinMode(surprisedPin, OUTPUT);
  pinMode(fearPin, OUTPUT);
  pinMode(disgustPin, OUTPUT);

  // Turn all off initially
  allEmotionsOff();

  // Show ready status
  blinkLed(statusLed, 3, 100);
  Serial.println("Emotion Controller Ready");
}

void loop() 
{
  if (Serial.available() > 0) 
  {
    char receivedChar = Serial.read();

    if (isDigit(receivedChar)) 
    {
      int emotionCode = receivedChar - '0';

      // Acknowledge receipt
      blinkLed(statusLed, 1, 100);
      Serial.print("Received code: ");
      Serial.println(emotionCode);

      // Process emotion
      handleEmotion(emotionCode);

      // Send confirmation
      Serial.print("Emotion ");
      Serial.print(emotionCode);
      Serial.println(" activated");
    }
  }
}

void handleEmotion(int code) {
  allEmotionsOff();  // Turn off all emotion pins
  int blinkBeepCount = 0;

  switch (code) {
    case 1:  // HAPPY
      digitalWrite(happyPin, LOW);
      Serial.println("Activated: HAPPY");
      blinkBeepCount = 1;
      break;
    case 2:  // SAD
      digitalWrite(sadPin, LOW);
      Serial.println("Activated: SAD");
      blinkBeepCount = 2;
      break;
    case 3:  // NEUTRAL
      digitalWrite(neutralPin, LOW);
      Serial.println("Activated: NEUTRAL");
      blinkBeepCount = 3;
      break;
    case 4:  // ANGRY
      digitalWrite(angryPin, LOW);
      Serial.println("Activated: ANGRY");
      blinkBeepCount = 4;
      break;
    case 5:  // SURPRISED
      digitalWrite(surprisedPin, LOW);
      Serial.println("Activated: SURPRISED");
      blinkBeepCount = 5;
      break;
    case 6:  // FEAR
      digitalWrite(fearPin, LOW);
      Serial.println("Activated: FEAR");
      blinkBeepCount = 6;
      break;
    case 7:  // DISGUST
      digitalWrite(disgustPin, LOW);
      Serial.println("Activated: DISGUST");
      blinkBeepCount = 7;
      break;
    default:
      Serial.println("Unknown emotion code");
      return;
  }

  // Blink LED and beep buzzer based on emotion
  blinkLed(statusLed, blinkBeepCount, 200);
  beepBuzzer(blinkBeepCount, 200);
}

void allEmotionsOff() {
  digitalWrite(happyPin, HIGH);
  digitalWrite(sadPin, HIGH);
  digitalWrite(neutralPin, HIGH);
  digitalWrite(angryPin, HIGH);
  digitalWrite(surprisedPin, HIGH);
  digitalWrite(fearPin, HIGH);
  digitalWrite(disgustPin, HIGH);
}

void blinkLed(int pin, int times, int delayMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(pin, LOW);
    delay(delayMs);
    digitalWrite(pin, HIGH);
    if (i < times - 1) delay(delayMs);
  }
}

void beepBuzzer(int times, int durationMs) {
  for (int i = 0; i < times; i++) {
    digitalWrite(buzzerPin, LOW);
    delay(durationMs);
    digitalWrite(buzzerPin, HIGH);
    if (i < times - 1) delay(durationMs);
  }
}
