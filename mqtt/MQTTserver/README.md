## In this repo : 

Basic conf for MQTT server.

- clientTester.c -> MQTT client in C for tests
- server.c -> Simple MQTT server who responds 1 or 0.



## Install Mosquitto Broker


```bash
sudo apt install mosquitto mosquitto-clients

```

## Launch MQTT Broker 

```bash
mosquitto -p 8080 -v

```

## Launch

1) run "make"

2) ./server => Server wait requests

3) ./clientTester www.vinted.com => Ask if www.vinted.com is a phishing site
