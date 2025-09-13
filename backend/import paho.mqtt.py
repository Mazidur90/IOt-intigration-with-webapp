import paho.mqtt.client as mqtt

# MQTT details
broker_address = "stack.lora.svc"
port = 1883
client_id = "HJTHFW5J5MOTUGL2FTCSMSECOUY6OGHYEPIDVPQ"
username = "mqtt-password-key-1744125230366_aws_jakob"
password = "NNSXS.DHD5O73ZAJWY77SRYQWHVFONTLBUYHGXO2WPFHY.FXDK7QG473UWTTUKUFHUMCF4ZKRX5FRZ2YGW6VSFCAYBHWI6TYBQ"

# Define callback function
def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print("Connected successfully!")
    else:
        print(f"Connection failed with code {rc}.")

# Create MQTT client instance
client = mqtt.Client(client_id=client_id)

# Set username and password
client.username_pw_set(username, password)

# Assign callback
client.on_connect = on_connect

# Connect to broker
try:
    client.connect(broker_address, port, 60)
    client.loop_forever()
except Exception as e:
    print(f"Could not connect: {e}")
