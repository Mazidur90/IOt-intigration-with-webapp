import csv
import time
from datetime import datetime
import os

INPUT_CSV = "leaf.csv"  
OUTPUT_CSV = "data/full_sensor_data.csv"
DEBUG_LOG = "data/csv_builder_debug.log"

os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)

last_air = None
last_leaf = None
last_combined_timestamp = None

seen_air_timestamps = set()
seen_leaf_timestamps = set()

def parse_row(row):
    return {
        "timestamp": row["timestamp"],
        "device_id": row["device_id"],
        "sensor_type": row["type"],
        "Air_Temperature": row.get("Air_Temperature", "").strip(),
        "Air_Humidity": row.get("Air_Humidity", "").strip(),
        "Leaf_Temperature": row.get("Leaf_Temperature", "").strip(),
        "Leaf_Moisture": row.get("Leaf_Moisture", "").strip(),
    }

def is_air(row): return row["sensor_type"] == "temp-hum"
def is_leaf(row): return row["sensor_type"] == "leaf"

def build_combined_row():
    global last_air, last_leaf, last_combined_timestamp
    if not last_air or not last_leaf:
        return None

    ts1 = datetime.fromisoformat(last_air["timestamp"].replace("Z", ""))
    ts2 = datetime.fromisoformat(last_leaf["timestamp"].replace("Z", ""))
    delta_range = f"{ts1.strftime('%H:%M:%S')}–{ts2.strftime('%H:%M:%S')}"
    delta_sec = abs((ts2 - ts1).total_seconds())

    # avoid duplicates
    if delta_range == last_combined_timestamp:
        return None

    last_combined_timestamp = delta_range

    return {
        "delta_range": delta_range,
        "delta_seconds": delta_sec,
        "Air_Temperature": last_air["Air_Temperature"],
        "Air_Humidity": last_air["Air_Humidity"],
        "Leaf_Temperature": last_leaf["Leaf_Temperature"],
        "Leaf_Moisture": last_leaf["Leaf_Moisture"],
    }

def main_loop():
    print(" Watching for new sensor data...")
    last_line_count = 0

    while True:
        with open(INPUT_CSV, newline='') as infile:
            reader = list(csv.DictReader(infile))
            new_rows = reader[last_line_count:]
            last_line_count = len(reader)

        for row in new_rows:
            data = parse_row(row)
            timestamp = data["timestamp"]

            if is_air(data):
                if timestamp in seen_air_timestamps:
                    continue
                seen_air_timestamps.add(timestamp)
                if last_air:
                    log_skip("air", last_air, data)
                last_air = data

            elif is_leaf(data):
                if timestamp in seen_leaf_timestamps:
                    continue
                seen_leaf_timestamps.add(timestamp)
                if last_leaf:
                    log_skip("leaf", last_leaf, data)
                last_leaf = data

            combined = build_combined_row()
            if combined:
                write_combined_row(combined)

        time.sleep(2)

def write_combined_row(row):
    write_header = not os.path.exists(OUTPUT_CSV)
    with open(OUTPUT_CSV, "a", newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row.keys())
        if write_header:
            writer.writeheader()
        writer.writerow(row)
    print(f" Row added: {row['delta_range']}")

def log_skip(sensor_type, old, new):
    with open(DEBUG_LOG, "a") as log:
        log.write(f" Skipped old {sensor_type} @ {old['timestamp']} → took newer @ {new['timestamp']}\n")

if __name__ == "__main__":
    main_loop()
