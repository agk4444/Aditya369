# EQM Data Collection Pipeline Plan

## Overview
This document outlines the comprehensive data collection strategy for the EQM (Aditya369) emotional intelligence system, focusing on real-time data ingestion from smart watches and rings.

## Supported Devices

### Primary Devices
1. **Apple Watch Series 6+**
   - Heart Rate, HRV, Blood Oxygen
   - Accelerometer, Gyroscope
   - HealthKit API integration

2. **Oura Ring**
   - Heart Rate, HRV, Temperature
   - Sleep patterns, Activity levels
   - REST API with webhooks

3. **Samsung Galaxy Watch 4+**
   - Heart Rate, HRV, ECG, SpO2
   - Samsung Health API integration

4. **Fitbit Versa/Inspire Series**
   - Heart Rate, Sleep tracking
   - Fitbit Web API

5. **Garmin Wearables**
   - Heart Rate, Stress levels
   - Garmin Connect API

6. **Smart Glasses**
   - Heart Rate (PPG), Eye gaze tracking, Pupil dilation, Blink rate, Head movement, Temperature
   - Device-specific SDK or cloud API
   - Real-time cognitive load assessment and attention monitoring

## Data Collection Architecture

### Real-time Streaming Pipeline
```
Device APIs → Authentication → Data Ingestion → Validation → Storage → Processing
```

#### 1. Authentication Layer
- **OAuth 2.0**: Industry-standard authentication for device APIs
- **API Keys**: Secure key management for third-party integrations
- **JWT Tokens**: User session management and data isolation

#### 2. Data Ingestion Methods

##### REST API Polling
```python
# Polling strategy for devices without webhooks
def poll_device_data(device_id, user_id):
    while active_session:
        data = fetch_latest_readings(device_id)
        if data:
            process_and_store(data, user_id)
        time.sleep(polling_interval)
```

##### WebSocket Streaming
```python
# Real-time data streaming
async def websocket_handler(websocket, path):
    async for message in websocket:
        data = json.loads(message)
        await process_streaming_data(data)
```

##### Webhook Integration
```python
# Handle incoming webhooks from devices
@app.post("/webhook/oura")
async def oura_webhook(request: Request):
    payload = await request.json()
    await validate_and_store(payload)
```

### Data Specifications

#### Sensor Data Types

| Sensor | Frequency | Units | Range | Precision |
|--------|-----------|-------|-------|-----------|
| Heart Rate | 1-10 Hz | BPM | 30-220 | ±1 BPM |
| HRV (RMSSD) | 1-5 Hz | ms | 1-200 | ±1 ms |
| Skin Temperature | 0.1-1 Hz | °C | 20-45 | ±0.1°C |
| GSR | 1-10 Hz | μS | 0.5-20 | ±0.1 μS |
| SpO2 | 0.1-1 Hz | % | 70-100 | ±1% |
| Accelerometer | 10-100 Hz | m/s² | -78 to 78 | ±0.1 m/s² |
| Eye Gaze (X,Y) | 30 Hz | pixels | 0-1920 | ±1 pixel |
| Pupil Dilation | 30 Hz | mm | 2-8 | ±0.1 mm |
| Blink Rate | 30 Hz | BPM | 10-30 | ±1 BPM |
| Head Movement | 30 Hz | degrees | -90 to 90 | ±1 degree |

#### Data Format Standards

##### JSON Schema for Sensor Data
```json
{
  "type": "object",
  "properties": {
    "device_id": {"type": "string"},
    "user_id": {"type": "string"},
    "timestamp": {"type": "string", "format": "date-time"},
    "session_id": {"type": "string"},
    "sensor_readings": {
      "type": "object",
      "properties": {
        "heart_rate": {"type": "number", "minimum": 30, "maximum": 220},
        "hrv": {"type": "number", "minimum": 0},
        "temperature": {"type": "number", "minimum": 20, "maximum": 45},
        "gsr": {"type": "number", "minimum": 0},
        "spo2": {"type": "number", "minimum": 70, "maximum": 100},
        "accelerometer": {
          "type": "object",
          "properties": {
            "x": {"type": "number"},
            "y": {"type": "number"},
            "z": {"type": "number"}
          }
        }
      }
    },
    "metadata": {
      "type": "object",
      "properties": {
        "device_model": {"type": "string"},
        "firmware_version": {"type": "string"},
        "battery_level": {"type": "number", "minimum": 0, "maximum": 100}
      }
    }
  },
  "required": ["device_id", "user_id", "timestamp", "sensor_readings"]
}
```

## Device Integration Details

### Apple Watch Integration

#### HealthKit API Setup
```python
# Using Apple's HealthKit framework via native iOS app
HKHealthStore.shared.requestAuthorization(toShare: nil, read: healthTypes) { (success, error) in
    if success {
        startHeartRateQuery()
    }
}
```

#### Data Collection Flow
1. Request HealthKit permissions
2. Set up background delivery for health data
3. Stream data via WebSocket to EQM backend
4. Handle iOS background app refresh limitations

### Oura Ring Integration

#### REST API Implementation
```python
class OuraClient:
    def __init__(self, access_token):
        self.base_url = "https://api.ouraring.com/v2"
        self.headers = {"Authorization": f"Bearer {access_token}"}

    async def get_heart_rate_data(self, start_date, end_date):
        endpoint = f"/usercollection/heartrate"
        params = {
            "start_date": start_date,
            "end_date": end_date
        }
        response = await self._make_request(endpoint, params)
        return response
```

#### Webhook Configuration
- Register webhook endpoints for real-time data
- Handle webhook signature verification
- Process incoming data batches

### Data Quality Management

#### Validation Rules
```python
def validate_sensor_data(data):
    validations = {
        'heart_rate': lambda x: 30 <= x <= 220,
        'hrv': lambda x: x >= 0,
        'temperature': lambda x: 20 <= x <= 45,
        'gsr': lambda x: x >= 0,
        'spo2': lambda x: 70 <= x <= 100
    }

    for sensor, validator in validations.items():
        if sensor in data and not validator(data[sensor]):
            raise ValidationError(f"Invalid {sensor} value: {data[sensor]}")
```

#### Noise Filtering
- **Outlier Detection**: Statistical methods (IQR, Z-score)
- **Signal Smoothing**: Moving averages, Kalman filtering
- **Gap Handling**: Interpolation for missing data points

### Privacy and Security

#### Data Anonymization
- User ID hashing for data storage
- Device ID encryption
- Location data exclusion (if accidentally collected)

#### Compliance Measures
- **GDPR Compliance**: Data subject rights implementation
- **HIPAA Compliance**: PHI data handling procedures
- **Data Retention**: Configurable data lifecycle policies

### Scalability Considerations

#### Rate Limiting
- Device API rate limits respect
- User-based throttling
- Backoff strategies for failed requests

#### Data Volume Management
- Expected data volume: ~1-10 KB per user per minute
- Compression strategies for storage
- Data partitioning by user and time

### Monitoring and Alerting

#### Key Metrics
- **Data Ingestion Rate**: Readings per second per device
- **Data Quality Score**: Percentage of valid readings
- **API Success Rate**: Successful API calls percentage
- **Latency**: End-to-end data pipeline latency

#### Alert Conditions
- Device disconnection alerts
- Data quality degradation
- API rate limit approaching
- Storage capacity warnings

## Implementation Roadmap

### Phase 1: Core Device Support (Month 1-2)
- Apple Watch integration
- Oura Ring integration
- Basic data validation and storage

### Phase 2: Extended Device Support (Month 3-4)
- Samsung Galaxy Watch integration
- Fitbit integration
- Garmin integration
- Smart Glasses integration

### Phase 3: Advanced Features (Month 5-6)
- Real-time streaming optimization
- Advanced data quality management
- Predictive data gap handling

### Phase 4: Enterprise Features (Month 7-8)
- Multi-tenant architecture
- Advanced monitoring and alerting
- Regulatory compliance automation

This data collection plan provides a solid foundation for building a robust, scalable emotional intelligence platform.