# Kaelum API Backend

Flask REST API for the Kaelum AI reasoning system.

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
python app.py
```

Server will start on `http://localhost:5000`

## API Endpoints

### Health Check
- `GET /api/health` - Check API status

### Configuration
- `GET /api/config` - Get current configuration
- `POST /api/config` - Update configuration

### Query Processing
- `POST /api/query` - Process reasoning query

### Metrics & Analytics
- `GET /api/metrics` - Get comprehensive metrics
- `GET /api/stats/router` - Neural router statistics
- `GET /api/stats/cache` - Cache and validation stats
- `GET /api/stats/calibration` - Threshold calibration data

### Workers
- `GET /api/workers` - List available expert workers

### Export
- `GET /api/export/training-data` - Export training dataset
