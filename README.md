# Fire and Smoke Incident Processing

This repository implements a video-based fire/smoke incident service with asynchronous event recording. The main design goal is to separate real-time incident processing from slower media finalization so that saving evidence does not block the detection path.

## Incident Flow

```text
active incident
  -> incident frames buffered in a temporary spool
  -> incident ends
  -> background MP4 finalization
  -> final video saved
  -> temporary frame spool removed
```

The final media directory contains MP4 incident recordings. Alert metadata remains available through the API and can be replaced by a persistent database or message-broker integration in a production deployment.

## API

The service exposes endpoints for:

- fire/smoke runtime status;
- video-storage status;
- recent alert metadata.

## Run

```powershell
python run_server.py
```

## Test

```powershell
python smoke_test.py
```

## Engineering Focus

The repository is mainly concerned with incident lifecycle management, asynchronous media handling, and API integration around a detection pipeline rather than with introducing a new fire/smoke model architecture.


## Goal

The repository isolates incident recording from the real-time detection path so that video encoding and file I/O do not stall alert processing.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run_server.py
```

On Windows, `run.ps1` provides the repository-specific launcher. Confirm the configured input source and writable media directories before starting the service.

## Working with the Repository

`fire_smoke_service.py` coordinates detection, `incident_recorder.py` manages incident state, and `async_video_storage.py` spools and finalizes media. Run `python smoke_test.py` after changes to the incident lifecycle. Generated recordings and temporary spool files should remain outside version control.
