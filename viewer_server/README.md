# EPS Viewer Backend

Run locally with:

```bash
./venv/bin/python -m uvicorn viewer_server.main:app --reload --port 8000
```

Available endpoints:

- `/api/runs`
- `/api/runs/{run_id}`
- `/api/runs/{run_id}/history`
- `/api/runs/{run_id}/predictions`
- `/api/compare?run_ids=...`
