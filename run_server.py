from __future__ import annotations

import logging

import uvicorn


if __name__ == "__main__":
    logging.getLogger(
        "watchfiles"
    ).setLevel(logging.WARNING)
    logging.getLogger(
        "watchfiles.main"
    ).setLevel(logging.WARNING)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info",
    )
