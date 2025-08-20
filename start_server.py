#!/usr/bin/env python3
import uvicorn
from puter_server import app

if __name__ == "__main__":
    print("🚀 Starting Puter Reverse OpenAI API Server...")
    uvicorn.run(app, host="0.0.0.0", port=8781, reload=False, log_level="info")


