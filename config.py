# Configuration for Puter Reverse API

PUTER_HEADERS = {
    "accept": "*/*",
    "accept-language": "en-US,en;q=0.8",
    "content-type": "application/json;charset=UTF-8",
    "origin": "https://docs.puter.com",
    "priority": "u=1, i",
    "referer": "https://docs.puter.com/",
    "sec-ch-ua": '"Not;A=Brand";v="99", "Brave";v="139", "Chromium";v="139"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "sec-fetch-dest": "empty",
    "sec-fetch-mode": "cors",
    "sec-fetch-site": "same-site",
    "sec-gpc": "1",
    "user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/139.0.0.0 Safari/537.36",
}

# Authorization bearer token from your curl (consider moving to env var for security)
PUTER_AUTH_BEARER = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ0IjoiYXUiLCJ2IjoiMC4wLjAiLCJ1dSI6IndNSVlyVkRwUUhTbHFmYzVDYjhwS1E9PSIsImF1IjoiaWRnL2ZEMDdVTkdhSk5sNXpXUGZhUT09IiwicyI6IkpmcE9lNzZ4dlorMWsrTWlvYWo5TGc9PSIsImlhdCI6MTc0ODA3NjU1NX0.sdF9zD4JlVQK71490InsnKxtjttD4lbPvNjEjXKjOHk"

# Map OpenAI model -> Puter driver and model
MODEL_MAPPING = {
    "claude-sonnet-4-20250514": {"driver": "claude", "puter_model": "claude-sonnet-4-20250514"},
    # Fallback
    "default": {"driver": "claude", "puter_model": "claude-sonnet-4-20250514"},
}

SERVER_CONFIG = {
    "host": "0.0.0.0",
    "port": 8781,
}


