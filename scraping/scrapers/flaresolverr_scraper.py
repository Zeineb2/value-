import requests
import os
import re
import json

def sanitize_filename(url):
    filename = url.strip().rstrip("/").split("/")[-1]
    if not filename or "." in filename:
        filename = re.sub(r"[^\w]", "_", url.strip().split("/")[-2]) if len(url.strip().split("/")) > 1 else "page"
    return filename + ".html"

def fetch_with_flaresolverr(url, save_dir="data/html"):
    payload = {
        "cmd": "request.get",
        "url": url,
        "maxTimeout": 60000
    }

    try:
        resp = requests.post("http://localhost:8191/v1", json=payload)
        if resp.status_code != 200:
            print(f"❌ FlareSolverr HTTP Error {resp.status_code} for {url}")
            _log_blocked_url(url)
            return

        data = resp.json()

        # Validate solution
        if "solution" not in data or "response" not in data["solution"]:
            print(f"❌ FlareSolverr: No solution returned for {url}")
            _log_blocked_url(url)
            return

        html = data["solution"]["response"]
        if not html or len(html.strip()) < 100:
            print(f"⚠️ FlareSolverr returned very short content for {url}")
            _log_blocked_url(url)
            return

        # Save result
        os.makedirs(save_dir, exist_ok=True)
        filename = sanitize_filename(url)
        path = os.path.join(save_dir, filename)
        with open(path, "w", encoding="utf-8") as f:
            f.write(html)

        print(f"✅ FlareSolverr saved {url} → {path}")

    except Exception as e:
        print(f"❌ FlareSolverr failed for {url}: {e}")
        _log_blocked_url(url)

def _log_blocked_url(url, log_path="output/blocked_urls.json"):
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"url": url}) + "\n")
