import requests
import json

url = "http://localhost:11434/api/generate"

data = {
    "model": "qwen2.5:7b",
    "prompt": "Say hello",
    "stream": False
}

response = requests.post(url, json=data)

print("Status:", response.status_code)
print("Raw response:")
print(json.dumps(response.json(), indent=2))