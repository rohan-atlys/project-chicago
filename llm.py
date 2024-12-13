import requests
import json

class LLava:
    def __init__(self):
        self.api_endpoint = "http://localhost:11434/api/generate"
        self.model_name = "llava:7b"
    
    def generate_analysis(self, images: list[str], prompt: str, system: str = ""):
        payload = json.dumps({
                "model": self.model_name,
                "images": images,
                "prompt": prompt,
                "system": system,
                "format": "json",
                "stream": False,
            })
        response = requests.post(
            self.api_endpoint,
            data=payload,
            headers={
            'Content-Type': 'application/json'
        }
        )
        
        response.raise_for_status()

        response = json.loads(response.json()["response"])

        return response
