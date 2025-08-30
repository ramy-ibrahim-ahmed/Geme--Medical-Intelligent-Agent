import requests
import base64
import json

from .state import State
from .prompt import ocr_prompt
from ..config import get_settings

settings = get_settings()


def OCR(state: State) -> State:
    try:
        image_path = state["image"]

        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")

        payload = {
            "model": settings.VISION_MODEL,
            "prompt": ocr_prompt,
            "images": [encoded_image],
            "stream": False,
        }

        headers = {"Content-Type": "application/json"}
        response = requests.post(settings.OLLAMA_URL, headers=headers, json=payload)
        response.raise_for_status()
        response_json = response.json()
        return {"transcription": response_json["response"], "image": ""}

    except FileNotFoundError:
        print(f"Error: Image file not found at {image_path}")
        return None
    except requests.exceptions.RequestException as e:
        print(f"Error: Request failed: {e}")
        return None
    except json.JSONDecodeError:
        print("Error: Invalid JSON response")
        return None
    except KeyError:
        print("Error: unexpected response format from ollama")
        return None
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return None
