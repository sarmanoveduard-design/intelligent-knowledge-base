"""Container preflight: model inventory only, no embeddings or model loading."""
import json
from time import sleep
from urllib.request import ProxyHandler, build_opener

ENDPOINT = "http://ollama:11434/api/tags"


def check_ollama(*, opener=None, attempts=6, pause=sleep):
    opener = opener or build_opener(ProxyHandler({}))
    for attempt in range(attempts):
        try:
            with opener.open(ENDPOINT, timeout=5) as response:
                inventory = json.loads(response.read(2 * 1024 * 1024))
            names = {model["name"] for model in inventory["models"]}
        except Exception:
            if attempt + 1 < attempts:
                pause(1)
            continue
        if names & {"bge-m3", "bge-m3:latest"}:
            return 0
        return 2
    return 3


def main():
    result = check_ollama()
    print({0: "Ollama ready; BGE-M3 present.",
           2: "BGE-M3 missing in project Ollama; no model was downloaded.",
           3: "Project Ollama is unavailable from reranker runtime."}[result])
    return result


if __name__ == "__main__":
    raise SystemExit(main())
