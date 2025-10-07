# OPENAI_CODEX

Dieses Repository enthält ein kleines CLI-Tool, um Fragen gegen eine bestehende Qdrant-Collection zu stellen.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Nutzung

Setzen Sie die benötigten Verbindungsdaten via Umgebungsvariablen oder Kommandozeilenargumente und starten Sie anschließend das Tool:

```bash
export QDRANT_URL="https://<ihr-cluster>.qdrant.tech"
export QDRANT_API_KEY="<Ihr_API_Key>"
export QDRANT_COLLECTION_NAME="meine_collection"
export QDRANT_TEXT_KEY="text"
python qdrant_qa.py
```

Alternativ können Sie alle Werte direkt angeben:

```bash
python qdrant_qa.py \
    --url https://<ihr-cluster>.qdrant.tech \
    --api-key <Ihr_API_Key> \
    --collection meine_collection \
    --text-key text
```

Nach dem Start geben Sie Fragen über die Eingabeaufforderung ein. Das Programm berechnet Embeddings mit einem konfigurierbaren Sentence-Transformer-Modell und ruft die ähnlichsten Einträge aus Qdrant ab.
