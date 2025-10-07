"""Command-line Werkzeug zur Fragebeantwortung mit Qdrant.

Dieses Skript stellt eine Verbindung zu einer Qdrant-Instanz her, lädt ein
Sentence-Transformer-Modell, um neue Fragen zu vektorisieren, und sucht nach
ähnlichen Einträgen in einer vorhandenen Collection. Ergebnisse werden mit ihren
Payloads ausgegeben, sodass der gespeicherte Kontext eingesehen werden kann.

Voraussetzungen:
    pip install -r requirements.txt

Beispielaufruf:
    python qdrant_qa.py --url http://localhost:6333 --collection meine_collection \
        --api-key <QDRANT_API_KEY> --text-key text

Sie können Konfigurationswerte auch über Umgebungsvariablen setzen:
    QDRANT_URL, QDRANT_HOST, QDRANT_PORT, QDRANT_API_KEY,
    QDRANT_COLLECTION_NAME, QDRANT_TEXT_KEY, QDRANT_TOP_K,
    QDRANT_EMBEDDING_MODEL.
"""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass
from typing import Iterable, List, NoReturn, Optional, Tuple

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse
from qdrant_client.models import PointStruct
from sentence_transformers import SentenceTransformer


@dataclass
class QdrantQAConfig:
    """Konfigurationswerte für das Qdrant-Fragebeantwortungswerkzeug."""

    collection_name: str
    url: Optional[str] = None
    host: Optional[str] = None
    port: int = 6333
    api_key: Optional[str] = None
    top_k: int = 5
    text_key: Optional[str] = None
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"

    @classmethod
    def from_env_and_args(cls, args: argparse.Namespace) -> "QdrantQAConfig":
        return cls(
            collection_name=args.collection
            or os.getenv("QDRANT_COLLECTION_NAME")
            or _die("Eine Collection muss angegeben werden (Argument --collection oder Umgebungsvariable QDRANT_COLLECTION_NAME)."),
            url=args.url or os.getenv("QDRANT_URL"),
            host=args.host or os.getenv("QDRANT_HOST"),
            port=int(args.port or os.getenv("QDRANT_PORT", 6333)),
            api_key=args.api_key or os.getenv("QDRANT_API_KEY"),
            top_k=int(args.top_k or os.getenv("QDRANT_TOP_K", 5)),
            text_key=args.text_key or os.getenv("QDRANT_TEXT_KEY"),
            embedding_model=args.embedding_model
            or os.getenv("QDRANT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2"),
        )


def _die(message: str) -> "NoReturn":
    """Beendet das Programm mit einer Fehlermeldung."""

    print(message, file=sys.stderr)
    sys.exit(1)


def build_client(config: QdrantQAConfig) -> QdrantClient:
    """Erzeugt einen QdrantClient anhand der Konfiguration."""

    if config.url:
        return QdrantClient(url=config.url, api_key=config.api_key)
    return QdrantClient(host=config.host or "localhost", port=config.port, api_key=config.api_key)


def load_embedder(model_name: str) -> SentenceTransformer:
    """Lädt das gewünschte Sentence-Transformer-Modell."""

    print(f"Lade Embedding-Modell '{model_name}'...")
    return SentenceTransformer(model_name)


def fetch_example_payload(client: QdrantClient, collection_name: str) -> Optional[PointStruct]:
    """Holt einen Beispielpunkt aus der Collection, um Payload-Keys zu zeigen."""

    scroll_result = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    # qdrant-client <1.9.0 gibt ein Tupel (Punkte, Offset) zurück,
    # neuere Versionen einen ScrollResponse. Beide Varianten werden hier unterstützt.
    if isinstance(scroll_result, tuple):
        points, _ = scroll_result
    else:  # ScrollResponse
        points = scroll_result.points

    if points:
        return points[0]
    return None


def search_collection(
    client: QdrantClient,
    config: QdrantQAConfig,
    embedder: SentenceTransformer,
    question: str,
) -> List[Tuple[str, float, dict]]:
    """Führt eine Ähnlichkeitssuche in Qdrant durch und liefert Ergebnisse."""

    query_vector = embedder.encode(question).astype(np.float32)

    search_result = client.search(
        collection_name=config.collection_name,
        query_vector=query_vector.tolist(),
        limit=config.top_k,
        with_payload=True,
        with_vectors=False,
    )

    formatted: List[Tuple[str, float, dict]] = []
    for point in search_result:
        payload = point.payload or {}
        formatted.append((str(point.id), point.score, payload))
    return formatted


def interactive_loop(client: QdrantClient, config: QdrantQAConfig, embedder: SentenceTransformer) -> None:
    """Startet eine einfache REPL zum Stellen von Fragen."""

    print("\nGeben Sie Ihre Frage ein. Mit 'exit' oder 'quit' beenden Sie das Programm.\n")
    if config.text_key:
        print(f"Es wird das Payload-Feld '{config.text_key}' zur Anzeige genutzt, falls vorhanden.\n")
    else:
        print("Es wird die gesamte Payload ausgegeben. Nutzen Sie --text-key, um ein spezifisches Feld zu wählen.\n")

    while True:
        try:
            question = input("Frage> ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nAuf Wiedersehen!")
            break

        if not question:
            continue
        if question.lower() in {"exit", "quit"}:
            print("Auf Wiedersehen!")
            break

        results = search_collection(client, config, embedder, question)
        if not results:
            print("Keine Ergebnisse gefunden. Prüfen Sie die Collection oder das Embedding-Modell.\n")
            continue

        for index, (point_id, score, payload) in enumerate(results, start=1):
            display_value = payload
            if config.text_key:
                display_value = payload.get(config.text_key, "<Payload enthält den angegebenen Key nicht>")
            print(f"#{index}: Punkt-ID={point_id}, Score={score:.4f}")
            print(f"    Kontext: {display_value}\n")


def parse_args(argv: Optional[Iterable[str]] = None) -> argparse.Namespace:
    """Parst Kommandozeilenargumente."""

    parser = argparse.ArgumentParser(description="Interaktive Fragebeantwortung mit Qdrant.")
    parser.add_argument("--collection", help="Name der Qdrant-Collection.")
    parser.add_argument("--url", help="Vollständige URL zur Qdrant-Instanz (z. B. https://xyz.qdrant.tech).")
    parser.add_argument("--host", help="Hostname der Qdrant-Instanz, wenn keine URL genutzt wird.")
    parser.add_argument("--port", type=int, help="Port der Qdrant-Instanz (Standard: 6333).")
    parser.add_argument("--api-key", help="API-Schlüssel für Qdrant, falls erforderlich.")
    parser.add_argument("--top-k", type=int, help="Anzahl der Ergebnisse pro Frage (Standard: 5).")
    parser.add_argument("--text-key", help="Payload-Feld, das als Antworttext angezeigt wird.")
    parser.add_argument(
        "--embedding-model",
        help="Name eines Sentence-Transformer-Modells für die Frage-Embeddings (Standard: sentence-transformers/all-MiniLM-L6-v2).",
    )
    return parser.parse_args(argv)


def main(argv: Optional[Iterable[str]] = None) -> None:
    args = parse_args(argv)
    config = QdrantQAConfig.from_env_and_args(args)

    try:
        client = build_client(config)
    except UnexpectedResponse as error:
        _die(f"Verbindung zu Qdrant fehlgeschlagen: {error}")

    try:
        info = client.get_collection(config.collection_name)
    except UnexpectedResponse as error:
        _die(f"Collection '{config.collection_name}' konnte nicht gelesen werden: {error}")

    print(
        f"Verbunden mit Collection '{config.collection_name}'. Anzahl gespeicherter Punkte: {info.points_count}."
    )

    example_point = fetch_example_payload(client, config.collection_name)
    if example_point is not None:
        print("Beispiel-Payload der Collection:")
        print(example_point.payload)
    else:
        print("Die Collection enthält aktuell keine Punkte.")

    embedder = load_embedder(config.embedding_model)

    interactive_loop(client, config, embedder)


if __name__ == "__main__":
    main()
