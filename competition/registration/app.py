"""HTTP entrypoint for registration and submission automation."""

import os
import re
from pathlib import Path

from flask import Flask, jsonify, request
from google.oauth2 import service_account
from googleapiclient.discovery import build

from competition.registration.webhook_receiver import process_registration_payload
from competition.ingestion.submission_webhook import process_submission_webhook
from competition.storage import SubmissionStore


_ENV_KEY_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


# The environment is now loaded via competition.config


def create_app() -> Flask:
    app = Flask(__name__)

    from competition.config import load_env
    load_env()
    print("[env] loaded from .env via competition.config")

    db_path = os.getenv("REGISTRATION_DB_PATH", "competition.db")
    expected_token = os.getenv("REGISTRATION_WEBHOOK_AUTH_TOKEN", "")
    credentials_file = os.getenv("LEADERBOARD_CREDENTIALS_FILE", os.getenv("GOOGLE_APPLICATION_CREDENTIALS", "secrets/service_account_credentials.json"))
    storage_dir = os.getenv("SUBMISSION_STORAGE_DIR", "submissions")


    @app.post("/register")
    def register_team():
        if expected_token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {expected_token}":
                return (
                    jsonify(
                        {
                            "status": "error",
                            "error_code": "UNAUTHORIZED",
                            "message": "Invalid or missing bearer token.",
                        }
                    ),
                    401,
                )

        payload = request.get_json(silent=True)
        if payload is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error_code": "INVALID_JSON",
                        "message": "Expected JSON body.",
                    }
                ),
                400,
            )

        store = SubmissionStore(db_path=db_path)
        result = process_registration_payload(payload=payload, store=store)
        if result.get("status") == "success":
            return jsonify(result), 200

        conflict_codes = {"TEAM_NAME_ALREADY_REGISTERED", "REGISTRATION_CONFLICT"}
        error_code = str(result.get("error_code", ""))
        if error_code in conflict_codes:
            return jsonify(result), 409
        return jsonify(result), 400

    @app.post("/submit")
    def submit_solution():
        if expected_token:
            auth_header = request.headers.get("Authorization", "")
            if auth_header != f"Bearer {expected_token}":
                return (
                    jsonify(
                        {
                            "status": "error",
                            "error_code": "UNAUTHORIZED",
                            "message": "Invalid or missing bearer token.",
                        }
                    ),
                    401,
                )

        payload = request.get_json(silent=True)
        if payload is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error_code": "INVALID_JSON",
                        "message": "Expected JSON body.",
                    }
                ),
                400,
            )

        # isolate drive_service
        drive_service = None
        if os.path.exists(credentials_file):
            try:
                creds = service_account.Credentials.from_service_account_file(
                    credentials_file,
                    scopes=["https://www.googleapis.com/auth/drive.readonly"],
                )
                drive_service = build("drive", "v3", credentials=creds)
            except Exception as e:
                print(f"Warning: Could not initialize Google Drive service: {e}")

        if drive_service is None:
            return (
                jsonify(
                    {
                        "status": "error",
                        "error_code": "SERVICE_UNAVAILABLE",
                        "message": "Google Drive service not initialized.",
                    }
                ),
                503,
            )

        store = SubmissionStore(db_path=db_path)
        ok, result = process_submission_webhook(
            request_json=payload,
            store=store,
            service=drive_service,
            storage_dir=storage_dir,
        )

        if ok:
            return jsonify(result), 200

        error_code = result.get("error", "UNKNOWN_ERROR")
        if error_code == "quota_exceeded":
            return jsonify(result), 429
        elif error_code == "auth_failed":
            return jsonify(result), 401
        else:
            return jsonify(result), 400

    return app


app = create_app()


if __name__ == "__main__":
    host = os.getenv("REGISTRATION_HOST", "0.0.0.0")
    port = int(os.getenv("REGISTRATION_PORT", "5000"))
    debug = os.getenv("REGISTRATION_DEBUG", "false").strip().lower() == "true"
    app.run(host=host, port=port, debug=debug)
