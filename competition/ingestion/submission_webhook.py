import os
import uuid
from datetime import datetime, timedelta, timezone
from typing import Optional, Tuple

from competition.config import VIETNAM_TZ
from competition.ingestion.collector import process_submission_item
from competition.integrations.notifications import update_google_sheets
from competition.storage import SubmissionStore


# Submission constants
MAX_SUBMISSIONS_PER_DAY = 3


def _env_flag(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def get_vietnam_day_identifier(now: Optional[datetime] = None) -> str:
    """
    Get the day identifier for Vietnam timezone (UTC+7).
    
    Resets at 7 AM Vietnam time each day.
    Returns format: "YYYY-MM-DD" where the day is defined as:
    - 2026-04-25 00:00 UTC+7 to 2026-04-25 23:59:59 UTC+7
    - Which is 2026-04-24 17:00 UTC to 2026-04-25 16:59:59 UTC
    
    Args:
        now: datetime.datetime in UTC. If None, uses datetime.now(timezone.utc)
    
    Returns:
        str: "YYYY-MM-DD" identifier for quota purposes
    """
    if now is None:
        now = datetime.now(timezone.utc)
    
    # Convert UTC to Vietnam time
    vietnam_time = now.astimezone(VIETNAM_TZ)
    
    # Apply 7 AM reset: times from 7 AM to 11:59 PM belong to this day,
    # times from 00:00 AM to 6:59 AM belong to the PREVIOUS day
    if vietnam_time.hour < 7:
        # Early morning (before 7 AM) = belongs to previous day
        vietnam_time = vietnam_time - timedelta(days=1)
    
    return vietnam_time.strftime("%Y-%m-%d")


def process_submission_webhook(
    request_json: dict,
    store: SubmissionStore,
    service,
    storage_dir: str = "submissions",
) -> Tuple[bool, dict]:
    # ... (existing docstring omitted for brevity in thought, but I'll keep it in implementation)
    
    # Extract required fields
    canonical_team_id = request_json.get("canonical_team_id", "").strip()
    submission_token = request_json.get("submission_token", "").strip()
    drive_file_id = request_json.get("drive_file_id", "").strip()
    changelog = request_json.get("changelog", "").strip()
    original_filename = request_json.get("original_filename", "")
    
    def _record_and_fail(error_code: str, reason: str):
        # Determine which team ID to use for logging (fallback to unknown_team)
        target_team_id = canonical_team_id
        try:
            # Check if team exists to avoid FK violation, fallback to system unknown_team
            if not target_team_id or store.get_team(target_team_id) is None:
                target_team_id = "unknown_team"
            
            # Use drive_file_id as response_id if available, otherwise use a new UUID
            resp_id = drive_file_id if drive_file_id else f"error-{uuid.uuid4()}"
            sub_id = str(uuid.uuid4())
            
            store.save_submission(
                submission_id=sub_id,
                canonical_team_id=target_team_id,
                response_id=resp_id,
                drive_file_id=drive_file_id or "",
                original_filename=original_filename or "unknown",
                sha256=None,
                uploaded_at=None,
                validation_status="invalid",
                validation_reason=f"{error_code}: {reason} (provided team_id: {canonical_team_id})",
                extracted_path=None,
                extracted_manifest_json=None
            )
            # Push to Google Sheets immediately
            update_google_sheets(
                db_path=store.db_path,
                spreadsheet_id=os.getenv("LEADERBOARD_SPREADSHEET_ID"),
            )
        except Exception as e:
            print(f"Warning: Could not record early failure to DB/Sheet: {e}")
        
        return False, {"error": error_code, "reason": reason}

    # Validate required fields
    if not canonical_team_id:
        return _record_and_fail("missing_field", "canonical_team_id")
    if not submission_token:
        return _record_and_fail("missing_field", "submission_token")
    if not drive_file_id:
        return _record_and_fail("missing_field", "drive_file_id")
    
    # Step 1: Verify token matches team
    team = store.get_team(canonical_team_id)
    if team is None:
        return _record_and_fail("auth_failed", "unknown_team")
    
    if team.status != "active":
        return _record_and_fail("auth_failed", f"team_not_active:{team.status}")
    
    if not store.verify_token(canonical_team_id, submission_token):
        return _record_and_fail("auth_failed", "token_mismatch")
    
    # Step 2: Check daily quota (Vietnam time, 7 AM reset)
    day_id = get_vietnam_day_identifier()
    quota_count = store.get_daily_quota_count(canonical_team_id, day_id)
    
    if quota_count >= MAX_SUBMISSIONS_PER_DAY:
        return _record_and_fail(
            "quota_exceeded",
            f"max {MAX_SUBMISSIONS_PER_DAY} submissions per day (Vietnam time, resets 7 AM)"
        )
    
    # Step 3: Invoke collector to download, validate, extract
    submission_metadata = {
        "drive_file_id": drive_file_id,
        "canonical_team_id": canonical_team_id,
        "submission_token": submission_token,
        "original_filename": original_filename,
    }
    
    ok, note = process_submission_item(
        service=service,
        store=store,
        storage_dir=storage_dir,
        item=submission_metadata,
    )
    
    if not ok:
        # Collector will have already saved the invalid submission record to DB
        try:
            update_google_sheets(
                db_path=store.db_path,
                spreadsheet_id=os.getenv("LEADERBOARD_SPREADSHEET_ID"),
            )
        except Exception as e:
            print(f"Warning: Could not record validation failure to Sheet: {e}")
        return False, {"error": "validation_failed", "reason": note}
    
    # Step 4: Increment daily quota counter
    store.increment_daily_quota(canonical_team_id, day_id)

    # Resolve actual stored submission id from source response identity.
    stored_record = store.get_submission_by_response_id(drive_file_id)
    submission_id = stored_record.submission_id if stored_record else ""

    # Step 5: Trigger immediate submission evaluation batch (auto-update leaderboard).
    evaluation_result = None
    if submission_id:
        try:
            from scripts.organizer.run_evaluation import run_submission_batch

            # Allow environment overrides; default to project leaderboard ID and credentials.
            sheet_spreadsheet_id = os.getenv(
                "LEADERBOARD_SPREADSHEET_ID",
                "1caRS0zqKovKqsL5ozzqNAtSWhseTVBT1LNBr0AVDrBE",
            )
            sheet_credentials_file = os.getenv("LEADERBOARD_CREDENTIALS_FILE", "secrets/service_account_credentials.json")
            sheet_range = os.getenv("LEADERBOARD_SHEET_RANGE", "Leaderboard!A1")
            parallel_workers = int(os.getenv("EVALUATION_PARALLEL_WORKERS", "1"))
            enable_gif = _env_flag("EVALUATION_ENABLE_GIF", True)
            enable_timing_logs = _env_flag("EVALUATION_TIMING_LOGS", True)
            startup_timeout_env = os.getenv("EVALUATION_STARTUP_TIMEOUT_S")
            startup_timeout_s = None if not startup_timeout_env else float(startup_timeout_env)

            evaluation_result = run_submission_batch(
                submission_id=submission_id,
                n_matches=12,
                db_path=store.db_path,
                update_sheet=True,
                sheet_credentials_file=sheet_credentials_file,
                sheet_spreadsheet_id=sheet_spreadsheet_id,
                sheet_range=sheet_range,
                parallel_workers=parallel_workers,
                enable_gif=enable_gif,
                enable_timing_logs=enable_timing_logs,
                startup_timeout_s=startup_timeout_s,
            )
        except Exception as e:
            evaluation_result = {
                "status": "error",
                "message": f"evaluation_trigger_failed:{e}",
            }
    
    return True, {
        "status": "success",
        "submission_id": submission_id,
        "reason": note,
        "remaining_today": MAX_SUBMISSIONS_PER_DAY - (quota_count + 1),
        "evaluation": evaluation_result,
    }
