import argparse
import concurrent.futures
import logging
import os
import random
import time
import fcntl
from pathlib import Path
from typing import Optional

from competition.storage import SubmissionStore
from competition.evaluation.match_runner import MatchRunner
from competition.integrations.notifications import update_google_sheets
from competition.integrations.drive_upload import upload_file_to_drive
from competition.evaluation.pool_manager import PoolManager
from competition.evaluation.ranking import RankingSystem
from competition.evaluation.runtime_guard import runtime_precheck


ROOT_DIR = Path(__file__).resolve().parent.parent.parent

from competition.config import load_env
load_env()

DEFAULT_DB_PATH = str(ROOT_DIR / "competition.db")
DEFAULT_LOG_DIR = str(ROOT_DIR / "logs")
EVALUATION_LOCK_FILE = ROOT_DIR / ".evaluation.lock"

DEFAULT_N_MATCHES = 25
DEFAULT_MAX_STEPS = 500
DEFAULT_NEAR_DELTA = 3.0
DEFAULT_TOP_K = 10
DEFAULT_OPPONENT_MAX_REPEAT = 5
DEFAULT_INFERENCE_TIMEOUT_S = 0.1

logger = logging.getLogger(__name__)


def _prewarm_drive_folders() -> None:
    """Pre-create the Drive folder structure (json/ and gifs/) in the parent process.

    When parallel workers all finish matches simultaneously and try to upload JSONs
    at the same time, they race to create the same Drive subfolder. Google Drive's
    files.list() is eventually consistent, so two workers can both see 'folder not
    found' and each create a duplicate. GIFs don't suffer because rendering time
    naturally staggers those uploads.

    Calling this once in the parent process before dispatching workers ensures the
    folders exist in Drive. Workers will find them via list() which has had the
    entire match execution time (~20 s) to propagate.
    """
    drive_folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
    if not drive_folder_id:
        return
    try:
        from datetime import datetime, timezone
        from competition.integrations.drive_upload import get_drive_service, ensure_drive_folder
        service = get_drive_service()
        date_str = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for artifact in ("json", "gifs"):
            root_id = ensure_drive_folder(service, drive_folder_id, artifact)
            ensure_drive_folder(service, root_id, date_str)
    except Exception as e:
        logger.warning("Drive folder pre-warm failed (uploads will still work, may create duplicate folders): %s", e)


def _score(mu: float, sigma: float) -> float:
    return float(mu) - 3.0 * float(sigma)


def _fetch_active_candidates(db_path: str):
    import sqlite3

    with sqlite3.connect(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            SELECT
                submission_id,
                canonical_team_id,
                extracted_path,
                mu,
                sigma,
                is_baseline,
                is_active,
                n_games,
                runtime_precheck_passed
            FROM submissions
            WHERE validation_status = 'valid' AND extracted_path IS NOT NULL AND is_active = 1
            """
        )
        rows = cursor.fetchall()

    candidates = []
    for row in rows:
        extracted_path = row[2]
        agent_path = Path(extracted_path) / "agent.py"
        if not agent_path.exists():
            continue
        candidates.append(
            {
                "submission_id": row[0],
                "canonical_team_id": row[1],
                "agent_path": str(agent_path),
                "mu": float(row[3]),
                "sigma": float(row[4]),
                "score": _score(row[3], row[4]),
                "is_baseline": bool(row[5]),
                "is_active": bool(row[6]),
                "n_games": int(row[7]),
                "runtime_precheck_passed": bool(row[8]),
            }
        )
    return candidates


def _runtime_filter_candidates(
    candidates: list[dict],
    store: SubmissionStore,
    inference_timeout_s: float,
    startup_timeout_s: Optional[float] = None,
) -> tuple[list[dict], dict[str, str]]:
    runnable = []
    failed = {}

    for item in candidates:
        if item.get("runtime_precheck_passed"):
            runnable.append(item)
            continue

        ok, note = runtime_precheck(
            agent_path=item["agent_path"],
            timeout_s=inference_timeout_s,
            startup_timeout_s=startup_timeout_s,
        )
        if ok:
            store.mark_submission_runtime_passed(item["submission_id"])
            runnable.append(item)
            continue

        failed[item["submission_id"]] = note
        if not item.get("is_baseline", False):
            store.mark_submission_runtime_invalid(
                submission_id=item["submission_id"],
                reason=f"invalid-runtime:{note}",
            )

    return runnable, failed


def _choose_from_bucket(
    rng: random.Random,
    bucket: list[dict],
    selected_ids: set[str],
    usage: dict[str, int],
    max_repeat: int,
):
    eligible = [
        item
        for item in bucket
        if item["submission_id"] not in selected_ids and usage.get(item["submission_id"], 0) < max_repeat
    ]
    if not eligible:
        return None
    return rng.choice(eligible)


def _sample_opponents(
    target: dict,
    opponents: list[dict],
    rng: random.Random,
    usage: dict[str, int],
    near_delta: float,
    top_k: int,
    max_repeat: int,
):
    near_bucket = [item for item in opponents if abs(item["score"] - target["score"]) <= near_delta]

    top_bucket = sorted(opponents, key=lambda x: x["score"], reverse=True)[:top_k]
    random_bucket = opponents[:]

    weights = [("near", 0.4), ("top", 0.3), ("random", 0.3)]
    buckets = {
        "near": near_bucket,
        "top": top_bucket,
        "random": random_bucket,
    }

    selected = []
    selected_ids = {target["submission_id"]}

    for _ in range(3):
        chosen = None

        for _ in range(5):
            bucket_key = rng.choices([w[0] for w in weights], weights=[w[1] for w in weights], k=1)[0]
            chosen = _choose_from_bucket(rng, buckets[bucket_key], selected_ids, usage, max_repeat)
            if chosen is not None:
                break

        if chosen is None:
            chosen = _choose_from_bucket(rng, opponents, selected_ids, usage, max_repeat)

        if chosen is None:
            eligible = [item for item in opponents if item["submission_id"] not in selected_ids]
            if not eligible:
                break
            chosen = rng.choice(eligible)

        selected.append(chosen)
        selected_ids.add(chosen["submission_id"])

    return selected


def _run_single_match_job(job: dict) -> dict:
    started_at = time.perf_counter()
    try:
        runner = MatchRunner(log_dir=job["log_dir"], enable_gif=job["enable_gif"])
        ranks, survival_steps, gif_path, json_path, gif_drive_url, json_drive_url = runner.run_match(
            agent_paths=job["agent_paths"],
            team_ids=job["submission_ids"],
            seed=job["match_seed"],
            max_steps=job["max_steps"],
            inference_timeout_s=job["inference_timeout_s"],
            startup_timeout_s=job.get("startup_timeout_s"),
        )
        return {
            "ok": True,
            "submission_ids": job["submission_ids"],
            "match_seed": job["match_seed"],
            "ranks": ranks,
            "survival_steps": survival_steps,
            "gif_path": gif_path,
            "json_path": json_path,
            "gif_drive_url": gif_drive_url,
            "json_drive_url": json_drive_url,
            "duration_s": time.perf_counter() - started_at,
        }
    except Exception as e:
        return {
            "ok": False,
            "error": str(e),
            "duration_s": time.perf_counter() - started_at,
        }


def _log_timing(enabled: bool, message: str):
    if enabled:
        logger.info("[timing] %s", message)


def run_submission_batch(
    submission_id: str,
    db_path: str = DEFAULT_DB_PATH,
    n_matches: int = DEFAULT_N_MATCHES,
    max_steps: int = DEFAULT_MAX_STEPS,
    near_delta: float = DEFAULT_NEAR_DELTA,
    top_k: int = DEFAULT_TOP_K,
    opponent_max_repeat: int = DEFAULT_OPPONENT_MAX_REPEAT,
    inference_timeout_s: float = DEFAULT_INFERENCE_TIMEOUT_S,
    startup_timeout_s: Optional[float] = None,
    update_sheet: bool = False,
    sheet_credentials_file: Optional[str] = None,
    sheet_spreadsheet_id: Optional[str] = None,
    sheet_range: Optional[str] = None,
    seed: Optional[int] = None,
    log_dir: str = DEFAULT_LOG_DIR,
    parallel_workers: int = 1,
    enable_gif: bool = True,
    enable_timing_logs: bool = True,
):
    lock_fd = None
    try:
        lock_fd = open(EVALUATION_LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    except Exception as e:
        logger.warning("Could not acquire lock: %s", e)

    def _cleanup_lock():
        nonlocal lock_fd
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
                lock_fd = None
            except Exception:
                pass

    total_started = time.perf_counter()
    parallel_workers = max(1, int(parallel_workers))
    timing_stages: dict[str, float] = {}

    stage_started = time.perf_counter()
    store = SubmissionStore(db_path=db_path)
    pool_manager = PoolManager(db_path=db_path)
    pool_summary = pool_manager.recompute_active_pool(recent_per_team=2, top_k=top_k)
    timing_stages["pool_refresh_initial_s"] = time.perf_counter() - stage_started
    _log_timing(enable_timing_logs, f"pool refresh (initial): {timing_stages['pool_refresh_initial_s']:.3f}s")

    stage_started = time.perf_counter()
    candidates = _fetch_active_candidates(db_path)
    candidates, runtime_failed = _runtime_filter_candidates(
        candidates=candidates,
        store=store,
        inference_timeout_s=inference_timeout_s,
        startup_timeout_s=startup_timeout_s,
    )
    timing_stages["candidate_fetch_and_precheck_s"] = time.perf_counter() - stage_started
    _log_timing(
        enable_timing_logs,
        f"candidate fetch + runtime precheck: {timing_stages['candidate_fetch_and_precheck_s']:.3f}s",
    )

    by_id = {item["submission_id"]: item for item in candidates}
    target = by_id.get(submission_id)

    def _refresh_sheet_if_requested():
        if not update_sheet:
            return None
        return update_google_sheets(
            db_path=db_path,
            credentials_file=sheet_credentials_file,
            spreadsheet_id=sheet_spreadsheet_id,
            sheet_range=sheet_range,
            include_baseline=True,
        )

    if target is None:
        target_reason = runtime_failed.get(submission_id)
        if target_reason:
            sheet_update = _refresh_sheet_if_requested()
            _cleanup_lock()
            return {
                "status": "error",
                "message": f"Target submission {submission_id} failed runtime precheck: {target_reason}",
                "pool_summary": pool_summary,
                "sheet_update": sheet_update,
            }
        sheet_update = _refresh_sheet_if_requested()
        _cleanup_lock()
        return {
            "status": "error",
            "message": f"Target submission {submission_id} not found in active valid pool.",
            "pool_summary": pool_summary,
            "sheet_update": sheet_update,
        }

    opponents = [item for item in candidates if item["submission_id"] != submission_id]
    if len(opponents) < 3:
        sheet_update = _refresh_sheet_if_requested()
        _cleanup_lock()
        return {
            "status": "error",
            "message": f"Not enough opponents in active pool. Need >=3, got {len(opponents)}.",
            "pool_summary": pool_summary,
            "sheet_update": sheet_update,
        }

    ranking = RankingSystem(db_path=db_path)
    rng = random.Random(seed)

    opponent_usage = {}
    success_count = 0
    fail_count = 0
    match_durations: list[float] = []

    jobs: list[dict] = []
    for match_idx in range(n_matches):
        sampled = _sample_opponents(
            target=target,
            opponents=opponents,
            rng=rng,
            usage=opponent_usage,
            near_delta=near_delta,
            top_k=top_k,
            max_repeat=opponent_max_repeat,
        )
        if len(sampled) < 3:
            fail_count += 1
            continue

        lineup = [target] + sampled
        rng.shuffle(lineup)

        agent_paths = [item["agent_path"] for item in lineup]
        submission_ids = [item["submission_id"] for item in lineup]
        match_seed = rng.randint(0, 1000000) if seed is None else seed + match_idx

        jobs.append(
            {
                "agent_paths": agent_paths,
                "submission_ids": submission_ids,
                "match_seed": match_seed,
                "max_steps": max_steps,
                "inference_timeout_s": inference_timeout_s,
                "startup_timeout_s": startup_timeout_s,
                "log_dir": log_dir,
                "enable_gif": enable_gif,
            }
        )

        # Reserve usage while scheduling so repeat caps still shape pairings when using parallel jobs.
        for item in sampled:
            sid = item["submission_id"]
            opponent_usage[sid] = opponent_usage.get(sid, 0) + 1

    stage_started = time.perf_counter()
    if parallel_workers > 1 and jobs:
        _prewarm_drive_folders()
        # Clear the cached Drive service so neither the forked workers nor the parent
        # reuse the TCP connections opened during pre-warm.  Forked workers inherit
        # open socket FDs; if they (or the parent) try to reuse those sockets after
        # another process has closed its copy, the result is [Errno 32] Broken pipe.
        # After this clear, every process builds a fresh service with its own sockets.
        try:
            from competition.integrations.drive_upload import _build_drive_service
            _build_drive_service.cache_clear()
        except Exception:
            pass
        _log_timing(enable_timing_logs, f"running {len(jobs)} matches with {parallel_workers} parallel workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(_run_single_match_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                job_result = future.result()
                match_durations.append(float(job_result.get("duration_s", 0.0)))
                if not job_result.get("ok"):
                    fail_count += 1
                    logger.error(f"Match failed: {job_result.get('error')}")
                    continue

                # If worker didn't successfully upload artifacts (common when workers lack
                # credentials), attempt to upload from the parent process where credentials
                # and env are available. Record upload errors in the job_result for diagnostics.
                drive_folder_id = os.getenv("DRIVE_FOLDER_ID", "").strip()
                if drive_folder_id and job_result.get("ok"):
                    if not job_result.get("json_drive_url") and job_result.get("json_path"):
                        try:
                            upload = upload_file_to_drive(None, drive_folder_id, job_result["json_path"])
                            job_result["json_drive_url"] = upload.get("web_view_link")
                        except Exception as e:
                            logger.exception("Parent upload of JSON failed for %s: %s", job_result.get("json_path"), e)
                            job_result["upload_error_json"] = str(e)

                    if not job_result.get("gif_drive_url") and job_result.get("gif_path"):
                        try:
                            upload = upload_file_to_drive(None, drive_folder_id, job_result["gif_path"])
                            job_result["gif_drive_url"] = upload.get("web_view_link")
                        except Exception as e:
                            logger.exception("Parent upload of GIF failed for %s: %s", job_result.get("gif_path"), e)
                            job_result["upload_error_gif"] = str(e)

                ranking.update_ratings(
                    submission_ids=job_result["submission_ids"],
                    ranks=job_result["ranks"],
                    steps=job_result["survival_steps"],
                    seed=job_result["match_seed"],
                    json_path=job_result["json_path"],
                    gif_path=job_result.get("gif_path"),
                    json_drive_url=job_result.get("json_drive_url"),
                    gif_drive_url=job_result.get("gif_drive_url"),
                    match_type="submission_batch",
                )
                success_count += 1
    else:
        runner = MatchRunner(log_dir=log_dir, enable_gif=enable_gif)
        for job in jobs:
            started_match = time.perf_counter()
            try:
                ranks, survival_steps, gif_path, json_path, gif_drive_url, json_drive_url = runner.run_match(
                    agent_paths=job["agent_paths"],
                    team_ids=job["submission_ids"],
                    seed=job["match_seed"],
                    max_steps=job["max_steps"],
                    inference_timeout_s=job["inference_timeout_s"],
                    startup_timeout_s=job.get("startup_timeout_s"),
                )
                ranking.update_ratings(
                    submission_ids=job["submission_ids"],
                    ranks=ranks,
                    steps=survival_steps,
                    seed=job["match_seed"],
                    json_path=json_path,
                    gif_path=gif_path,
                    json_drive_url=json_drive_url,
                    gif_drive_url=gif_drive_url,
                    match_type="submission_batch",
                )
                success_count += 1
            except Exception:
                fail_count += 1
            finally:
                match_durations.append(time.perf_counter() - started_match)
    timing_stages["match_execution_and_rating_update_s"] = time.perf_counter() - stage_started
    _log_timing(
        enable_timing_logs,
        f"match execution + rating updates: {timing_stages['match_execution_and_rating_update_s']:.3f}s",
    )

    stage_started = time.perf_counter()
    pool_manager.recompute_active_pool(recent_per_team=2, top_k=top_k)
    timing_stages["pool_refresh_final_s"] = time.perf_counter() - stage_started
    _log_timing(enable_timing_logs, f"pool refresh (final): {timing_stages['pool_refresh_final_s']:.3f}s")

    result = {
        "status": "success",
        "submission_id": submission_id,
        "matches_requested": n_matches,
        "matches_successful": success_count,
        "matches_failed": fail_count,
        "runtime_precheck_failed": len(runtime_failed),
        "pool_summary": pool_summary,
        "parallel_workers": parallel_workers,
        "enable_gif": enable_gif,
    }

    if match_durations:
        result["avg_match_duration_s"] = sum(match_durations) / len(match_durations)
        result["max_match_duration_s"] = max(match_durations)

    stage_started = time.perf_counter()
    if update_sheet and success_count > 0:
        result["sheet_update"] = update_google_sheets(
            db_path=db_path,
            credentials_file=sheet_credentials_file,
            spreadsheet_id=sheet_spreadsheet_id,
            sheet_range=sheet_range,
            include_baseline=True,
        )
        timing_stages["sheet_update_s"] = time.perf_counter() - stage_started
    else:
        timing_stages["sheet_update_s"] = 0.0

    result["timings"] = {
        "stages": timing_stages,
        "total_s": time.perf_counter() - total_started,
    }
    _log_timing(enable_timing_logs, f"total run_submission_batch: {result['timings']['total_s']:.3f}s")

    _cleanup_lock()
    return result


def run_background_cycle(
    db_path: str = DEFAULT_DB_PATH,
    n_matches: int = 100,
    max_steps: int = DEFAULT_MAX_STEPS,
    inference_timeout_s: float = DEFAULT_INFERENCE_TIMEOUT_S,
    startup_timeout_s: Optional[float] = None,
    update_sheet: bool = False,
    sheet_credentials_file: Optional[str] = None,
    sheet_spreadsheet_id: Optional[str] = None,
    sheet_range: Optional[str] = None,
    seed: Optional[int] = None,
    log_dir: str = DEFAULT_LOG_DIR,
    parallel_workers: int = 1,
    enable_gif: bool = True,
    enable_timing_logs: bool = True,
):
    lock_fd = None
    try:
        lock_fd = open(EVALUATION_LOCK_FILE, "w")
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        if lock_fd:
            lock_fd.close()
        return {"status": "skipped", "message": "Yielding to active evaluation (lock is busy)."}
    except Exception as e:
        if lock_fd:
            lock_fd.close()
        return {"status": "skipped", "message": f"Could not acquire lock: {e}"}

    def _cleanup_lock():
        nonlocal lock_fd
        if lock_fd:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
                lock_fd.close()
                lock_fd = None
            except Exception:
                pass

    total_started = time.perf_counter()
    parallel_workers = max(1, int(parallel_workers))
    timing_stages: dict[str, float] = {}

    stage_started = time.perf_counter()
    store = SubmissionStore(db_path=db_path)
    pool_manager = PoolManager(db_path=db_path)
    pool_manager.recompute_active_pool(recent_per_team=2, top_k=10)
    candidates = _fetch_active_candidates(db_path)
    candidates, runtime_failed = _runtime_filter_candidates(
        candidates=candidates,
        store=store,
        inference_timeout_s=inference_timeout_s,
        startup_timeout_s=startup_timeout_s,
    )
    timing_stages["prepare_candidates_s"] = time.perf_counter() - stage_started
    _log_timing(enable_timing_logs, f"background prepare candidates: {timing_stages['prepare_candidates_s']:.3f}s")

    if len(candidates) < 4:
        return {"status": "error", "message": f"Not enough active submissions: {len(candidates)}"}

    ranking = RankingSystem(db_path=db_path)
    rng = random.Random(seed)

    jobs: list[dict] = []
    success = 0
    
    student_candidates = [c for c in candidates if not c.get("is_baseline")]

    if not student_candidates:
        return {"status": "skipped", "message": "No student agents in active pool. All-baseline matches are skipped."}

    for i in range(n_matches):
        if len(candidates) >= 4:
            # Force at least one student agent to avoid useless all-baseline matches
            student = rng.choice(student_candidates)
            remaining = [c for c in candidates if c["submission_id"] != student["submission_id"]]
            lineup = [student] + rng.sample(remaining, 3)
            rng.shuffle(lineup) # randomize player positions
        else:
            lineup = rng.sample(candidates, 4)

        agent_paths = [item["agent_path"] for item in lineup]
        submission_ids = [item["submission_id"] for item in lineup]
        match_seed = rng.randint(0, 1000000) if seed is None else seed + i

        jobs.append(
            {
                "agent_paths": agent_paths,
                "submission_ids": submission_ids,
                "match_seed": match_seed,
                "max_steps": max_steps,
                "inference_timeout_s": inference_timeout_s,
                "startup_timeout_s": startup_timeout_s,
                "log_dir": log_dir,
                "enable_gif": enable_gif,
            }
        )

    stage_started = time.perf_counter()
    if parallel_workers > 1 and jobs:
        _prewarm_drive_folders()
        try:
            from competition.integrations.drive_upload import _build_drive_service
            _build_drive_service.cache_clear()
        except Exception:
            pass
        _log_timing(enable_timing_logs, f"running background {len(jobs)} matches with {parallel_workers} workers")
        with concurrent.futures.ProcessPoolExecutor(max_workers=parallel_workers) as executor:
            futures = [executor.submit(_run_single_match_job, job) for job in jobs]
            for future in concurrent.futures.as_completed(futures):
                job_result = future.result()
                if not job_result.get("ok"):
                    logger.error(f"Match failed: {job_result.get('error')}")
                    continue
                ranking.update_ratings(
                    submission_ids=job_result["submission_ids"],
                    ranks=job_result["ranks"],
                    steps=job_result["survival_steps"],
                    seed=job_result["match_seed"],
                    json_path=job_result["json_path"],
                    gif_path=job_result.get("gif_path"),
                    json_drive_url=job_result.get("json_drive_url"),
                    gif_drive_url=job_result.get("gif_drive_url"),
                    match_type="background",
                )
                success += 1
    else:
        runner = MatchRunner(log_dir=log_dir, enable_gif=enable_gif)
        for job in jobs:
            try:
                ranks, survival_steps, gif_path, json_path, gif_drive_url, json_drive_url = runner.run_match(
                    agent_paths=job["agent_paths"],
                    team_ids=job["submission_ids"],
                    seed=job["match_seed"],
                    max_steps=job["max_steps"],
                    inference_timeout_s=job["inference_timeout_s"],
                    startup_timeout_s=job.get("startup_timeout_s"),
                )
                ranking.update_ratings(
                    submission_ids=job["submission_ids"],
                    ranks=ranks,
                    steps=survival_steps,
                    seed=job["match_seed"],
                    json_path=json_path,
                    gif_path=gif_path,
                    json_drive_url=json_drive_url,
                    gif_drive_url=gif_drive_url,
                    match_type="background",
                )
                success += 1
            except Exception:
                continue
    timing_stages["match_execution_and_rating_update_s"] = time.perf_counter() - stage_started
    _log_timing(
        enable_timing_logs,
        f"background match execution + rating updates: {timing_stages['match_execution_and_rating_update_s']:.3f}s",
    )

    stage_started = time.perf_counter()
    pool_manager.recompute_active_pool(recent_per_team=2, top_k=10)
    timing_stages["pool_refresh_final_s"] = time.perf_counter() - stage_started
    result = {
        "status": "success",
        "matches_requested": n_matches,
        "matches_successful": success,
        "runtime_precheck_failed": len(runtime_failed),
        "parallel_workers": parallel_workers,
        "enable_gif": enable_gif,
    }

    if update_sheet:
        result["sheet_update"] = update_google_sheets(
            db_path=db_path,
            credentials_file=sheet_credentials_file,
            spreadsheet_id=sheet_spreadsheet_id,
            sheet_range=sheet_range,
            include_baseline=True,
            update_feedback=False,
        )

    result["timings"] = {
        "stages": timing_stages,
        "total_s": time.perf_counter() - total_started,
    }
    _log_timing(enable_timing_logs, f"total run_background_cycle: {result['timings']['total_s']:.3f}s")

    _cleanup_lock()
    return result


def _print_leaderboard(db_path: str, limit: int = 20):
    ranking = RankingSystem(db_path=db_path)
    rows = ranking.get_leaderboard(include_baseline=True)[:limit]
    print("\n--- Submission Leaderboard ---")
    for i, row in enumerate(rows, start=1):
        print(
            f"{i}. {row['team_name']} [{row['submission_id']}] "
            f"score={row['score']:.2f} mu={row['mu']:.2f} sigma={row['sigma']:.2f} "
            f"games={row['n_games']} W/D/L={row['wins']}/{row['draws']}/{row['losses']}"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--db_path", default=DEFAULT_DB_PATH)
    parser.add_argument("--log_dir", default=DEFAULT_LOG_DIR)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--inference_timeout_ms", type=int, default=100)
    parser.add_argument("--startup_timeout_ms", type=int, default=int(float(os.getenv("EVALUATION_STARTUP_TIMEOUT_S", "20")) * 1000))
    parser.add_argument("--update_sheet", action="store_true")
    parser.add_argument("--sheet_credentials_file", default=None)
    parser.add_argument("--sheet_spreadsheet_id", default=None)
    parser.add_argument("--sheet_range", default=None)
    parser.add_argument("--parallel_workers", type=int, default=1)
    parser.add_argument("--disable_gif", action="store_true")
    parser.add_argument("--disable_timing_logs", action="store_true")

    sub = parser.add_subparsers(dest="command", required=True)

    batch_cmd = sub.add_parser("submission-batch")
    batch_cmd.add_argument("--submission_id", required=True)
    batch_cmd.add_argument("--matches", type=int, default=DEFAULT_N_MATCHES)
    batch_cmd.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)

    bg_cmd = sub.add_parser("background")
    bg_cmd.add_argument("--matches", type=int, default=5)
    bg_cmd.add_argument("--max_steps", type=int, default=DEFAULT_MAX_STEPS)

    lb_cmd = sub.add_parser("leaderboard")
    lb_cmd.add_argument("--limit", type=int, default=20)

    if hasattr(os, "geteuid") and os.geteuid() != 0:
        print("ERROR: For security, evaluation must be run with sudo to enable sandboxing.")
        import sys
        sys.exit(1)

    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    if args.command == "submission-batch":
        result = run_submission_batch(
            submission_id=args.submission_id,
            db_path=args.db_path,
            n_matches=args.matches,
            max_steps=args.max_steps,
            inference_timeout_s=max(0.001, args.inference_timeout_ms / 1000.0),
            startup_timeout_s=max(0.001, args.startup_timeout_ms / 1000.0),
            update_sheet=args.update_sheet,
            sheet_credentials_file=args.sheet_credentials_file,
            sheet_spreadsheet_id=args.sheet_spreadsheet_id,
            sheet_range=args.sheet_range,
            seed=args.seed,
            log_dir=args.log_dir,
            parallel_workers=args.parallel_workers,
            enable_gif=not args.disable_gif,
            enable_timing_logs=not args.disable_timing_logs,
        )
        print(result)
    elif args.command == "background":
        # Allow environment variables to override or provide defaults for the background worker
        update_sheet = args.update_sheet or os.getenv("EVALUATION_UPDATE_SHEET", "false").lower() == "true"
        matches = args.matches if args.matches != 5 else int(os.getenv("BACKGROUND_EVAL_MATCHES", "5"))
        parallel_workers = args.parallel_workers if args.parallel_workers != 1 else int(os.getenv("EVALUATION_PARALLEL_WORKERS", "1"))
        
        # Enable GIF/Timing by default unless explicitly disabled by flag OR by env var
        enable_gif = not args.disable_gif and os.getenv("EVALUATION_ENABLE_GIF", "1") not in ("0", "false", "False")
        enable_timing_logs = not args.disable_timing_logs and os.getenv("EVALUATION_TIMING_LOGS", "1") not in ("0", "false", "False")

        result = run_background_cycle(
            db_path=args.db_path,
            n_matches=matches,
            max_steps=args.max_steps,
            inference_timeout_s=max(0.001, args.inference_timeout_ms / 1000.0),
            startup_timeout_s=max(0.001, args.startup_timeout_ms / 1000.0),
            update_sheet=update_sheet,
            sheet_credentials_file=args.sheet_credentials_file,
            sheet_spreadsheet_id=args.sheet_spreadsheet_id,
            sheet_range=args.sheet_range,
            seed=args.seed,
            log_dir=args.log_dir,
            parallel_workers=parallel_workers,
            enable_gif=enable_gif,
            enable_timing_logs=enable_timing_logs,
        )
        print(result)
    else:
        _print_leaderboard(db_path=args.db_path, limit=args.limit)
