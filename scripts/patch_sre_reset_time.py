#!/usr/bin/env python3
"""Skip SRE 0.994b's one-time 1999-to-present maintenance replay."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from bbs_gym.sre_data import patch_sre_reset_time


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--system",
        type=Path,
        default=Path("runtime/sbbs/xtrn/sre/DATA/SYSTEM.II"),
        help="path to the freshly reset SRE DATA/SYSTEM.II",
    )
    parser.add_argument(
        "--executable",
        type=Path,
        default=Path("runtime/sbbs/xtrn/sre/SRE.EXE"),
        help="path to the matching SRE 0.994b executable",
    )
    parser.add_argument(
        "--galaxy",
        type=Path,
        default=Path("runtime/sbbs/xtrn/sre/DATA/GALAXY.II"),
        help="path to the freshly reset SRE DATA/GALAXY.II",
    )
    parser.add_argument(
        "--backup-directory",
        type=Path,
        help="optional empty destination for exclusive-create backups",
    )
    parser.add_argument("--timestamp", type=int, help="Unix timestamp override, primarily for reproducible tests")
    args = parser.parse_args()

    lock_path = args.system.parent.parent / "inuse.sr"
    if lock_path.exists():
        parser.error(f"SRE appears to be in use; remove only a confirmed stale lock: {lock_path}")

    system_backup_path = None
    galaxy_backup_path = None
    executable_backup_path = None
    if args.backup_directory is not None:
        system_backup_path = args.backup_directory / "SYSTEM.II"
        galaxy_backup_path = args.backup_directory / "GALAXY.II"
        executable_backup_path = args.backup_directory / "SRE.EXE"

    result = patch_sre_reset_time(
        args.system,
        args.executable,
        galaxy_path=args.galaxy,
        timestamp=args.timestamp,
        system_backup_path=system_backup_path,
        galaxy_backup_path=galaxy_backup_path,
        executable_backup_path=executable_backup_path,
    )
    previous_system = datetime.fromtimestamp(result.previous_system_timestamps[0], tz=timezone.utc).isoformat()
    previous_maintenance = datetime.fromtimestamp(
        result.previous_maintenance_timestamps[0],
        tz=timezone.utc,
    ).isoformat()
    updated = datetime.fromtimestamp(result.timestamp, tz=timezone.utc).isoformat()
    tournament_start = datetime.fromtimestamp(result.tournament_start_timestamp, tz=timezone.utc).isoformat()
    hourly_maintenance = datetime.fromtimestamp(result.hourly_maintenance_timestamp, tz=timezone.utc).isoformat()
    executable_action = "patched" if result.executable_changed else "already-patched"
    backup = f" backups={args.backup_directory}" if args.backup_directory is not None else ""
    print(
        f"system={result.system_path} galaxy={result.galaxy_path} executable={result.executable_path} "
        f"executable_action={executable_action} previous_system={previous_system} "
        f"previous_maintenance={previous_maintenance} updated={updated} tournament_start={tournament_start} "
        f"hourly_maintenance={hourly_maintenance}{backup}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
