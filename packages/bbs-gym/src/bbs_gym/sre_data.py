"""Maintenance helpers for Solar Realms Elite data files."""

from __future__ import annotations

import os
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path


SRE_EXECUTABLE_SIZE = 443_568
SRE_EMPIRE_RECORD_COUNT = 25
SRE_EMPIRE_RECORD_SIZE = 764
SRE_GALAXY_RECORD_SIZE = 966
SRE_PIRATE_RECORD_COUNT = 9
SRE_PIRATE_RECORD_SIZE = 262
SRE_SYSTEM_RECORD_SIZE = 322
SRE_SYSTEM_TIME_OFFSETS = (0x120, 0x124)
SRE_GALAXY_DAILY_MAINTENANCE_OFFSET = 0x40
SRE_GALAXY_TOURNAMENT_START_OFFSET = 0x44
SRE_GALAXY_HOURLY_MAINTENANCE_OFFSET = 0x3BA
SRE_GALAXY_MAINTENANCE_TIME_OFFSETS = (
    SRE_GALAXY_DAILY_MAINTENANCE_OFFSET,
    SRE_GALAXY_TOURNAMENT_START_OFFSET,
    SRE_GALAXY_HOURLY_MAINTENANCE_OFFSET,
)
SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET = 0x12A
SRE_GALAXY_EMPIRE_DESCRIPTOR_OFFSET = 0x54
SRE_GALAXY_PIRATE_DESCRIPTOR_OFFSET = 0x2AC
SRE_INTEGRITY_DESCRIPTOR_SIZE = 24
_MIN_SUPPORTED_EPOCH = 315_532_800  # 1980-01-01 UTC
_MAX_PATCHED_EPOCH = 2_147_483_646

# SRE 0.994b rejects maintenance timestamps after 945,000,000 even though the
# rest of the program stores them as signed 32-bit Unix timestamps. These four
# comparisons are the upper-bound halves of its two timestamp range checks.
# Raising the exclusive limit to 0x7fffffff preserves the original lower-bound
# and structural validation while allowing dates through January 2038.
_Y2K_VALIDATION_PATCHES = (
    (
        0x63586,
        bytes.fromhex("26 81 bf 26 01 53 38 7f 54 7c 09"),
        bytes.fromhex("26 81 bf 26 01 ff 7f 7f 54 7c 09"),
    ),
    (
        0x63591,
        bytes.fromhex("26 81 bf 24 01 40 8e 73 49"),
        bytes.fromhex("26 81 bf 24 01 ff ff 73 49"),
    ),
    (
        0x635B4,
        bytes.fromhex("26 81 bf 22 01 53 38 7f 26 7c 09"),
        bytes.fromhex("26 81 bf 22 01 ff 7f 7f 26 7c 09"),
    ),
    (
        0x635BF,
        bytes.fromhex("26 81 bf 20 01 40 8e 73 1b"),
        bytes.fromhex("26 81 bf 20 01 ff ff 73 1b"),
    ),
)


@dataclass(frozen=True)
class SreResetTimePatch:
    """Description of one reset-time and executable patch."""

    system_path: Path
    galaxy_path: Path
    executable_path: Path
    previous_system_timestamps: tuple[int, int]
    previous_maintenance_timestamps: tuple[int, int, int]
    timestamp: int
    tournament_start_timestamp: int
    hourly_maintenance_timestamp: int
    executable_changed: bool
    system_backup_path: Path | None = None
    galaxy_backup_path: Path | None = None
    executable_backup_path: Path | None = None


def patch_sre_reset_time(
        system_path: str | Path,
        executable_path: str | Path,
        *,
        galaxy_path: str | Path | None = None,
        timestamp: int | None = None,
        system_backup_path: str | Path | None = None,
        galaxy_backup_path: str | Path | None = None,
        executable_backup_path: str | Path | None = None,
) -> SreResetTimePatch:
    """Advance a freshly reset SRE 0.994b world to the current hour.

    The author's reset workaround creates the world under a 1999 clock. SRE
    otherwise replays every intervening hourly interval on first entry. Merely
    changing ``DATA/SYSTEM.II`` is insufficient: SRE 0.994b explicitly rejects
    its two System timestamps after December 1999, while the authoritative
    hourly and daily maintenance clocks live inside encrypted ``GALAXY.II``.
    This function narrowly raises the executable validation limits, advances
    both System values and the Galaxy maintenance cursors. SRE compares its
    tournament-start field with a current-day value normalized to midnight,
    not the current clock, so the tournament start is set to the beginning of
    the current UTC day. Setting it to the current hour would leave the game
    closed until the following day. The function then rebuilds the Galaxy
    integrity descriptor stored at the end of ``SYSTEM.II``.

    This operation is intended only for a freshly reset, idle world. SRE's DOS
    launcher must also set ``TZ=UTC0`` so its Watcom runtime and this helper use
    the same epoch year-round.

    Args:
        system_path: Path to ``DATA/SYSTEM.II``.
        executable_path: Path to the matching SRE 0.994b ``SRE.EXE``.
        galaxy_path: Path to ``DATA/GALAXY.II``. Defaults to the file beside
            ``SYSTEM.II``.
        timestamp: Unix timestamp to store. Defaults to the start of the
            current hour.
        system_backup_path: Optional exclusive-create backup for ``SYSTEM.II``.
        galaxy_backup_path: Optional exclusive-create backup for ``GALAXY.II``.
        executable_backup_path: Optional exclusive-create backup for
            ``SRE.EXE``.

    Returns:
        Details of the values and files changed.

    Raises:
        FileExistsError: If any backup already exists.
        ValueError: If a file or timestamp is not recognized as safe.
    """

    resolved_system_path = Path(system_path)
    resolved_galaxy_path = Path(galaxy_path) if galaxy_path is not None else resolved_system_path.with_name("GALAXY.II")
    resolved_executable_path = Path(executable_path)
    system_data = bytearray(resolved_system_path.read_bytes())
    galaxy_ciphertext = resolved_galaxy_path.read_bytes()
    executable_data = bytearray(resolved_executable_path.read_bytes())

    if len(system_data) != SRE_SYSTEM_RECORD_SIZE:
        raise ValueError(
            f"unexpected SRE SYSTEM.II size {len(system_data)}; expected {SRE_SYSTEM_RECORD_SIZE}"
        )
    if len(galaxy_ciphertext) != SRE_GALAXY_RECORD_SIZE:
        raise ValueError(
            f"unexpected SRE GALAXY.II size {len(galaxy_ciphertext)}; expected {SRE_GALAXY_RECORD_SIZE}"
        )
    if len(executable_data) != SRE_EXECUTABLE_SIZE:
        raise ValueError(
            f"unexpected SRE.EXE size {len(executable_data)}; expected {SRE_EXECUTABLE_SIZE}"
        )

    previous_system = tuple(
        struct.unpack_from("<I", system_data, offset)[0]
        for offset in SRE_SYSTEM_TIME_OFFSETS
    )
    if previous_system[0] != previous_system[1]:
        raise ValueError(f"SRE System timestamps disagree: {previous_system[0]} != {previous_system[1]}")
    if not _MIN_SUPPORTED_EPOCH <= previous_system[0] <= _MAX_PATCHED_EPOCH:
        raise ValueError(f"unrecognized SRE System timestamp: {previous_system[0]}")

    descriptor_end = SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET + SRE_INTEGRITY_DESCRIPTOR_SIZE
    galaxy_descriptor = bytes(system_data[SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET:descriptor_end])
    galaxy_plaintext = bytearray(decode_sre_record(galaxy_ciphertext, galaxy_descriptor))
    previous_maintenance = tuple(
        struct.unpack_from("<I", galaxy_plaintext, offset)[0]
        for offset in SRE_GALAXY_MAINTENANCE_TIME_OFFSETS
    )
    previous_daily = previous_maintenance[0]
    previous_start = previous_maintenance[1]
    previous_hourly = previous_maintenance[2]
    legacy_layout = previous_daily == previous_start and previous_hourly in {
        previous_daily,
        previous_daily - 3600,
    }
    normalized_layout = (
        previous_daily == previous_hourly
        and previous_start <= previous_daily
        and previous_start % 86_400 == 0
        and previous_daily - previous_start < 86_400
    )
    if not (legacy_layout or normalized_layout):
        rendered = ", ".join(str(value) for value in previous_maintenance)
        raise ValueError(f"SRE Galaxy reset and maintenance timestamps are inconsistent: {rendered}")
    if not all(_MIN_SUPPORTED_EPOCH <= value <= _MAX_PATCHED_EPOCH for value in previous_maintenance):
        raise ValueError(f"unrecognized SRE Galaxy maintenance timestamps: {previous_maintenance}")

    effective_timestamp = int(time.time()) if timestamp is None else timestamp
    effective_timestamp -= effective_timestamp % 3600
    if not _MIN_SUPPORTED_EPOCH <= effective_timestamp <= _MAX_PATCHED_EPOCH:
        raise ValueError(f"timestamp is outside SRE's patched signed 32-bit range: {effective_timestamp}")
    tournament_start_timestamp = effective_timestamp - (effective_timestamp % 86_400)
    hourly_maintenance_timestamp = effective_timestamp
    previous_latest = max(previous_system[0], *previous_maintenance)
    if effective_timestamp < previous_latest:
        raise ValueError(
            f"refusing to move SRE reset time backwards from {previous_latest} to {effective_timestamp}"
        )

    executable_states = []
    for offset, original, patched in _Y2K_VALIDATION_PATCHES:
        current = bytes(executable_data[offset : offset + len(original)])
        if current == original:
            executable_states.append("original")
        elif current == patched:
            executable_states.append("patched")
        else:
            raise ValueError(
                f"SRE.EXE does not match the supported 0.994b validation code at offset 0x{offset:x}"
            )
    if len(set(executable_states)) != 1:
        raise ValueError("SRE.EXE contains a partial Y2K validation patch")

    resolved_system_backup = Path(system_backup_path) if system_backup_path is not None else None
    resolved_galaxy_backup = Path(galaxy_backup_path) if galaxy_backup_path is not None else None
    resolved_executable_backup = Path(executable_backup_path) if executable_backup_path is not None else None
    backup_paths = [
        path
        for path in (resolved_system_backup, resolved_galaxy_backup, resolved_executable_backup)
        if path is not None
    ]
    if len(set(backup_paths)) != len(backup_paths):
        raise ValueError("SRE System, Galaxy, and executable backups must use different paths")
    for backup_path in backup_paths:
        if backup_path.exists():
            raise FileExistsError(backup_path)

    _write_backup(resolved_system_backup, system_data)
    _write_backup(resolved_galaxy_backup, galaxy_ciphertext)
    _write_backup(resolved_executable_backup, executable_data)

    executable_changed = executable_states[0] == "original"
    if executable_changed:
        for offset, original, patched in _Y2K_VALIDATION_PATCHES:
            executable_data[offset : offset + len(original)] = patched
        _replace_bytes(resolved_executable_path, executable_data)

    for offset in SRE_SYSTEM_TIME_OFFSETS:
        struct.pack_into("<I", system_data, offset, effective_timestamp)
    struct.pack_into(
        "<I",
        galaxy_plaintext,
        SRE_GALAXY_DAILY_MAINTENANCE_OFFSET,
        effective_timestamp,
    )
    struct.pack_into(
        "<I",
        galaxy_plaintext,
        SRE_GALAXY_TOURNAMENT_START_OFFSET,
        tournament_start_timestamp,
    )
    struct.pack_into(
        "<I",
        galaxy_plaintext,
        SRE_GALAXY_HOURLY_MAINTENANCE_OFFSET,
        hourly_maintenance_timestamp,
    )
    patched_galaxy, patched_descriptor = encode_sre_record(
        galaxy_plaintext,
        nonce=galaxy_descriptor[8:16],
    )
    system_data[SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET:descriptor_end] = patched_descriptor

    _replace_bytes(resolved_galaxy_path, patched_galaxy)
    _replace_bytes(resolved_system_path, system_data)

    return SreResetTimePatch(
        system_path=resolved_system_path,
        galaxy_path=resolved_galaxy_path,
        executable_path=resolved_executable_path,
        previous_system_timestamps=previous_system,
        previous_maintenance_timestamps=previous_maintenance,
        timestamp=effective_timestamp,
        tournament_start_timestamp=tournament_start_timestamp,
        hourly_maintenance_timestamp=hourly_maintenance_timestamp,
        executable_changed=executable_changed,
        system_backup_path=resolved_system_backup,
        galaxy_backup_path=resolved_galaxy_backup,
        executable_backup_path=resolved_executable_backup,
    )


def decode_sre_record(ciphertext: bytes | bytearray, descriptor: bytes | bytearray) -> bytes:
    """Decrypt and validate one SRE integrity-protected record."""

    descriptor_bytes = bytes(descriptor)
    if len(descriptor_bytes) != SRE_INTEGRITY_DESCRIPTOR_SIZE:
        raise ValueError(
            f"unexpected SRE integrity descriptor size {len(descriptor_bytes)}; "
            f"expected {SRE_INTEGRITY_DESCRIPTOR_SIZE}"
        )

    expected_weak, expected_crc = struct.unpack_from("<II", descriptor_bytes)
    plaintext, verifier = _transform_sre_record(ciphertext, descriptor_bytes[8:16], decrypt=True)
    if verifier != descriptor_bytes[16:24]:
        raise ValueError("SRE record encryption verifier does not match")
    if _sre_weak_checksum(plaintext) != expected_weak:
        raise ValueError("SRE record weak checksum does not match")
    if _sre_crc32(plaintext) != expected_crc:
        raise ValueError("SRE record CRC does not match")
    return plaintext


def encode_sre_record(
        plaintext: bytes | bytearray,
        *,
        nonce: bytes | bytearray,
) -> tuple[bytes, bytes]:
    """Encrypt one SRE record and return its 24-byte integrity descriptor."""

    nonce_bytes = bytes(nonce)
    if len(nonce_bytes) != 8:
        raise ValueError(f"unexpected SRE record nonce size {len(nonce_bytes)}; expected 8")
    ciphertext, verifier = _transform_sre_record(plaintext, nonce_bytes, decrypt=False)
    descriptor = struct.pack("<II", _sre_weak_checksum(plaintext), _sre_crc32(plaintext)) + nonce_bytes + verifier
    return ciphertext, descriptor


def _transform_sre_record(
        data: bytes | bytearray,
        nonce: bytes,
        *,
        decrypt: bool,
) -> tuple[bytes, bytes]:
    """Apply SRE's evolving eight-byte XOR transform."""

    state = bytearray(nonce)
    output = bytearray()
    counter = 0
    for index, value in enumerate(data):
        position = index & 7
        plaintext = value ^ state[position] if decrypt else value
        output.append(plaintext if decrypt else plaintext ^ state[position])
        state[position] = (((state[position] << 3) & 0xFF) | (state[position] >> 5)) ^ counter ^ plaintext
        counter = (counter + 7) & 0xFF
    return bytes(output), bytes(state)


def _sre_weak_checksum(data: bytes | bytearray) -> int:
    """Return SRE's 16-bit cumulative checksum in its 32-bit field."""

    cumulative = 0
    running = 0
    for value in data:
        running = (running + value) & 0xFFFF
        cumulative = (cumulative + running) & 0xFFFF
    return cumulative


def _sre_crc32(data: bytes | bytearray) -> int:
    """Return SRE's non-final-XOR form of the standard reflected CRC-32."""

    return (~zlib.crc32(data)) & 0xFFFFFFFF


def _write_backup(path: str | Path | None, data: bytes | bytearray) -> Path | None:
    """Exclusively create an optional backup."""

    if path is None:
        return None
    resolved_path = Path(path)
    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    with resolved_path.open("xb") as backup:
        backup.write(data)
    return resolved_path


def _replace_bytes(path: Path, data: bytes | bytearray) -> None:
    """Atomically replace a file while retaining its permission bits."""

    temp_path = path.with_name(f".{path.name}.spree-patch.tmp")
    try:
        temp_path.write_bytes(data)
        os.chmod(temp_path, path.stat().st_mode)
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)
