import struct

import pytest

from bbs_gym.sre_data import (
    SRE_EXECUTABLE_SIZE,
    SRE_GALAXY_DAILY_MAINTENANCE_OFFSET,
    SRE_GALAXY_HOURLY_MAINTENANCE_OFFSET,
    SRE_GALAXY_RECORD_SIZE,
    SRE_GALAXY_TOURNAMENT_START_OFFSET,
    SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET,
    SRE_SYSTEM_RECORD_SIZE,
    decode_sre_record,
    encode_sre_record,
    patch_sre_reset_time,
)


ORIGINAL_VALIDATION_CODE = (
    (0x63586, bytes.fromhex("26 81 bf 26 01 53 38 7f 54 7c 09")),
    (0x63591, bytes.fromhex("26 81 bf 24 01 40 8e 73 49")),
    (0x635B4, bytes.fromhex("26 81 bf 22 01 53 38 7f 26 7c 09")),
    (0x635BF, bytes.fromhex("26 81 bf 20 01 40 8e 73 1b")),
)

PATCHED_VALIDATION_CODE = (
    (0x63586, bytes.fromhex("26 81 bf 26 01 ff 7f 7f 54 7c 09")),
    (0x63591, bytes.fromhex("26 81 bf 24 01 ff ff 73 49")),
    (0x635B4, bytes.fromhex("26 81 bf 22 01 ff 7f 7f 26 7c 09")),
    (0x635BF, bytes.fromhex("26 81 bf 20 01 ff ff 73 1b")),
)


def galaxy_record(
        timestamp: int,
        tournament_start: int | None = None,
        hourly_timestamp: int | None = None,
) -> bytearray:
    data = bytearray((index * 17) & 0xFF for index in range(SRE_GALAXY_RECORD_SIZE))
    struct.pack_into("<I", data, SRE_GALAXY_DAILY_MAINTENANCE_OFFSET, timestamp)
    struct.pack_into(
        "<I",
        data,
        SRE_GALAXY_TOURNAMENT_START_OFFSET,
        timestamp if tournament_start is None else tournament_start,
    )
    struct.pack_into(
        "<I",
        data,
        SRE_GALAXY_HOURLY_MAINTENANCE_OFFSET,
        timestamp if hourly_timestamp is None else hourly_timestamp,
    )
    return data


def system_record(timestamp: int, galaxy_descriptor: bytes) -> bytearray:
    data = bytearray((index * 17) & 0xFF for index in range(SRE_SYSTEM_RECORD_SIZE))
    struct.pack_into("<I", data, 0x120, timestamp)
    struct.pack_into("<I", data, 0x124, timestamp)
    data[SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET:] = galaxy_descriptor
    return data


def executable(validation_code=ORIGINAL_VALIDATION_CODE) -> bytearray:
    data = bytearray(SRE_EXECUTABLE_SIZE)
    for offset, code in validation_code:
        data[offset : offset + len(code)] = code
    return data


def write_world(
        tmp_path,
        *,
        timestamp=946_616_400,
        tournament_start: int | None = None,
        hourly_timestamp: int | None = None,
        validation_code=ORIGINAL_VALIDATION_CODE,
):
    system_path = tmp_path / "SYSTEM.II"
    galaxy_path = tmp_path / "GALAXY.II"
    executable_path = tmp_path / "SRE.EXE"
    galaxy_plaintext = galaxy_record(timestamp, tournament_start, hourly_timestamp)
    galaxy_ciphertext, galaxy_descriptor = encode_sre_record(
        galaxy_plaintext,
        nonce=bytes.fromhex("0366c2b6af939ba1"),
    )
    system_path.write_bytes(system_record(915_204_627, galaxy_descriptor))
    galaxy_path.write_bytes(galaxy_ciphertext)
    executable_path.write_bytes(executable(validation_code))
    return system_path, galaxy_path, executable_path, galaxy_plaintext


def test_sre_record_codec_validates_checksums_and_verifier():
    plaintext = b"Spree Galaxy\0" + bytes(range(64))
    ciphertext, descriptor = encode_sre_record(
        plaintext,
        nonce=bytes.fromhex("0366c2b6af939ba1"),
    )

    assert descriptor.hex() == "edec00005381d6f50366c2b6af939ba175075e206d219e60"
    assert decode_sre_record(ciphertext, descriptor) == plaintext

    corrupted = bytearray(ciphertext)
    corrupted[10] ^= 1
    with pytest.raises(ValueError, match="does not match"):
        decode_sre_record(corrupted, descriptor)


def test_patch_sre_reset_time_updates_world_executable_and_backups(tmp_path):
    system_path, galaxy_path, executable_path, original_galaxy = write_world(tmp_path)
    system_backup_path = tmp_path / "backup" / "SYSTEM.II"
    galaxy_backup_path = tmp_path / "backup" / "GALAXY.II"
    executable_backup_path = tmp_path / "backup" / "SRE.EXE"
    original_system = system_path.read_bytes()
    original_ciphertext = galaxy_path.read_bytes()
    original_executable = executable_path.read_bytes()

    result = patch_sre_reset_time(
        system_path,
        executable_path,
        galaxy_path=galaxy_path,
        timestamp=1_785_263_999,
        system_backup_path=system_backup_path,
        galaxy_backup_path=galaxy_backup_path,
        executable_backup_path=executable_backup_path,
    )

    patched_system = system_path.read_bytes()
    patched_executable = executable_path.read_bytes()
    patched_descriptor = patched_system[SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET:]
    patched_galaxy = decode_sre_record(galaxy_path.read_bytes(), patched_descriptor)
    expected_galaxy = bytearray(original_galaxy)
    struct.pack_into(
        "<I",
        expected_galaxy,
        SRE_GALAXY_DAILY_MAINTENANCE_OFFSET,
        1_785_261_600,
    )
    struct.pack_into(
        "<I",
        expected_galaxy,
        SRE_GALAXY_TOURNAMENT_START_OFFSET,
        1_785_196_800,
    )
    struct.pack_into(
        "<I",
        expected_galaxy,
        SRE_GALAXY_HOURLY_MAINTENANCE_OFFSET,
        1_785_261_600,
    )

    assert struct.unpack_from("<I", patched_system, 0x120)[0] == 1_785_261_600
    assert struct.unpack_from("<I", patched_system, 0x124)[0] == 1_785_261_600
    assert patched_system[:0x120] == original_system[:0x120]
    assert (
        patched_system[0x128:SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET]
        == original_system[0x128:SRE_SYSTEM_GALAXY_DESCRIPTOR_OFFSET]
    )
    assert patched_galaxy == expected_galaxy
    for offset, code in PATCHED_VALIDATION_CODE:
        assert patched_executable[offset : offset + len(code)] == code
    assert system_backup_path.read_bytes() == original_system
    assert galaxy_backup_path.read_bytes() == original_ciphertext
    assert executable_backup_path.read_bytes() == original_executable
    assert result.previous_system_timestamps == (915_204_627, 915_204_627)
    assert result.previous_maintenance_timestamps == (946_616_400, 946_616_400, 946_616_400)
    assert result.timestamp == 1_785_261_600
    assert result.tournament_start_timestamp == 1_785_196_800
    assert result.hourly_maintenance_timestamp == 1_785_261_600
    assert result.executable_changed is True


def test_patch_sre_reset_time_is_idempotent_for_an_already_patched_world(tmp_path):
    system_path, galaxy_path, executable_path, _ = write_world(
        tmp_path,
        timestamp=1_785_261_600,
        tournament_start=1_785_196_800,
        hourly_timestamp=1_785_261_600,
        validation_code=PATCHED_VALIDATION_CODE,
    )
    system_data = bytearray(system_path.read_bytes())
    struct.pack_into("<I", system_data, 0x120, 1_785_261_600)
    struct.pack_into("<I", system_data, 0x124, 1_785_261_600)
    system_path.write_bytes(system_data)
    original_system = system_path.read_bytes()
    original_galaxy = galaxy_path.read_bytes()

    result = patch_sre_reset_time(
        system_path,
        executable_path,
        galaxy_path=galaxy_path,
        timestamp=1_785_263_999,
    )

    assert result.executable_changed is False
    assert system_path.read_bytes() == original_system
    assert galaxy_path.read_bytes() == original_galaxy


def test_patch_sre_reset_time_rejects_unknown_records(tmp_path):
    system_path, galaxy_path, executable_path, _ = write_world(tmp_path)
    system_path.write_bytes(b"short")

    with pytest.raises(ValueError, match="unexpected SRE SYSTEM.II size"):
        patch_sre_reset_time(
            system_path,
            executable_path,
            galaxy_path=galaxy_path,
            timestamp=1_785_261_600,
        )


def test_patch_sre_reset_time_rejects_unknown_executables(tmp_path):
    system_path, galaxy_path, executable_path, _ = write_world(tmp_path)
    executable_path.write_bytes(bytearray(SRE_EXECUTABLE_SIZE))

    with pytest.raises(ValueError, match="does not match the supported 0.994b validation code"):
        patch_sre_reset_time(
            system_path,
            executable_path,
            galaxy_path=galaxy_path,
            timestamp=1_785_261_600,
        )


def test_patch_sre_reset_time_does_not_replace_existing_backups(tmp_path):
    system_path, galaxy_path, executable_path, _ = write_world(tmp_path)
    system_backup_path = tmp_path / "system-backup"
    galaxy_backup_path = tmp_path / "galaxy-backup"
    executable_backup_path = tmp_path / "executable-backup"
    original_system = system_path.read_bytes()
    original_galaxy = galaxy_path.read_bytes()
    original_executable = executable_path.read_bytes()
    system_backup_path.write_bytes(b"keep me")

    with pytest.raises(FileExistsError):
        patch_sre_reset_time(
            system_path,
            executable_path,
            galaxy_path=galaxy_path,
            timestamp=1_785_261_600,
            system_backup_path=system_backup_path,
            galaxy_backup_path=galaxy_backup_path,
            executable_backup_path=executable_backup_path,
        )

    assert system_path.read_bytes() == original_system
    assert galaxy_path.read_bytes() == original_galaxy
    assert executable_path.read_bytes() == original_executable
    assert system_backup_path.read_bytes() == b"keep me"
    assert not galaxy_backup_path.exists()
    assert not executable_backup_path.exists()
