import pytest

from maskcast.mask import (
    CHUNK_SIZE,
    PROTOCOL_KEY,
    data_packets,
    datcp_command,
    dats_command,
    pad16,
    parse_notify,
    play_command,
)


def test_key_is_one_aes_block():
    assert len(PROTOCOL_KEY) == 16


def test_pad16():
    assert pad16(b"\x05DATCP") == b"\x05DATCP" + b"\x00" * 10
    with pytest.raises(ValueError):
        pad16(b"x" * 17)


def test_commands_fit_in_one_block():
    for cmd in (dats_command(8100), datcp_command(), play_command()):
        assert len(pad16(cmd)) == 16
        assert cmd[0] == len(cmd) - 1  # first byte is the payload length


def test_dats_encodes_length_big_endian_twice():
    cmd = dats_command(0x1FA4)
    assert cmd[1:5] == b"DATS"
    assert cmd[5:7] == cmd[7:9] == b"\x1f\xa4"


def test_data_packets_chunking():
    data = bytes(range(256)) * 2  # 512 bytes
    pkts = list(data_packets(data))
    assert len(pkts) == -(-len(data) // CHUNK_SIZE)
    assert [p[1] for p in pkts] == list(range(len(pkts)))
    assert all(p[0] == len(p) - 1 for p in pkts)
    assert b"".join(p[2:] for p in pkts) == data


def test_parse_notify():
    assert parse_notify(bytes([6]) + b"DATSOK" + b"\x00" * 9) == "DATSOK"
