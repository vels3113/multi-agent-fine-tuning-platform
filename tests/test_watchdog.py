import struct
import time
import pytest
from multiprocessing.shared_memory import SharedMemory


STRUCT_FMT = ">dddii8x"
STRUCT_SIZE = struct.calcsize(STRUCT_FMT)


def test_struct_size():
    assert STRUCT_SIZE == 40


def test_watchdog_starts_and_writes():
    from watchdog import Watchdog
    wdog = Watchdog(shm_name="test_wdog_basic", interval=0.1)
    wdog.start(initial_workers=2, initial_batch=0)
    time.sleep(0.3)
    shm = SharedMemory(name="test_wdog_basic", create=False)
    data = bytes(shm.buf[:STRUCT_SIZE])
    shm.close()
    wdog.stop()
    ts, alloc, reserved, workers, batch = struct.unpack(STRUCT_FMT, data)
    assert ts > 0
    assert workers == 2
    assert batch == 0


def test_watchdog_batch_index_updates():
    from watchdog import Watchdog
    wdog = Watchdog(shm_name="test_wdog_batch", interval=0.05)
    wdog.start(initial_workers=0, initial_batch=0)
    time.sleep(0.1)
    wdog.update_batch(42)
    time.sleep(0.15)
    shm = SharedMemory(name="test_wdog_batch", create=False)
    data = bytes(shm.buf[:STRUCT_SIZE])
    shm.close()
    wdog.stop()
    _, _, _, _, batch = struct.unpack(STRUCT_FMT, data)
    assert batch == 42


def test_watchdog_stop_cleans_up():
    from watchdog import Watchdog
    wdog = Watchdog(shm_name="test_wdog_cleanup", interval=0.05)
    wdog.start(initial_workers=0, initial_batch=0)
    wdog.stop()
    with pytest.raises(FileNotFoundError):
        SharedMemory(name="test_wdog_cleanup", create=False)


def test_read_snapshot_parses_correctly():
    from watchdog import Watchdog, read_snapshot
    wdog = Watchdog(shm_name="test_wdog_read", interval=0.05)
    wdog.start(initial_workers=3, initial_batch=7)
    time.sleep(0.15)
    snap = read_snapshot("test_wdog_read")
    wdog.stop()
    assert snap["workers"] == 3
    assert snap["batch"] == 7
    assert snap["watchdog_ts"] > 0
