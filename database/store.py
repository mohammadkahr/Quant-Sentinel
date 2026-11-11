"""Append-only research event store; every connection applies WAL pragmas."""
import json
import sqlite3
import time
from contextlib import closing
from core.config import settings


class EventStore:
    def __init__(self, path=None):
        self.path = path or settings.DB_PATH
        with closing(self.connect()) as conn:
            with conn:
                conn.execute('CREATE TABLE IF NOT EXISTS events '
                             '(id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, epoch TEXT, '
                             'kind TEXT NOT NULL, timestamp REAL NOT NULL, payload TEXT NOT NULL)')
                conn.execute('CREATE INDEX IF NOT EXISTS events_run ON events(run_id, kind)')

    def connect(self):
        conn = sqlite3.connect(self.path, timeout=60)
        conn.execute('PRAGMA journal_mode=WAL')
        conn.execute('PRAGMA busy_timeout=60000')
        conn.execute('PRAGMA synchronous=NORMAL')
        return conn

    def append(self, run_id, epoch, kind, payload):
        serialized = json.dumps(payload, ensure_ascii=False, allow_nan=False, default=str)
        conn = self.connect()
        try:
            with conn:
                conn.execute('INSERT INTO events(run_id,epoch,kind,timestamp,payload) VALUES (?,?,?,?,?)',
                             (run_id, epoch, kind, time.time(), serialized))
        finally:
            conn.close()

    def read(self, run_id, kind):
        conn = self.connect()
        try:
            return [{**json.loads(p), "epoch": e, "event_timestamp": t} for e, t, p in conn.execute(
                'SELECT epoch,timestamp,payload FROM events WHERE run_id=? AND kind=? ORDER BY id', (run_id, kind))]
        finally:
            conn.close()
