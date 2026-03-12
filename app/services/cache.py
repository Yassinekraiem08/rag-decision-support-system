import time


class TTLCache:
    def __init__(self, ttl_seconds=300):
        self.ttl_seconds = ttl_seconds
        self.store = {}

    def get(self, key):
        if key not in self.store:
            return None

        value, expires_at = self.store[key]

        if time.time() > expires_at:
            del self.store[key]
            return None

        return value

    def set(self, key, value):
        self.store[key] = (value, time.time() + self.ttl_seconds)