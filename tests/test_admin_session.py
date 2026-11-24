import base64
import json
import shutil
import subprocess
import unittest
import time

from services.admin_session import (
    _EPOCH_CACHE,
    _EPOCH_CACHE_LOCK,
    build_admin_session_clear_script,
    build_admin_session_store_script,
    clear_admin_session_epoch_cache,
    create_admin_session_token,
    get_admin_session_epoch,
    verify_admin_session_token,
)


class _Snapshot:
    def __init__(self, data):
        self.data = data

    def to_dict(self):
        return self.data


class _Document:
    def __init__(self, client):
        self.client = client

    def get(self, *, timeout, retry):
        assert timeout > 0
        assert retry is None
        self.client.reads += 1
        if self.client.unavailable:
            raise OSError("unavailable")
        return _Snapshot({"epoch": self.client.epoch})


class _Collection:
    def __init__(self, client):
        self.client = client

    def document(self, _name):
        return _Document(self.client)


class _Client:
    def __init__(self, epoch=0, unavailable=False):
        self.epoch = epoch
        self.unavailable = unavailable
        self.reads = 0

    def collection(self, _name):
        return _Collection(self)


class AdminSessionTests(unittest.TestCase):
    def setUp(self):
        clear_admin_session_epoch_cache()
        self.secrets = {"ENCRYPTION_KEY": "test-encryption-key"}
        self.uid = "abcdef0123456789"

    def test_valid_token_binds_uid_expiry_and_epoch(self):
        token = create_admin_session_token(
            self.uid, 4, self.secrets, now=1_000
        )

        self.assertTrue(
            verify_admin_session_token(
                token, self.uid, 4, self.secrets, now=1_001
            )
        )
        self.assertFalse(
            verify_admin_session_token(
                token, self.uid, 4, self.secrets, now=1_000 + 30 * 24 * 60 * 60
            )
        )
        self.assertFalse(
            verify_admin_session_token(
                token, "fedcba9876543210", 4, self.secrets, now=1_001
            )
        )
        self.assertFalse(
            verify_admin_session_token(
                token, self.uid, 5, self.secrets, now=1_001
            )
        )

    def test_damaged_token_and_missing_signing_secret_are_rejected(self):
        token = create_admin_session_token(self.uid, 0, self.secrets, now=1_000)
        decoded = base64.urlsafe_b64decode(token + "=" * (-len(token) % 4))
        token_uid, expiry, signature = decoded.decode("ascii").split("|")
        damaged_signature = ("0" if signature[0] != "0" else "1") + signature[1:]
        damaged_token = base64.urlsafe_b64encode(
            f"{token_uid}|{expiry}|{damaged_signature}".encode("ascii")
        ).rstrip(b"=").decode("ascii")

        self.assertFalse(
            verify_admin_session_token(
                damaged_token,
                self.uid,
                0,
                self.secrets,
                now=1_001,
            )
        )
        self.assertFalse(
            verify_admin_session_token(token, self.uid, 0, {}, now=1_001)
        )

    def test_browser_scripts_store_and_clear_both_session_copies(self):
        token = create_admin_session_token(self.uid, 0, self.secrets, now=1_000)
        store_script = build_admin_session_store_script(token)
        clear_script = build_admin_session_clear_script()

        self.assertIn("SameSite=Lax", store_script)
        self.assertIn("localStorage.setItem", store_script)
        self.assertIn(token, store_script)
        self.assertIn("Max-Age=0", clear_script)
        self.assertIn("localStorage.removeItem", clear_script)

    @unittest.skipUnless(shutil.which("node"), "Node.js is not installed")
    def test_browser_storage_scripts_report_success_and_partial_failure(self):
        harness = r"""
const script = process.argv[1];
const scenario = process.argv[2];
let cookie = "jh_adm=old-token";
const storage = new Map();
storage.set("jh_adm", "old-token");
globalThis.location = {protocol: "https:"};
globalThis.document = {};
Object.defineProperty(document, "cookie", {
  get() { return cookie; },
  set(value) {
    if (scenario === "cookie-failure") return;
    cookie = value.includes("Max-Age=0") ? "" : value.split(";")[0];
  }
});
globalThis.localStorage = {
  setItem(key, value) { storage.set(key, value); },
  removeItem(key) {
    if (scenario === "storage-failure") throw new Error("storage unavailable");
    storage.delete(key);
  }
};
console.log(JSON.stringify({result: eval(script), cookie, stored: storage.size}));
"""
        token = create_admin_session_token(self.uid, 0, self.secrets, now=1_000)
        cases = (
            (build_admin_session_store_script(token), "normal", "stored", 1, 1),
            (build_admin_session_clear_script(), "normal", "cleared", 0, 0),
            (
                build_admin_session_clear_script(),
                "storage-failure",
                "cookie-cleared",
                0,
                1,
            ),
            (
                build_admin_session_clear_script(),
                "cookie-failure",
                "error",
                1,
                1,
            ),
        )
        for script, scenario, expected, cookie_present, stored in cases:
            with self.subTest(scenario=scenario, expected=expected):
                completed = subprocess.run(
                    [shutil.which("node"), "-e", harness, script, scenario],
                    check=True,
                    capture_output=True,
                    text=True,
                )
                result = json.loads(completed.stdout)
                self.assertEqual(result["result"], expected)
                self.assertEqual(bool(result["cookie"]), bool(cookie_present))
                self.assertEqual(result["stored"], stored)

    def test_epoch_read_is_cached_and_refreshed_after_five_minutes(self):
        client = _Client(epoch=2)
        self.assertEqual(get_admin_session_epoch(client), 2)
        client.epoch = 3
        self.assertEqual(get_admin_session_epoch(client), 2)
        self.assertEqual(client.reads, 1)

        self.assertEqual(get_admin_session_epoch(client, force_refresh=True), 3)
        self.assertEqual(client.reads, 2)

    def test_warm_epoch_background_check_does_not_read_firestore(self):
        client = _Client(epoch=2)
        self.assertEqual(get_admin_session_epoch(client), 2)
        client.epoch = 3

        self.assertEqual(
            get_admin_session_epoch(client, background_refresh=True),
            2,
        )
        self.assertEqual(client.reads, 1)

    def test_unavailable_firestore_fails_closed(self):
        self.assertIsNone(get_admin_session_epoch(None))
        client = _Client(unavailable=True)

        self.assertIsNone(get_admin_session_epoch(client))
        self.assertIsNone(get_admin_session_epoch(client))
        self.assertEqual(client.reads, 1)

    def test_expired_epoch_cache_is_refreshed(self):
        client = _Client(epoch=6)
        with _EPOCH_CACHE_LOCK:
            _EPOCH_CACHE[id(client)] = (time.monotonic() - 301.0, 5)
        self.assertEqual(get_admin_session_epoch(client), 6)
        self.assertEqual(client.reads, 1)

    def test_firestore_unavailable_epoch_cannot_validate_token(self):
        token = create_admin_session_token(self.uid, 0, self.secrets, now=1_000)
        epoch = get_admin_session_epoch(None)

        self.assertFalse(
            verify_admin_session_token(
                token, self.uid, epoch, self.secrets, now=1_001
            )
        )

    def test_token_persists_after_logout(self):
        """Verify that token remains valid after 'logout' (admin_authenticated=False only)."""
        token = create_admin_session_token(self.uid, 0, self.secrets, now=1_000)
        # Simulate logout: only admin_authenticated is set to False
        # Token in cookie/localStorage remains
        self.assertTrue(
            verify_admin_session_token(
                token, self.uid, 0, self.secrets, now=1_001
            )
        )
        # Even after "re-login" (just setting admin_authenticated=True), token still valid
        self.assertTrue(
            verify_admin_session_token(
                token, self.uid, 0, self.secrets, now=1_001
            )
        )

    def test_revoke_all_sessions_invalidates_token(self):
        """Verify that epoch increment (revoke all sessions) invalidates existing tokens."""
        token = create_admin_session_token(self.uid, 5, self.secrets, now=1_000)
        # Token valid at epoch 5
        self.assertTrue(
            verify_admin_session_token(
                token, self.uid, 5, self.secrets, now=1_001
            )
        )
        # After revoke, epoch becomes 6 -> token invalid
        self.assertFalse(
            verify_admin_session_token(
                token, self.uid, 6, self.secrets, now=1_001
            )
        )
        # New token at epoch 6 works
        new_token = create_admin_session_token(self.uid, 6, self.secrets, now=1_000)
        self.assertTrue(
            verify_admin_session_token(
                new_token, self.uid, 6, self.secrets, now=1_001
            )
        )


if __name__ == "__main__":
    unittest.main()
