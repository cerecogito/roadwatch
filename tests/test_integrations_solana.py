"""Solana receipt tests. RPC is a MOCK; nothing is submitted to any network. A throwaway keypair is generated in memory."""
import json
import types

import pytest

from roadwatch.integrations import config
from roadwatch.integrations import solana_receipt as sr

WL = b"rank,block_id\n1,52878\n2,48264\n"


@pytest.fixture
def man(tmp_path):
    m = sr.build_manifest(WL, "2026-03-16", "strict_topk", "1pct", "primary__LGBM | retrain 2026-03-02 | model sha256 1e8448d92f01",
                          root=tmp_path, generated_utc="2026-10-25T00:00:00Z")
    p = tmp_path / "x.manifest.json"
    p.write_text(json.dumps(m, sort_keys=True), encoding="utf-8")
    (tmp_path / "wl.csv").write_bytes(WL)
    return p


def test_manifest_is_deterministic_and_minimal(tmp_path):
    a = sr.build_manifest(WL, "2026-03-16", "p", "1pct", "mv", root=tmp_path, generated_utc="t")
    b = sr.build_manifest(WL, "2026-03-16", "p", "1pct", "mv", root=tmp_path, generated_utc="t")
    assert sr.manifest_digest(a) == sr.manifest_digest(b) and a["worklist_rows"] == 2
    assert set(a) == {"format", "forecast_week", "selection_policy", "capacity", "model_version", "selection_record_sha256",
                      "source_manifest_sha256", "worklist_sha256", "worklist_rows", "generated_utc"}
    assert sr.manifest_digest(a) != sr.manifest_digest({**a, "worklist_sha256": "0" * 64})
    memo = sr.memo_text(a)
    assert memo.startswith("roadwatch:v1:") and len(memo.encode()) <= 566 and "AVE" not in memo


def test_local_verification_detects_tampering(man, tmp_path):
    ok = sr.verify_local(man, tmp_path / "wl.csv")
    assert ok["worklist_matches_manifest"] and ok["local_only"] and "not a blockchain receipt" in ok["on_chain"]
    (tmp_path / "wl.csv").write_bytes(WL + b"3,1\n")
    assert not sr.verify_local(man, tmp_path / "wl.csv")["worklist_matches_manifest"]


def test_memo_transaction_builds_offline_with_throwaway_key():
    from solders.keypair import Keypair
    from solders.transaction import Transaction
    kp = Keypair()
    b64, payer = sr.build_memo_transaction(bytes(kp), "roadwatch:v1:abc:2026-03-16", "11111111111111111111111111111111")
    import base64
    tx = Transaction.from_bytes(base64.b64decode(b64))
    ix = tx.message.instructions[0]
    assert str(tx.message.account_keys[ix.program_id_index]) == sr.MEMO_PROGRAM_ID and ix.data == b"roadwatch:v1:abc:2026-03-16" and payer == str(kp.pubkey())


def test_anchor_refuses_mainnet_and_requires_confirmation_and_wallet(man, monkeypatch):
    assert sr.anchor(man, confirm=True, rpc_url="https://api.mainnet-beta.solana.com")["status"] == "refused"
    assert sr.anchor(man, confirm=False)["status"] == "not_submitted"
    monkeypatch.delenv("SOLANA_KEYPAIR_PATH", raising=False)
    assert sr.anchor(man, confirm=True)["status"] == "missing_wallet"


def _rpc_mock(calls, balance=1_000_000_000, status="confirmed", http_error=False):
    def post(url, json=None, timeout=None):
        calls.append(json["method"])
        if http_error:
            r = types.SimpleNamespace(raise_for_status=lambda: (_ for _ in ()).throw(
                __import__("requests").HTTPError(response=types.SimpleNamespace(status_code=429))))
            return r
        res = {"getLatestBlockhash": {"value": {"blockhash": "11111111111111111111111111111111"}}, "getBalance": {"value": balance},
               "sendTransaction": "5" * 88, "getSignatureStatuses": {"value": [{"confirmationStatus": status, "err": None}]}}[json["method"]]
        return types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"result": res})
    return post


def test_anchor_flow_with_mock_rpc_writes_receipt_only_on_confirmation(man, tmp_path, monkeypatch):
    from solders.keypair import Keypair
    kpf = tmp_path / "id.json"
    kpf.write_text(json.dumps(list(bytes(Keypair()))), encoding="utf-8")
    monkeypatch.setenv("SOLANA_KEYPAIR_PATH", str(kpf))
    calls = []
    rc = sr.anchor(man, confirm=True, post=_rpc_mock(calls), sleep=lambda s: None)
    assert rc["status"] == "confirmed" and rc["explorer"].endswith("?cluster=devnet") and "sendTransaction" in calls
    assert (tmp_path / "x.receipt.json").exists() and str(kpf) not in json.dumps(rc)
    (tmp_path / "x.receipt.json").unlink()
    c2 = []
    assert sr.anchor(man, confirm=True, post=_rpc_mock(c2, balance=0), sleep=lambda s: None)["status"] == "insufficient_funds" and "sendTransaction" not in c2
    assert sr.anchor(man, confirm=True, post=_rpc_mock([], http_error=True))["status"] == "rpc_error"
    assert sr.anchor(man, confirm=True, post=_rpc_mock([], status="processed"), sleep=lambda s: None)["status"] == "unconfirmed"
    assert not (tmp_path / "x.receipt.json").exists()                                 # nothing recorded unless confirmed


def test_verify_onchain_compares_memo(man):
    expect = sr.memo_text(json.loads(man.read_text()))
    tx = {"slot": 5, "meta": {"err": None}, "transaction": {"message": {"instructions": [{"programId": sr.MEMO_PROGRAM_ID, "parsed": expect}]}}}
    post = lambda url, json=None, timeout=None: types.SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"result": tx})
    assert sr.verify_onchain(man, "sig", post=post)["on_chain_match"]
    tx["transaction"]["message"]["instructions"][0]["parsed"] = "roadwatch:v1:other:2026-03-16"
    assert not sr.verify_onchain(man, "sig", post=post)["on_chain_match"]
    with pytest.raises(ValueError):
        sr.verify_onchain(man, "sig", rpc_url="https://api.mainnet-beta.solana.com")
