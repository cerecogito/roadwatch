"""Verifiable inspection-list receipts: canonical manifest -> sha256 digest -> optional Memo anchor on Solana DEVNET.

Two distinct things, never conflated:
  * LOCAL verification: recompute the worklist checksum and the manifest digest from saved files (no network).
  * ON-CHAIN receipt: a Memo transaction that really exists on Devnet; verified by fetching it back by signature.
The memo contains only a version tag, the digest and the forecast week. No street-level records, complaint text, personal
data or credentials. Signing needs a Solana CLI keypair FILE (path in SOLANA_KEYPAIR_PATH, outside the repository) and an
explicit confirm=True; the private key is never read into logs or printed.
Memo program id (SPL Memo, from the Solana program docs): MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr
CLI:  python -I -m roadwatch.integrations.solana_receipt create|anchor|verify ...
"""
from __future__ import annotations

import argparse
import base64
import datetime as dt
import hashlib
import json
import sys
import time
from pathlib import Path

import requests

from . import config

MEMO_PROGRAM_ID = "MemoSq4gqABAXKb96qnH8TysNcWxMyWCqXgDLGmfcHr"
DEVNET_RPC = "https://api.devnet.solana.com"
FORMAT_VERSION = "roadwatch-receipt/1"
RECEIPT_DIR = config.ROOT / "outputs" / "integrations" / "receipts"
MAX_MEMO_BYTES = 566


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_file(p: Path) -> str | None:
    return sha256_bytes(p.read_bytes()) if p.exists() else None


def canonical_json(obj: dict) -> bytes:
    """Deterministic encoding: sorted keys, no whitespace, UTF-8."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def build_manifest(worklist_bytes: bytes, forecast_week: str, policy_id: str, capacity: str, model_version: str,
                   root: Path | None = None, generated_utc: str | None = None) -> dict:
    root = root or config.ROOT
    return {
        "format": FORMAT_VERSION,
        "forecast_week": forecast_week,
        "selection_policy": policy_id,
        "capacity": capacity,
        "model_version": model_version,                              # includes the model sha256 prefix and retrain date
        "selection_record_sha256": sha256_file(root / "data" / "processed" / "stage4" / "selection.json"),
        "source_manifest_sha256": sha256_file(root / "data" / "raw" / "MANIFEST.jsonl"),
        "worklist_sha256": sha256_bytes(worklist_bytes),
        "worklist_rows": worklist_bytes.count(b"\n") - 1,
        "generated_utc": generated_utc or dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }


def manifest_digest(manifest: dict) -> str:
    return sha256_bytes(canonical_json(manifest))


def memo_text(manifest: dict) -> str:
    m = f"roadwatch:v1:{manifest_digest(manifest)}:{manifest['forecast_week']}"
    assert len(m.encode()) <= MAX_MEMO_BYTES
    return m


def explorer_url(signature: str) -> str:
    return f"https://explorer.solana.com/tx/{signature}?cluster=devnet"


# ----------------------------------------------------------------- local verification
def verify_local(manifest_path: Path, worklist_path: Path) -> dict:
    """Recompute the worklist checksum and the digest; compare with the saved manifest (and receipt, if any)."""
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    wl_ok = sha256_file(worklist_path) == man["worklist_sha256"]
    digest = manifest_digest(man)
    out = {"worklist_matches_manifest": wl_ok, "manifest_digest": digest, "local_only": True,
           "on_chain": "not checked (a local checksum is not a blockchain receipt)"}
    rp = manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".receipt.json"))
    if rp.exists():
        rc = json.loads(rp.read_text(encoding="utf-8"))
        out["receipt_digest_matches"] = rc.get("digest") == digest
    return out


# ----------------------------------------------------------------- RPC
def _rpc(method: str, params: list, rpc_url: str, post=None, timeout: int = 20):
    r = (post or requests.post)(rpc_url, json={"jsonrpc": "2.0", "id": 1, "method": method, "params": params}, timeout=timeout)
    r.raise_for_status()
    j = r.json()
    if "error" in j:
        raise RuntimeError(f"RPC {method}: {j['error'].get('message', j['error'])}")
    return j["result"]


def check_devnet(rpc_url: str) -> None:
    if "devnet" not in rpc_url or "mainnet" in rpc_url:
        raise ValueError("RoadWatch receipts are Devnet-only; refusing a non-Devnet RPC URL.")


def build_memo_transaction(keypair_bytes: bytes, memo: str, recent_blockhash: str) -> tuple[str, str]:
    """Return (base64 transaction, payer pubkey). Signs locally with solders; nothing is sent."""
    from solders.hash import Hash
    from solders.instruction import AccountMeta, Instruction
    from solders.keypair import Keypair
    from solders.message import Message
    from solders.pubkey import Pubkey
    from solders.transaction import Transaction
    kp = Keypair.from_bytes(keypair_bytes)
    ix = Instruction(Pubkey.from_string(MEMO_PROGRAM_ID), memo.encode("utf-8"), [AccountMeta(kp.pubkey(), True, False)])
    bh = Hash.from_string(recent_blockhash)
    tx = Transaction([kp], Message.new_with_blockhash([ix], kp.pubkey(), bh), bh)
    return base64.b64encode(bytes(tx)).decode(), str(kp.pubkey())


def load_keypair_bytes(path: str | Path) -> bytes:
    arr = json.loads(Path(path).read_text(encoding="utf-8"))
    if not (isinstance(arr, list) and len(arr) == 64):
        raise ValueError("keypair file must be a Solana CLI JSON array of 64 bytes")
    return bytes(arr)


def anchor(manifest_path: Path, confirm: bool = False, rpc_url: str | None = None, post=None, sleep=time.sleep) -> dict:
    """Submit the manifest digest as a Memo on Devnet. Requires confirm=True and a funded Devnet keypair. Never raises:
    returns {"status": ...} with a precise reason when it cannot proceed."""
    rpc_url = rpc_url or config.secret("SOLANA_RPC_URL") or DEVNET_RPC
    try:
        check_devnet(rpc_url)
    except ValueError as e:
        return {"status": "refused", "reason": str(e)}
    if not confirm:
        return {"status": "not_submitted", "reason": "explicit confirmation required (--yes / the confirm button)"}
    kp_path = config.secret("SOLANA_KEYPAIR_PATH")
    if not kp_path or not Path(kp_path).exists():
        return {"status": "missing_wallet", "reason": "SOLANA_KEYPAIR_PATH is not set or the file does not exist"}
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    memo = memo_text(man)
    try:
        kb = load_keypair_bytes(kp_path)
        bh = _rpc("getLatestBlockhash", [{"commitment": "finalized"}], rpc_url, post)["value"]["blockhash"]
        tx_b64, payer = build_memo_transaction(kb, memo, bh)
        bal = _rpc("getBalance", [payer], rpc_url, post)["value"]
        if bal < 5000:
            return {"status": "insufficient_funds", "reason": f"payer {payer} has {bal} lamports on Devnet; fund it with a Devnet faucet"}
        sig = _rpc("sendTransaction", [tx_b64, {"encoding": "base64", "preflightCommitment": "confirmed"}], rpc_url, post)
        status = None
        for _ in range(30):
            st = _rpc("getSignatureStatuses", [[sig], {"searchTransactionHistory": True}], rpc_url, post)["value"][0]
            if st and st.get("err"):
                return {"status": "failed", "signature": sig, "reason": f"transaction error: {st['err']}"}
            if st and st.get("confirmationStatus") in ("confirmed", "finalized"):
                status = st["confirmationStatus"]
                break
            sleep(2)
        if status is None:
            return {"status": "unconfirmed", "signature": sig, "reason": "not confirmed within 60 s; check the explorer later"}
    except requests.HTTPError as e:
        return {"status": "rpc_error", "reason": f"HTTP {e.response.status_code} (rate limit?)"}
    except Exception as e:
        return {"status": "error", "reason": f"{type(e).__name__}: {e}"}
    rc = {"status": "confirmed", "signature": sig, "cluster": "devnet", "digest": manifest_digest(man), "memo": memo,
          "payer": payer, "confirmation": status, "explorer": explorer_url(sig),
          "anchored_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")}
    manifest_path.with_name(manifest_path.name.replace(".manifest.json", ".receipt.json")).write_text(json.dumps(rc, indent=1), encoding="utf-8")
    return rc


def verify_onchain(manifest_path: Path, signature: str, rpc_url: str | None = None, post=None) -> dict:
    """Fetch the transaction from Devnet and compare its Memo with the digest recomputed from the manifest."""
    rpc_url = rpc_url or DEVNET_RPC
    check_devnet(rpc_url)
    man = json.loads(manifest_path.read_text(encoding="utf-8"))
    expect = memo_text(man)
    tx = _rpc("getTransaction", [signature, {"encoding": "jsonParsed", "maxSupportedTransactionVersion": 0, "commitment": "confirmed"}],
              rpc_url, post)
    if tx is None:
        return {"on_chain_match": False, "reason": "transaction not found on Devnet"}
    memos = [i.get("parsed") for i in tx["transaction"]["message"]["instructions"] if i.get("programId") == MEMO_PROGRAM_ID]
    return {"on_chain_match": expect in memos, "expected_memo": expect, "found_memos": memos, "slot": tx.get("slot"),
            "failed": bool(tx.get("meta", {}).get("err")), "explorer": explorer_url(signature)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="roadwatch.integrations.solana_receipt")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("create", help="write a manifest for a saved worklist CSV (local only)")
    c.add_argument("--worklist", type=Path, required=True)
    c.add_argument("--week", required=True)
    c.add_argument("--policy", required=True)
    c.add_argument("--capacity", required=True)
    c.add_argument("--model-version", required=True)
    a = sub.add_parser("anchor", help="submit the digest as a Devnet Memo (needs SOLANA_KEYPAIR_PATH and --yes)")
    a.add_argument("--manifest", type=Path, required=True)
    a.add_argument("--yes", action="store_true")
    v = sub.add_parser("verify", help="recompute digests from saved files; add --signature to check Devnet")
    v.add_argument("--manifest", type=Path, required=True)
    v.add_argument("--worklist", type=Path, required=True)
    v.add_argument("--signature")
    args = ap.parse_args(argv)
    if args.cmd == "create":
        wl = args.worklist.read_bytes()
        man = build_manifest(wl, args.week, args.policy, args.capacity, args.model_version)
        RECEIPT_DIR.mkdir(parents=True, exist_ok=True)
        out = RECEIPT_DIR / f"{args.week}_{args.policy}_{args.capacity}.manifest.json"
        out.write_text(json.dumps(man, indent=1, sort_keys=True), encoding="utf-8")
        print(json.dumps({"manifest": str(out), "digest": manifest_digest(man), "memo": memo_text(man), "on_chain": "no (local only)"}, indent=1))
        return 0
    if args.cmd == "anchor":
        res = anchor(args.manifest, confirm=args.yes)
        print(json.dumps(res, indent=1))
        return 0 if res["status"] == "confirmed" else 1
    res = verify_local(args.manifest, args.worklist)
    if args.signature:
        res.update(verify_onchain(args.manifest, args.signature))
    print(json.dumps(res, indent=1))
    return 0 if res["worklist_matches_manifest"] and res.get("on_chain_match", True) else 1


if __name__ == "__main__":
    sys.exit(main())
