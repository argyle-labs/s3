<p align="center">
  <img src="assets/icon-256.png" width="120" alt="s3" />
</p>

# s3

Registers an S3-compatible object-store `StorageBackend` — it mounts an S3 bucket into orca's storage domain as a userspace FUSE filesystem.

A first-party [orca](https://github.com/argyle-labs/orca) plugin (storage-backend).

This is a **backend/adapter** — it has no service of its own; it wires an existing system into orca.

---

## How it differs from nfs/smb

s3 joins the same `StorageBackend` trait for the single-pane view, but diverges on the mount mechanic. Where nfs/smb are **kernel mounts** driven through autofs, s3 is a `MountStyle::UserspaceProcess`:

- **No kernel mount, no autofs, no failover.** The bucket is exposed by a long-lived userspace FUSE process (`mountpoint-s3`, `rclone mount`, or `goofys`) supervised like a service/unit.
- **`usage` is an S3 API call**, not `statvfs` against a kernel mount table.
- **`StorageKind::Object`**, capabilities `List` / `Usage` / `Mount` / `Unmount`. No `RecoverStale` (there is no stale-NFS-handle equivalent) and no source failover.

## Credentials

S3 access-key / secret-key credentials are supplied through the storage domain's `SecretRef` mechanism — the secret is resolved by orca's secrets domain at mount time and is never written into a world-readable location or logged. Endpoint, region, and bucket are non-secret configuration carried in the mount spec.

## Configuration

Any S3-compatible service works the same way; there is no per-provider code. The mount spec carries:

- `source`: `s3://bucket[/prefix]`
- `endpoint=`: `https://` URL (required); `http://` only for loopback or with `allow_insecure`
- `region=` (required)
- `addressing=path|virtual-host` (default `path`)
- `storage_class=` (optional)
- `access_key=<secret name>`, plus the secret access key as the credential `SecretRef` (or `credential=<secret name>`); both are names resolved through orca secrets, never values, and must live in the `s3.` namespace (e.g. `s3.backup.access_key`) so a mount spec cannot pull another plugin's secret
- extra tool flags: per-tool allowlist only; TLS-off, `allow-other` and mode/umask flags are rejected. Mounts are always file `0600` / dir `0700`.

With rclone, all of this is passed as `RCLONE_S3_*` environment (`provider=Other`, `RCLONE_CONFIG=/dev/null`), so no rclone config is read or written.

## Run it without orca

There's nothing to deploy: this plugin drives a FUSE mount tool you already run — one of [mountpoint-s3](https://github.com/awslabs/mountpoint-s3), [rclone](https://rclone.org/), or [goofys](https://github.com/kahing/goofys). Install/configure that directly, then register this plugin with orca.

## With orca

orca drives this plugin through its generic storage surface — the backend contributes the object mount as a share, reports usage via the S3 API, and mounts/unmounts the supervised FUSE process.

## Layout

- `src/` — the plugin (pure Rust): the `S3Backend` `StorageBackend` descriptor + `validate_spec` / `mount` / `unmount` / `usage` / `list_shares`.
- `assets/` — plugin icon.
