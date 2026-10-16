//! S3-compatible object-store integration. A *thin* domain adapter: it owns only
//! what is S3-specific — the bucket/endpoint/region/credentials config grammar,
//! selecting and launching a userspace FUSE mount tool (`mountpoint-s3`,
//! `rclone mount`, `goofys`), and tearing that mount down — and reaches
//! everything generic (the `storage` domain model, the serve loop, process
//! spawning) through the shared `plugin_toolkit` surface.
//!
//! ## How s3 diverges from nfs/smb
//!
//! nfs and smb are **kernel mounts** realized through core's autofs applier
//! ([`MountStyle::KernelMount`]). s3 is a [`MountStyle::UserspaceProcess`]: the
//! bucket is exposed by a long-lived userspace FUSE daemon that this backend
//! launches and supervises, *not* a kernel mount table entry. There is no
//! autofs, no source failover, and no stale-NFS-handle recovery
//! (`RecoverStale`) — an object store has none of those semantics. `usage` is an
//! S3 API call rather than `statvfs`, and [`StorageKind`] is [`StorageKind::Object`].
//!
//! ## The `mount()` entry point
//!
//! Per the locked Phase 1 decision, the trait's `mount(id, target)` method — a
//! vestigial no-op for kernel-mount backends (autofs owns their mechanics) — is
//! the home for a [`MountStyle::UserspaceProcess`] backend's helper-process path.
//! s3 implements it: it launches the configured FUSE tool at `target`.
//!
//! ## Credential safety
//!
//! S3 access-key / secret-key material arrives as a [`SecretRef`] the secrets
//! domain resolves; the resolved secret is passed to the FUSE tool through its
//! environment / credentials file, never rendered into the world-readable option
//! string or logged. `validate_spec` rejects a spec that declares inline secret
//! material in the option string.

use std::path::Path;
use std::sync::Arc;

use plugin_toolkit::orca_async;
use plugin_toolkit::path::which;
use plugin_toolkit::prelude::*;
use plugin_toolkit::process::Command;
use plugin_toolkit::storage::{
    Capability, MountOutcome, MountSpec, MountStyle, NormalizedSpec, OptionSet, SecretRef, Share,
    StorageBackend, StorageError, StorageKind,
};

/// S3 tool / transport errors. Expressed through the orca-native
/// `#[orca_error]` abstraction — the plugin names no error crate; the macro
/// emits `Display` + `std::error::Error` (with the `Io` source chain) + the
/// `From<std::io::Error>` conversion.
#[orca_error]
pub enum S3Error {
    #[orca(display = "required FUSE mount tool not found on PATH: {0}")]
    MissingTool(String),
    #[orca(display = "s3 mount tool failed: {tool} (exit {code:?}): {stderr}")]
    ToolFailed {
        tool: String,
        code: Option<i32>,
        stderr: String,
    },
    #[orca(display = "io: {0}", from)]
    Io(std::io::Error),
    #[orca(display = "unsupported on this platform")]
    Unsupported,
}

/// The userspace FUSE mount tool that realizes the object mount. Each maps a
/// bucket onto a local mountpoint as a long-lived process; the backend picks one
/// per the declared `tool=` option (defaulting to [`MountTool::MountpointS3`]).
#[orca_struct]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[orca(rename_all = "kebab-case")]
pub enum MountTool {
    /// AWS's `mountpoint-s3` (binary: `mount-s3`).
    MountpointS3,
    /// `rclone mount` against an S3 remote.
    Rclone,
    /// `goofys`.
    Goofys,
}

impl MountTool {
    /// Parse the declared `tool=` option value. Unknown values are rejected at
    /// declare time so a typo surfaces before a mount is ever attempted.
    fn parse(s: &str) -> Result<Self, StorageError> {
        match s {
            "mountpoint-s3" | "mount-s3" => Ok(MountTool::MountpointS3),
            "rclone" => Ok(MountTool::Rclone),
            "goofys" => Ok(MountTool::Goofys),
            other => Err(StorageError::Other(format!(
                "s3: unsupported mount tool `{other}` (accepted: mountpoint-s3, rclone, goofys)"
            ))),
        }
    }

    /// The executable this tool invokes, checked against `PATH` before a mount.
    fn binary(self) -> &'static str {
        match self {
            MountTool::MountpointS3 => "mount-s3",
            MountTool::Rclone => "rclone",
            MountTool::Goofys => "goofys",
        }
    }

    fn as_option_value(self) -> &'static str {
        match self {
            MountTool::MountpointS3 => "mountpoint-s3",
            MountTool::Rclone => "rclone",
            MountTool::Goofys => "goofys",
        }
    }
}

/// How bucket requests are addressed. Every S3-compatible service accepts one or
/// both; which is configuration, never inferred from the endpoint.
#[orca_struct]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
#[orca(rename_all = "kebab-case")]
pub enum Addressing {
    /// `https://endpoint/bucket/key`.
    #[default]
    Path,
    /// `https://bucket.endpoint/key`.
    VirtualHost,
}

impl Addressing {
    fn parse(s: &str) -> Result<Self, StorageError> {
        match s {
            "path" => Ok(Addressing::Path),
            "virtual-host" => Ok(Addressing::VirtualHost),
            other => Err(StorageError::Other(format!(
                "s3: unsupported addressing `{other}` (accepted: path, virtual-host)"
            ))),
        }
    }

    fn as_option_value(self) -> &'static str {
        match self {
            Addressing::Path => "path",
            Addressing::VirtualHost => "virtual-host",
        }
    }
}

/// The validated S3 mount configuration parsed out of a [`MountSpec`]. The
/// `storage` contract has no typed `OptionSet::S3` variant, so the backend owns
/// this grammar itself and normalizes the raw option string into it; the
/// normalized spec carries the options back through as [`OptionSet::Raw`] (the
/// canonicalized string) so the wire contract is unchanged.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3Config {
    /// The bucket (S3 `source`, e.g. `s3://my-bucket/prefix` or `my-bucket`).
    pub bucket: String,
    /// Key prefix inside the bucket, without leading/trailing `/`; empty for the root.
    pub prefix: String,
    pub endpoint: String,
    pub allow_insecure: bool,
    /// Region (`us-east-1`, `auto`, …).
    pub region: String,
    pub addressing: Addressing,
    /// Storage class for written objects; the provider default when unset.
    pub storage_class: Option<String>,
    /// Userspace FUSE tool that realizes the mount.
    pub tool: MountTool,
    /// Access-key id (non-secret half of the credential pair).
    pub access_key_id: String,
    /// Secret access key, resolved by the secrets domain from a [`SecretRef`].
    /// Never rendered into the option string or logged.
    pub secret_access_key: SecretRef,
    pub extra: Vec<String>,
}

/// Normalize a bucket source into a bare bucket name, accepting either the bare
/// name or the `s3://bucket` URL form. Rejects an empty bucket.
fn normalize_bucket(source: &str) -> Result<(String, String), StorageError> {
    let s = source.trim();
    let s = s.strip_prefix("s3://").unwrap_or(s);
    let (bucket, prefix) = s.split_once('/').unwrap_or((s, ""));
    let bucket = bucket.trim();
    if bucket.is_empty() {
        return Err(StorageError::Other("s3 bucket is empty".into()));
    }
    let prefix: Vec<&str> = prefix.split('/').filter(|c| !c.is_empty()).collect();
    if prefix.iter().any(|c| *c == "." || *c == "..") {
        return Err(StorageError::Other(
            "s3 prefix must not contain . or ..".into(),
        ));
    }
    Ok((bucket.to_string(), prefix.join("/")))
}

/// Parse + validate an S3 mount spec into a typed [`S3Config`], enforcing the
/// backend's grammar: bucket (from `source`), and `endpoint=` / `region=` /
/// `access_key=` from the option string are all required; the secret access key
/// arrives as the spec's `credential` [`SecretRef`]. A `tool=` selects the FUSE
/// helper (default `mountpoint-s3`). Inline secret material in the option string
/// is rejected — the secret is a `SecretRef`, never a plaintext option.
pub fn validate_s3_config(spec: &MountSpec) -> Result<S3Config, StorageError> {
    let (bucket, prefix) = normalize_bucket(&spec.source)?;

    let mut endpoint: Option<String> = None;
    let mut region: Option<String> = None;
    let mut addressing = Addressing::default();
    let mut storage_class: Option<String> = None;
    let mut access_key_id: Option<String> = None;
    let mut tool = MountTool::MountpointS3;
    let mut allow_insecure = false;
    let mut extra: Vec<String> = Vec::new();

    let raw = spec.options.as_deref().unwrap_or("");
    for opt in raw.split(',').map(str::trim).filter(|o| !o.is_empty()) {
        let (key, value) = match opt.split_once('=') {
            Some((k, v)) => (k.trim(), Some(v.trim().to_string())),
            None => (opt, None),
        };
        match (key, value) {
            ("endpoint", Some(v)) => endpoint = Some(v),
            ("region", Some(v)) => region = Some(v),
            ("addressing", Some(v)) => addressing = Addressing::parse(&v)?,
            ("storage_class", Some(v)) if !v.is_empty() => storage_class = Some(v),
            ("access_key" | "access_key_id", Some(v)) => access_key_id = Some(v),
            ("tool", Some(v)) => tool = MountTool::parse(&v)?,
            ("allow_insecure", None) => allow_insecure = true,
            ("allow_insecure", Some(v)) => allow_insecure = v == "true",
            ("secret_key" | "secret_access_key" | "password", _) => {
                return Err(StorageError::Other(
                    "s3: the secret access key must be supplied as a credential SecretRef, \
                     never as an inline option"
                        .to_string(),
                ));
            }
            _ => extra.push(opt.to_string()),
        }
    }

    let endpoint = endpoint.ok_or_else(|| {
        StorageError::Other("s3: `endpoint=` is required in the mount options".into())
    })?;
    match endpoint.strip_prefix("http://") {
        Some(rest) if !allow_insecure && !is_loopback(rest) => {
            return Err(StorageError::Other(format!(
                "s3: `endpoint={endpoint}` must be https:// (http:// only for loopback or allow_insecure)"
            )));
        }
        None if !endpoint.starts_with("https://") => {
            return Err(StorageError::Other(format!(
                "s3: `endpoint={endpoint}` must be an https:// URL"
            )));
        }
        _ => {}
    }
    let region = region.filter(|r| !r.is_empty()).ok_or_else(|| {
        StorageError::Other("s3: `region=` is required in the mount options".into())
    })?;
    let access_key_id = access_key_id.ok_or_else(|| {
        StorageError::Other("s3: `access_key=` is required in the mount options".into())
    })?;
    for e in &extra {
        check_extra(tool, e)?;
    }
    let secret_access_key = spec.credential.clone().ok_or_else(|| {
        StorageError::Other("s3: a credential SecretRef (the secret access key) is required".into())
    })?;

    Ok(S3Config {
        bucket,
        prefix,
        endpoint,
        allow_insecure,
        region,
        addressing,
        storage_class,
        tool,
        access_key_id,
        secret_access_key,
        extra,
    })
}

fn is_loopback(rest: &str) -> bool {
    let authority = rest.split(['/', '?', '#']).next().unwrap_or("");
    if authority.contains(['@', '\\']) || authority.chars().any(char::is_whitespace) {
        return false;
    }
    let (host, port) = match authority.strip_prefix('[') {
        Some(v6) => match v6.split_once(']') {
            Some((h, p)) => (h, p),
            None => return false,
        },
        None => match authority.split_once(':') {
            Some((h, p)) => (h, &authority[h.len()..][..p.len() + 1]),
            None => (authority, ""),
        },
    };
    if !(port.is_empty()
        || port
            .strip_prefix(':')
            .is_some_and(|p| !p.is_empty() && p.bytes().all(|b| b.is_ascii_digit())))
    {
        return false;
    }
    host.eq_ignore_ascii_case("localhost")
        || host
            .parse::<std::net::IpAddr>()
            .is_ok_and(|ip| ip.is_loopback())
}

const DENIED_FLAGS: &[&str] = &[
    "no-check-certificate",
    "no-verify-ssl",
    "ca-bundle",
    "insecure",
    "allow-other",
    "allow_other",
    "umask",
    "file-mode",
    "dir-mode",
    "file-perms",
    "dir-perms",
];

fn allowed_flags(tool: MountTool) -> &'static [&'static str] {
    match tool {
        MountTool::MountpointS3 => &[
            "read-only",
            "allow-delete",
            "allow-overwrite",
            "auto-unmount",
            "uid",
            "gid",
            "cache",
            "max-cache-size",
            "metadata-ttl",
            "part-size",
            "max-threads",
        ],
        MountTool::Rclone => &[
            "read-only",
            "uid",
            "gid",
            "vfs-cache-mode",
            "vfs-cache-max-size",
            "vfs-cache-max-age",
            "dir-cache-time",
            "buffer-size",
            "transfers",
        ],
        MountTool::Goofys => &[
            "uid",
            "gid",
            "stat-cache-ttl",
            "type-cache-ttl",
            "cheap",
            "sse",
        ],
    }
}

fn flag_name(opt: &str) -> &str {
    let opt = opt.trim_start_matches('-');
    opt.split_once('=').map_or(opt, |(k, _)| k)
}

fn check_extra(tool: MountTool, opt: &str) -> Result<(), StorageError> {
    let name = flag_name(opt);
    if DENIED_FLAGS.contains(&name) || !allowed_flags(tool).contains(&name) {
        return Err(StorageError::Other(format!(
            "s3: option `{opt}` is not allowed for {}",
            tool.as_option_value()
        )));
    }
    Ok(())
}

fn extra_args(cfg: &S3Config) -> impl Iterator<Item = String> + '_ {
    cfg.extra
        .iter()
        .map(|e| format!("--{}", e.trim_start_matches('-')))
}

/// Render a validated [`S3Config`] back into the canonical, comma-joined option
/// string. CREDENTIAL SAFETY: the secret access key ([`SecretRef`]) is NEVER
/// rendered — only the non-secret `endpoint` / `region` / `access_key` / `tool`
/// and passthrough `extra` appear. The access-key *id* is non-secret (it is not
/// a credential on its own) and is safe to render.
pub fn render_s3_options(cfg: &S3Config) -> String {
    let mut parts = vec![
        format!("endpoint={}", cfg.endpoint),
        format!("region={}", cfg.region),
        format!("addressing={}", cfg.addressing.as_option_value()),
        format!("access_key={}", cfg.access_key_id),
        format!("tool={}", cfg.tool.as_option_value()),
    ];
    if let Some(sc) = &cfg.storage_class {
        parts.push(format!("storage_class={sc}"));
    }
    if cfg.allow_insecure {
        parts.push("allow_insecure=true".to_string());
    }
    parts.extend(cfg.extra.iter().cloned());
    parts.join(",")
}

/// rclone's `s3` backend configured entirely from env, so no rclone config file
/// is read or written. `provider=Other` keeps rclone generic: endpoint, region and
/// addressing come only from `cfg`. `secret` is the resolved secret access key.
pub fn rclone_env(cfg: &S3Config, secret: &str) -> Vec<(String, String)> {
    let mut env = vec![
        ("RCLONE_CONFIG".to_string(), "/dev/null".to_string()),
        ("RCLONE_S3_PROVIDER".to_string(), "Other".to_string()),
        ("RCLONE_S3_ENV_AUTH".to_string(), "false".to_string()),
        ("RCLONE_S3_ENDPOINT".to_string(), cfg.endpoint.clone()),
        ("RCLONE_S3_REGION".to_string(), cfg.region.clone()),
        (
            "RCLONE_S3_FORCE_PATH_STYLE".to_string(),
            (cfg.addressing == Addressing::Path).to_string(),
        ),
        (
            "RCLONE_S3_ACCESS_KEY_ID".to_string(),
            cfg.access_key_id.clone(),
        ),
        (
            "RCLONE_S3_SECRET_ACCESS_KEY".to_string(),
            secret.to_string(),
        ),
    ];
    if let Some(sc) = &cfg.storage_class {
        env.push(("RCLONE_S3_STORAGE_CLASS".to_string(), sc.clone()));
    }
    env
}

/// `:s3:bucket[/prefix]`, the on-the-fly remote `rclone_env` configures.
pub fn rclone_remote(cfg: &S3Config) -> String {
    if cfg.prefix.is_empty() {
        format!(":s3:{}", cfg.bucket)
    } else {
        format!(":s3:{}/{}", cfg.bucket, cfg.prefix)
    }
}

/// rclone `crypt` layered over the `:s3:` remote, configured from env only, so
/// objects are encrypted client-side before they reach the provider. Both
/// passwords must already be `rclone obscure`d, the form `RCLONE_CRYPT_PASSWORD`
/// expects; losing them makes every object unrecoverable, so the caller holds
/// them in orca secrets. Add these to [`rclone_env`] and address
/// [`RCLONE_CRYPT_REMOTE`].
pub fn rclone_crypt_env(
    cfg: &S3Config,
    obscured_password: &str,
    obscured_salt: &str,
) -> Vec<(String, String)> {
    vec![
        ("RCLONE_CRYPT_REMOTE".to_string(), rclone_remote(cfg)),
        (
            "RCLONE_CRYPT_FILENAME_ENCRYPTION".to_string(),
            "standard".to_string(),
        ),
        (
            "RCLONE_CRYPT_DIRECTORY_NAME_ENCRYPTION".to_string(),
            "true".to_string(),
        ),
        (
            "RCLONE_CRYPT_PASSWORD".to_string(),
            obscured_password.to_string(),
        ),
        (
            "RCLONE_CRYPT_PASSWORD2".to_string(),
            obscured_salt.to_string(),
        ),
    ]
}

/// The on-the-fly crypt remote [`rclone_crypt_env`] configures.
pub const RCLONE_CRYPT_REMOTE: &str = ":crypt:";

// ── storage domain backend ──────────────────────────────────────────────────

/// S3-compatible object-store backend for the `storage` domain.
///
/// [`StorageKind::Object`], [`MountStyle::UserspaceProcess`]. Capabilities:
/// [`Capability::List`], [`Capability::Usage`], [`Capability::Mount`],
/// [`Capability::Unmount`]. It does NOT advertise `RecoverStale` (no stale-handle
/// semantics) and carries no source failover.
pub struct S3Backend {
    name: String,
}

impl S3Backend {
    pub fn new(name: impl Into<String>) -> Self {
        Self { name: name.into() }
    }
}

impl Default for S3Backend {
    fn default() -> Self {
        Self::new("s3")
    }
}

/// Launch the configured FUSE tool to mount `bucket` at `target`. The resolved
/// secret is passed through the child's environment (never the command line /
/// option string). Returns once the tool has been spawned; long-lived
/// supervision (restart-on-crash) is orca core's job once the UserspaceProcess
/// applier lands — see the module docs and README.
///
/// `secret` is the *resolved* secret access key. In the wired-up flow orca's
/// secrets domain resolves `cfg.secret_access_key` before calling the backend;
/// until that seam exists, callers pass the raw `SecretRef` inner string.
pub fn mount_args(cfg: &S3Config, target: &str) -> Vec<String> {
    let mut a: Vec<String> = Vec::new();
    match cfg.tool {
        MountTool::MountpointS3 => {
            a.extend([
                cfg.bucket.clone(),
                target.to_string(),
                "--endpoint-url".into(),
                cfg.endpoint.clone(),
                "--region".into(),
                cfg.region.clone(),
                "--file-mode".into(),
                "0600".into(),
                "--dir-mode".into(),
                "0700".into(),
            ]);
            if cfg.addressing == Addressing::Path {
                a.push("--force-path-style".into());
            }
            if !cfg.prefix.is_empty() {
                a.extend(["--prefix".into(), format!("{}/", cfg.prefix)]);
            }
            a.extend(extra_args(cfg));
        }
        MountTool::Rclone => {
            a.extend([
                "mount".into(),
                rclone_remote(cfg),
                target.to_string(),
                "--file-perms".into(),
                "0600".into(),
                "--dir-perms".into(),
                "0700".into(),
            ]);
            a.extend(extra_args(cfg));
        }
        MountTool::Goofys => {
            a.extend([
                "--endpoint".into(),
                cfg.endpoint.clone(),
                "--region".into(),
                cfg.region.clone(),
                "--file-mode".into(),
                "0600".into(),
                "--dir-mode".into(),
                "0700".into(),
            ]);
            if cfg.addressing == Addressing::VirtualHost {
                a.push("--subdomain".into());
            }
            a.extend(extra_args(cfg));
            a.push(if cfg.prefix.is_empty() {
                cfg.bucket.clone()
            } else {
                format!("{}:{}", cfg.bucket, cfg.prefix)
            });
            a.push(target.to_string());
        }
    }
    a
}

/// Launch the configured FUSE tool to mount `bucket` at `target`. The resolved
/// secret is passed through the child's environment, never the command line.
async fn spawn_mount(cfg: &S3Config, target: &str, secret: &str) -> Result<(), S3Error> {
    let bin = cfg.tool.binary();
    which(bin).ok_or_else(|| S3Error::MissingTool(bin.to_string()))?;
    let mut cmd = Command::new(bin).args(mount_args(cfg, target));
    if cfg.tool == MountTool::Rclone {
        for (k, v) in rclone_env(cfg, secret) {
            cmd = cmd.env(k, v);
        }
    } else {
        cmd = cmd
            .env("AWS_ACCESS_KEY_ID", &cfg.access_key_id)
            .env("AWS_SECRET_ACCESS_KEY", secret);
    }
    cmd.spawn()?;
    Ok(())
}

/// Tear down a userspace FUSE mount at `target` via `fusermount -u` (Linux) or
/// `umount` (macOS / fallback). Unmounting the FUSE filesystem signals the
/// supervising daemon to exit.
async fn teardown_mount(target: &str) -> Result<(), S3Error> {
    #[cfg(target_os = "linux")]
    let (tool, args): (&str, Vec<&str>) = ("fusermount", vec!["-u", target]);
    #[cfg(not(target_os = "linux"))]
    let (tool, args): (&str, Vec<&str>) = ("umount", vec![target]);

    which(tool).ok_or_else(|| S3Error::MissingTool(tool.to_string()))?;
    let out = Command::new(tool).args(&args).output().await?;
    if out.status.success {
        Ok(())
    } else {
        Err(S3Error::ToolFailed {
            tool: tool.to_string(),
            code: out.status.code,
            stderr: String::from_utf8_lossy(&out.stderr).into_owned(),
        })
    }
}

#[orca_async]
impl StorageBackend for S3Backend {
    fn name(&self) -> &str {
        &self.name
    }

    fn kind(&self) -> StorageKind {
        StorageKind::Object
    }

    fn capabilities(&self) -> Vec<Capability> {
        vec![
            Capability::List,
            Capability::Usage,
            Capability::Mount,
            Capability::Unmount,
        ]
    }

    fn endpoint(&self) -> String {
        "s3://local".to_string()
    }

    /// s3 is realized by a supervised userspace FUSE process, not a kernel mount.
    fn mount_style(&self) -> MountStyle {
        MountStyle::UserspaceProcess
    }

    /// Parse + validate the S3 mount spec (bucket, endpoint, region, access key,
    /// credential SecretRef, tool selection), rejecting a spec that is missing a
    /// required field or that declares inline secret material. The normalized
    /// spec carries the canonicalized option string back through as
    /// [`OptionSet::Raw`] (the contract has no typed S3 option variant).
    async fn validate_spec(&self, spec: &MountSpec) -> Result<NormalizedSpec, StorageError> {
        let cfg = validate_s3_config(spec)?;
        Ok(NormalizedSpec {
            backend: spec.backend.clone(),
            target: spec.target.clone(),
            fstype: "fuse.s3".to_string(),
            source: cfg.bucket.clone(),
            failover_sources: Vec::new(), // object stores have no source failover
            options: OptionSet::Raw {
                options: Some(render_s3_options(&cfg)),
            },
            credential: spec.credential.clone(),
            secret_file: None, // s3 auth rides the credential SecretRef; no on-disk secret file
            remount_policy: spec.remount_policy.clone(),
            enabled: spec.enabled,
        })
    }

    /// Bring the object mount up at `target` by launching the configured FUSE
    /// tool. `id` is the declarative option string (the same grammar
    /// `validate_spec` accepts) so this method is self-contained until core's
    /// UserspaceProcess applier passes a normalized spec through.
    ///
    /// NOTE: the resolved secret is taken from the parsed [`SecretRef`]'s inner
    /// string. Once orca core resolves the SecretRef through the secrets domain
    /// before invoking the backend, that resolved value flows in here unchanged.
    async fn mount(&self, id: &str, target: &str) -> Result<MountOutcome, StorageError> {
        // `id` carries the declarative spec: bucket in `source` position is not
        // available here, so callers pass the option string; the backend still
        // needs a bucket. We accept `id` as `"<bucket>;<options>"` — bucket, then
        // a `;`, then the option string — so `mount` is drivable standalone.
        let (bucket, options) = match id.split_once(';') {
            Some((b, o)) => (b.to_string(), o.to_string()),
            None => (id.to_string(), String::new()),
        };
        let spec = MountSpec {
            backend: self.name.clone(),
            target: target.to_string(),
            fstype: "fuse.s3".to_string(),
            source: bucket,
            failover_sources: Vec::new(),
            options: Some(options),
            credential: None,
            remount_policy: None,
            enabled: true,
        };
        let cfg = validate_s3_config(&spec)?;
        let secret = cfg.secret_access_key.0.clone();
        spawn_mount(&cfg, target, &secret)
            .await
            .map_err(|e| StorageError::Transport(e.to_string()))?;
        Ok(MountOutcome {
            target: target.to_string(),
            mounted: true,
            recovered: false,
            detail: Some(format!(
                "launched {} FUSE mount",
                cfg.tool.as_option_value()
            )),
        })
    }

    async fn unmount(&self, target: &str) -> Result<MountOutcome, StorageError> {
        teardown_mount(target)
            .await
            .map_err(|e| StorageError::Other(format!("unmount {target}: {e}")))?;
        Ok(MountOutcome {
            target: target.to_string(),
            mounted: false,
            recovered: false,
            detail: None,
        })
    }

    /// Report the object mount(s) this backend manages. It reads the live kernel
    /// mount table filtered to FUSE-S3 filesystem types via the shared storage
    /// primitive, so a mount launched by this backend shows up as a share.
    async fn list_shares(&self) -> Result<Vec<Share>, StorageError> {
        // FUSE-S3 mounts appear in the mount table with a `fuse.*` fstype
        // (`fuse.mount-s3`, `fuse.rclone`, `fuse.goofys`). Enumerate them.
        const S3_FUSE_FSTYPES: &[&str] = &[
            "fuse.mount-s3",
            "fuse.rclone",
            "fuse.goofys",
            "fuse.s3",
            "fuse",
        ];
        let mounts = plugin_toolkit::storage::mount_table_of(S3_FUSE_FSTYPES)
            .map_err(|e| StorageError::Transport(e.to_string()))?;
        Ok(mounts
            .into_iter()
            .map(|m| Share {
                id: m.mountpoint.clone(),
                source: m.source,
                target: Some(m.mountpoint),
                fstype: m.fstype,
                mounted: true,
            })
            .collect())
    }

    /// Capacity/usage for an object mount. Object stores expose usage through the
    /// S3 API (e.g. summing object sizes / bucket metrics), NOT `statvfs` against
    /// the FUSE mount — a FUSE-S3 filesystem reports a synthetic, meaningless
    /// `statvfs`. A real S3 API client is out of scope for this phase; wiring one
    /// is the documented follow-up. Until then this is a fail-closed stub.
    async fn usage(&self, id: &str) -> Result<plugin_toolkit::storage::Usage, StorageError> {
        let _ = Path::new(id);
        Err(StorageError::Other(format!(
            "s3 usage for `{id}` requires an S3 API client (list-objects / bucket metrics); \
             not implemented in this phase — see README"
        )))
    }
}

/// Register the s3 storage backend with the process-global `storage` registry.
/// Retained for the `rlib` shape (in-process embedding / tests); the subprocess
/// plugin path contributes the backend via the serve loop's `backends()` seam
/// instead.
pub fn bootstrap() {
    plugin_toolkit::storage::register_backend(Arc::new(S3Backend::default()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use plugin_toolkit::serde_json;

    /// Build an S3 `MountSpec` for the grammar tests. `options` is the raw
    /// declared option string; `credential` is the secret-access-key `SecretRef`.
    fn s3_spec(source: &str, options: Option<&str>, credential: Option<&str>) -> MountSpec {
        MountSpec {
            backend: "s3".into(),
            target: "/mnt/objects".into(),
            fstype: "fuse.s3".into(),
            source: source.into(),
            failover_sources: vec![],
            options: options.map(str::to_string),
            credential: credential.map(|s| SecretRef(s.to_string())),
            remount_policy: None,
            enabled: true,
        }
    }

    const HAPPY_OPTS: &str = "endpoint=https://s3.example.com,region=us-east-1,access_key=AKIA";

    #[test]
    fn mount_style_is_userspace_process() {
        assert_eq!(
            S3Backend::default().mount_style(),
            MountStyle::UserspaceProcess
        );
    }

    #[test]
    fn kind_is_object() {
        assert_eq!(S3Backend::default().kind(), StorageKind::Object);
    }

    #[test]
    fn capabilities_are_object_semantics_no_recover_stale() {
        let caps = S3Backend::default().capabilities();
        assert!(caps.contains(&Capability::List));
        assert!(caps.contains(&Capability::Usage));
        assert!(caps.contains(&Capability::Mount));
        assert!(caps.contains(&Capability::Unmount));
        // Object stores have no stale-handle recovery and no create/remove here.
        assert!(!caps.contains(&Capability::RecoverStale));
        assert!(!caps.contains(&Capability::Create));
        assert!(!caps.contains(&Capability::Remove));
    }

    #[test]
    fn validate_happy_path_parses_all_fields() {
        let spec = s3_spec(
            "s3://my-bucket",
            Some(HAPPY_OPTS),
            Some("op://vault/s3-secret"),
        );
        let cfg = validate_s3_config(&spec).expect("happy path validates");
        assert_eq!(cfg.bucket, "my-bucket");
        assert_eq!(cfg.endpoint, "https://s3.example.com");
        assert_eq!(cfg.region, "us-east-1");
        assert_eq!(cfg.access_key_id, "AKIA");
        assert_eq!(cfg.tool, MountTool::MountpointS3); // default
        assert_eq!(
            cfg.secret_access_key,
            SecretRef("op://vault/s3-secret".into())
        );
    }

    #[test]
    fn validate_accepts_bare_bucket_name() {
        let spec = s3_spec("my-bucket", Some(HAPPY_OPTS), Some("op://x"));
        let cfg = validate_s3_config(&spec).expect("bare bucket accepted");
        assert_eq!(cfg.bucket, "my-bucket");
    }

    #[test]
    fn validate_rejects_missing_bucket() {
        let spec = s3_spec("s3://", Some(HAPPY_OPTS), Some("op://x"));
        let err = validate_s3_config(&spec).expect_err("empty bucket rejected");
        assert!(err.to_string().contains("bucket is empty"), "got: {err}");
    }

    #[test]
    fn validate_rejects_missing_endpoint() {
        let spec = s3_spec(
            "b",
            Some("region=us-east-1,access_key=AKIA"),
            Some("op://x"),
        );
        let err = validate_s3_config(&spec).expect_err("missing endpoint rejected");
        assert!(err.to_string().contains("endpoint"), "got: {err}");
    }

    #[test]
    fn validate_rejects_missing_region() {
        let spec = s3_spec(
            "b",
            Some("endpoint=https://s3.example.com,access_key=AKIA"),
            Some("op://x"),
        );
        let err = validate_s3_config(&spec).expect_err("missing region rejected");
        assert!(err.to_string().contains("region"), "got: {err}");
    }

    #[test]
    fn validate_rejects_missing_access_key() {
        let spec = s3_spec(
            "b",
            Some("endpoint=https://s3.example.com,region=us-east-1"),
            Some("op://x"),
        );
        let err = validate_s3_config(&spec).expect_err("missing access key rejected");
        assert!(err.to_string().contains("access_key"), "got: {err}");
    }

    #[test]
    fn validate_rejects_missing_credential_secretref() {
        // Everything present except the secret-access-key SecretRef.
        let spec = s3_spec("b", Some(HAPPY_OPTS), None);
        let err = validate_s3_config(&spec).expect_err("missing credential rejected");
        assert!(err.to_string().contains("SecretRef"), "got: {err}");
    }

    #[test]
    fn validate_rejects_inline_secret_material() {
        let spec = s3_spec(
            "b",
            Some("endpoint=https://s3.example.com,region=us-east-1,access_key=AKIA,secret_key=hunter2"),
            Some("op://x"),
        );
        let err = validate_s3_config(&spec).expect_err("inline secret rejected");
        assert!(err.to_string().contains("SecretRef"), "got: {err}");
    }

    #[test]
    fn validate_selects_and_rejects_tools() {
        for (name, expect) in [
            ("mountpoint-s3", MountTool::MountpointS3),
            ("rclone", MountTool::Rclone),
            ("goofys", MountTool::Goofys),
        ] {
            let spec = s3_spec(
                "b",
                Some(&format!("{HAPPY_OPTS},tool={name}")),
                Some("op://x"),
            );
            assert_eq!(validate_s3_config(&spec).unwrap().tool, expect);
        }
        let spec = s3_spec(
            "b",
            Some(&format!("{HAPPY_OPTS},tool=nope")),
            Some("op://x"),
        );
        assert!(validate_s3_config(&spec).is_err(), "unknown tool rejected");
    }

    #[test]
    fn validate_keeps_allowlisted_extra_options() {
        let spec = s3_spec(
            "b",
            Some(&format!("{HAPPY_OPTS},read-only,uid=1000")),
            Some("op://x"),
        );
        let cfg = validate_s3_config(&spec).unwrap();
        assert_eq!(cfg.extra, ["read-only", "uid=1000"]);
        let args = mount_args(&cfg, "/mnt/o");
        assert!(args.contains(&"--read-only".to_string()));
        assert!(args.contains(&"--uid=1000".to_string()));
    }

    #[test]
    fn validate_rejects_denied_and_unlisted_extra_options() {
        for tool in ["mountpoint-s3", "rclone", "goofys"] {
            for opt in [
                "--no-check-certificate",
                "no-verify-ssl",
                "ca-bundle=/tmp/ca.pem",
                "--insecure",
                "allow-other",
                "allow_other",
                "umask=000",
                "file-mode=0777",
                "dir-mode=0777",
                "--file-perms=0777",
                "--dir-perms=0777",
                "--endpoint-url=http://evil",
                "-o",
            ] {
                let spec = s3_spec(
                    "b",
                    Some(&format!("{HAPPY_OPTS},tool={tool},{opt}")),
                    Some("op://x"),
                );
                assert!(validate_s3_config(&spec).is_err(), "{tool} {opt}");
            }
        }
    }

    #[test]
    fn http_endpoint_requires_loopback_or_allow_insecure() {
        for ep in [
            "http://s3.example.com",
            "http://10.0.0.5:9000",
            "http://localhost.evil.com",
            "http://localhost:x@evil.com",
            "http://[::1]@evil.com",
            "http://[::1]:9000@evil.com",
            "http://127.0.0.1:80@evil.com",
            "http://localhost\\@evil.com",
            "http://127.0.0.1.nip.io",
            "http://[::ffff:10.0.0.5]",
            "HTTP://s3.example.com",
        ] {
            let opts = format!("endpoint={ep},region=r,access_key=A");
            assert!(
                validate_s3_config(&s3_spec("b", Some(&opts), Some("op://x"))).is_err(),
                "{ep}"
            );
        }
        for ep in [
            "http://localhost:9000",
            "http://127.0.0.1:9000/",
            "http://[::1]:9000",
        ] {
            let opts = format!("endpoint={ep},region=r,access_key=A");
            assert!(
                validate_s3_config(&s3_spec("b", Some(&opts), Some("op://x"))).is_ok(),
                "{ep}"
            );
        }
        let c = cfg("endpoint=http://10.0.0.5:9000,region=r,access_key=A,allow_insecure");
        assert!(c.allow_insecure);
        assert!(render_s3_options(&c).contains("allow_insecure=true"));
        assert!(!cfg(HAPPY_OPTS).allow_insecure);
    }

    #[test]
    fn mounts_default_to_private_modes() {
        for (tool, file, dir) in [
            ("mountpoint-s3", "--file-mode", "--dir-mode"),
            ("rclone", "--file-perms", "--dir-perms"),
            ("goofys", "--file-mode", "--dir-mode"),
        ] {
            let args = mount_args(&cfg(&format!("{HAPPY_OPTS},tool={tool}")), "/mnt/o");
            let after = |f: &str| args[args.iter().position(|a| a == f).unwrap() + 1].clone();
            assert_eq!(after(file), "0600", "{tool}");
            assert_eq!(after(dir), "0700", "{tool}");
        }
    }

    #[test]
    fn render_never_emits_the_secret() {
        let secret = "op://vault/s3-secret";
        let spec = s3_spec("my-bucket", Some(HAPPY_OPTS), Some(secret));
        let cfg = validate_s3_config(&spec).unwrap();
        let rendered = render_s3_options(&cfg);
        assert!(!rendered.contains(secret), "secret ref must not render");
        assert!(!rendered.contains("op://"), "no secret scheme leak");
        assert!(!rendered.contains("secret"), "no secret_key rendered");
        // Non-secret fields still render.
        assert!(rendered.contains("endpoint=https://s3.example.com"));
        assert!(rendered.contains("region=us-east-1"));
        assert!(rendered.contains("access_key=AKIA"));
        assert!(rendered.contains("tool=mountpoint-s3"));
    }

    #[tokio::test]
    async fn validate_spec_normalizes_to_object_mount_no_failover() {
        let backend = S3Backend::default();
        let spec = s3_spec("s3://my-bucket", Some(HAPPY_OPTS), Some("op://x"));
        let normalized = backend.validate_spec(&spec).await.expect("validate");
        assert_eq!(normalized.source, "my-bucket");
        assert_eq!(normalized.fstype, "fuse.s3");
        assert!(
            normalized.failover_sources.is_empty(),
            "object stores carry no failover"
        );
        // The secret never rides in the normalized option string. `OptionSet`
        // has a single `Raw` variant, so this destructure is irrefutable.
        let OptionSet::Raw { options } = &normalized.options;
        let opts = options.as_deref().unwrap_or("");
        assert!(!opts.contains("op://"), "secret must not be in options");
    }

    #[tokio::test]
    async fn validate_spec_backend_method_rejects_missing_creds() {
        let backend = S3Backend::default();
        let spec = s3_spec("b", Some(HAPPY_OPTS), None);
        assert!(backend.validate_spec(&spec).await.is_err());
    }

    #[tokio::test]
    async fn usage_is_documented_stub() {
        // Usage requires an S3 API client (out of scope this phase); it must
        // fail closed with a clear message rather than fake a statvfs number.
        let backend = S3Backend::default();
        let err = backend.usage("/mnt/objects").await.expect_err("stubbed");
        assert!(err.to_string().contains("S3 API client"), "got: {err}");
    }

    #[test]
    fn mount_tool_round_trips_through_serde() {
        for t in [
            MountTool::MountpointS3,
            MountTool::Rclone,
            MountTool::Goofys,
        ] {
            let j = serde_json::to_string(&t).unwrap();
            let back: MountTool = serde_json::from_str(&j).unwrap();
            assert_eq!(back, t);
        }
    }

    #[test]
    fn s3_error_display_covers_each_variant() {
        let e = S3Error::MissingTool("mount-s3".into());
        assert!(e.to_string().contains("mount-s3"));
        let e = S3Error::ToolFailed {
            tool: "rclone".into(),
            code: Some(1),
            stderr: "boom".into(),
        };
        assert!(e.to_string().contains("boom"));
        let e = S3Error::Unsupported;
        assert!(e.to_string().contains("unsupported"));
        let io: S3Error = std::io::Error::other("x").into();
        assert!(io.to_string().starts_with("io:"));
    }

    fn cfg(opts: &str) -> S3Config {
        validate_s3_config(&s3_spec(
            "s3://b/backups/immich/",
            Some(opts),
            Some("op://x"),
        ))
        .unwrap()
    }

    #[test]
    fn source_prefix_is_kept_and_normalized() {
        let c = cfg(HAPPY_OPTS);
        assert_eq!(
            (c.bucket.as_str(), c.prefix.as_str()),
            ("b", "backups/immich")
        );
        assert_eq!(rclone_remote(&c), ":s3:b/backups/immich");
        let spec = s3_spec("s3://b/../x", Some(HAPPY_OPTS), Some("op://x"));
        assert!(validate_s3_config(&spec).is_err());
    }

    #[test]
    fn addressing_defaults_to_path_and_is_configurable() {
        assert_eq!(cfg(HAPPY_OPTS).addressing, Addressing::Path);
        let vh = cfg(&format!("{HAPPY_OPTS},addressing=virtual-host"));
        assert_eq!(vh.addressing, Addressing::VirtualHost);
        assert!(render_s3_options(&vh).contains("addressing=virtual-host"));
        let spec = s3_spec(
            "b",
            Some(&format!("{HAPPY_OPTS},addressing=dns")),
            Some("op://x"),
        );
        assert!(validate_s3_config(&spec).is_err());
    }

    #[test]
    fn endpoint_must_be_a_url_and_region_non_empty() {
        for opts in [
            "endpoint=s3.example.com,region=r,access_key=A",
            "endpoint=https://s3.example.com,region=,access_key=A",
        ] {
            assert!(
                validate_s3_config(&s3_spec("b", Some(opts), Some("op://x"))).is_err(),
                "{opts}"
            );
        }
    }

    #[test]
    fn rclone_env_is_provider_neutral_and_config_driven() {
        let c = cfg(&format!(
            "{HAPPY_OPTS},addressing=virtual-host,storage_class=GLACIER_IR"
        ));
        let env: std::collections::HashMap<_, _> = rclone_env(&c, "s3cr3t").into_iter().collect();
        assert_eq!(env["RCLONE_CONFIG"], "/dev/null");
        assert_eq!(env["RCLONE_S3_PROVIDER"], "Other");
        assert_eq!(env["RCLONE_S3_ENV_AUTH"], "false");
        assert_eq!(env["RCLONE_S3_ENDPOINT"], "https://s3.example.com");
        assert_eq!(env["RCLONE_S3_REGION"], "us-east-1");
        assert_eq!(env["RCLONE_S3_FORCE_PATH_STYLE"], "false");
        assert_eq!(env["RCLONE_S3_STORAGE_CLASS"], "GLACIER_IR");
        assert_eq!(env["RCLONE_S3_SECRET_ACCESS_KEY"], "s3cr3t");
        assert_eq!(
            rclone_env(&cfg(HAPPY_OPTS), "k")
                .into_iter()
                .find(|(k, _)| k == "RCLONE_S3_FORCE_PATH_STYLE")
                .unwrap()
                .1,
            "true"
        );
    }

    #[test]
    fn crypt_wraps_the_s3_remote_with_env_only_keys() {
        let c = cfg(HAPPY_OPTS);
        let env: std::collections::HashMap<_, _> = rclone_crypt_env(&c, "obs-pass", "obs-salt")
            .into_iter()
            .collect();
        assert_eq!(env["RCLONE_CRYPT_REMOTE"], rclone_remote(&c));
        assert_eq!(env["RCLONE_CRYPT_FILENAME_ENCRYPTION"], "standard");
        assert_eq!(env["RCLONE_CRYPT_DIRECTORY_NAME_ENCRYPTION"], "true");
        assert_eq!(env["RCLONE_CRYPT_PASSWORD"], "obs-pass");
        assert_eq!(env["RCLONE_CRYPT_PASSWORD2"], "obs-salt");
        assert!(!RCLONE_CRYPT_REMOTE.contains("obs-"));
    }

    #[test]
    fn secret_stays_out_of_the_remote_and_options() {
        let c = cfg(HAPPY_OPTS);
        assert!(!rclone_remote(&c).contains("op://"));
        assert!(!render_s3_options(&c).contains("op://"));
    }
}
