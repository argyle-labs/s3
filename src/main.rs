//! Dynamic (subprocess) entrypoint for the s3 plugin.
//!
//! The toolkit's `serve_storage_plugin!` emits `fn main`, serving this plugin over the orca
//! socket. Dynamic replacement for the retired cdylib export — the plugin is a
//! `[[bin]]`, owns no runtime, and reaches orca only through the socket.
plugin_toolkit::serve_storage_plugin! {
    name: "s3",
    target_compat: "any",
    backend: s3::S3Backend::new("s3"),
}
