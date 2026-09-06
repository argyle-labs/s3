//! Dynamic (subprocess) entrypoint for the s3 plugin.
//!
//! Serves this plugin over the orca socket via the typed `Plugin` builder. The plugin is a
//! `[[bin]]`, owns no runtime, and reaches orca only through the socket.
plugin_toolkit::instrument::bootstrap!();
use plugin_toolkit::plugin::Plugin;

fn main() -> plugin_toolkit::anyhow::Result<()> {
    Plugin::named("s3")
        .version(env!("CARGO_PKG_VERSION"))
        .storage(s3::S3Backend::new("s3"))
        .serve()
}
