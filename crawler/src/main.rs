pub(crate) mod utils;
mod workers;

use std::collections::{hash_map::Entry, HashMap, HashSet};

use clap::Parser;
use futures::TryStreamExt;
use mongodb::{
    bson::{doc, Bson, Document},
    error::ErrorKind,
    options::{
        ClientOptions, CreateIndexOptions, FindOneAndUpdateOptions, IndexOptions, InsertManyOptions,
    },
    Client, Collection, IndexModel,
};
use tracing::metadata::LevelFilter;
use tracing_subscriber::{prelude::*, Registry};
use tracing_tree::HierarchicalLayer;

use crate::{
    utils::get_duplicate_indices,
    workers::{
        BrowserWorker, BrowserWorkerArgs, CoreWorker, CoreWorkerArgs, DomainWorker,
        DomainWorkerArgs,
    },
};

#[derive(Parser)]
#[clap(author, version, about, long_about = None)]
struct Args {
    /// MongoDB connection string
    #[clap(
        short,
        long,
        default_value = "mongodb://127.0.0.1:27017/?appName=crawler"
    )]
    connection_string: String,

    /// MongoDB database name
    #[clap(long, default_value = "phishing")]
    database: String,

    /// Number of workers to spawn in parallel
    #[clap(short, long, default_value = "0")]
    workers: u16,

    /// Stop the crawler when it is idle due to a lack of tasks
    #[clap(long)]
    exit_if_idle: bool,

    /// Clear database before starting crawler
    #[clap(long)]
    drop_dbs: bool,

    /// Verbose logging
    #[clap(short, long)]
    verbose: bool,

    #[clap(subcommand)]
    worker_args: WorkerArgs,
}

#[derive(clap::Subcommand)]
enum WorkerArgs {
    /// Run a worker which extracts core (ie. URL and HTML) features
    Core {
        /// Depth after which documents will not be processed
        #[clap(long, default_value = "1")]
        max_depth: u8,
    },
    /// Run a worker which extracts browser-based features
    Browser,
    /// Run a worker which extracts domain-level features
    Domain,
    /// Add a list of URLs to the pages to fetch
    Add {
        /// Mark the added pages as phishing pages
        #[clap(long)]
        is_phishing: bool,

        urls: Vec<String>,
    },
    /// Streams all entries as CSV on stdout
    Extract {
        /// Depth after which documents will not be processed
        #[clap(long, default_value = "1")]
        max_depth: u8,
    },
}

/// Entry point of the app, which sets up the state and handles cancellation.
#[tokio::main]
async fn main() -> anyhow::Result<()> {
    // Parse args.
    let args = Args::parse();
    let exit_if_idle = args.exit_if_idle;

    // Set-up tracing.
    let level_filter = if args.verbose {
        LevelFilter::TRACE
    } else {
        LevelFilter::INFO
    };

    tracing::subscriber::set_global_default(
        Registry::default().with(
            HierarchicalLayer::new(2)
                .with_thread_ids(true)
                .with_bracketed_fields(false)
                .with_filter(level_filter),
        ),
    )?;

    // Connect to MongoDB.
    let client_options = ClientOptions::parse(&args.connection_string).await?;
    let client = Client::with_options(client_options)?;
    let db = client.database(&args.database);

    let pages_collection = db.collection("pages");
    let domains_collection = db.collection("domains");

    // Create channel to notify main loop when shutting down.
    let (shutdown_tx, shutdown_rx) = tokio::sync::broadcast::channel(1);

    if args.drop_dbs {
        tracing::info!("dropping existing databases");

        // Drop the collection to restart from scratch.
        let (_, _) = tokio::join!(pages_collection.drop(None), domains_collection.drop(None));
    }

    if !has_index(&pages_collection, "url").await? {
        tracing::info!(collection = "pages", "creating indexes");

        pages_collection
            .create_indexes(
                [
                    IndexModel::builder()
                        .keys(doc! { "url": 1 })
                        .options(
                            IndexOptions::builder()
                                .name("url".to_string())
                                .unique(true)
                                .build(),
                        )
                        .build(),
                    // `"depth": 1` means that `depth` should be sorted in
                    // ascending order, which is what we want to process pages
                    // with a lower depth first.
                    IndexModel::builder()
                        .keys(doc! { "depth": 1 })
                        .options(
                            IndexOptions::builder()
                                .name("depth".to_string())
                                .unique(false)
                                .build(),
                        )
                        .build(),
                    IndexModel::builder()
                        .keys(doc! { "depth": 1, "status_code": 1 })
                        .options(
                            IndexOptions::builder()
                                .name("depth_and_status".to_string())
                                .unique(false)
                                .build(),
                        )
                        .build(),
                ],
                CreateIndexOptions::builder().build(),
            )
            .await?;
    }

    if !has_index(&domains_collection, "domain").await? {
        tracing::info!(collection = "domains", "creating indexes");

        domains_collection
            .create_indexes(
                [
                    IndexModel::builder()
                        .keys(doc! { "domain": 1 })
                        .options(
                            IndexOptions::builder()
                                .name("domain".to_string())
                                .unique(true)
                                .build(),
                        )
                        .build(),
                    IndexModel::builder()
                        .keys(doc! { "is_cert_valid": 1 })
                        .options(
                            IndexOptions::builder()
                                .name("is_cert_valid".to_string())
                                .unique(false)
                                .build(),
                        )
                        .build(),
                ],
                CreateIndexOptions::builder().build(),
            )
            .await?;
    }

    // Spawn requested worker pool.
    let mut main_loop_handle = match args.worker_args {
        WorkerArgs::Core { max_depth } => {
            let workers_count = if args.workers == 0 {
                256
            } else {
                args.workers as usize
            };
            let args = CoreWorkerArgs {
                domains_collection,
                pages_collection: pages_collection.clone(),
            };

            let future = workers::run::<CoreWorker, _, _>(
                workers_count,
                args,
                move || {
                    // Find any item which hasn't been processed yet
                    // (`status_code == null`) and which should be processed
                    // (`depth <= max_depth`) then set its `status_code` to 0
                    // (to identify it as processing), and then process it.
                    let filter =
                        doc! { "status_code": null, "depth": { "$lte": max_depth as u32 } };
                    let update = doc! { "$set": { "status_code": 0 } };

                    get_next_document(pages_collection.clone(), filter, update)
                },
                exit_if_idle,
                shutdown_rx,
            );

            tokio::spawn(future)
        }
        WorkerArgs::Browser => {
            let workers_count = if args.workers == 0 {
                12
            } else {
                args.workers as usize
            };
            let args = BrowserWorkerArgs {
                pages_collection: pages_collection.clone(),
            };

            let future = workers::run::<BrowserWorker, _, _>(
                workers_count,
                args,
                move || {
                    // Find any item which has been marked as needing further
                    // processing (`state == PendingBrowser`) then set its
                    // and then process it.
                    let filter = doc! { "state": ProcessingState::PendingBrowser as u32 };
                    let update = doc! { "$set": { "state": ProcessingState::Complete as u32 } };

                    get_next_document(pages_collection.clone(), filter, update)
                },
                exit_if_idle,
                shutdown_rx,
            );

            tokio::spawn(future)
        }
        WorkerArgs::Domain => {
            let workers_count = if args.workers == 0 {
                32
            } else {
                args.workers as usize
            };
            let args = DomainWorkerArgs {
                domains_collection: domains_collection.clone(),
            };

            let future = workers::run::<DomainWorker, _, _>(
                workers_count,
                args,
                move || {
                    // Find any item which hasn't been processed yet.
                    let filter = doc! { "is_cert_valid": null };
                    let update = doc! { "$set": { "is_cert_valid": false } };

                    get_next_document(domains_collection.clone(), filter, update)
                },
                exit_if_idle,
                shutdown_rx,
            );

            tokio::spawn(future)
        }
        WorkerArgs::Add { is_phishing, urls } => {
            let urls = urls
                .iter()
                .map(|x| {
                    url::Url::parse(x).map(|mut url| {
                        url.set_fragment(None);
                        url
                    })
                })
                .collect::<Result<HashSet<_>, _>>()?
                .into_iter()
                .collect::<Vec<_>>();
            let docs = urls
                .iter()
                .map(|x| doc! { "url": x.as_str(), "depth": 0, "is_phishing": is_phishing });

            let duplicate_indices = get_duplicate_indices(
                pages_collection
                    .insert_many(docs, InsertManyOptions::builder().ordered(false).build())
                    .await,
            )?;

            if let Some(duplicate_indices) = duplicate_indices {
                tracing::info!(
                    "{} document(s) already existed, updating length to 0...",
                    duplicate_indices.len(),
                );

                let duplicate_urls = Bson::Array(
                    duplicate_indices
                        .iter()
                        .map(|&x| Bson::String(urls[x].as_str().to_string()))
                        .collect(),
                );

                pages_collection
                    .update_many(
                        doc! { "url": { "$in": duplicate_urls } },
                        doc! { "$set": { "depth": 0, "is_phishing": is_phishing } },
                        None,
                    )
                    .await?;
            }

            return Ok(());
        }
        WorkerArgs::Extract { max_depth } => {
            // Open writer and write header.
            let mut writer = csv::Writer::from_writer(std::io::stdout());

            write_page_document_fields(&mut writer)?;

            // Print all CSV records.
            let mut domains = HashMap::new();
            let mut cursor = pages_collection
                .find(
                    doc! { "status_code": { "$gt": 0 }, "depth": { "$lte": max_depth as u32 } },
                    None,
                )
                .await?;
            let mut erroneous_docs = HashMap::new();

            while let Some(document) = cursor.try_next().await? {
                // Parse URL.
                let url_str = match document.get_str("url") {
                    Ok(url) => url,
                    Err(_) => {
                        let id = match document.get_object_id("_id") {
                            Ok(id) => id.to_hex(),
                            Err(_) => "<unknown>".into(),
                        };
                        erroneous_docs.insert(id, anyhow::anyhow!("does not have an url"));
                        continue;
                    }
                };
                let url = url::Url::parse(url_str)?;

                // Resolve corresponding domain.
                let domain_doc = match domains.entry(url_str.to_string()) {
                    Entry::Occupied(entry) => entry.into_mut(),
                    Entry::Vacant(entry) => {
                        let domain = url
                            .host_str()
                            .ok_or_else(|| anyhow::anyhow!("invalid url {url}"))?;
                        let domain_doc = domains_collection
                            .find_one(doc! { "domain": domain }, None)
                            .await?
                            .ok_or_else(|| anyhow::anyhow!("missing domain for {url}"))?;

                        entry.insert(domain_doc)
                    }
                };

                // Write fields.
                writer.write_field(url_str)?;

                if let Err(err) =
                    write_page_document(&document, domain_doc, max_depth, &mut writer).await
                {
                    erroneous_docs.insert(url_str.to_string(), err);
                }

                // Write record terminator.
                writer.write_record(None::<&str>)?;
            }

            writer.flush()?;

            // Print errors.
            for (doc_id, err) in erroneous_docs {
                tracing::error!(doc_id = doc_id.as_str(), error = ?err, "document could not be written");
            }

            return Ok(());
        }
    };

    // Shutdown and dispose of the app when a cancellation signal is received.
    tokio::select! {
        _ = tokio::signal::ctrl_c() => {
            tracing::info!("cancellation requested, shutting down");

            shutdown_tx.send(())?;

            main_loop_handle.await??;
        }
        result = &mut main_loop_handle => result??
    }

    Ok(())
}

/// Returns the next `document` to match the given `filter` in `collection`,
/// updating it with `update` atomically to ensure no other workers process it.
async fn get_next_document(
    collection: Collection<Document>,
    filter: Document,
    update: Document,
) -> Option<Document> {
    let options = FindOneAndUpdateOptions::builder().build();

    loop {
        match collection
            .find_one_and_update(filter.clone(), update.clone(), options.clone())
            .await
        {
            Ok(site) => return site,
            Err(err) => {
                tracing::error!(error = ?err, "error calling find_one_and_update");

                tokio::time::sleep(tokio::time::Duration::from_millis(1_000)).await;
                continue;
            }
        }
    }
}

/// The state of a document to process or that has been processed.
pub enum ProcessingState {
    /// Not yet processed.
    Pending = 0,
    /// Fully processed.
    Complete = 1,
    /// Partially processed, awaiting browser processing.
    PendingBrowser = 2,
}

/// Returns whether the specified `collection` has an index with the given name.
async fn has_index(
    collection: &Collection<Document>,
    index_name: &str,
) -> mongodb::error::Result<bool> {
    match collection.list_index_names().await {
        Ok(names) => Ok(names.iter().any(|x| x == index_name)),
        Err(error) => {
            if let ErrorKind::Command(command_error) = &*error.kind {
                if command_error.code == 26
                /* NamespaceNotFound */
                {
                    return Ok(false);
                }
            }

            Err(error)
        }
    }
}

fn write_page_document_fields(writer: &mut csv::Writer<std::io::Stdout>) -> csv::Result<()> {
    writer.write_record(&[
        "url",
        "depth",
        "is_phishing",
        "status_code",
        "redirects",
        // URL:
        "is_https",
        "is_ip_address",
        "is_error_page",
        "url_length",
        "domain_url_depth",
        "domain_url_length",
        "has_sub_domain",
        "has_at_symbol",
        "dashes_count",
        "path_starts_with_url",
        // Content:
        "is_valid_html",
        "anchors_count",
        "forms_count",
        "javascript_count",
        "self_anchors_count",
        "has_form_with_url",
        "has_iframe",
        "use_mouseover",
        // Domain:
        "is_cert_valid",
        "has_dns_record",
        "has_whois",
        "cert_country",
        "cert_reliability",
        "domain_age",
        "domain_end_period",
        "domain_creation_date",
        // Refs:
        "refs",
    ])
}

async fn write_page_document(
    doc: &Document,
    domain_doc: &Document,
    max_depth: u8,
    writer: &mut csv::Writer<std::io::Stdout>,
) -> anyhow::Result<()> {
    // Helpers.
    type W<'a> = &'a mut csv::Writer<std::io::Stdout>;

    let write_bool = |w: W, doc: &Document, name| {
        w.write_field(match doc.get_bool(name) {
            Ok(x) => {
                if x {
                    "true"
                } else {
                    "false"
                }
            }
            Err(_) => "",
        })
    };
    let write_i32 = |w: W, doc: &Document, name| {
        w.write_field(match doc.get_i32(name) {
            Ok(x) => format!("{x}"),
            Err(_) => String::new(),
        })
    };
    let write_i64 = |w: W, doc: &Document, name| {
        w.write_field(match doc.get_i64(name) {
            Ok(x) => format!("{x}"),
            Err(_) => String::new(),
        })
    };
    let write_datetime = |w: W, doc: &Document, name| {
        w.write_field(match doc.get_datetime(name) {
            Ok(x) => x.to_rfc3339_string(),
            Err(_) => String::new(),
        })
    };
    let write_str = |w: W, doc: &Document, name| w.write_field(doc.get_str(name).unwrap_or(""));

    // Write metadata fields.
    let depth = doc.get_i32("depth")?;

    writer.write_field(format!("{depth}"))?;

    write_bool(writer, doc, "is_phishing")?;
    write_i32(writer, doc, "status_code")?;
    write_i32(writer, doc, "redirects")?;

    // Write URL fields.
    write_bool(writer, doc, "is_https")?;
    write_bool(writer, doc, "is_ip_address")?;
    write_bool(writer, doc, "is_error_page")?;
    write_i32(writer, doc, "url_length")?;
    write_i32(writer, doc, "domain_url_depth")?;
    write_i32(writer, doc, "domain_url_length")?;
    write_bool(writer, doc, "has_sub_domain")?;
    write_bool(writer, doc, "has_at_symbol")?;
    write_i32(writer, doc, "dashes_count")?;
    write_bool(writer, doc, "path_starts_with_url")?;

    // Write contents fields.
    write_bool(writer, doc, "is_valid_html")?;
    write_i32(writer, doc, "anchors_count")?;
    write_i32(writer, doc, "forms_count")?;
    write_i32(writer, doc, "javascript_count")?;
    write_i32(writer, doc, "self_anchors_count")?;
    write_bool(writer, doc, "has_form_with_url")?;
    write_bool(writer, doc, "has_iframe")?;
    write_bool(writer, doc, "use_mouseover")?;

    // Write domain fields.
    write_bool(writer, domain_doc, "is_cert_valid")?;
    write_bool(writer, domain_doc, "has_dns_record")?;
    write_bool(writer, domain_doc, "has_whois")?;
    write_str(writer, domain_doc, "cert_country")?;
    write_i32(writer, domain_doc, "cert_reliability")?;
    write_i64(writer, domain_doc, "domain_age")?;
    write_i64(writer, domain_doc, "domain_end_period")?;
    write_datetime(writer, domain_doc, "domain_creation_date")?;

    // Write refs, or an empty array if the `max_depth` has been reached.
    if depth == max_depth as i32 {
        writer.write_field("[]")?;
    } else {
        let refs = match doc.get_array("refs") {
            Ok(refs) => refs,
            Err(err) => {
                writer.write_field("")?;

                return Err(err.into());
            }
        };
        let mut refs_json = Vec::new();

        for ref_doc in refs.iter().filter_map(|x| x.as_document()) {
            let url = match ref_doc.get_str("url") {
                Ok(url) => url,
                Err(_) => continue,
            };
            let value = serde_json::json!({
                "url": url,
                "is_same_domain": ref_doc.get_bool("is_same_domain").unwrap_or(false),
                "is_anchor": ref_doc.get_bool("is_anchor").unwrap_or(false),
                "is_form": ref_doc.get_bool("is_form").unwrap_or(false),
                "is_iframe": ref_doc.get_bool("is_iframe").unwrap_or(false),
            });

            refs_json.push(value);
        }

        let refs_json_str = format!("{}", serde_json::Value::Array(refs_json));

        writer.write_field(refs_json_str)?;
    }

    Ok(())
}
