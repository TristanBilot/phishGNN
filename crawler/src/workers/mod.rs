mod browser;
mod common;
mod core;
mod domain;

use std::collections::HashMap;
use std::sync::Arc;

use futures::Future;
use mongodb::bson::Document;
use tokio::sync::mpsc::unbounded_channel;
use tokio::sync::{broadcast::Receiver, Mutex};
use tracing::Instrument;

use self::common::{Worker, WorkerPool};

pub use self::{
    browser::{BrowserWorker, BrowserWorkerArgs},
    core::{CoreWorker, CoreWorkerArgs},
    domain::{DomainWorker, DomainWorkerArgs},
};

/// Run up to `capacity` workers of type `W` in parallel with the given
/// `worker_args`. When a worker is ready to process a document, a document will
/// be obtained by calling `next_document` for the worker. End of the processing
/// should be notified by sending a message through `shutdown_rx`.
///
/// If `next_document` returns `None`, it will be retried after a short amount
/// of time.
pub async fn run<W, F, FR>(
    capacity: usize,
    worker_args: W::Args,
    next_document: F,
    exit_if_idle: bool,
    mut shutdown_rx: Receiver<()>,
) -> anyhow::Result<()>
where
    W: Worker,
    F: Fn() -> FR,
    FR: Future<Output = Option<Document>>,
{
    let (drop_tx, mut drop_rx) = unbounded_channel();
    let worker_pool = WorkerPool::<W>::new(drop_tx, capacity, worker_args);

    let dropper_handle = tokio::spawn(async move {
        while let Some(drop_handle) = drop_rx.recv().await {
            let _ = drop_handle.await;
        }
    });

    let tasks = Arc::new(Mutex::new(HashMap::new()));
    let mut tasks_count = 0usize;
    let mut idle_count = 0u8;

    loop {
        let document = tokio::select! {
            document = next_document() => match document {
                Some(document) => document,
                None => {
                    if exit_if_idle {
                        if idle_count == 9 {
                            tracing::info!("crawler is idle, exiting");

                            break;
                        }

                        idle_count += 1;
                    }

                    tokio::time::sleep(tokio::time::Duration::from_millis(100)).await;
                    continue;
                },
            },
            _ = shutdown_rx.recv() => break,
        };

        idle_count = 0;

        let task_id = tasks_count;
        tasks_count += 1;

        let task_span = tracing::trace_span!("task", task_id);
        let _guard = task_span.enter();
        let task_span = task_span.clone();
        let tasks_copy = tasks.clone();
        let worker = worker_pool.get().instrument(task_span.clone()).await?;

        tasks.lock().await.insert(
            task_id,
            tokio::spawn(async move {
                if let Err(err) = worker
                    .process(document)
                    .instrument(tracing::trace_span!(parent: task_span.clone(), "work", task_id))
                    .await
                {
                    tracing::error!(error = ?err, "error processing document");
                }

                tasks_copy.lock().await.remove(&task_id);
            }),
        );
    }

    let tasks = std::mem::take(&mut *tasks.lock().await);

    // We can't simply assign do `let Crawler { bg_handle, .. } = crawler;`,
    // because this will **not** drop the other fields of `crawler`, which
    // will lead to some `Arc`s not being dropped, and a deadlock below.

    futures::future::join_all(tasks.into_values()).await;

    drop(worker_pool);

    dropper_handle.await?;

    anyhow::Ok(())
}
