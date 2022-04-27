use deadpool::managed::{Object, PoolError};
use mongodb::{bson::Document, Collection};
use tokio::{sync::mpsc::UnboundedSender, task::JoinHandle};
use tracing::Instrument;

#[async_trait::async_trait]
pub trait Worker: Sized + Send + Sync + 'static {
    type Args: Send + Sync;

    async fn new(args: &Self::Args) -> anyhow::Result<Self>;
    async fn process(&self, document: Document) -> anyhow::Result<()>;
    async fn clear(&mut self) -> anyhow::Result<()>;
    async fn destroy(self) -> anyhow::Result<()>;
}

/// A wrapper around a [`Worker`] which spawns a task that will
/// [destroy](Worker::destroy) the worker on [`Drop`].
pub struct WorkerWrapper<W: Worker> {
    worker: Option<W>,
    drop_tx: UnboundedSender<JoinHandle<()>>,
}

impl<W: Worker> AsRef<W> for WorkerWrapper<W> {
    fn as_ref(&self) -> &W {
        // SAFETY: inner value is always `Some`, except after dropping.
        unsafe { self.worker.as_ref().unwrap_unchecked() }
    }
}

impl<W: Worker> AsMut<W> for WorkerWrapper<W> {
    fn as_mut(&mut self) -> &mut W {
        // SAFETY: inner value is always `Some`, except after dropping.
        unsafe { self.worker.as_mut().unwrap_unchecked() }
    }
}

impl<W: Worker> std::ops::Deref for WorkerWrapper<W> {
    type Target = W;

    fn deref(&self) -> &W {
        self.as_ref()
    }
}

impl<W: Worker> std::ops::DerefMut for WorkerWrapper<W> {
    fn deref_mut(&mut self) -> &mut W {
        self.as_mut()
    }
}

impl<W: Worker> Drop for WorkerWrapper<W> {
    fn drop(&mut self) {
        let worker = self.worker.take().unwrap();

        let _ = self.drop_tx.send(tokio::spawn(async move {
            if let Err(err) = worker.destroy().await {
                tracing::error!(error = ?err, "error destroying worker");
            }
        }));
    }
}

/// A [`Manager`](deadpool::managed::Manager) for a [`WorkerPool`].
pub struct WorkerManager<W: Worker> {
    args: W::Args,
    drop_tx: UnboundedSender<JoinHandle<()>>,
}

#[async_trait::async_trait]
impl<W: Worker> deadpool::managed::Manager for WorkerManager<W> {
    type Type = WorkerWrapper<W>;
    type Error = anyhow::Error;

    async fn create(&self) -> Result<Self::Type, Self::Error> {
        Ok(WorkerWrapper {
            worker: Some(W::new(&self.args).await?),
            drop_tx: self.drop_tx.clone(),
        })
    }

    async fn recycle(
        &self,
        wrapper: &mut Self::Type,
    ) -> deadpool::managed::RecycleResult<Self::Error> {
        wrapper.clear().await?;

        Ok(())
    }
}

pub struct WorkerPool<W: Worker> {
    pool: deadpool::managed::Pool<WorkerManager<W>>,
}

impl<W: Worker> WorkerPool<W> {
    pub fn new(drop_tx: UnboundedSender<JoinHandle<()>>, capacity: usize, args: W::Args) -> Self {
        Self {
            pool: deadpool::managed::Pool::builder(WorkerManager { args, drop_tx })
                .max_size(capacity)
                .build()
                .unwrap(),
        }
    }

    pub async fn get(&self) -> anyhow::Result<Object<WorkerManager<W>>> {
        match self
            .pool
            .get()
            .instrument(tracing::trace_span!("worker creation"))
            .await
        {
            Ok(worker) => Ok(worker),
            Err(PoolError::Backend(err)) => Err(err),
            Err(PoolError::Closed) => Err(PoolError::<NeverError>::Closed.into()),
            Err(PoolError::NoRuntimeSpecified) => {
                Err(PoolError::<NeverError>::NoRuntimeSpecified.into())
            }
            Err(_) => unreachable!(),
        }
    }
}

enum NeverError {}

impl std::fmt::Display for NeverError {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unreachable!()
    }
}

impl std::fmt::Debug for NeverError {
    fn fmt(&self, _f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unreachable!()
    }
}

impl std::error::Error for NeverError {}

/// Updates the document matching the given `filter` in the given `collection`
/// with the given `update` document.
pub async fn update_document(
    collection: &Collection<Document>,
    filter: Document,
    update: Document,
) -> mongodb::error::Result<()> {
    collection
        .update_one(
            filter,
            mongodb::bson::doc! { "$set": update },
            mongodb::options::UpdateOptions::builder().build(),
        )
        .await?;

    Ok(())
}

/// Returns the given domain name without its prefix (or subdomain).
pub fn without_prefix<'a>(domain_name: &'a addr::domain::Name<'a>) -> &'a str {
    if let Some(prefix) = domain_name.prefix() {
        &domain_name.as_str()[prefix.len() + 1..]
    } else {
        domain_name.as_str()
    }
}

#[macro_export]
macro_rules! insert_into_doc {
    ( $doc: expr, $($id: ident),* $(,)? ) => {
        $($doc.insert(stringify!($id), $id);)*
    };
}

#[cfg(test)]
pub static TEST_DATA: once_cell::sync::Lazy<tokio::sync::Mutex<sled::Db>> =
    once_cell::sync::Lazy::new(|| {
        let test_data_path =
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("test/testdata.sled");

        sled::open(test_data_path).unwrap().into()
    });
