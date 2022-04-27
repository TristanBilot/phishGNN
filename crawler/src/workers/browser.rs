use mongodb::{
    bson::{doc, Document},
    Collection,
};
use reqwest::Url;
use thirtyfour::{DesiredCapabilities, WebDriver};

use super::common::{update_document, Worker};

#[derive(Clone)]
pub struct BrowserWorkerArgs {
    pub pages_collection: Collection<Document>,
}

pub struct BrowserWorker {
    args: BrowserWorkerArgs,
    webdriver: WebDriver,
}

#[async_trait::async_trait]
impl Worker for BrowserWorker {
    type Args = BrowserWorkerArgs;

    async fn new(args: &BrowserWorkerArgs) -> anyhow::Result<Self> {
        let mut caps = DesiredCapabilities::chrome();
        caps.set_headless()?;
        let webdriver = WebDriver::new("http://127.0.0.1:9515", &caps).await?;

        Ok(BrowserWorker {
            args: args.clone(),
            webdriver,
        })
    }

    async fn process(&self, document: Document) -> anyhow::Result<()> {
        let req_url_str = document.get_str("url").unwrap();
        let req_url = Url::parse(req_url_str)?;
        let mut update_doc = Document::new();

        BrowserBasedFeatureExtractor::new(&self.webdriver, &mut update_doc, &req_url)
            .await?
            .extract_features()
            .await?;

        update_document(
            &self.args.pages_collection,
            doc! { "url": req_url_str },
            update_doc,
        )
        .await?;

        Ok(())
    }

    async fn clear(&mut self) -> anyhow::Result<()> {
        self.webdriver.delete_all_cookies().await?;

        Ok(())
    }

    async fn destroy(self) -> anyhow::Result<()> {
        if let Err(err) = self.webdriver.quit().await {
            tracing::error!(error = ?err, "error dropping webdriver");
        }

        Ok(())
    }
}

/// A heavy, [WebDriver](https://chromedriver.chromium.org/)-based feature
/// extractor.
#[allow(dead_code)]
struct BrowserBasedFeatureExtractor<'a> {
    doc: &'a mut Document,
    req_url: &'a Url,
    driver: &'a WebDriver,
}

impl<'a> BrowserBasedFeatureExtractor<'a> {
    async fn new(
        driver: &'a WebDriver,
        doc: &'a mut Document,
        req_url: &'a Url,
    ) -> anyhow::Result<BrowserBasedFeatureExtractor<'a>> {
        driver.get(req_url.as_str()).await?;

        Ok(BrowserBasedFeatureExtractor {
            doc,
            req_url,
            driver,
        })
    }

    async fn extract_features(&self) -> anyhow::Result<()> {
        // TODO

        Ok(())
    }
}
