use std::{
    collections::{hash_map::Entry, HashMap, HashSet},
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use encoding_rs::Encoding;
use mime::Mime;
use mongodb::{
    bson::{doc, Bson, Document},
    options::{InsertManyOptions, InsertOneOptions, UpdateOptions},
    Collection,
};
use reqwest::{redirect::Policy, Url};
use scraper::{ElementRef, Html};
use unicode_segmentation::UnicodeSegmentation;

use crate::{
    utils::{get_duplicate_indices, ignore_duplicate_error},
    workers::common::update_document,
    ProcessingState,
};

use super::common::{without_prefix, Worker};

#[derive(Clone)]
pub struct CoreWorkerArgs {
    pub pages_collection: Collection<Document>,
    pub domains_collection: Collection<Document>,
}

pub struct CoreWorker {
    args: CoreWorkerArgs,
}

impl CoreWorker {
    const REV: u32 = 6;
}

#[async_trait::async_trait]
impl Worker for CoreWorker {
    type Args = CoreWorkerArgs;

    async fn new(args: &CoreWorkerArgs) -> anyhow::Result<Self> {
        Ok(CoreWorker { args: args.clone() })
    }

    async fn process(&self, document: Document) -> anyhow::Result<()> {
        let url_str = document.get_str("url")?;
        let depth = document.get_i32("depth")?;

        tracing::info!(url = url_str, depth, "processing page");

        // Connect to the site and extract features.
        let url = Url::parse(url_str)?;
        let (doc, refs) = match get_counting_redirects(&url).await {
            Ok((resp, redirects)) => {
                let resp_status = resp.status().as_u16();
                let resp_addr = resp.remote_addr().map(|x| x.to_string());

                match read_text_with_limit(resp, DEFAULT_MAX_BYTES).await {
                    Ok(Ok(resp_text)) => {
                        extract_features(&url, redirects, resp_text, resp_status, resp_addr)
                    }
                    Ok(Err(err)) => match err {
                        ReadTextError::TooLarge => {
                            (doc! { "is_size_error": true }, Default::default())
                        }
                        ReadTextError::InvalidData => {
                            (doc! { "is_read_error": true }, Default::default())
                        }
                    },
                    Err(_) => (doc! { "is_read_error": true }, Default::default()),
                }
            }
            Err(err) => {
                let doc = if err.is_timeout() {
                    doc! { "is_timeout_error": true }
                } else if err.is_redirect() {
                    doc! { "is_redirect_error": true }
                } else if err.is_status() {
                    doc! { "is_status_error": true }
                } else {
                    doc! { "is_unknown_error": true }
                };

                (doc, Default::default())
            }
        };

        // Update the document representing the site with all extracted features.
        update_document(&self.args.pages_collection, doc! { "url": url_str }, doc).await?;

        // Create new documents for referenced sites.
        if !refs.is_empty() {
            // Set `ordered = false` to ignore errors (thrown when a document
            // already exists for the given URL) and keep writing.
            let urls = refs
                .iter()
                .map(|x| x.url.as_str())
                .collect::<HashSet<_>>()
                .into_iter()
                .collect::<Vec<_>>();
            let duplicate_indices = get_duplicate_indices(
                self.args
                    .pages_collection
                    .insert_many(
                        urls.iter()
                            .map(|url| doc! { "url": url, "depth": depth + 1 }),
                        InsertManyOptions::builder().ordered(false).build(),
                    )
                    .await,
            )?;

            if let Some(duplicate_indices) = duplicate_indices {
                // Some documents already existed. To make sure we process them
                // if needed, we update their depth.
                let urls = Bson::Array(
                    duplicate_indices
                        .iter()
                        .map(|i| Bson::String(urls[*i].to_string()))
                        .collect(),
                );

                self.args
                    .pages_collection
                    .update_many(
                        doc! { "url": { "$in": urls } },
                        doc! { "$min": { "depth": depth + 1 } },
                        UpdateOptions::builder().build(),
                    )
                    .await?;
            }
        }

        // Create a new document for the domain.
        ignore_duplicate_error(
            self.args
                .domains_collection
                .insert_one(
                    doc! { "domain": url.host_str().unwrap() },
                    InsertOneOptions::builder().build(),
                )
                .await,
        )?;

        Ok(())
    }

    async fn clear(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn destroy(self) -> anyhow::Result<()> {
        Ok(())
    }
}

fn extract_features(
    url: &Url,
    redirects: u16,
    resp_text: String,
    resp_status: u16,
    resp_addr: Option<String>,
) -> (Document, Vec<Ref>) {
    // Create the basic document.
    let mut doc = doc! {
        "rev": CoreWorker::REV,
        "status_code": resp_status as u32,
        "addr": resp_addr,
        "redirects": redirects as u32,
        "is_error_page": !(200..=299).contains(&resp_status),
    };

    // Extract basic features.
    extract_url_features(url, &mut doc);

    // Parse the contents of the site as HTML.
    let html = Html::parse_document(&resp_text);

    // Remember if the HTML is fully valid. We keep going despite
    // encountering errors because some errors are insignificant and do not
    // hinder processing. If errors are significant (e.g. the document is
    // actually JSON), the `root_element` will be a dummy node without any
    // contents in its `<body>`, so we can process it nonetheless.
    doc.insert("is_valid_html", html.errors.is_empty());

    // Extract HTML features.
    let refs = extract_content_features(html.root_element(), url, &mut doc);

    // Update state.
    let require_browser_processing = false; // TODO

    doc.insert(
        "state",
        if require_browser_processing {
            ProcessingState::PendingBrowser
        } else {
            ProcessingState::Complete
        } as u32,
    );

    (doc, refs)
}

/// A reference from one page to another.
#[derive(PartialEq, Eq, Hash)]
struct Ref {
    url: Url,
    is_same_domain: bool,
    source: RefSource,
}

#[derive(PartialEq, Eq, Hash)]
enum RefSource {
    Form,
    Anchor,
    Iframe,
}

impl Ref {
    fn new(url: Url, source_url: &Url, source: RefSource) -> Self {
        let is_same_domain = match (
            url.host_str().and_then(|x| addr::parse_domain_name(x).ok()),
            source_url
                .host_str()
                .and_then(|x| addr::parse_domain_name(x).ok()),
        ) {
            (Some(domain), Some(source_domain)) => {
                without_prefix(&domain) == without_prefix(&source_domain)
            }
            _ => url.host() == source_url.host(),
        };

        Self {
            is_same_domain,
            url,
            source,
        }
    }

    fn to_document(&self) -> Document {
        doc! {
            "url": self.url.as_str(),
            "is_same_domain": self.is_same_domain,
            "is_form": self.source == RefSource::Form,
            "is_anchor": self.source == RefSource::Anchor,
            "is_iframe": self.source == RefSource::Iframe,
        }
    }
}

fn extract_url_features(url: &Url, doc: &mut Document) {
    // Scheme-based features.
    doc.insert("is_https", url.scheme() == "https");

    // Host-based features.
    let host = url.host().unwrap(); // Host must be set to be processed.

    if let url::Host::Domain(domain) = host {
        doc.insert("is_ip_address", false);
        doc.insert("domain_url_length", domain.len() as i32);

        let domain_url_depth = domain.chars().filter(|&x| x == '.').count() as i32;

        crate::insert_into_doc!(doc, domain_url_depth);

        doc.insert("has_sub_domain", domain_url_depth >= 2);
        doc.insert(
            "dashes_count",
            domain.chars().filter(|&x| x == '-').count() as i32,
        );
    } else {
        doc.insert("is_ip_address", true);
    }

    // Full URL features.
    doc.insert("has_at_symbol", url.as_str().contains('@'));
    doc.insert("url_length", url.as_str().len() as u32);

    let path_starts_with_url = if let Some(name) = url
        .path_segments()
        .and_then(|mut x| x.next())
        .and_then(|x| addr::parse_domain_name(x).ok())
    {
        name.has_known_suffix()
    } else {
        false
    };

    crate::insert_into_doc!(doc, path_starts_with_url);
}

fn extract_content_features(root: ElementRef, current_url: &Url, doc: &mut Document) -> Vec<Ref> {
    // Count words and derive features.
    let words_hist = count_words(root);
    let distinct_words_count = words_hist.len() as u32;
    let mean_word_length = if words_hist.is_empty() {
        0
    } else {
        (words_hist.keys().map(|x| x.len()).sum::<usize>() / words_hist.len()) as u32
    };

    crate::insert_into_doc!(doc, distinct_words_count, mean_word_length);

    // Count refs and other elements.
    let mut refs = Vec::new();
    let mut invalid_refs_count = 0;
    let mut self_anchors_count = 0;
    let mut has_iframe = false;
    let mut use_mouseover = false;
    let mut has_form_with_url = false;
    let mut anchors_count = 0;
    let mut forms_count = 0;
    let mut javascript_count = 0;

    // For each element in the page with an `href` attribute, create a link to
    // the site. Also perform other analysis at the same time.
    for (node, el) in root
        .descendants()
        .filter_map(|x| Some((x, x.value().as_element()?)))
    {
        let (source, path) = match el.name() {
            "a" => {
                anchors_count += 1;

                let href = match el.attr("href") {
                    Some(href) => href,
                    None => continue,
                };

                if href.starts_with('#') {
                    self_anchors_count += 1;
                }

                (RefSource::Anchor, href)
            }
            "iframe" => {
                has_iframe = true;

                let src = match el.attr("src") {
                    Some(src) => src,
                    None => continue,
                };

                (RefSource::Iframe, src)
            }
            "form" => {
                forms_count += 1;

                if let Some(href) = el.attr("action") {
                    if current_url.join(href).is_ok() {
                        has_form_with_url = true;
                    }
                }

                let action = match el.attr("action") {
                    Some(action) => action,
                    None => continue,
                };

                (RefSource::Form, action)
            }
            "script" => {
                use_mouseover = use_mouseover
                    || node.children().any(|x| {
                        x.value()
                            .as_text()
                            .map(|x| x.contains("mouseover"))
                            .unwrap_or(false)
                    });

                continue;
            }
            _ => continue,
        };

        let mut url = match current_url.join(path) {
            Ok(url) => url,
            Err(_) => {
                invalid_refs_count += 1;
                continue;
            }
        };

        // Note: `scheme()` is always in lower case.
        if url.scheme() == "javascript" {
            javascript_count += 1;

            continue;
        }

        if url.host().is_none() || (url.scheme() != "http" && url.scheme() != "https") {
            continue;
        }

        url.set_fragment(None);

        refs.push(Ref::new(url, current_url, source));
    }

    crate::insert_into_doc!(
        doc,
        has_iframe,
        use_mouseover,
        has_form_with_url,
        anchors_count,
        forms_count,
        javascript_count,
        self_anchors_count,
        invalid_refs_count,
    );

    doc.insert(
        "refs",
        refs.iter().map(Ref::to_document).collect::<Vec<_>>(),
    );

    refs.into_iter().collect()
}

fn count_words(root: ElementRef) -> HashMap<String, usize> {
    let mut words_hist = HashMap::new();
    let mut add_word = |word| match words_hist.entry(word) {
        Entry::Occupied(mut entry) => *entry.get_mut() += 1usize,
        Entry::Vacant(entry) => {
            entry.insert(1);
        }
    };
    let mut curr_word = String::new();
    let mut last_word = "";

    for text in root.text() {
        let words = text.unicode_words();
        last_word = "";

        for word in words {
            if !curr_word.is_empty() {
                // We can extend the previous text, as it ended with a word.
                if word.as_ptr() == text.as_ptr() {
                    // But only do it if the text starts with this word.
                    curr_word.push_str(word);
                    add_word(curr_word.to_lowercase());
                    curr_word.truncate(curr_word.len() - word.len());
                }
            }

            add_word(word.to_lowercase());
            last_word = word;
        }

        if (last_word.as_ptr() as usize) + last_word.len() == (text.as_ptr() as usize) + text.len()
        {
            // The text ends with the last word, so we can extend it during the
            // next iteration.
            curr_word.push_str(last_word);
        } else {
            curr_word.clear();
        }
    }

    if !curr_word.is_empty() && last_word.is_empty() {
        add_word(curr_word);
    }

    words_hist
}

/// Wrapper around [`reqwest::get`] which also counts the number of redirects
/// followed before getting to the response. Errors after 10 redirects (the
/// default behavior of [`reqwest::redirect::Policy`] at the time of writing).
///
/// Additionally:
/// - The request times out after 10 seconds.
/// - Only HTML is accepted (though the text of the response may not be HTML).
async fn get_counting_redirects(url: &Url) -> reqwest::Result<(reqwest::Response, u16)> {
    let redirects = Arc::new(std::sync::atomic::AtomicU16::new(0));
    let redirects_result = redirects.clone();

    let client = reqwest::Client::builder()
        .timeout(Duration::from_secs(10))
        .redirect(Policy::custom(move |attempt| {
            if attempt.previous().len() == 10 {
                attempt.error("too many redirects")
            } else {
                redirects.store(attempt.previous().len() as u16, Ordering::Relaxed);

                attempt.follow()
            }
        }))
        .build()?;
    let resp = client
        .get(url.clone())
        .header("Accept", "text/html,application/xhtml+xml")
        .send()
        .await?;
    drop(client);
    let redirects_result = Arc::try_unwrap(redirects_result).unwrap().into_inner();

    Ok((resp, redirects_result))
}

#[cfg_attr(test, derive(Debug, Clone, Copy, PartialEq))]
enum ReadTextError {
    TooLarge,
    InvalidData,
}

/// Same as [`reqwest::Response::text()`], but limits the number of bytes in the
/// response.
async fn read_text_with_limit(
    mut resp: reqwest::Response,
    max_bytes: usize,
) -> reqwest::Result<Result<String, ReadTextError>> {
    // Allocate resulting string if possible.
    let mut result = if let Some(content_length) = resp.content_length() {
        if content_length > max_bytes as u64 {
            return Ok(Err(ReadTextError::TooLarge));
        }

        String::with_capacity(content_length as usize)
    } else {
        String::new()
    };

    // Determine encoding of response body; logic extracted from
    // `reqwest::Response::text`.
    let content_type = resp
        .headers()
        .get(reqwest::header::CONTENT_TYPE)
        .and_then(|value| value.to_str().ok())
        .and_then(|value| value.parse::<Mime>().ok());
    let encoding_name = content_type
        .as_ref()
        .and_then(|mime| mime.get_param("charset").map(|charset| charset.as_str()));
    let encoding = encoding_name
        .and_then(|name| Encoding::for_label(name.as_bytes()))
        .unwrap_or(encoding_rs::UTF_8);
    let mut decoder = encoding.new_decoder();

    // Read and decode response.
    let mut read_bytes = 0;
    let mut last_chunk = None;

    loop {
        let (chunk, is_last) = match resp.chunk().await? {
            Some(chunk) => {
                if read_bytes + chunk.len() > max_bytes {
                    return Ok(Err(ReadTextError::TooLarge));
                }

                match std::mem::replace(&mut last_chunk, Some(chunk)) {
                    Some(chunk) => (chunk, false),
                    None => continue,
                }
            }
            None => match last_chunk.take() {
                Some(last_chunk) => (last_chunk, true),
                None => break,
            },
        };

        let max_utf8_buffer_length = match decoder.max_utf8_buffer_length(chunk.len()) {
            Some(len) => len,
            None => return Ok(Err(ReadTextError::TooLarge)),
        };

        result.reserve(max_utf8_buffer_length);

        let (result, _, is_invalid) = decoder.decode_to_string(&chunk, &mut result, is_last);

        if is_invalid {
            return Ok(Err(ReadTextError::InvalidData));
        }

        match result {
            encoding_rs::CoderResult::InputEmpty => (),
            encoding_rs::CoderResult::OutputFull => {
                tracing::error!("output full when reading response chunk");

                return Ok(Err(ReadTextError::TooLarge));
            }
        }

        if is_last {
            break;
        }

        read_bytes += chunk.len();
    }

    Ok(Ok(result))
}

/// The default number of bytes to read before giving up in
/// [`read_text_with_limit`] (1MiB).
const DEFAULT_MAX_BYTES: usize = 1 << 20;

#[cfg(test)]
mod tests {
    use mongodb::bson::Document;
    use reqwest::Url;

    use crate::workers::core::{read_text_with_limit, ReadTextError, DEFAULT_MAX_BYTES};

    use super::{get_counting_redirects, Ref};

    async fn extract_features(url: &str) -> Document {
        let req_url = Url::parse(url).unwrap();
        let test_data = crate::workers::common::TEST_DATA.lock().await;

        let (raw_html, status_code, addr, redirects) = match test_data.get(url).unwrap() {
            Some(data) => {
                // Recover saved data.
                let mut cursor = std::io::Cursor::new(data);
                let doc = Document::from_reader(&mut cursor).unwrap();

                (
                    doc.get_str("html").unwrap().to_owned(),
                    doc.get_i32("status").unwrap() as u16,
                    doc.get_str("addr").map(str::to_string).ok(),
                    doc.get_i32("redirects").unwrap() as u16,
                )
            }
            None => {
                let (resp, redirects) = get_counting_redirects(&req_url).await.unwrap();
                let status_code = resp.status().as_u16();
                let resp_addr = resp.remote_addr().map(|x| x.to_string());
                let resp_text = read_text_with_limit(resp, DEFAULT_MAX_BYTES)
                    .await
                    .unwrap()
                    .unwrap();

                // Persist result to make sure repeats of this test process the
                // same data.
                let doc = mongodb::bson::doc! {
                    "html": resp_text.as_str(),
                    "date": mongodb::bson::DateTime::now().to_rfc3339_string(),
                    "status": status_code as u32,
                    "redirects": redirects as u32,
                    "addr": resp_addr.clone(),
                };
                let mut data = Vec::new();
                doc.to_writer(&mut data).unwrap();

                test_data.insert(url, data).unwrap();

                (resp_text, status_code, resp_addr, redirects)
            }
        };

        super::extract_features(&req_url, redirects, raw_html, status_code, addr).0
    }

    #[tokio::test]
    async fn basic() {
        let doc = extract_features("https://en.wikipedia.org").await;

        assert_eq!(doc.get_i32("invalid_refs_count"), Ok(0));
        assert_eq!(doc.get_i32("self_anchors_count"), Ok(2));

        let refs = doc.get_array("refs").unwrap();

        for r in refs.iter().map(|x| x.as_document().unwrap()) {
            let url = r.get_str("url").unwrap();

            assert_eq!(
                r.get_bool("is_same_domain"),
                Ok(url.contains("wikipedia.org/"))
            );
            assert!(
                r.get_bool("is_form").unwrap()
                    || r.get_bool("is_anchor").unwrap()
                    || r.get_bool("is_iframe").unwrap()
            );
        }

        assert_eq!(doc.get_bool("is_https"), Ok(true));
        assert_eq!(doc.get_bool("is_ip_address"), Ok(false));
        assert_eq!(doc.get_i32("domain_url_length"), Ok(16));
        assert_eq!(doc.get_i32("domain_url_depth"), Ok(2));
        assert_eq!(doc.get_bool("has_sub_domain"), Ok(true));
        assert_eq!(doc.get_i32("dashes_count"), Ok(0));
        assert_eq!(doc.get_bool("has_at_symbol"), Ok(false));
        assert_eq!(doc.get_bool("is_error_page"), Ok(false));

        assert_eq!(doc.get_bool("has_iframe"), Ok(false));
        assert_eq!(doc.get_bool("has_form_with_url"), Ok(true));
        assert_eq!(doc.get_i32("redirects"), Ok(1));
    }

    #[tokio::test]
    async fn by_ip() {
        let doc = extract_features("https://1.1.1.1").await;

        assert_eq!(doc.get_bool("is_https"), Ok(true));
        assert_eq!(doc.get_bool("is_ip_address"), Ok(true));
        assert_eq!(doc.get_bool("has_at_symbol"), Ok(false));
        assert_eq!(doc.get_bool("is_error_page"), Ok(false));

        assert_eq!(doc.get_bool("has_iframe"), Ok(false));
        assert_eq!(doc.get_bool("has_form_with_url"), Ok(false));
        assert_eq!(doc.get_i32("redirects"), Ok(0));

        for k in [
            "domain_url_length",
            "domain_url_depth",
            "has_sub_domain",
            "dashes_count",
        ] {
            assert!(!doc.contains_key(k));
        }
    }

    #[tokio::test]
    async fn has_error() {
        let doc = extract_features("https://httpbingo.org/status/404").await;

        assert_eq!(doc.get_bool("is_error_page"), Ok(true));
        assert_eq!(doc.get_bool("has_form_with_url"), Ok(false));
    }

    #[tokio::test]
    async fn has_form() {
        let doc = extract_features("https://httpbingo.org/forms/post").await;

        assert_eq!(doc.get_bool("has_form_with_url"), Ok(true));
    }

    #[tokio::test]
    async fn count_redirects() {
        assert_eq!(
            extract_features("https://httpbingo.org/redirect/1")
                .await
                .get_i32("redirects"),
            Ok(1)
        );
        assert_eq!(
            extract_features("https://httpbingo.org/redirect/4")
                .await
                .get_i32("redirects"),
            Ok(4)
        );
    }

    #[tokio::test]
    async fn read_text_with_limit_decodes_well() {
        async fn assert_same_result(url: &str, max_bytes: usize) {
            let with_reqwest = reqwest::get(url).await.unwrap().text().await.unwrap();
            let without_reqwest = read_text_with_limit(reqwest::get(url).await.unwrap(), max_bytes)
                .await
                .unwrap()
                .unwrap();

            assert_eq!(with_reqwest, without_reqwest);
        }

        assert_same_result("https://en.wikipedia.org/", DEFAULT_MAX_BYTES).await;
        assert_same_result("https://httpbingo.org/encoding/utf8", DEFAULT_MAX_BYTES).await;
        assert_same_result("https://262.ecma-international.org/12.0/", 1 << 24).await;
    }

    #[tokio::test]
    async fn read_text_with_limit_respects_limit() {
        async fn read_with_limit(url: &str, max_bytes: usize) -> Result<String, ReadTextError> {
            read_text_with_limit(reqwest::get(url).await.unwrap(), max_bytes)
                .await
                .unwrap()
        }

        assert!(read_with_limit("https://httpbingo.org/", DEFAULT_MAX_BYTES)
            .await
            .is_ok());
        assert_eq!(
            read_with_limit("https://httpbingo.org/", 1000).await,
            Err(ReadTextError::TooLarge)
        );
    }

    #[test]
    fn count_words() {
        fn words(html: &str) -> Vec<(String, usize)> {
            let root = scraper::Html::parse_fragment(html);
            let mut words = super::count_words(root.root_element())
                .into_iter()
                .collect::<Vec<_>>();

            words.sort();
            words
        }

        assert_eq!(
            words("<a>abc def!ghi <b>abc</b></a>"),
            &[("abc".into(), 2), ("def".into(), 1), ("ghi".into(), 1)],
        );

        assert_eq!(
            words("<span><i>a</i><i>b</i><i>c </i><i>d</i> e</span>"),
            &[
                ("a".into(), 1),
                ("ab".into(), 1),
                ("abc".into(), 1),
                ("b".into(), 1),
                ("c".into(), 1),
                ("d".into(), 1),
                ("e".into(), 1),
            ],
        );
    }

    #[test]
    fn create_ref() {
        fn mkref(url: &str, source_url: &str) -> Ref {
            Ref::new(
                Url::parse(url).unwrap(),
                &Url::parse(source_url).unwrap(),
                super::RefSource::Anchor,
            )
        }

        // Same URL.
        assert!(mkref("https://en.wikipedia.org/", "https://en.wikipedia.org/").is_same_domain);

        // Same domain.
        assert!(
            mkref(
                "https://en.wikipedia.org/foo",
                "https://en.wikipedia.org/bar"
            )
            .is_same_domain
        );

        // Same domain (no subdomain).
        assert!(mkref("https://wikipedia.org/", "https://wikipedia.org/").is_same_domain);
        assert!(mkref("https://wikipedia.org/", "https://en.wikipedia.org/").is_same_domain);

        // Same domain, different prefix.
        assert!(mkref("https://en.wikipedia.org/", "https://fr.wikipedia.org/").is_same_domain);

        // Same domain, different capitalization and prefix.
        assert!(mkref("https://en.Wikipedia.Org/", "https://fr.wikipediA.orG/").is_same_domain);

        // Same domain, different prefix depth.
        assert!(mkref("https://en.wikipedia.org/", "https://en.m.wikipedia.org/").is_same_domain);
        assert!(mkref("https://en.wikipedia.org/", "https://fr.m.wikipedia.org/").is_same_domain);

        // Same as above, but reversed.
        assert!(mkref("https://en.m.wikipedia.org/", "https://en.wikipedia.org/").is_same_domain);
        assert!(mkref("https://fr.m.wikipedia.org/", "https://en.wikipedia.org/").is_same_domain);

        // Different domains.
        assert!(!mkref("https://www.google.com/", "https://en.wikipedia.org/").is_same_domain);
        assert!(!mkref("https://www.google.com/", "https://www.google.fr/").is_same_domain);

        // Same domain, different prefix (with complex suffix).
        assert!(mkref("https://en.wikipedia.co.kr/", "https://fr.wikipedia.co.kr/").is_same_domain);
        assert!(mkref("https://wikipedia.co.kr/", "https://wikipedia.co.kr/").is_same_domain);
        assert!(!mkref("https://wikipedia.co.kr/", "https://google.co.kr/").is_same_domain);
    }
}
