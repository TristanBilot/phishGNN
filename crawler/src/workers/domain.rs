use std::{collections::HashMap, sync::Arc, time::Duration};

use mongodb::{
    bson::{doc, DateTime, Document},
    Collection,
};
use once_cell::sync::Lazy;
use tokio::{io::AsyncWriteExt, net::TcpStream};
use tokio_rustls::{
    rustls::{ClientConfig, ClientConnection, OwnedTrustAnchor, RootCertStore},
    TlsConnector,
};
use whois_rust::{WhoIs, WhoIsLookupOptions};
use x509_parser::{prelude::X509Certificate, traits::FromDer};

use super::common::{update_document, without_prefix, Worker};

#[derive(Clone)]
pub struct DomainWorkerArgs {
    pub domains_collection: Collection<Document>,
}

pub struct DomainWorker {
    args: DomainWorkerArgs,
}

impl DomainWorker {
    // 1 -> 2: added cert_reliability
    const REV: u32 = 2;
}

#[async_trait::async_trait]
impl Worker for DomainWorker {
    type Args = DomainWorkerArgs;

    async fn new(args: &DomainWorkerArgs) -> anyhow::Result<Self> {
        Ok(DomainWorker { args: args.clone() })
    }

    async fn process(&self, document: Document) -> anyhow::Result<()> {
        let domain = document.get_str("domain")?;

        tracing::info!(domain, "processing domain");

        let update_doc = extract_features(domain).await?;

        update_document(
            &self.args.domains_collection,
            doc! { "domain": domain },
            update_doc,
        )
        .await?;

        Ok(())
    }

    async fn clear(&mut self) -> anyhow::Result<()> {
        Ok(())
    }

    async fn destroy(self) -> anyhow::Result<()> {
        Ok(())
    }
}

async fn extract_features(domain: &str) -> anyhow::Result<Document> {
    let mut doc = doc! { "rev": DomainWorker::REV };
    let domain_url = url::Url::parse(&format!("http://{domain}"))?;

    // Whois features.
    if let Some(url::Host::Domain(_)) = domain_url.host() {
        if let Ok(domain_name) = addr::parse_domain_name(domain) {
            extract_whois_features(&mut doc, domain_name).await?;
        }
    }

    // Certificate features.
    let mut is_cert_valid = false;

    match get_domain_tls_connection(domain).await {
        Ok(tls_conn) => match get_certificates(&tls_conn) {
            Ok(certs) => {
                if !certs.is_empty() {
                    let server_cert = &certs[0];

                    is_cert_valid = server_cert.validity.is_valid();

                    doc.insert(
                        "cert_country",
                        server_cert
                            .subject()
                            .iter_country()
                            .next()
                            .or_else(|| server_cert.issuer().iter_country().next())
                            .and_then(|x| x.as_str().ok()),
                    );

                    let auth_cert = certs.last().unwrap();

                    doc.insert("cert_reliability", get_certificate_reliability(auth_cert));
                }
            }
            Err(err) => {
                tracing::error!(domain, error = ?err, "error parsing certificate");
            }
        },
        Err(err) => {
            let is_invalid_cert_error = err.downcast_ref::<tokio_rustls::webpki::Error>().is_some();

            if !is_invalid_cert_error {
                tracing::error!(domain, error = ?err, "error fetching certificate");
            }
        }
    }

    crate::insert_into_doc!(doc, is_cert_valid);

    Ok(doc)
}

static WHOIS: Lazy<WhoIs> =
    Lazy::new(|| WhoIs::from_string(include_str!("./domain_whois_servers.json")).unwrap());

async fn extract_whois_features(
    document: &mut Document,
    domain_name: addr::domain::Name<'_>,
) -> anyhow::Result<()> {
    let domain_without_subdomain = without_prefix(&domain_name);
    let mut whois_lookup_options = WhoIsLookupOptions::from_str(domain_without_subdomain).unwrap();
    let mut has_whois = false;
    let mut has_dns_record = false;

    whois_lookup_options.timeout = Some(Duration::from_millis(1_000));

    match WHOIS.lookup_async(whois_lookup_options).await {
        Ok(whois) => {
            // TODO: a lot of domains cannot be found; do something about this.
            // out of 1,080 tested domains, only 67 returned a valid whois.

            if !whois.starts_with("No match")
                && !whois.starts_with("no match")
                && !whois.starts_with("NOT FOUND")
            {
                let mut data = HashMap::new();

                // Attempt to parse WHOIS string.
                let mut chars = whois.char_indices();
                let is_valid = 'outer: loop {
                    // Parse key.
                    let mut key_start = None;
                    let key = loop {
                        match chars.next() {
                            Some((_, ' ')) if key_start.is_none() => {
                                // Skip leading spaces.
                                continue;
                            }
                            Some((_, '>')) if key_start.is_none() => {
                                // Probably ">>>", which marks the end of the data.
                                break 'outer true;
                            }
                            Some((key_end, ':')) => match key_start {
                                Some(key_start) => {
                                    break &whois[key_start..key_end];
                                }
                                None => break 'outer false,
                            },
                            Some((position, _)) => {
                                if key_start.is_none() {
                                    key_start = Some(position);
                                }
                            }
                            None => {
                                break 'outer key_start.is_none();
                            }
                        }
                    };

                    // Parse value.
                    let mut value_start = None;
                    let mut value_end = None;

                    let value = loop {
                        match chars.next() {
                            Some((_, ' ')) => {
                                if value_start.is_none() {
                                    // Skip leading spaces.
                                    continue;
                                }
                            }
                            Some((value_end, '\n')) => {
                                break &whois[value_start.unwrap_or(value_end)..value_end];
                            }
                            Some((position, _)) => {
                                if value_start.is_none() {
                                    value_start = Some(position);
                                }

                                value_end = Some(position);
                            }
                            None => {
                                break match (value_start, value_end) {
                                    (Some(start), Some(end)) => &whois[start..end],
                                    _ => "",
                                };
                            }
                        }
                    };

                    // Add entry.
                    data.insert(key.trim_end().to_ascii_lowercase(), value.trim_end());
                };

                has_whois = is_valid;

                // Find whois information.
                if is_valid {
                    let creation_date = data
                        .get("creation date")
                        .or_else(|| data.get("registered"))
                        .or_else(|| data.get("entry created"))
                        .or_else(|| data.get("created"))
                        .and_then(|x| DateTime::parse_rfc3339_str(x).ok());

                    if let Some(creation_date) = creation_date {
                        document.insert("domain_creation_date", creation_date);
                    }

                    let end_date = data
                        .get("registry expiry date")
                        .or_else(|| data.get("expires"))
                        .or_else(|| data.get("expire"))
                        .or_else(|| data.get("renewal date"))
                        .and_then(|x| DateTime::parse_rfc3339_str(x).ok());

                    if let Some(end_date) = end_date {
                        document.insert(
                            "domain_end_period",
                            (end_date.timestamp_millis() - DateTime::now().timestamp_millis())
                                / 1_000,
                        );

                        let updated_date = data
                            .get("updated date")
                            .or_else(|| data.get("last modified"))
                            .or_else(|| data.get("changed"))
                            .or_else(|| data.get("entry updated"))
                            .and_then(|x| DateTime::parse_rfc3339_str(x).ok());

                        if let Some(updated_date) = updated_date {
                            document.insert(
                                "domain_age",
                                (end_date.timestamp_millis() - updated_date.timestamp_millis())
                                    / 1_000,
                            );
                        }
                    }

                    has_dns_record = data.contains_key("name server");
                }
            }
        }
        Err(err) => {
            tracing::error!(domain = domain_without_subdomain, error = ?err, "error looking up whois");
        }
    }

    if !has_whois {
        tracing::trace!(domain = domain_without_subdomain, "no whois found");
    }

    crate::insert_into_doc!(document, has_whois, has_dns_record);

    Ok(())
}

/// Returns an arbitrary integer representing the "reliability" of the given
/// (certificate authority) certificate.
///
/// Source: https://github.com/abhishekdid/detecting-phishing-websites/blob/ee6cecc45a3fdaae4ffe9e25c70f2b1425bf4a34/inputScript.py#L65
fn get_certificate_reliability(auth_cert: &X509Certificate) -> u32 {
    let issuer = auth_cert.issuer();
    let auth = match issuer
        .iter_common_name()
        .next()
        .and_then(|x| x.as_str().ok())
    {
        Some(auth) => auth,
        None => return 0,
    };
    let trusted = &[
        "Buypass ",
        "Comodo ",
        "Deutsche Telekom ",
        "DigiCert ",
        "Doster ",
        "Entrust ",
        "GeoTrust ",
        "GlobalSign ",
        "GoDaddy ",
        "IdenTrust ",
        "Network Solutions ",
        "QuoVadis ",
        "Secom ",
        "StartCom ",
        "SwissSign ",
        "Symantec ",
        "Thawte ",
        "Trustwave ",
        "TWCA ",
        "Unizeto ",
        "VeriSign ",
        "Verizon ",
    ];
    let is_trusted = trusted.iter().any(|x| auth.starts_with(x));
    let is_trusted_score = if is_trusted { 1 } else { 0 };
    let certificate_duration =
        (auth_cert.validity().not_after - auth_cert.validity().not_before).unwrap_or_default();
    let certificate_duration_score = if certificate_duration.whole_days() > 365 {
        1
    } else {
        0
    };

    is_trusted_score + certificate_duration_score
}

/// Returns the [`ClientConnection`] obtained after connecting to the given
/// `domain`.
async fn get_domain_tls_connection(domain: &str) -> anyhow::Result<ClientConnection> {
    // Build client config.
    let mut root_store = RootCertStore::empty();
    root_store.add_server_trust_anchors(webpki_roots::TLS_SERVER_ROOTS.0.iter().map(|x| {
        OwnedTrustAnchor::from_subject_spki_name_constraints(x.subject, x.spki, x.name_constraints)
    }));
    let config = ClientConfig::builder()
        .with_safe_defaults()
        .with_root_certificates(root_store)
        .with_no_client_auth();
    let config = Arc::new(config);

    // Connect to domain.
    let stream = TcpStream::connect((domain, 443)).await?;
    let connector = TlsConnector::from(config);
    let mut tls = connector.connect(domain.try_into()?, stream).await?;

    // Send basic GET request to /.
    let req_str = format!(
        "GET / HTTP/1.0\r\n\
         Host: {}\r\n\
         Connection: close\r\n\
         Accept-Encoding: identity\r\n\r\n",
        domain,
    );
    tls.write_all(req_str.as_bytes()).await?;

    // Obtain certificates.
    Ok(tls.into_inner().1)
}

/// Returns the X509 certificates associated with the given
/// [`ClientConnection`].
fn get_certificates(conn: &ClientConnection) -> anyhow::Result<Vec<X509Certificate<'_>>> {
    if let Some(certificates) = conn.peer_certificates() {
        certificates
            .iter()
            .map(|cert| Ok(X509Certificate::from_der(&cert.0)?.1))
            .collect()
    } else {
        Ok(Vec::new())
    }
}

#[cfg(test)]
mod tests {
    use mongodb::bson::DateTime;

    use super::extract_features;

    #[tokio::test]
    async fn basic() {
        let doc = extract_features("google.com").await.unwrap();

        assert_eq!(doc.get_bool("is_cert_valid"), Ok(true));
        assert_eq!(doc.get_str("cert_country"), Ok("US"));

        assert_eq!(doc.get_bool("has_whois"), Ok(true));
        assert_eq!(doc.get_bool("has_dns_record"), Ok(true));
        assert_eq!(
            doc.get_datetime("domain_creation_date"),
            Ok(&DateTime::parse_rfc3339_str("1997-09-15T04:00:00Z").unwrap())
        );
        assert!(doc.contains_key("domain_end_period"));
        assert!(doc.contains_key("domain_age"));
    }

    #[tokio::test]
    async fn badssl() {
        for subdomain in [
            "expired",
            // TODO: "revoked",
            "self-signed",
            "untrusted-root",
            "wrong.host",
        ] {
            let domain = format!("{subdomain}.badssl.com");

            // Compare `&domain` as well to get better error messages.
            assert_eq!(
                (
                    &domain,
                    extract_features(&domain)
                        .await
                        .unwrap()
                        .get_bool("is_cert_valid")
                ),
                (&domain, Ok(false)),
            );
        }
    }
}
