//! Network guard (DESIGN §12.6): http(s) only, and no loopback/private/link-local/metadata targets
//! unless `web.allow_private_network` is set, in which case they need approval.

use std::net::{IpAddr, Ipv4Addr, Ipv6Addr, SocketAddr};

use url::Url;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AddrClass {
    Public,
    Loopback,
    Private,
    LinkLocal,
    Metadata,
    Unspecified,
    Multicast,
    Reserved,
}

pub fn classify_ip(ip: IpAddr) -> AddrClass {
    match ip {
        IpAddr::V4(v4) => classify_v4(v4),
        IpAddr::V6(v6) => classify_v6(v6),
    }
}

fn classify_v4(ip: Ipv4Addr) -> AddrClass {
    let o = ip.octets();
    if matches!(
        o,
        [169, 254, 169, 254] | [169, 254, 170, 2] | [100, 100, 100, 200]
    ) {
        AddrClass::Metadata
    } else if ip.is_loopback() {
        AddrClass::Loopback
    } else if ip.is_unspecified() || o[0] == 0 {
        AddrClass::Unspecified
    } else if ip.is_private() || (o[0] == 100 && (64..128).contains(&o[1])) {
        AddrClass::Private
    } else if ip.is_link_local() {
        AddrClass::LinkLocal
    } else if ip.is_multicast() {
        AddrClass::Multicast
    } else if ip.is_broadcast() || o[0] >= 240 {
        AddrClass::Reserved
    } else {
        AddrClass::Public
    }
}

fn classify_v6(ip: Ipv6Addr) -> AddrClass {
    if let Some(v4) = ip.to_ipv4_mapped() {
        return classify_v4(v4);
    }
    let s = ip.segments();
    // NAT64 well-known prefix 64:ff9b::/96 embeds an IPv4 address.
    if s[..6] == [0x64, 0xff9b, 0, 0, 0, 0] {
        let [a, b] = s[6].to_be_bytes();
        let [c, d] = s[7].to_be_bytes();
        return classify_v4(Ipv4Addr::new(a, b, c, d));
    }
    if ip == Ipv6Addr::new(0xfd00, 0xec2, 0, 0, 0, 0, 0, 0x254) {
        AddrClass::Metadata
    } else if ip.is_loopback() {
        AddrClass::Loopback
    } else if ip.is_unspecified() {
        AddrClass::Unspecified
    } else if (s[0] & 0xfe00) == 0xfc00 {
        AddrClass::Private
    } else if (s[0] & 0xffc0) == 0xfe80 {
        AddrClass::LinkLocal
    } else if ip.is_multicast() {
        AddrClass::Multicast
    } else {
        AddrClass::Public
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NetVerdict {
    /// Connect only to `addrs` to avoid DNS rebinding between check and use.
    Allow {
        addrs: Vec<SocketAddr>,
    },
    Ask {
        reason: String,
        addrs: Vec<SocketAddr>,
    },
    Block {
        reason: String,
    },
}

#[derive(Debug, Clone, Copy)]
pub struct NetworkGuard {
    allow_private: bool,
}

impl NetworkGuard {
    pub fn new(allow_private_network: bool) -> Self {
        Self {
            allow_private: allow_private_network,
        }
    }

    /// Validates scheme and host; returns `(host, port)`.
    pub fn target(&self, raw: &str) -> Result<(String, u16), String> {
        let url = Url::parse(raw).map_err(|e| format!("invalid URL: {e}"))?;
        if !matches!(url.scheme(), "http" | "https") {
            return Err(format!(
                "scheme `{}` is not allowed; use http or https",
                url.scheme()
            ));
        }
        let host = url
            .host_str()
            .filter(|h| !h.is_empty())
            .ok_or("URL has no host")?;
        let port = url.port_or_known_default().ok_or("URL has no port")?;
        Ok((
            host.trim_start_matches('[')
                .trim_end_matches(']')
                .to_owned(),
            port,
        ))
    }

    pub fn judge(&self, addrs: &[SocketAddr]) -> NetVerdict {
        if addrs.is_empty() {
            return NetVerdict::Block {
                reason: "host did not resolve".into(),
            };
        }
        let restricted: Vec<(SocketAddr, AddrClass)> = addrs
            .iter()
            .map(|a| (*a, classify_ip(a.ip())))
            .filter(|(_, c)| *c != AddrClass::Public)
            .collect();
        match restricted.first() {
            None => NetVerdict::Allow {
                addrs: addrs.to_vec(),
            },
            Some((addr, class)) => {
                let reason = format!("{} resolves to a {class:?} address", addr.ip());
                if self.allow_private {
                    NetVerdict::Ask {
                        reason,
                        addrs: addrs.to_vec(),
                    }
                } else {
                    NetVerdict::Block { reason }
                }
            }
        }
    }

    pub async fn check(&self, raw: &str) -> NetVerdict {
        let (host, port) = match self.target(raw) {
            Ok(t) => t,
            Err(reason) => return NetVerdict::Block { reason },
        };
        match tokio::net::lookup_host((host.as_str(), port)).await {
            Ok(addrs) => self.judge(&addrs.collect::<Vec<_>>()),
            Err(e) => NetVerdict::Block {
                reason: format!("cannot resolve {host}: {e}"),
            },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn class(s: &str) -> AddrClass {
        classify_ip(s.parse().unwrap())
    }

    #[test]
    fn classifies_addresses() {
        for (ip, want) in [
            ("8.8.8.8", AddrClass::Public),
            ("2606:4700::1111", AddrClass::Public),
            ("127.0.0.1", AddrClass::Loopback),
            ("127.8.9.10", AddrClass::Loopback),
            ("::1", AddrClass::Loopback),
            ("10.1.2.3", AddrClass::Private),
            ("172.16.0.1", AddrClass::Private),
            ("192.168.1.1", AddrClass::Private),
            ("100.64.0.1", AddrClass::Private),
            ("fd12:3456::1", AddrClass::Private),
            ("169.254.1.1", AddrClass::LinkLocal),
            ("fe80::1", AddrClass::LinkLocal),
            ("169.254.169.254", AddrClass::Metadata),
            ("fd00:ec2::254", AddrClass::Metadata),
            ("0.0.0.0", AddrClass::Unspecified),
            ("::", AddrClass::Unspecified),
            ("224.0.0.1", AddrClass::Multicast),
            ("255.255.255.255", AddrClass::Reserved),
            ("::ffff:127.0.0.1", AddrClass::Loopback),
            ("::ffff:169.254.169.254", AddrClass::Metadata),
            ("64:ff9b::a00:1", AddrClass::Private),
        ] {
            assert_eq!(class(ip), want, "{ip}");
        }
    }

    #[test]
    fn rejects_bad_schemes_and_urls() {
        let g = NetworkGuard::new(false);
        for bad in [
            "file:///etc/passwd",
            "ftp://x.com/a",
            "gopher://x",
            "javascript:alert(1)",
            "not a url",
            "http://",
        ] {
            assert!(g.target(bad).is_err(), "{bad}");
        }
        assert_eq!(
            g.target("https://example.com/a?b").unwrap(),
            ("example.com".into(), 443)
        );
        assert_eq!(
            g.target("http://[::1]:8080/").unwrap(),
            ("::1".into(), 8080)
        );
    }

    #[test]
    fn mixed_resolution_is_restricted() {
        let public: SocketAddr = "93.184.216.34:443".parse().unwrap();
        let private: SocketAddr = "10.0.0.5:443".parse().unwrap();
        assert!(matches!(
            NetworkGuard::new(false).judge(&[public]),
            NetVerdict::Allow { .. }
        ));
        assert!(matches!(
            NetworkGuard::new(false).judge(&[public, private]),
            NetVerdict::Block { .. }
        ));
        assert!(matches!(
            NetworkGuard::new(true).judge(&[public, private]),
            NetVerdict::Ask { .. }
        ));
        assert!(matches!(
            NetworkGuard::new(false).judge(&[]),
            NetVerdict::Block { .. }
        ));
    }

    #[tokio::test]
    async fn localhost_and_metadata_are_blocked() {
        let g = NetworkGuard::new(false);
        for url in [
            "http://127.0.0.1/",
            "http://localhost:11434/api",
            "http://169.254.169.254/latest/meta-data",
            "http://[::1]/",
        ] {
            assert!(
                matches!(g.check(url).await, NetVerdict::Block { .. }),
                "{url}"
            );
        }
        assert!(matches!(
            NetworkGuard::new(true).check("http://127.0.0.1/").await,
            NetVerdict::Ask { .. }
        ));
    }
}
