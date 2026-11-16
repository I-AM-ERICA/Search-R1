"""Bounded HTML excerpts from explicitly supported public policy websites."""
from html.parser import HTMLParser
import ipaddress
import re
from pathlib import Path
import shutil
import socket
import subprocess
import tempfile
from urllib.parse import urlsplit, unquote, parse_qs


# Exact hosts only. These records identify publishers, not correctness/currentness
# of individual pages. Keep verification evidence alongside additions.
OFFICIAL_REGISTRY = {
    "www.semas.or.kr": ("소상공인시장진흥공단", "https://www.semas.or.kr/"),
    "semas.or.kr": ("소상공인시장진흥공단", "https://www.semas.or.kr/"),
    "ols.semas.or.kr": ("소상공인정책자금", "https://ols.semas.or.kr/"),
    "www.mss.go.kr": ("중소벤처기업부", "https://www.mss.go.kr/"),
    "www.bizinfo.go.kr": ("기업마당", "https://www.bizinfo.go.kr/"),
    "www.seoulshinbo.co.kr": ("서울신용보증재단", "https://www.nabis.go.kr/localSocietyDetailView.do?menucd=248&societyIdx=501"),
    "www.gcgf.or.kr": ("경기신용보증재단", "https://www.gcgf.or.kr/gcgf/main.do"),
    "www.sbiz24.kr": ("소상공인24", "https://www.sbiz24.kr/?menu_id=010100"),
}
OFFICIAL_HOSTS = frozenset(OFFICIAL_REGISTRY)


class PageUnavailable(RuntimeError):
    def __init__(self, reason, *, http_status=None, content_type=None, transport_code=None):
        super().__init__(reason)
        # Numeric/status metadata only. Never serialize curl stderr, response
        # bodies or headers (which may contain cookies or reflected credentials).
        self.details = {}
        if isinstance(http_status, str) and re.fullmatch(r"[1-5][0-9]{2}", http_status):
            self.details["http_status"] = int(http_status)
        if isinstance(content_type, str):
            media = content_type.split(";")[0].strip().lower()
            if re.fullmatch(r"[a-z0-9.+-]+/[a-z0-9.+-]+", media):
                self.details["content_type"] = media
        if type(transport_code) is int:
            self.details["curl_exit_code"] = transport_code


def public_page_url(url):
    """Syntax gate; DNS addresses are separately checked and pinned before fetching."""
    try:
        parts = urlsplit(url)
        host = parts.hostname or ""
        if (parts.scheme != "https" or parts.port not in (None, 443) or parts.username
                or parts.password or not host or "." not in host or len(url) > 4096
                or any(ord(c) < 32 for c in url)):
            return False
        try:
            ipaddress.ip_address(host)
            return False  # only named websites, never literal addresses
        except ValueError:
            return not host.endswith((".localhost", ".local", ".internal"))
    except (ValueError, TypeError):
        return False


def official_url(url):
    try:
        parts = urlsplit(url)
        return (parts.scheme == "https" and parts.hostname in OFFICIAL_HOSTS
                and parts.port in (None, 443) and not parts.username and not parts.password
                and not any(ord(c) < 32 for c in url) and len(url) <= 4096)
    except (ValueError, TypeError):
        return False


def unsupported_document_url(url):
    """Known binary/download resources cannot be consumed by the HTML reader.

    Inspect URL paths/explicit filename fields, never the result title: an HTML
    announcement can legitimately mention an attached .hwp in its title.
    """
    parts = urlsplit(url)
    path = unquote(parts.path).lower()
    endpoints = {
        "www.bizinfo.go.kr": {"/cmm/fms/filedown.do", "/cmm/fms/getimagefile.do"},
        "www.mss.go.kr": {"/common/board/download.do"},
    }
    if path in endpoints.get(parts.hostname, set()):
        return True
    names = [path]
    for key, values in parse_qs(parts.query).items():
        if key.casefold() in {"strefilenm", "filename", "file_name"}:
            names.extend(value.casefold() for value in values)
    return any(re.search(r"\.(?:pdf|hwp|hwpx|docx?|xlsx?|pptx?|zip|jpe?g|png|gif|webp)$", name)
               for name in names)


class ContentParser(HTMLParser):
    """Prefer a content container; never substitute whole navigation-only pages."""
    VOID = {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
    SKIP = {"script", "style", "nav", "header", "footer", "noscript", "form", "svg"}

    def __init__(self, allow_editor=True):
        super().__init__(convert_charrefs=True)
        self.stack, self.candidates = [], []
        self.editor_parts = None
        self.allow_editor = allow_editor

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        hidden = (tag in self.SKIP or "hidden" in attrs or attrs.get("aria-hidden") == "true"
                  or any(node[1] for node in self.stack))
        parent = self.stack[-1][0] if self.stack else None
        section_start = tag in {"h1", "h2", "h3", "h4", "h5", "h6", "dt"} or (
            tag == "strong" and parent in {"dd", "div", "section"})
        if section_start and not hidden:
            for _, _, active in self.stack:
                if active is not None and active["parts"] and active["parts"][-1] != "":
                    active["parts"].append("")
        classes = set((attrs.get("class") or "").split())
        rank = (4 if "board_view" in classes or attrs.get("id") == "contsView" else
                3 if tag in {"main", "article"} else
                2 if classes & {"contents", "board_view", "view_cont", "view_content"} else
                1 if attrs.get("id") in {"content", "contents", "sub_contents"} else 0)
        # MSS renders its article from an escaped editor payload. Interpret only
        # this known article field as inert HTML; never execute the page's JS.
        if self.allow_editor and tag == "textarea" and attrs.get("id") == "editContents" and not hidden:
            self.editor_parts = []
            hidden = True
        elif tag == "textarea":
            hidden = True
        candidate = None
        if rank and not hidden:
            candidate = {"rank": rank, "parts": [], "length": 0}
            self.candidates.append(candidate)
        if tag not in self.VOID:
            self.stack.append((tag, hidden, candidate))

    def handle_startendtag(self, tag, attrs):
        self.handle_starttag(tag, attrs)
        if tag not in self.VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag):
        if tag == "textarea" and self.editor_parts is not None:
            embedded = ContentParser(allow_editor=False)
            embedded.feed("<article>" + "".join(self.editor_parts) + "</article>")
            for item in embedded.candidates:
                for _, hidden, active in self.stack:
                    if active is not None and not hidden:
                        active["parts"].extend(item["parts"])
                        active["length"] += item["length"]
                break
            self.editor_parts = None
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                del self.stack[index:]
                break

    def handle_data(self, data):
        if self.editor_parts is not None:
            if sum(map(len, self.editor_parts)) < 20000:
                self.editor_parts.append(data[:20000])
            return
        if not self.stack or self.stack[-1][1]:
            return
        text = " ".join(data.split())
        if text:
            for _, _, candidate in self.stack:
                if candidate is not None and candidate["length"] < 20000:
                    candidate["parts"].append(text)
                    candidate["length"] += len(text)


def extract_html(raw):
    # Supported sites use UTF-8 or Korean legacy encodings; no replacement-decoding.
    for encoding in ("utf-8-sig", "cp949"):
        try:
            html = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    else:
        raise PageUnavailable("unsupported_encoding")
    parser = ContentParser()
    parser.feed(html)
    candidates = [(c["rank"], re.sub(r"\n{3,}", "\n\n", "\n".join(c["parts"])).strip()) for c in parser.candidates]
    candidates = [(rank, text) for rank, text in candidates if len(text) >= 120]
    if not candidates:
        raise PageUnavailable("no_html_body")
    return max(candidates, key=lambda c: (c[0], len(c[1])))[1][:5000]


def fetch_official_page(url, timeout=8):
    if not official_url(url):
        raise PageUnavailable("unsupported_host")
    return fetch_public_page(url, timeout)


def fetch_public_page(url, timeout=8):
    if not public_page_url(url):
        raise PageUnavailable("unsupported_url")
    if unsupported_document_url(url):
        raise PageUnavailable("unsupported_resource")
    if timeout <= 0:
        raise PageUnavailable("deadline")
    curl = shutil.which("curl")
    if not curl:
        raise PageUnavailable("curl_missing")
    host = urlsplit(url).hostname
    try:
        addresses = socket.getaddrinfo(host, 443, family=socket.AF_INET, type=socket.SOCK_STREAM)
        ips = [row[4][0] for row in addresses]
        if not ips or any(not ipaddress.ip_address(ip).is_global for ip in ips):
            raise PageUnavailable("nonpublic_address")
        # Pin the validated IP; keep hostname for TLS validation. No proxies, cookies,
        # curl config, redirects, API credentials or insecure TLS options.
        with tempfile.TemporaryDirectory(prefix="policy-page-") as tmp:
            output = Path(tmp) / "page.html"
            result = subprocess.run([
                curl, "--disable", "--silent", "--show-error", "--noproxy", "*",
                "--proto", "=https", "--resolve", f"{host}:443:{ips[0]}",
                "--max-time", str(min(timeout, 8)), "--max-filesize", "1048576",
                "--output", str(output), "--write-out", "%{http_code}\n%{content_type}", url,
            ], capture_output=True, timeout=min(timeout, 8) + 1,
               env={"PATH": "/usr/bin:/bin", "LANG": "C"})
            if result.returncode:
                raise PageUnavailable("page_request_failed", transport_code=result.returncode)
            status, _, content_type = result.stdout.decode("ascii", "replace").partition("\n")
            if status != "200" or content_type.split(";")[0].strip().lower() not in {"text/html", "application/xhtml+xml"}:
                raise PageUnavailable("not_html_200", http_status=status, content_type=content_type)
            if output.stat().st_size > 1048576:
                raise PageUnavailable("page_too_large")
            return extract_html(output.read_bytes())
    except (OSError, subprocess.SubprocessError):
        raise PageUnavailable("page_request_failed") from None
