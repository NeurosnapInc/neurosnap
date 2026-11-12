"""Export generated SDK articles for the platform docs shell.

The standalone Sphinx HTML remains usable. The backend consumes only the
versioned manifest and serves source pages and static assets from the same build.
"""
import html
import json
from html.parser import HTMLParser
from pathlib import Path


class Article(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=False)
        self.active = False
        self.parts = []
        self.text = []
        self.sections = []
        self.headings = []
        self.heading = None
        self.skip_h1 = False

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == "article" and "bd-article" in attrs.get("class", "").split():
            self.active = True
            return
        if not self.active:
            return
        if tag == "h1":
            self.skip_h1 = True
            return
        if self.skip_h1:
            return
        self.parts.append(self.get_starttag_text())
        if tag == "section":
            self.sections.append(attrs.get("id", ""))
        if tag == "h2":
            self.heading = {"id": attrs.get("id") or (self.sections[-1] if self.sections else ""), "title": ""}

    def handle_endtag(self, tag):
        if not self.active:
            return
        if tag == "article":
            self.active = False
            return
        if tag == "h1":
            self.skip_h1 = False
            return
        if self.skip_h1:
            return
        self.parts.append("</" + tag + ">")
        if tag == "h2" and self.heading:
            self.heading["title"] = html.unescape(self.heading["title"]).rstrip("#¶ ").strip()
            if self.heading["id"]:
                self.headings.append(self.heading)
            self.heading = None
        if tag == "section" and self.sections:
            self.sections.pop()
        self.text.append(" ")

    def handle_data(self, data):
        if self.active and not self.skip_h1:
            self.parts.append(data)
            self.text.append(data)
            if self.heading is not None:
                self.heading["title"] += data

    def handle_entityref(self, name):
        self.handle_data("&" + name + ";")

    def handle_charref(self, name):
        self.handle_data("&#" + name + ";")

    def handle_comment(self, data):
        if self.active and not self.skip_h1:
            self.parts.append("<!--" + data + "-->")


def export_reference(app, exception):
    if exception is not None or app.builder.name != "html":
        return
    output = Path(app.outdir)
    pages = []
    # env.found_docs excludes old, stale HTML left behind by a previous build.
    for name in sorted(app.env.found_docs):
        if "/" in name:
            raise ValueError("SDK reference exporter currently requires flat document names")
        parser = Article()
        parser.feed((output / (name + ".html")).read_text(encoding="utf-8"))
        body = "".join(parser.parts).strip()
        if not body:
            raise ValueError("Missing Sphinx article body for " + name)
        title = app.env.titles[name].astext()
        if name == "index":
            title = "Python SDK reference"
        elif name == "modules":
            title = "Module index"
        elif name.startswith("neurosnap"):
            title = name
        pages.append({
            "name": name + ".html", "title": title,
            "html": body, "text": " ".join(html.unescape("".join(parser.text)).split()),
            "headings": parser.headings,
        })
    pages.sort(key=lambda p: (p["name"] != "index.html", p["name"]))
    artifact = {"version": app.config.release, "pages": pages}
    (output / "neurosnap-reference.json").write_text(json.dumps(artifact, ensure_ascii=False) + "\n", encoding="utf-8")


def setup(app):
    app.connect("build-finished", export_reference)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
