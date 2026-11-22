import os
import re
import subprocess
import sys
import tempfile

from markdown_it import MarkdownIt

# technical_report.md dosyasını PDF'e çevirir (Markdown -> HTML -> Chrome ile PDF).
# Kullanım (proje kökünden): python report/build_pdf.py
# Gereken: markdown-it-py ve Google Chrome (macOS yolu aşağıda).

REPORT_DIR = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.join(REPORT_DIR, "technical_report.md")
OUTPUT = os.path.join(REPORT_DIR, "technical_report.pdf")
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = """
@page { size: A4; margin: 1.2cm 1.4cm; }
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 10pt; line-height: 1.25; color: #111; }
h1 { font-size: 15pt; margin: 0 0 3pt; line-height: 1.2; }
h2 { font-size: 12pt; margin: 7pt 0 3pt; padding-bottom: 1pt; border-bottom: 1px solid #bbb; break-after: avoid; }
p { margin: 3pt 0; }
ul, ol { margin: 2pt 0; padding-left: 15pt; }
li { margin: 1pt 0; }
table { border-collapse: collapse; font-size: 8.5pt; margin: 4pt 0; }
tr { break-inside: avoid; }
table:has(th:nth-child(10)) { break-inside: avoid; }
p:has(+ table:has(th:nth-child(10))) { break-after: avoid; }
th, td { border: 1px solid #c8c8c4; padding: 2pt 4pt; text-align: left; vertical-align: top; }
th { background: #efefec; }
td:not(:first-child) { }
code { font-family: Menlo, monospace; font-size: 8.5pt; }
img { display: block; margin: 4pt auto 0; break-inside: avoid; }
img[alt="Figure 1"] { width: 70%; }
img[alt="Figure 2"] { width: 56%; }
em { color: #333; }
p:has(> img) { margin: 0; }
p:has(> strong:only-child) { break-after: avoid; }
"""

markdown = MarkdownIt("commonmark").enable("table")
body = markdown.render(open(SOURCE, encoding="utf-8").read())
html = f"<!doctype html><html><head><meta charset='utf-8'><style>{CSS}</style></head><body>{body}</body></html>"

# HTML, görsellerin göreli yolları çalışsın diye rapor klasörüne geçici olarak yazılır
with tempfile.NamedTemporaryFile("w", suffix=".html", dir=REPORT_DIR, delete=False, encoding="utf-8") as file:
    file.write(html)
    html_path = file.name

try:
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
                    f"--print-to-pdf={OUTPUT}", f"file://{html_path}"],
                   check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
finally:
    os.remove(html_path)

pages = len(re.findall(rb"/Type\s*/Page[^s]", open(OUTPUT, "rb").read()))
print(f"Saved {OUTPUT} ({pages} pages)")
sys.exit(0 if pages <= 4 else 1)
