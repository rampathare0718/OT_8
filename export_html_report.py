"""
Converts REPORT.md into a standalone styled HTML report (REPORT.html)
with KaTeX math rendering, styled tables, code highlighting,
and embedded benchmark plots ready to Print to PDF (Ctrl+P).
"""

import os
import re

def markdown_to_html(md_text):
    # Simple converter for headers, bold, italics, lists, code, tables
    html = md_text

    # Code blocks
    html = re.sub(r'```(\w+)?\n(.*?)```', r'<pre><code>\2</code></pre>', html, flags=re.DOTALL)
    html = re.sub(r'`([^`]+)`', r'<code>\1</code>', html)

    # Headers
    html = re.sub(r'^# (.*?)$', r'<h1>\1</h1>', html, flags=re.MULTILINE)
    html = re.sub(r'^## (.*?)$', r'<h2>\1</h2>', html, flags=re.MULTILINE)
    html = re.sub(r'^### (.*?)$', r'<h3>\1</h3>', html, flags=re.MULTILINE)
    html = re.sub(r'^#### (.*?)$', r'<h4>\1</h4>', html, flags=re.MULTILINE)

    # Bold and italics
    html = re.sub(r'\*\*(.*?)\*\*', r'<strong>\1</strong>', html)
    html = re.sub(r'\*(.*?)\*', r'<em>\1</em>', html)

    # Blockquotes
    html = re.sub(r'^> (.*?)$', r'<blockquote>\1</blockquote>', html, flags=re.MULTILINE)

    # Horizontal rules
    html = re.sub(r'^---$', r'<hr>', html, flags=re.MULTILINE)

    # Convert simple markdown tables to HTML tables
    lines = html.split('\n')
    in_table = False
    table_lines = []
    output_lines = []

    for line in lines:
        if line.strip().startswith('|') and line.strip().endswith('|'):
            if not in_table:
                in_table = True
                table_lines = [line]
            else:
                table_lines.append(line)
        else:
            if in_table:
                in_table = False
                # Convert table_lines to HTML
                table_html = ['<table>']
                for r_idx, row in enumerate(table_lines):
                    cells = [c.strip() for c in row.strip()[1:-1].split('|')]
                    if r_idx == 0:
                        table_html.append('<thead><tr>' + ''.join(f'<th>{c}</th>' for c in cells) + '</tr></thead><tbody>')
                    elif r_idx == 1 and all(set(c).issubset({'-', ':', ' '}) for c in cells):
                        continue
                    else:
                        table_html.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
                table_html.append('</tbody></table>')
                output_lines.append('\n'.join(table_html))
            output_lines.append(line)

    if in_table:
        table_html = ['<table><tbody>']
        for row in table_lines:
            cells = [c.strip() for c in row.strip()[1:-1].split('|')]
            table_html.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
        table_html.append('</tbody></table>')
        output_lines.append('\n'.join(table_html))

    html = '\n'.join(output_lines)

    # Paragraphs (lines separated by blank lines that aren't tags)
    paragraphs = html.split('\n\n')
    processed_p = []
    for p in paragraphs:
        p_str = p.strip()
        if not p_str:
            continue
        if p_str.startswith('<h') or p_str.startswith('<pre') or p_str.startswith('<table') or p_str.startswith('<hr') or p_str.startswith('<ul') or p_str.startswith('<ol'):
            processed_p.append(p_str)
        else:
            processed_p.append(f'<p>{p_str}</p>')

    return '\n\n'.join(processed_p)

def generate_html_report():
    with open("REPORT.md", "r", encoding="utf-8") as f:
        md_content = f.read()

    body_html = markdown_to_html(md_content)

    full_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Technical Report: Optimization Techniques & Vibe-Coding Challenge</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.8/dist/contrib/auto-render.min.js"
    onload="renderMathInElement(document.body, {{delimiters: [
        {{left: '$$', right: '$$', display: true}},
        {{left: '$', right: '$', display: false}},
        {{left: '\\(', right: '\\)', display: false}},
        {{left: '\\[', right: '\\]', display: true}}
    ]}});"></script>
<style>
    body {{
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, "Helvetica Neue", Arial, sans-serif;
        line-height: 1.65;
        color: #24292f;
        background-color: #ffffff;
        max-width: 900px;
        margin: 0 auto;
        padding: 40px 25px;
    }}
    h1 {{ color: #0969da; border-bottom: 2px solid #eaecef; padding-bottom: 0.3em; margin-top: 1.5em; }}
    h2 {{ color: #1f2328; border-bottom: 1px solid #eaecef; padding-bottom: 0.3em; margin-top: 1.4em; }}
    h3 {{ color: #1f2328; margin-top: 1.2em; }}
    table {{
        border-collapse: collapse;
        width: 100%;
        margin: 20px 0;
        font-size: 0.92em;
    }}
    th, td {{
        border: 1px solid #d0d7de;
        padding: 8px 12px;
        text-align: left;
    }}
    th {{
        background-color: #f6f8fa;
        font-weight: 600;
    }}
    tr:nth-child(even) {{
        background-color: #fcfcfc;
    }}
    pre {{
        background-color: #f6f8fa;
        border-radius: 6px;
        padding: 16px;
        overflow: auto;
        font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, Courier, monospace;
        font-size: 0.88em;
        line-height: 1.45;
    }}
    code {{
        background-color: #eff1f3;
        padding: 0.2em 0.4em;
        border-radius: 4px;
        font-size: 0.9em;
        font-family: monospace;
    }}
    pre code {{
        background: transparent;
        padding: 0;
    }}
    blockquote {{
        margin: 0;
        padding: 0 1em;
        color: #57606a;
        border-left: 0.25em solid #d0d7de;
    }}
    hr {{
        border: none;
        border-top: 1px solid #d0d7de;
        margin: 24px 0;
    }}
    @media print {{
        body {{
            max-width: 100%;
            padding: 0;
            color: #000;
        }}
        pre, blockquote, table {{
            page-break-inside: avoid;
        }}
    }}
</style>
</head>
<body>
{body_html}
</body>
</html>
"""

    with open("REPORT.html", "w", encoding="utf-8") as f:
        f.write(full_html)
    print("Report generated successfully as REPORT.html!")

if __name__ == "__main__":
    generate_html_report()
