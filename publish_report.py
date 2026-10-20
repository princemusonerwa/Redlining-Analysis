"""Build a self-contained tabbed HTML report from the integrated s11 report.

Prefer: python run_all.py --publish (also validates current prerequisites).
Standalone: python publish_report.py --output-dir outputs
"""
import argparse
import base64
import html
import json
from pathlib import Path
import pandas as pd
import config


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir', type=Path, default=config.OUT)
    a = p.parse_args()
    root = a.output_dir.resolve()
    source = root / 's11_report/sections.json'
    if not source.exists():
        p.error(f'Missing {source}; run through s11_report first.')
    data = json.loads(source.read_text(encoding='utf-8'))
    esc = html.escape
    buttons, panes = [], []
    for i, s in enumerate(data['sections']):
        key = esc(s['key'], quote=True)
        buttons.append(
            f'<button type="button" data-pane="{key}" aria-pressed="{str(i == 0).lower()}">{esc(s["title"])}</button>')
        blocks = [f'<h2>{esc(s["title"])}</h2><p>{esc(s["explanation"])}</p>']
        for table in s['tables']:
            df = pd.read_csv(root / table['path'])
            blocks.append(
                f'<h3>{esc(table["caption"])}</h3><div class="table">{df.round(4).to_html(index=False, escape=True, border=0)}</div>')
        for figure in s['figures']:
            image = base64.b64encode(
                (root / figure['path']).read_bytes()).decode('ascii')
            caption = esc(figure['caption'])
            blocks.append(
                f'<figure><img alt="{caption}" src="data:image/png;base64,{image}"><figcaption>{caption}</figcaption></figure>')
        panes.append(
            f'<section id="{key}" {"hidden" if i else ""}>{"".join(blocks)}</section>')
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>HMDA analysis</title><style>
body{margin:0;background:#f5f7fa;color:#172338;font:16px/1.6 system-ui,sans-serif}header{padding:32px 5%;background:#102a43;color:white}
h1{font-size:32px;margin:0}header p{max-width:1000px}main{display:grid;grid-template-columns:240px minmax(0,1fr);gap:24px;padding:24px 3%}
nav{display:flex;flex-direction:column;gap:7px;align-self:start;position:sticky;top:20px}button{text-align:left;border:1px solid #cbd5e1;border-radius:6px;background:white;padding:12px;cursor:pointer;color:#172338}
button[aria-pressed=true]{background:#164e63;color:white}section{background:white;border-radius:10px;padding:28px;min-width:0}section[hidden]{display:none}
h2{margin-top:0}.table{overflow:auto}table{border-collapse:collapse;font-size:12px;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:right;white-space:nowrap}th{background:#e8eff5}
figure{margin:24px 0}img{max-width:100%;height:auto}figcaption{font-size:14px;color:#52616f}p{max-width:1000px}
@media(max-width:800px){main{display:block}nav{position:static;margin-bottom:20px}section{padding:16px}}@media print{nav{display:none}main{display:block}section[hidden]{display:block}section{break-before:page}}
</style>'''
    page += f'<header><h1>{esc(data["title"])}</h1><p>{esc(data["subtitle"])}</p></header><main><nav aria-label="Report sections">{"".join(buttons)}</nav><article>{"".join(panes)}</article></main>'
    page += '''<script>document.querySelectorAll('button[data-pane]').forEach(b=>b.addEventListener('click',()=>{document.querySelectorAll('section').forEach(s=>s.hidden=s.id!==b.dataset.pane);document.querySelectorAll('button[data-pane]').forEach(x=>x.setAttribute('aria-pressed',String(x===b)));}));</script></html>'''
    target = root / 'published/hmda_fair_lending_report.html'
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(page, encoding='utf-8')
    print(f'Published locally: {target}')


if __name__ == '__main__':
    main()
