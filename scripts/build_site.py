#!/usr/bin/env python3
"""Build documentation pages using a small stdlib renderer for this repository.

No external assets, Node toolchain or network access. Figures are already supplied.
"""
from pathlib import Path
import html
import re
import shutil

ROOT=Path(__file__).resolve().parents[1]


def inline(text):
    text=html.escape(text)
    text=re.sub(r'`([^`]+)`',r'<code>\1</code>',text)
    text=re.sub(r'\*\*([^*]+)\*\*',r'<strong>\1</strong>',text)
    text=re.sub(r'\[([^\]]+)\]\(([^)]+)\)',lambda m:'<a href="'+m[2].replace('.md','.html')+'">'+m[1]+'</a>',text)
    return text


def render(text):
    lines=text.splitlines();out=[];i=0
    while i<len(lines):
        line=lines[i]
        if not line.strip():i+=1;continue
        if line.startswith('```'):
            code=[];i+=1
            while i<len(lines) and not lines[i].startswith('```'):code.append(lines[i]);i+=1
            out.append('<pre><code>'+html.escape('\n'.join(code))+'</code></pre>');i+=1;continue
        match=re.match(r'^(#{1,6}) (.+)',line)
        if match:
            n=len(match[1]);out.append(f'<h{n}>'+inline(match[2])+f'</h{n}>');i+=1;continue
        if line.startswith('|') and i+1<len(lines) and re.match(r'^\|[ :|\-]+$',lines[i+1]):
            cells=[x.strip() for x in line.strip('|').split('|')];out.append('<table><thead><tr>'+''.join('<th>'+inline(x)+'</th>' for x in cells)+'</tr></thead><tbody>');i+=2
            while i<len(lines) and lines[i].startswith('|'):
                out.append('<tr>'+''.join('<td>'+inline(x.strip())+'</td>' for x in lines[i].strip('|').split('|'))+'</tr>');i+=1
            out.append('</tbody></table>');continue
        if re.match(r'^(- |\d+\. )',line):
            ordered=bool(re.match(r'^\d+\. ',line));tag='ol' if ordered else 'ul';out.append('<'+tag+'>')
            while i<len(lines) and re.match(r'^(- |\d+\. )',lines[i]):
                out.append('<li>'+inline(re.sub(r'^(- |\d+\. )','',lines[i]))+'</li>');i+=1
            out.append('</'+tag+'>');continue
        para=[line];i+=1
        while i<len(lines) and lines[i].strip() and not re.match(r'^(#|```|\||- |\d+\. )',lines[i]):para.append(lines[i]);i+=1
        out.append('<p>'+inline(' '.join(para))+'</p>')
    return '\n'.join(out)


def main():
    site=ROOT/'website';count=0
    for src in sorted((ROOT/'docs').rglob('*.md')):
        rel=src.relative_to(ROOT/'docs');dest=site/'docs'/rel.with_suffix('.html');dest.parent.mkdir(parents=True,exist_ok=True)
        prefix='../'*(len(rel.parts))
        title=src.read_text().splitlines()[0].lstrip('# ')
        page='<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>'+html.escape(title)+' — SHRED-MPC</title><link rel="stylesheet" href="'+prefix+'styles.css"></head><body><main class="doc"><div class="doc-nav"><a href="'+prefix+'index.html">← SHRED-MPC project</a><a href="'+src.name+'">Markdown source</a></div>'+render(src.read_text())+'</main></body></html>\n'
        dest.write_text(page);shutil.copy2(src,dest.with_suffix('.md'));count+=1
    shutil.copytree(ROOT/'LICENSES',site/'LICENSES',dirs_exist_ok=True)
    (site/'.nojekyll').touch()
    print(f'Prepared {count} documentation pages and static site assets.')

if __name__=='__main__':main()
