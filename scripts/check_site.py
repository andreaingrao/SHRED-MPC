#!/usr/bin/env python3
"""Check static website links, fragments, accessibility basics and project paths."""
from pathlib import Path
from html.parser import HTMLParser
from urllib.parse import urlsplit, unquote
import re

ROOT=Path(__file__).resolve().parents[1]/'website'

class Page(HTMLParser):
    def __init__(self):super().__init__();self.links=[];self.ids=set();self.errors=[];self.title=False;self.lang=False
    def handle_starttag(self,tag,attrs):
        a=dict(attrs)
        if 'id' in a:self.ids.add(a['id'])
        if tag=='title':self.title=True
        if tag=='html':self.lang=a.get('lang')=='en'
        for k in ['href','src','poster']:
            if k in a:self.links.append(a[k])
        if tag=='img' and not a.get('alt'):self.errors.append('Image lacks alt text')
        if tag=='video' and ('autoplay' in a or 'controls' not in a):self.errors.append('Video must use controls without autoplay')


def main():
    pages={}
    for file in ROOT.rglob('*.html'):
        p=Page();p.feed(file.read_text());pages[file.resolve()]=p
    errors=[];checked=0
    for file,page in pages.items():
        errors += [f'{file.name}: {e}' for e in page.errors]
        if not(page.title and page.lang):errors.append(f'{file}: missing title or English language')
        for link in page.links:
            url=urlsplit(link)
            if url.scheme or url.netloc:continue
            if url.path.startswith('/'):errors.append(f'{file}: root-relative URL breaks project prefix: {link}');continue
            target=(file.parent/unquote(url.path)).resolve() if url.path else file
            if not target.is_relative_to(ROOT.resolve()):errors.append(f'{file}: link escapes website: {link}');continue
            if not target.exists():errors.append(f'{file}: missing {link}')
            elif url.fragment and target in pages and unquote(url.fragment) not in pages[target].ids:errors.append(f'{file}: missing fragment {link}')
            checked+=1
    css=(ROOT/'styles.css').read_text()
    assert 'prefers-reduced-motion' in css
    assert '@media(max-width:600px)' in css
    if errors:raise SystemExit('\n'.join(errors))
    print(f'Checked {len(pages)} HTML pages and {checked} local links/assets; project-relative paths, alt text and motion controls passed.')

if __name__=='__main__':main()
