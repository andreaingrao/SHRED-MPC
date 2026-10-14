# Local preview and GitHub Pages

The website is static HTML/CSS plus local images/video and copied Markdown documentation. It needs no Node build, backend, external fonts, analytics or hosted service. The source target is [andreaingrao/SHRED-MPC](https://github.com/andreaingrao/SHRED-MPC), kept private by Andrea’s explicit instruction. Pages activation and deployment remain disabled; only a private source upload is authorized.

```bash
python scripts/build_site.py
python scripts/check_site.py
python -m http.server 8000 --bind 127.0.0.1 --directory website
```

Open http://127.0.0.1:8000/. For a project-path check, serve the repository parent and open `/REPOSITORY_DIRECTORY/website/`; all site asset/document links are relative and work below a repository prefix.

## After approval only

1. Review `docs/release.md` and `PUBLIC_FILES.txt`. Confirm rights/license and whether the repository and website should be public; a private repository does not by itself guarantee a private Pages site.
2. Use the private `andreaingrao/SHRED-MPC` repository. Confirm the intended repository/site visibility before changing either.
3. Review/commit/push only this isolated repository. Never stage from the original workspace or home-directory repository.
4. In GitHub Settings → Pages, select **GitHub Actions** as the source.
5. Manually run the prepared **Deploy project website** workflow. It has `workflow_dispatch` only, so a push does not automatically publish.

The workflow builds/copies documentation, checks local links, uploads `website/`, and uses GitHub’s Pages deployment actions with the required permissions and environment. Relative URLs support `https://OWNER.github.io/REPOSITORY/` without hard-coded domain names. The source link identifies the private repository and requires access.

Official references: [GitHub custom Pages workflows](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages), [Pages limits](https://docs.github.com/en/pages/getting-started-with-github-pages/github-pages-limits). Hosting itself was not exercised; local files, HTTP paths and browser rendering are the review target.
