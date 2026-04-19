# Report source

Quarto site documenting the `roman_l2_job` pipeline and its results.

## Build locally

```bash
# first time only:
pixi install --environment report

# render the site:
pixi run --environment report quarto render report
# output lands in report/_site/

# preview with live-reload:
pixi run --environment report quarto preview report
```

## Publish to GitHub Pages

```bash
pixi run --environment report quarto publish gh-pages report
```

First publish on a new clone will prompt for confirmation and create the
`gh-pages` branch. Once configured in the GitHub repo's Settings → Pages →
Source → "Deploy from a branch" → `gh-pages`, published output appears at
`https://roman-grs-pit.github.io/roman_l2_job/`.
