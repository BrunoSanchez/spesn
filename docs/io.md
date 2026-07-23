# I/O module usage

The I/O helpers are in `spesn.io` and are designed around `visit` + `detector`.

## Quick start

```python
from spesn.io import (
    SaturationConfig,
    build_visit_detector_saturation_table,
    get_butler,
    get_visit_summary,
)

visit = 7024060900051
detector = 42

config = SaturationConfig(
    band="g",
    min_snr=3.0,
    max_sep_arcsec=1.0,
    source_flux_col="psfFlux",
    source_flux_err_col="psfFluxErr",
    monster_flux_col="monster_ComCam_g_flux",
)

# End-to-end retrieval + crossmatch + metrics.
df = build_visit_detector_saturation_table(
    visit=visit,
    detector=detector,
    config=config,
)

# Visit summary metadata.
butler = get_butler(repo="dp1", collections="LSSTComCam/DP1")
summary = get_visit_summary(butler, visit=visit, detector=detector, dataset_type="visitSummary")
```

The returned table `df` contains:

- Source columns from `dp1.Source` (for the selected visit/detector).
- Monster match information (`monster_index`, `match_sep_arcsec`, `match_monster`).
- Saturation diagnostics:
  - `source_mag`
  - `expected_mag`
  - `delta_mag` (`source_mag - expected_mag`)
  - `flux_pull` (`(source_flux - expected_flux) / source_flux_err`)

## Step-by-step functions

```python
from spesn.io import (
    add_saturation_metrics,
    crossmatch_sources_to_monster,
    fetch_monster_catalog_for_sources,
    fetch_source_table_for_visit_detector,
    get_butler,
)

visit = 7024060900051
detector = 42

sources = fetch_source_table_for_visit_detector(
    visit=visit,
    detector=detector,
    band="g",
    min_snr=3.0,
)

butler = get_butler()
monster = fetch_monster_catalog_for_sources(butler, sources)
matched = crossmatch_sources_to_monster(sources, monster, max_sep_arcsec=1.0)
result = add_saturation_metrics(
    matched,
    source_flux_col="psfFlux",
    source_flux_err_col="psfFluxErr",
    monster_flux_col="monster_ComCam_g_flux",
)
```

## Build docs locally

```bash
pip install -e .[docs]
sphinx-build -b html docs docs/_build/html
```
