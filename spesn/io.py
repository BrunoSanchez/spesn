"""I/O helpers for DP1 bright-star saturation analysis.

This module keeps the API small: input is typically ``visit`` + ``detector``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd


DP1_COLLECTION = "LSSTComCam/DP1"
MONSTER_DATASET = "the_monster_20250219"


@dataclass(frozen=True)
class SaturationConfig:
    """Configuration for saturation extraction."""

    band: str | None = "g"
    min_snr: float | None = 3.0
    max_sep_arcsec: float = 1.0
    source_flux_col: str = "psfFlux"
    source_flux_err_col: str = "psfFluxErr"
    monster_flux_col: str = "monster_ComCam_g_flux"


def get_butler(repo: str = "dp1", collections: str = DP1_COLLECTION) -> Any:
    """Return an LSST Butler instance."""
    import lsst.daf.butler as dafButler

    return dafButler.Butler(repo, collections=collections)


def get_tap_service(service_name: str = "tap") -> Any:
    """Return an RSP TAP service client."""
    from lsst.rsp import get_tap_service as _get_tap_service

    service = _get_tap_service(service_name)
    if service is None:
        raise RuntimeError(f"Could not get TAP service '{service_name}'.")
    return service


def fetch_source_table_for_visit_detector(
    visit: int,
    detector: int,
    *,
    band: str | None = "g",
    min_snr: float | None = 3.0,
    tap_service: Any | None = None,
) -> pd.DataFrame:
    """Fetch DP1 source rows for a visit+detector pair from ``dp1.Source``."""
    service = tap_service or get_tap_service("tap")

    where_parts = [f"visit = {int(visit)}", f"detector = {int(detector)}"]
    if band:
        where_parts.append(f"band = '{band}'")
    if min_snr is not None:
        where_parts.append(f"(psfFlux / psfFluxErr) >= {float(min_snr)}")
    where_clause = " AND ".join(where_parts)

    query = (
        "SELECT sourceId, visit, detector, band, ra, dec, psfFlux, psfFluxErr "
        "FROM dp1.Source "
        f"WHERE {where_clause}"
    )

    job = service.submit_job(query)
    job.run()
    job.wait(phases=["COMPLETED", "ERROR"])
    if job.phase == "ERROR":
        job.raise_if_error()
    if job.phase != "COMPLETED":
        raise RuntimeError(f"TAP job failed with phase={job.phase}")

    table = job.fetch_result().to_table()
    df = _as_pandas(table)
    return _normalize_source_positions(df)


def fetch_monster_catalog_for_sources(
    butler: Any,
    sources: pd.DataFrame,
    *,
    monster_dataset: str = MONSTER_DATASET,
    htm_level: int = 7,
) -> pd.DataFrame:
    """Fetch monster-catalog rows near source positions using HTM sharding."""
    import esutil
    from astropy.table import vstack

    src = _normalize_source_positions(_as_pandas(sources))
    if src.empty:
        return pd.DataFrame()

    htm = esutil.htm.HTM(htm_level)
    htm_ids = set(htm.lookup_id(src["ra"].to_numpy(), src["dec"].to_numpy()))

    monster_tables = [
        butler.get(monster_dataset, htm7=int(htm_id)).asAstropy()
        for htm_id in sorted(htm_ids)
    ]
    if not monster_tables:
        return pd.DataFrame()

    monster = vstack(monster_tables)
    if "coord_ra" in monster.colnames:
        monster["coord_ra_deg"] = np.rad2deg(monster["coord_ra"])
    if "coord_dec" in monster.colnames:
        monster["coord_dec_deg"] = np.rad2deg(monster["coord_dec"])
    return _as_pandas(monster)


def crossmatch_sources_to_monster(
    sources: pd.DataFrame,
    monster: pd.DataFrame,
    *,
    max_sep_arcsec: float = 1.0,
) -> pd.DataFrame:
    """Nearest-neighbor sky match between source rows and monster rows."""
    from astropy.coordinates import SkyCoord
    import astropy.units as u

    src = _normalize_source_positions(_as_pandas(sources))
    mon = _normalize_monster_positions(_as_pandas(monster))
    if src.empty or mon.empty:
        out = src.copy()
        out["monster_index"] = -1
        out["match_sep_arcsec"] = np.nan
        out["match_monster"] = False
        return out

    stars = SkyCoord(ra=src["ra"].to_numpy() * u.deg, dec=src["dec"].to_numpy() * u.deg)
    monsters = SkyCoord(
        ra=mon["coord_ra_deg"].to_numpy() * u.deg,
        dec=mon["coord_dec_deg"].to_numpy() * u.deg,
    )
    idx, d2d, _ = stars.match_to_catalog_sky(monsters)

    matched_mon = mon.iloc[idx].reset_index(drop=True)
    out = src.reset_index(drop=True).copy()
    out["monster_index"] = idx
    out["match_sep_arcsec"] = d2d.arcsec
    out["match_monster"] = out["match_sep_arcsec"] < float(max_sep_arcsec)

    # Keep source columns first, then append non-overlapping monster columns.
    overlap = set(out.columns).intersection(matched_mon.columns)
    mon_only_cols = [c for c in matched_mon.columns if c not in overlap]
    return pd.concat([out, matched_mon[mon_only_cols]], axis=1)


def add_saturation_metrics(
    matched: pd.DataFrame,
    *,
    source_flux_col: str = "psfFlux",
    source_flux_err_col: str = "psfFluxErr",
    monster_flux_col: str = "monster_ComCam_g_flux",
) -> pd.DataFrame:
    """Add magnitude and pull diagnostics for saturation checks."""
    out = _as_pandas(matched).copy()

    src_flux = _to_njy_array(out[source_flux_col])
    src_flux_err = _to_njy_array(out[source_flux_err_col])
    mon_flux = _to_njy_array(out[monster_flux_col])

    out["source_mag"] = _abmag_from_njy(src_flux)
    out["expected_mag"] = _abmag_from_njy(mon_flux)
    out["delta_mag"] = out["source_mag"] - out["expected_mag"]
    out["flux_pull"] = (src_flux - mon_flux) / src_flux_err

    return out


def get_visit_summary(
    butler: Any,
    visit: int,
    detector: int | None = None,
    *,
    dataset_type: str = "visitSummary",
) -> dict[str, Any]:
    """Get visit-summary metadata as a plain dict.

    If your repo uses another dataset type (for example ``finalVisitSummary``),
    pass it via ``dataset_type``.
    """
    data_id: dict[str, Any] = {"visit": int(visit)}
    if detector is not None:
        data_id["detector"] = int(detector)

    obj = butler.get(dataset_type, dataId=data_id)
    return _object_to_dict(obj)


def build_visit_detector_saturation_table(
    visit: int,
    detector: int,
    *,
    config: SaturationConfig = SaturationConfig(),
    butler: Any | None = None,
    tap_service: Any | None = None,
    monster_dataset: str = MONSTER_DATASET,
) -> pd.DataFrame:
    """End-to-end helper: source rows -> monster match -> saturation metrics."""
    src = fetch_source_table_for_visit_detector(
        visit,
        detector,
        band=config.band,
        min_snr=config.min_snr,
        tap_service=tap_service,
    )
    btlr = butler or get_butler()
    monster = fetch_monster_catalog_for_sources(btlr, src, monster_dataset=monster_dataset)
    matched = crossmatch_sources_to_monster(src, monster, max_sep_arcsec=config.max_sep_arcsec)
    return add_saturation_metrics(
        matched,
        source_flux_col=config.source_flux_col,
        source_flux_err_col=config.source_flux_err_col,
        monster_flux_col=config.monster_flux_col,
    )


def _normalize_source_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "ra" not in out.columns and "coord_ra" in out.columns:
        out["ra"] = _to_degree_values(out["coord_ra"])
    if "dec" not in out.columns and "coord_dec" in out.columns:
        out["dec"] = _to_degree_values(out["coord_dec"])
    return out


def _normalize_monster_positions(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "coord_ra_deg" not in out.columns and "coord_ra" in out.columns:
        out["coord_ra_deg"] = _to_degree_values(out["coord_ra"])
    if "coord_dec_deg" not in out.columns and "coord_dec" in out.columns:
        out["coord_dec_deg"] = _to_degree_values(out["coord_dec"])
    return out


def _to_degree_values(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(values)
    try:
        import astropy.units as u

        if hasattr(values, "to"):
            return np.asarray(values.to(u.deg).value, dtype=float)
    except Exception:
        pass

    out = np.asarray(arr, dtype=float)
    finite = np.isfinite(out)
    if finite.any() and np.nanmax(np.abs(out[finite])) <= 2.0 * np.pi + 0.1:
        out = np.rad2deg(out)
    return out


def _to_njy_array(values: Iterable[Any]) -> np.ndarray:
    arr = np.asarray(values)
    try:
        import astropy.units as u

        if hasattr(values, "to"):
            return np.asarray(values.to(u.nJy).value, dtype=float)
    except Exception:
        pass

    if arr.dtype == object and arr.size > 0 and hasattr(arr.flat[0], "to_value"):
        try:
            import astropy.units as u

            return np.array([v.to_value(u.nJy) for v in arr], dtype=float)
        except Exception:
            return np.array([float(v) for v in arr], dtype=float)
    return np.asarray(arr, dtype=float)


def _abmag_from_njy(flux_njy: np.ndarray) -> np.ndarray:
    out = np.full_like(flux_njy, np.nan, dtype=float)
    good = np.isfinite(flux_njy) & (flux_njy > 0)
    out[good] = 31.4 - 2.5 * np.log10(flux_njy[good])
    return out


def _as_pandas(table_like: Any) -> pd.DataFrame:
    if isinstance(table_like, pd.DataFrame):
        return table_like.copy()
    if hasattr(table_like, "to_pandas"):
        return table_like.to_pandas()
    if hasattr(table_like, "asAstropy"):
        return table_like.asAstropy().to_pandas()
    return pd.DataFrame(table_like)


def _object_to_dict(obj: Any) -> dict[str, Any]:
    if isinstance(obj, dict):
        return obj
    if hasattr(obj, "toDict"):
        return dict(obj.toDict())
    if hasattr(obj, "asDict"):
        return dict(obj.asDict())
    if hasattr(obj, "to_pandas"):
        df = obj.to_pandas()
        if len(df) == 1:
            return df.iloc[0].to_dict()
        return {"rows": df.to_dict(orient="records")}
    if hasattr(obj, "asAstropy"):
        table = obj.asAstropy()
        if len(table) == 1:
            return {name: table[name][0] for name in table.colnames}
        return {"rows": table.to_pandas().to_dict(orient="records")}
    return {"value": obj}
