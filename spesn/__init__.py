"""spesn - Saturation Point Extraction for SuperNovae."""

__version__ = "0.1.0"

from .io import (  # noqa: F401
    DP1_COLLECTION,
    MONSTER_DATASET,
    SaturationConfig,
    add_saturation_metrics,
    build_visit_detector_saturation_table,
    crossmatch_sources_to_monster,
    fetch_monster_catalog_for_sources,
    fetch_source_table_for_visit_detector,
    get_butler,
    get_tap_service,
    get_visit_summary,
)
