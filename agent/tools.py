"""
Country Data Tool
Wraps the public REST Countries API with retries, timeout, and normalisation.
"""

from __future__ import annotations

import logging
from typing import Optional

import httpx
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

logger = logging.getLogger(__name__)

_BASE_URL = "https://restcountries.com/v3.1"

# Only request the fields we might ever use – keeps payloads small.
_FIELDS = (
    "name,capital,population,currencies,languages,region,subregion,"
    "flags,area,timezones,borders,continents,tld,cca2,cca3,callingCodes,latlng"
)


class CountryAPIError(Exception):
    """Raised when the Countries API cannot be queried successfully."""


class CountryNotFoundError(CountryAPIError):
    """Raised when the API returns no result for the given name."""


@retry(
    retry=retry_if_exception_type(httpx.TransportError),
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=1, max=8),
    reraise=True,
)
def fetch_country_data(country_name: str) -> dict:
    """
    Fetch structured data for *country_name* from the REST Countries API.

    Parameters
    ----------
    country_name : str
        The country name as understood by the end user (e.g. "Germany").

    Returns
    -------
    dict
        Normalised country record with well-typed values.

    Raises
    ------
    CountryNotFoundError
        When the API returns 404 or an empty list.
    CountryAPIError
        For any other non-2xx response or network failure.
    """
    url = f"{_BASE_URL}/name/{country_name.strip()}"
    params = {"fields": _FIELDS, "fullText": "false"}

    logger.info("Querying REST Countries API: country=%r", country_name)

    try:
        with httpx.Client(timeout=10.0) as client:
            response = client.get(url, params=params)
    except httpx.TransportError as exc:
        logger.warning("Network error querying Countries API: %s", exc)
        raise

    if response.status_code == 404:
        raise CountryNotFoundError(
            f"No country found for '{country_name}'. "
            "Please check the spelling and try again."
        )

    if response.status_code != 200:
        raise CountryAPIError(
            f"REST Countries API returned HTTP {response.status_code} "
            f"for '{country_name}'."
        )

    records = response.json()
    if not records:
        raise CountryNotFoundError(f"Empty result for country '{country_name}'.")

    # When multiple results are returned (e.g. "Guinea" matches several),
    # prefer an exact match on the common name; fall back to first result.
    exact = _pick_best_match(country_name, records)
    return _normalise(exact)


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _pick_best_match(query: str, records: list[dict]) -> dict:
    """Return the record whose common/official name best matches *query*."""
    query_lower = query.lower().strip()
    for record in records:
        name_block = record.get("name", {})
        common = name_block.get("common", "").lower()
        official = name_block.get("official", "").lower()
        if query_lower in (common, official):
            return record
    # Fall back to the first (typically most-relevant) result.
    return records[0]


def _normalise(record: dict) -> dict:
    """
    Flatten and type-coerce the raw API payload into a clean dict.
    Missing values become ``None`` so callers can use ``.get()`` safely.
    """
    name_block = record.get("name", {})

    # Currencies: {"USD": {"name": "US Dollar", "symbol": "$"}, ...}
    currencies_raw = record.get("currencies") or {}
    currencies = [
        f"{info.get('name', code)} ({info.get('symbol', '')})"
        for code, info in currencies_raw.items()
    ]

    # Languages: {"eng": "English", "fra": "French"}
    languages = list((record.get("languages") or {}).values())

    # Capital is an array
    capital = record.get("capital") or []

    # Calling codes
    idd = record.get("idd") or {}
    root = idd.get("root", "")
    suffixes = idd.get("suffixes") or []
    calling_codes = [f"{root}{s}" for s in suffixes] if root else []

    return {
        "common_name": name_block.get("common"),
        "official_name": name_block.get("official"),
        "cca2": record.get("cca2"),
        "cca3": record.get("cca3"),
        "capital": capital[0] if capital else None,
        "region": record.get("region"),
        "subregion": record.get("subregion"),
        "continents": record.get("continents") or [],
        "population": record.get("population"),
        "area_km2": record.get("area"),
        "currencies": currencies,
        "languages": languages,
        "timezones": record.get("timezones") or [],
        "borders": record.get("borders") or [],
        "tld": record.get("tld") or [],
        "calling_codes": calling_codes,
        "flag_emoji": record.get("flags", {}).get("alt", ""),
        "flag_png": record.get("flags", {}).get("png", ""),
        "latlng": record.get("latlng") or [],
    }
