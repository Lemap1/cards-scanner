#!/usr/bin/env python3
"""
Pokémon TCG card image scraper using the TCGdex API (https://tcgdex.dev).

Downloads card images for every set of every series, in both qualities
(high/low) and every available extension (png, jpg, webp).

API reference used:
  - List sets:        GET https://api.tcgdex.net/v2/{lang}/sets
  - Set detail+cards: GET https://api.tcgdex.net/v2/{lang}/sets/{setId}
  - Card image URL:   {card.image}/{quality}.{extension}
      quality   -> "high" | "low"
      extension -> "png" | "webp" | "jpg"

Before downloading anything, the script checks whether the target file
already exists on disk (and is non-empty) and skips it if so, so the
script can be re-run safely / resumed at any time.

Images are organized on disk as:
    {output}/{lang}/{setId}/{localId}_{quality}.{extension}

You can scrape one or several languages in the same run. Some old/promo
sets have no translated card data in certain languages (TCGdex's database
is community-translated); when that happens the script automatically
retries with --fallback-lang (English by default) so you still get images
instead of silently skipping the set.

Usage examples:
    python pokemon_tcg_scraper.py
    python pokemon_tcg_scraper.py --langs en --output ./cards --workers 12
    python pokemon_tcg_scraper.py --langs en fr de      # multiple languages
    python pokemon_tcg_scraper.py --qualities high --extensions png webp
    python pokemon_tcg_scraper.py --sets base1 swsh3    # only specific sets
    python pokemon_tcg_scraper.py --fallback-lang ""    # disable the fallback
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from requests.adapters import HTTPAdapter, Retry
from tqdm import tqdm

API_BASE = "https://api.tcgdex.net/v2"
DEFAULT_QUALITIES = ["high", "low"]
DEFAULT_EXTENSIONS = ["png", "jpg", "webp"]

logging.basicConfig(
    filename="tcgdex_scraper.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)


def build_session() -> requests.Session:
    """Requests session with retry/backoff for flaky network conditions."""
    session = requests.Session()
    # urllib3 renamed "method_whitelist" to "allowed_methods" in v2; support both.
    try:
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            allowed_methods=["GET"],
        )
    except TypeError:
        retries = Retry(
            total=5,
            backoff_factor=0.5,
            status_forcelist=[429, 500, 502, 503, 504],
            method_whitelist=["GET"],
        )
    adapter = HTTPAdapter(max_retries=retries, pool_maxsize=32)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    # Some CDNs/WAFs silently reject requests that look scripted (missing
    # Accept/Referer, generic User-Agent). Use browser-like headers so image
    # downloads from assets.tcgdex.net aren't blocked differently from the
    # API calls to api.tcgdex.net.
    session.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
            "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
        ),
        "Accept": "image/webp,image/png,image/jpeg,application/json,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": "https://tcgdex.dev/",
    })
    return session


class TCGdexError(Exception):
    """Raised when the TCGdex API returns something we can't use."""


def get_all_sets(session: requests.Session, lang: str) -> list:
    """Fetch the list of every set (brief info) for a given language."""
    url = f"{API_BASE}/{lang}/sets"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as exc:
        raise TCGdexError(f"Invalid JSON from {url}: {exc}") from exc

    if not isinstance(data, list):
        raise TCGdexError(f"Unexpected response shape for {url} (expected a list, got {type(data).__name__})")
    return data


def get_set_detail(session: requests.Session, lang: str, set_id: str) -> dict:
    """Fetch full detail of a set, including its list of cards (with image URLs)."""
    url = f"{API_BASE}/{lang}/sets/{set_id}"
    resp = session.get(url, timeout=30)
    resp.raise_for_status()
    try:
        data = resp.json()
    except ValueError as exc:
        raise TCGdexError(f"Invalid JSON from {url}: {exc}") from exc

    if not isinstance(data, dict):
        raise TCGdexError(f"Unexpected response shape for {url} (expected an object, got {type(data).__name__})")
    return data


def download_image(session: requests.Session, url: str, filepath: str) -> tuple:
    """
    Download a single image to filepath, skipping if it already exists.
    Returns a (status, detail) tuple. status is one of:
    "skipped", "downloaded", "missing" (404 = not available for this card,
    e.g. some promos don't have a "low" quality), "error". detail is a
    short human-readable reason, only meaningful for "error".
    """
    if not url:
        logging.error("Empty/invalid image URL for target %s", filepath)
        return "error", "empty image URL"

    if os.path.exists(filepath) and os.path.getsize(filepath) > 0:
        return "skipped", None

    tmp_path = filepath + ".part"
    try:
        resp = session.get(url, timeout=30, stream=True)
        if resp.status_code == 404:
            return "missing", None
        if resp.status_code in (401, 403):
            detail = f"HTTP {resp.status_code} (blocked/forbidden) on {url}"
            logging.error(detail)
            return "error", detail
        resp.raise_for_status()

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=65536):
                if chunk:
                    f.write(chunk)

        if os.path.getsize(tmp_path) == 0:
            raise IOError("downloaded file is empty")

        os.replace(tmp_path, filepath)
        return "downloaded", None
    except requests.RequestException as exc:
        detail = f"network error on {url}: {exc}"
        logging.error("Network error downloading %s -> %s: %s", url, filepath, exc)
        return "error", detail
    except OSError as exc:
        # Disk full, permission denied, invalid path/filename, etc.
        detail = f"filesystem error saving {filepath}: {exc}"
        logging.error("Filesystem error saving %s -> %s: %s", url, filepath, exc)
        return "error", detail
    except Exception as exc:  # noqa: BLE001 - last-resort safety net so one bad file never crashes the run
        detail = f"unexpected error on {url}: {exc}"
        logging.error("Unexpected error downloading %s -> %s: %s", url, filepath, exc)
        return "error", detail
    finally:
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except OSError:
                pass


def sanitize(name) -> str:
    """Turn a value into a filesystem-safe string; never returns an empty string."""
    text = str(name) if name is not None else ""
    cleaned = "".join(c if c.isalnum() or c in "-_." else "_" for c in text).strip("_")
    return cleaned or "unknown"


def process_set(
    session: requests.Session,
    lang: str,
    set_brief: dict,
    output_dir: str,
    qualities: list,
    extensions: list,
    workers: int,
    fallback_lang: str = None,
) -> dict:
    """Download every image of one set. Returns a counters dict, never raises."""
    counters = {"downloaded": 0, "skipped": 0, "missing": 0, "error": 0, "no_image": 0}

    if not isinstance(set_brief, dict) or not set_brief.get("id"):
        logging.error("Malformed set entry (missing 'id') for lang %s: %r", lang, set_brief)
        print(f"  [!] Skipping a malformed set entry for '{lang}' (no id).")
        return counters

    set_id = set_brief["id"]
    set_name = set_brief.get("name", set_id)

    try:
        set_detail = get_set_detail(session, lang, set_id)
    except (requests.RequestException, TCGdexError) as exc:
        logging.error("Could not fetch set detail for %s/%s: %s", lang, set_id, exc)
        print(f"  [!] Skipping set {set_id} ({set_name}) for lang '{lang}': {exc}")
        return counters

    cards = set_detail.get("cards")
    used_lang = lang  # the language whose card data we actually end up using

    if not isinstance(cards, list) or not cards:
        # Very common for old/promo sets: the set exists but has no
        # translated card data for this language yet in TCGdex's database.
        print(f"  [i] Set {set_id} ({set_name}) has no card data in '{lang}'.", end="")
        if fallback_lang and fallback_lang != lang:
            print(f" Retrying with fallback language '{fallback_lang}'...")
            try:
                fallback_detail = get_set_detail(session, fallback_lang, set_id)
                fallback_cards = fallback_detail.get("cards")
            except (requests.RequestException, TCGdexError) as exc:
                logging.error("Fallback fetch failed for %s/%s: %s", fallback_lang, set_id, exc)
                fallback_cards = None
            if isinstance(fallback_cards, list) and fallback_cards:
                cards = fallback_cards
                used_lang = fallback_lang
                print(f"      -> using '{fallback_lang}' card data instead ({len(cards)} cards).")
            else:
                print(f"      -> no card data in '{fallback_lang}' either. Skipping this set.")
                logging.info("Set %s has no cards in '%s' or fallback '%s'.", set_id, lang, fallback_lang)
                return counters
        else:
            print(" Skipping this set (no --fallback-lang configured, or already using it).")
            logging.info("Set %s/%s has no cards to download.", lang, set_id)
            return counters

    # {output}/{lang}/{setId}/...
    set_dir = os.path.join(output_dir, sanitize(lang), sanitize(set_id))

    # Build the full list of (url, filepath) download tasks for this set,
    # skipping/logging any card that doesn't have what we need instead of
    # letting one bad entry crash the whole set.
    tasks = []
    seen_filenames = set()
    for card in cards:
        if not isinstance(card, dict):
            logging.warning("Skipping malformed card entry in %s/%s: %r", lang, set_id, card)
            counters["no_image"] += 1
            continue

        card_image_base = card.get("image")
        card_ref = card.get("id") or card.get("localId") or "unknown"
        if not card_image_base:
            logging.info("Card %s in %s/%s has no image URL, skipping.", card_ref, lang, set_id)
            counters["no_image"] += 1
            continue

        # Defensive normalization: the API is documented to always return a
        # fully-qualified "https://assets.tcgdex.net/..." base, but guard
        # against a protocol-relative ("//assets...") or bare-host/path
        # variant so a format change doesn't silently produce broken URLs.
        if card_image_base.startswith("//"):
            card_image_base = "https:" + card_image_base
        elif "://" not in card_image_base:
            first_segment = card_image_base.split("/", 1)[0]
            if "." in first_segment:
                # Already looks like "assets.tcgdex.net/..." - just add the scheme.
                card_image_base = "https://" + card_image_base
            else:
                # A bare path with no host at all.
                card_image_base = "https://assets.tcgdex.net/" + card_image_base.lstrip("/")

        local_id = sanitize(card.get("localId") or card.get("id"))

        for quality in qualities:
            for ext in extensions:
                url = f"{card_image_base}/{quality}.{ext}"
                filename = f"{local_id}_{quality}.{ext}"
                # Guard against duplicate localIds colliding on the same filename.
                if filename in seen_filenames:
                    filename = f"{local_id}_{sanitize(card_ref)}_{quality}.{ext}"
                seen_filenames.add(filename)
                filepath = os.path.join(set_dir, filename)
                tasks.append((url, filepath))

    if not tasks:
        return counters

    desc = f"[{lang}] {set_id} - {set_name}"[:40]
    error_samples = []

    try:
        with tqdm(total=len(tasks), desc=desc, unit="img", leave=True) as pbar:
            with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
                futures = {
                    executor.submit(download_image, session, url, filepath): (url, filepath)
                    for url, filepath in tasks
                }
                for future in as_completed(futures):
                    try:
                        result, detail = future.result()
                    except Exception as exc:  # noqa: BLE001 - a worker must never take down the whole run
                        url, filepath = futures[future]
                        logging.error("Worker crashed for %s -> %s: %s", url, filepath, exc)
                        result, detail = "error", f"worker crashed: {exc}"
                    counters[result] = counters.get(result, 0) + 1
                    if result == "error" and detail and len(error_samples) < 3:
                        error_samples.append(detail)
                    pbar.set_postfix({k: v for k, v in counters.items() if v}, refresh=False)
                    pbar.update(1)
    except KeyboardInterrupt:
        print(f"\n  Interrupted while downloading set {set_id} ({lang}). Partial progress is kept.")
        raise

    if counters["no_image"]:
        print(f"  Note: {counters['no_image']} card(s) had no usable image URL in this set.")
    if counters["error"]:
        print(f"  [!] {counters['error']} download(s) failed in {set_id} ({lang}). Example reason(s):")
        for sample in error_samples:
            print(f"      - {sample}")

    logging.info("Set %s/%s done: %s", lang, set_id, counters)
    return counters


def parse_args():
    parser = argparse.ArgumentParser(description="Scrape all Pokémon TCG card images from TCGdex.")
    parser.add_argument(
        "--langs",
        nargs="+",
        default=["en"],
        help="One or more language codes to scrape, e.g. --langs en fr de (default: en). "
        "Each language gets its own subfolder in the output directory.",
    )
    parser.add_argument("--output", default="pokemon_cards", help="Output directory (default: pokemon_cards)")
    parser.add_argument(
        "--qualities",
        nargs="+",
        default=DEFAULT_QUALITIES,
        choices=["high", "low"],
        help="Image qualities to download (default: high low)",
    )
    parser.add_argument(
        "--extensions",
        nargs="+",
        default=DEFAULT_EXTENSIONS,
        choices=["png", "jpg", "webp"],
        help="Image extensions to download (default: png jpg webp)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of concurrent download threads per set (default: 8)",
    )
    parser.add_argument(
        "--sets",
        nargs="*",
        default=None,
        help="Optional list of specific set IDs to download (default: all sets)",
    )
    parser.add_argument(
        "--fallback-lang",
        default="en",
        help="Language to retry with when a set has no translated card data in the requested "
        "language (common for old/promo sets). Set to an empty string to disable. Default: en.",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Delay in seconds between processing sets, to be gentle on the API (default: 0)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.workers < 1:
        print("--workers must be at least 1; using 1.")
        args.workers = 1

    session = build_session()
    try:
        os.makedirs(args.output, exist_ok=True)
    except OSError as exc:
        print(f"Fatal error: cannot create output directory '{args.output}': {exc}")
        sys.exit(1)

    grand_total = {"downloaded": 0, "skipped": 0, "missing": 0, "error": 0, "no_image": 0}
    failed_langs = []

    try:
        for lang_idx, lang in enumerate(args.langs, start=1):
            print(f"\n=== Language {lang_idx}/{len(args.langs)}: '{lang}' ===")
            print(f"Fetching set list for language '{lang}'...")
            try:
                all_sets = get_all_sets(session, lang)
            except (requests.RequestException, TCGdexError) as exc:
                print(f"  [!] Could not fetch set list for '{lang}': {exc} -- skipping this language.")
                logging.error("Could not fetch set list for lang %s: %s", lang, exc)
                failed_langs.append(lang)
                continue

            sets_to_process = [s for s in all_sets if isinstance(s, dict) and s.get("id")]
            skipped_malformed = len(all_sets) - len(sets_to_process)
            if skipped_malformed:
                print(f"  Warning: {skipped_malformed} set entr(y/ies) from the API had no usable 'id' and were ignored.")

            if args.sets:
                wanted = set(args.sets)
                sets_to_process = [s for s in sets_to_process if s["id"] in wanted]
                missing = wanted - {s["id"] for s in sets_to_process}
                if missing:
                    print(f"  Warning: these set IDs were not found for '{lang}': {', '.join(sorted(missing))}")

            print(f"Found {len(sets_to_process)} set(s) to process for '{lang}'.")

            for i, set_brief in enumerate(sets_to_process, start=1):
                set_label = f"{set_brief.get('id', '?')} - {set_brief.get('name', '?')}"
                print(f"\n[{lang}][{i}/{len(sets_to_process)}] Processing set: {set_label}")
                try:
                    result = process_set(
                        session=session,
                        lang=lang,
                        set_brief=set_brief,
                        output_dir=args.output,
                        qualities=args.qualities,
                        extensions=args.extensions,
                        workers=args.workers,
                        fallback_lang=args.fallback_lang or None,
                    )
                    for key, value in result.items():
                        grand_total[key] = grand_total.get(key, 0) + value
                except KeyboardInterrupt:
                    raise
                except Exception as exc:  # noqa: BLE001 - never let one bad set kill the whole run
                    logging.error("Unexpected error processing set %s/%s: %s", lang, set_brief.get("id"), exc)
                    print(f"  [!] Unexpected error on set {set_label}, skipping it: {exc}")

                if args.delay:
                    time.sleep(args.delay)
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Already-downloaded files are kept; re-run the script to resume.")
        sys.exit(130)

    print("\n" + "=" * 60)
    print("Done! Images saved under:", os.path.abspath(args.output))
    print("Folder structure: {output}/{lang}/{setId}/{localId}_{quality}.{extension}")
    print(
        f"Totals -> downloaded: {grand_total['downloaded']}, "
        f"already existed: {grand_total['skipped']}, "
        f"not available (404): {grand_total['missing']}, "
        f"no image URL: {grand_total['no_image']}, "
        f"errors: {grand_total['error']}"
    )
    if failed_langs:
        print(f"Languages that could not be fetched at all: {', '.join(failed_langs)}")
    if grand_total["error"]:
        print("Some downloads failed -- see tcgdex_scraper.log for details. Re-run the script to retry them.")
    print("Full log: tcgdex_scraper.log")


if __name__ == "__main__":
    main()