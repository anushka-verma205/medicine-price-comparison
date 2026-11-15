import re
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

ALIASES = {
    "product_name": ["product_name", "name", "medicine", "medicine_name", "drug_name", "title", "item_name", "generic_name"],
    "active_ingredient": ["active_ingredient", "active_ingredients", "ingredient", "salt_name"],
    "search_composition": ["search_composition", "search_salt", "search_ingredient"],
    "composition": ["composition", "salt", "salt_composition", "active_ingredients", "ingredients"],
    "strength": ["strength", "dosage", "dose", "medicine_strength", "concentration"],
    "dosage_form": ["dosage_form", "form", "product_form", "medicine_form", "type"],
    "pack_size": ["pack_size", "pack", "quantity", "unit_size", "package_size", "packaging"],
    "mrp": ["mrp", "list_price", "original_price", "maximum_retail_price", "market_price"],
    "selling_price": ["selling_price", "price", "sale_price", "discounted_price", "current_price", "final_price"],
    "discount_percent": ["discount_percent", "discount", "discount_percentage", "off"],
    "pharmacy": ["pharmacy", "pharmacy_platform", "source", "platform", "website", "store", "seller"],
    "brand": ["brand", "brand_name", "manufacturer", "company"],
    "category": ["category", "therapeutic_category", "medicine_category", "group_name"],
    "url": ["url", "product_url", "link", "product_link"],
    "scraped_at": ["scraped_at", "date", "collected_at", "crawl_date", "updated_at"],
    "medicine_key": ["medicine_key", "canonical_key", "comparison_key", "match_key"],
}


def clean_col(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def find_column(columns, candidates):
    normalized = {clean_col(column): column for column in columns}
    for candidate in candidates:
        if candidate in normalized:
            return normalized[candidate]
    return None


def numeric(series):
    cleaned = series.astype(str).str.replace(",", "", regex=False)
    cleaned = cleaned.str.replace(r"[^0-9.\-]", "", regex=True)
    return pd.to_numeric(cleaned, errors="coerce")


def normalize_one(df, source_hint="Unknown"):
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    out = pd.DataFrame(index=df.index)

    for target, aliases in ALIASES.items():
        column = find_column(df.columns, aliases)
        out[target] = df[column] if column else np.nan

    # Keep the official catalogue identifier when it is available.
    drug_code_col = find_column(df.columns, ["drug_code", "medicine_code", "product_code"])
    sr_no_col = find_column(df.columns, ["sr_no", "serial_no", "serial_number"])
    out["drug_code"] = df[drug_code_col] if drug_code_col else ""
    out["serial_number"] = df[sr_no_col] if sr_no_col else ""

    for column in ["product_name", "active_ingredient", "search_composition", "composition", "strength", "dosage_form", "pack_size", "brand", "category", "url"]:
        out[column] = out[column].fillna("").astype(str).str.strip()

    out["mrp"] = numeric(out["mrp"])
    out["selling_price"] = numeric(out["selling_price"])
    out["discount_percent"] = numeric(out["discount_percent"])

    # Jan Aushadhi catalogue exports provide MRP, not a separate discounted selling price.
    # Use that listed catalogue price for Phase 1 price analysis, without presenting it as
    # a live retail pharmacy selling price.
    is_jan_catalogue = "jan" in str(source_hint).lower() and "aushadhi" in str(source_hint).lower()
    if is_jan_catalogue:
        out["selling_price"] = out["selling_price"].fillna(out["mrp"])
        out["pharmacy"] = "Jan Aushadhi"
        out["price_basis"] = "Jan Aushadhi catalogue MRP"
    else:
        out["pharmacy"] = out["pharmacy"].replace("", np.nan).fillna(source_hint).astype(str).str.strip()
        out["price_basis"] = "Uploaded listed/selling price"

    calculated_discount = np.where(
        out["mrp"].notna() & (out["mrp"] > 0) & out["selling_price"].notna(),
        ((out["mrp"] - out["selling_price"]) / out["mrp"]) * 100,
        np.nan,
    )
    out["discount_percent"] = out["discount_percent"].fillna(pd.Series(calculated_discount, index=out.index))
    # Use structured ingredient fields when composition is absent.
    out["composition"] = out["composition"].replace("", np.nan).fillna(out["active_ingredient"]).fillna("")
    out["search_composition"] = out["search_composition"].replace("", np.nan).fillna(out["active_ingredient"]).fillna("")

    # Some exports put a long free-text description in the composition column
    # (e.g. "Active Ingredient: ... Excipients: ..."). Use the structured
    # ingredient instead so the equivalence key stays meaningful.
    long_text = out["composition"].str.len() > 100
    has_ingredient = out["active_ingredient"].str.strip().ne("")
    out.loc[long_text & has_ingredient, "composition"] = out.loc[long_text & has_ingredient, "active_ingredient"]
    out["composition"] = out["composition"].str.replace(r"^\s*Active Ingredients?\s*:\s*", "", regex=True, flags=re.I)

    # Jan Aushadhi supplies medicine details inside Generic Name rather than
    # separate composition, strength and dosage-form columns. Extract them.
    if is_jan_catalogue:
        generic_names = out["product_name"].fillna("").astype(str)

        out["strength"] = generic_names.str.findall(
            r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:ml|mg))?",
            flags=re.I
        ).apply(lambda values: "+".join(re.sub(r"\s+", "", v.lower()) for v in values))

        def jan_form(value):
            value = str(value).lower()
            if re.search(r"\btablets?\b", value):
                return "Tablet"
            if re.search(r"\bcapsules?\b", value):
                return "Capsule"
            if "suspension" in value:
                return "Suspension"
            if "syrup" in value:
                return "Syrup"
            if "drops" in value:
                return "Drops"
            return ""

        out["dosage_form"] = generic_names.apply(jan_form)
        out["composition"] = generic_names

    # Some pharmacy exports omit the strength column even though it is in
    # the product name (for example, Dolo-650 Tablet).
    volume_only = out["strength"].fillna("").astype(str).str.fullmatch(r"\s*\d+(?:\.\d+)?\s*ml\s*", case=False)
    missing_strength = out["strength"].fillna("").astype(str).str.strip().eq("") | volume_only
    if missing_strength.any():
        names = out.loc[missing_strength, "product_name"].fillna("").astype(str)

        def extract_name_strength(value):
            value = str(value)
            found = re.findall(
                r"\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)(?:\s*/\s*\d+(?:\.\d+)?\s*(?:ml|mg))?",
                value,
                flags=re.I
            )
            # A bare volume such as "60 ml" is pack volume, not medicine strength.
            found = [item for item in found if not re.fullmatch(r"\d+(?:\.\d+)?\s*ml", item, flags=re.I)]
            if found:
                return "+".join(re.sub(r"\s+", "", item.lower()) for item in found)

            # Some Apollo titles omit the unit, e.g. Dolo-650 Tablet.
            # Look only before the dosage-form word so pack counts such as 15's
            # are not mistaken for medicine strength.
            before_form = re.split(
                r"\b(?:tablets?|tabs?|capsules?|suspension|syrup|drops?)\b",
                value,
                maxsplit=1,
                flags=re.I
            )[0]
            unitless = re.findall(
                r"(?<!\d)(250|325|500|650|1000)(?!\d)",
                before_form,
                flags=re.I
            )
            return "+".join(item + "mg" for item in unitless)

        extracted = names.apply(extract_name_strength)
        out.loc[missing_strength, "strength"] = extracted

    # Fill a missing dosage form from the product name only.
    missing_form = out["dosage_form"].fillna("").astype(str).str.strip().eq("")
    if missing_form.any():
        def infer_form_from_name(value):
            value = str(value).lower()
            for term, label in [("tablet", "Tablet"), ("capsule", "Capsule"), ("suspension", "Suspension"), ("syrup", "Syrup"), ("drops", "Drops")]:
                if term in value:
                    return label
            return ""

        out.loc[missing_form, "dosage_form"] = out.loc[missing_form, "product_name"].apply(infer_form_from_name)

    # Category is not supplied by every pharmacy export. Infer a broad analytics category
    # from ingredient/name keywords instead of leaving the dashboard cell blank.
    def infer_category(row):
        existing = str(row.get("category", "")).strip()
        if existing and existing.lower() not in {"nan", "none"}:
            return existing
        text = " ".join(str(row.get(c, "")) for c in ["search_composition", "active_ingredient", "composition", "product_name"]).lower()
        rules = [
            ("Analgesic / Antipyretic", ["paracetamol", "acetaminophen", "ibuprofen", "aceclofenac", "diclofenac"]),
            ("Antibiotic", ["azithromycin", "amoxicillin", "cefixime", "ofloxacin", "ciprofloxacin"]),
            ("Antiallergic", ["cetirizine", "levocetirizine", "fexofenadine", "loratadine"]),
            ("Gastrointestinal", ["pantoprazole", "omeprazole", "rabeprazole", "domperidone"]),
            ("Diabetes", ["metformin", "glimepiride", "sitagliptin"]),
            ("Cardiovascular", ["amlodipine", "telmisartan", "atorvastatin", "losartan"]),
        ]
        for label, words in rules:
            if any(word in text for word in words):
                return label
        return "Other / Unclassified"

    out["category"] = out.apply(infer_category, axis=1)

    # Canonicalize category labels so spelling/case/spacing variants collapse
    # into one dashboard category (including catalogue group-name variants).
    def normalize_category(value):
        raw = re.sub(r"\s+", " ", str(value)).strip()
        key = re.sub(r"[^a-z0-9]+", " ", raw.casefold()).strip()

        if not key or key in {"nan", "none", "unknown"}:
            return "Other / Unclassified"

        aliases = {
            # Analgesic categories
            "analgesic": "Analgesic / Antipyretic",
            "analgesics": "Analgesic / Antipyretic",
            "antipyretic": "Analgesic / Antipyretic",
            "analgesic antipyretic": "Analgesic / Antipyretic",
            "analgesic and antipyretic": "Analgesic / Antipyretic",
            "analgesic antipyretic anti inflammatory": "Analgesic / Antipyretic",
            "analgesic anti inflammatory": "Analgesic / Antipyretic",
            "pain killer": "Analgesic / Antipyretic",
            "painkillers": "Analgesic / Antipyretic",

            # Gastrointestinal categories
            "gastro": "Gastrointestinal",
            "gastrointestinal": "Gastrointestinal",
            "gastrointestinal git": "Gastrointestinal",

            # Other categories
            "antibiotics": "Antibiotic",
            "anti allergic": "Antiallergic",
            "anti allergy": "Antiallergic",
            "cardiac": "Cardiovascular",
            "cardiovascular system cvs": "Cardiovascular",
            "cardiovascular": "Cardiovascular",
            "anti diabetic": "Diabetes",
            "antidiabetic": "Diabetes",
            "other": "Other / Unclassified",
            "others": "Other / Unclassified",
            "other unclassified": "Other / Unclassified",
        }

        return aliases.get(key, raw.title())
    out["category"] = out["category"].apply(normalize_category)
    out["scraped_at"] = pd.to_datetime(out["scraped_at"], errors="coerce", dayfirst=True)

    supplied_key = out["medicine_key"].fillna("").astype(str).str.strip()

    def norm_text(value):
        return re.sub(r"\s+", " ", str(value).lower()).strip()

    # Normalize ingredient text while retaining all active ingredients.
    # Paracetamol and Acetaminophen are treated as synonyms.
    def normalize_composition(value):
        text = norm_text(value)
        text = re.sub(r"\bacetaminophen\b", "paracetamol", text)
        text = re.sub(r"\b(ip|usp|bp)\b", " ", text)
        text = re.sub(r"\([^)]*\)", " ", text)
        text = re.sub(r"\b\d+(?:\.\d+)?\s*(?:mg|mcg|g|ml|iu)\b", " ", text)
        text = re.sub(r"\b(tablets?|tabs?|capsules?|caps?|syrup|suspension|drops?)\b", " ", text)
        text = re.sub(r"[^a-z0-9+]+", " ", text)
        parts = [re.sub(r"\s+", " ", part).strip() for part in re.split(r"\s*\+\s*|\s+and\s+", text)]
        parts = sorted(part for part in parts if part)
        return "+".join(parts)

    def normalize_strength(value):
        return re.sub(r"\s+", "", norm_text(value))

    def normalize_form(value):
        value = norm_text(value)
        form_map = {"tablets": "tablet", "tabs": "tablet", "tab": "tablet", "capsules": "capsule", "caps": "capsule", "cap": "capsule"}
        return form_map.get(value, value)

    def make_key(row):
        if row["_provided_key"]:
            return norm_text(row["_provided_key"])
        parts = [normalize_composition(row["composition"]), normalize_strength(row["strength"]), normalize_form(row["dosage_form"])]
        if all(str(value).strip() and str(value).lower() not in {"nan", "none"} for value in parts):
            return "|".join(parts)
        return ""

    out["_provided_key"] = supplied_key
    out["comparison_key"] = out.apply(make_key, axis=1)

    # Equivalence matching deliberately excludes pack size, but requires
    # confirmed composition, strength and dosage form. Missing fields never match.
    def make_equivalent_key(row):
        parts = [normalize_composition(row["composition"]), normalize_strength(row["strength"]), normalize_form(row["dosage_form"])]
        if not all(str(value).strip() and str(value).lower() not in {"nan", "none"} for value in parts):
            return ""
        return "|".join(parts)

    out["equivalent_key"] = out.apply(make_equivalent_key, axis=1)

    # Stable ID for each pharmacy product listing; distinct from medicine matching.
    def make_product_id(row):
        identity = "|".join(
            str(row.get(field, "") or "").strip().casefold()
            for field in [
                "pharmacy", "product_name", "brand", "strength",
                "dosage_form", "pack_size", "url"
            ]
        )
        digest = hashlib.sha1(identity.encode("utf-8")).hexdigest()[:10].upper()
        return "PRD-" + digest

    out["product_id"] = out.apply(make_product_id, axis=1)
    out.drop(columns=["medicine_key", "_provided_key"], inplace=True)
    out["data_source_file"] = source_hint
    return out.reset_index(drop=True)


def read_file(file_or_path):
    name = str(getattr(file_or_path, "name", file_or_path)).lower()
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(file_or_path)
    # utf-8-sig handles common Excel CSV exports; fallback encodings help with Indian exports.
    try:
        return pd.read_csv(file_or_path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        if hasattr(file_or_path, "seek"):
            file_or_path.seek(0)
        return pd.read_csv(file_or_path, encoding="cp1252")


def load_uploaded_files(uploaded_files, source_hint):
    frames = []
    for file in uploaded_files or []:
        try:
            frames.append(normalize_one(read_file(file), source_hint=f"{source_hint}: {file.name}"))
        except Exception as exc:
            raise ValueError(f"Could not read {file.name}: {exc}") from exc
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def load_local_folder(folder, source_hint):
    frames = []
    folder_path = Path(folder)
    if not folder_path.exists():
        return pd.DataFrame()
    for path in folder_path.iterdir():
        if path.suffix.lower() in [".csv", ".xlsx", ".xls"]:
            try:
                frames.append(normalize_one(read_file(path), f"{source_hint}: {path.name}"))
            except Exception:
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def combine_sources(janaushadhi, hubscrape):
    parts = [part for part in [janaushadhi, hubscrape] if part is not None and not part.empty]
    if not parts:
        return pd.DataFrame()
    df = pd.concat(parts, ignore_index=True)
    df["selling_price"] = pd.to_numeric(df["selling_price"], errors="coerce")
    df["mrp"] = pd.to_numeric(df["mrp"], errors="coerce")
    # Keep named catalogue rows even when price is missing/zero so data-quality
    # reporting can account for them. Price charts filter to valid positive values.
    df = df[df["product_name"].fillna("").astype(str).str.strip().ne("")].copy()
    for column in ["product_name", "composition", "strength", "dosage_form", "pack_size"]:
        df[column] = df[column].fillna("").astype(str).str.strip()
    df["pharmacy"] = df["pharmacy"].replace("", "Unknown").fillna("Unknown").astype(str).str.strip()
    pharmacy_key = df["pharmacy"].str.casefold().str.replace(r"\s+", " ", regex=True)
    pharmacy_aliases = {
        "apollo": "Apollo Pharmacy", "apollo pharmacy": "Apollo Pharmacy",
        "tata1mg": "Tata 1mg", "tata 1mg": "Tata 1mg", "1mg": "Tata 1mg",
        "jan aushadhi": "Jan Aushadhi", "janaushadhi": "Jan Aushadhi",
    }
    df["pharmacy"] = pharmacy_key.map(pharmacy_aliases).fillna(df["pharmacy"])
    df["price_per_unit"] = np.nan

    # For simple count-based packs such as 10's or 10 tablets, calculate price per item.
    pack_text = df["pack_size"].astype(str).str.strip()
    pack_count = pack_text.str.extract(
        r"^(\d+(?:\.\d+)?)\s*(?:'s|tabs?|tablets?|capsules?|caps?|sachets?|strips?)(?:\s+er)?$", flags=re.I
    )[0]
    # Scraped exports sometimes give just the number (e.g. 15.0); that is a unit
    # count only for solid forms, not for liquids where it is a volume.
    solid = df["dosage_form"].astype(str).str.casefold().isin(["tablet", "capsule"])
    bare = pack_text.str.extract(r"^(\d+(?:\.\d+)?)$")[0].where(solid)
    pack_count = pd.to_numeric(pack_count.fillna(bare), errors="coerce")
    valid = pack_count.notna() & (pack_count > 0)
    df.loc[valid, "price_per_unit"] = df.loc[valid, "selling_price"] / pack_count[valid]
    return df.reset_index(drop=True)


def price_comparison(df):
    """Compare the cheapest listing of each pharmacy for identical medicines.

    Identical = same composition + strength + dosage form. To keep the comparison
    like-for-like, price PER UNIT is used when the pack is a countable tablet/capsule
    pack; otherwise listings are compared only within the same pack size.
    """
    if df.empty:
        return pd.DataFrame()
    eligible = df[df["comparison_key"].fillna("").astype(str).str.len() > 0].copy()
    eligible["selling_price"] = pd.to_numeric(eligible["selling_price"], errors="coerce")
    eligible = eligible[eligible["selling_price"].notna() & (eligible["selling_price"] > 0)]
    if eligible.empty:
        return pd.DataFrame()

    unit = pd.to_numeric(eligible["price_per_unit"], errors="coerce")
    has_unit = unit.notna() & (unit > 0)
    eligible["compare_price"] = np.where(has_unit, unit, eligible["selling_price"])
    eligible["price_basis_label"] = np.where(has_unit, "per tablet/capsule", "per pack")
    pack_norm = eligible["pack_size"].astype(str).str.casefold().str.replace(r"\s+", "", regex=True)
    eligible["group_key"] = np.where(
        has_unit, eligible["comparison_key"], eligible["comparison_key"] + "|" + pack_norm
    )

    rows = []
    for _, group in eligible.groupby("group_key"):
        # cheapest listing per pharmacy, so one source can never be compared with itself
        best = group.loc[group.groupby("pharmacy")["compare_price"].idxmin()]
        if len(best) < 2:
            continue
        best = best.sort_values(["compare_price", "pharmacy"])
        low, high = best.iloc[0], best.iloc[-1]
        rows.append({
            "comparison_key": low["comparison_key"],
            "product_name": low["product_name"],
            "composition": low["composition"],
            "strength": low["strength"],
            "dosage_form": low["dosage_form"],
            "pack_size": low["pack_size"],
            "price_basis": low["price_basis_label"],
            "lowest_pharmacy": low["pharmacy"],
            "lowest_price": low["compare_price"],
            "highest_pharmacy": high["pharmacy"],
            "highest_price": high["compare_price"],
            "potential_saving": high["compare_price"] - low["compare_price"],
            "pharmacies_compared": len(best),
            "records_compared": len(group),
        })
    return pd.DataFrame(rows)
