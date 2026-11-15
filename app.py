from pathlib import Path

from flask import Flask, render_template, request, redirect, url_for, flash, make_response

import pandas as pd
import numpy as np
import re
from src.processor import load_local_folder, combine_sources, price_comparison

from scipy.stats import f_oneway ,f as f_dist, pearsonr, spearmanr

BASE_DIR = Path(__file__).resolve().parent

JAN_DIR = BASE_DIR / "data" / "raw" / "janaushadhi"

HUB_DIR = BASE_DIR / "data" / "raw" / "hubscrape"

app = Flask(__name__)

app.secret_key = "medicine-price-analysis-local-app"

def load_data():

    jan = load_local_folder(JAN_DIR, "Jan Aushadhi")

    hub = load_local_folder(HUB_DIR, "HubScrape")

    return jan, hub, combine_sources(jan, hub)

PHARMACIES = ["Jan Aushadhi", "Tata 1mg", "Apollo Pharmacy"]

def calculate_matched_anova(data):
    """Like-for-like ANOVA: price PER UNIT (tablet/capsule) compared across pharmacies
    after controlling for the medicine itself (same composition + strength + form).

    Implemented as a randomized-block / two-way additive ANOVA (medicine = block,
    pharmacy = factor) using an F-test of the full model against a medicine-only
    model, so unequal numbers of listings per pharmacy are handled correctly.
    """
    needed = ["equivalent_key", "pharmacy", "price_per_unit"]
    if data.empty or not all(c in data.columns for c in needed):
        return {"error": "No price data is available for the matched ANOVA."}

    df = data[needed].copy()
    df["price_per_unit"] = pd.to_numeric(df["price_per_unit"], errors="coerce")
    df = df[df["price_per_unit"].notna() & (df["price_per_unit"] > 0)
            & df["equivalent_key"].fillna("").astype(str).ne("")
            & df["pharmacy"].isin(PHARMACIES)]
    # keep only medicines sold by at least two of the pharmacies
    df = df[df.groupby("equivalent_key")["pharmacy"].transform("nunique") >= 2]
    if df.empty or df["pharmacy"].nunique() < 2:
        return {"error": "No medicine has a matched price per unit in at least two pharmacies."}

    pharmacies = [p for p in PHARMACIES if p in set(df["pharmacy"])]
    keys = sorted(df["equivalent_key"].unique())
    y = df["price_per_unit"].to_numpy(float)
    block = pd.get_dummies(df["equivalent_key"]).reindex(columns=keys, fill_value=0).to_numpy(float)
    treat = np.column_stack([(df["pharmacy"] == p).to_numpy(float) for p in pharmacies[1:]])
    full = np.hstack([block, treat])

    def fit(x):
        beta = np.linalg.lstsq(x, y, rcond=None)[0]
        resid = y - x @ beta
        return beta, float(resid @ resid), int(np.linalg.matrix_rank(x))

    beta_f, rss_f, rank_f = fit(full)
    _, rss_r, rank_r = fit(block)
    df1, df2 = rank_f - rank_r, len(y) - rank_f
    if df1 < 1 or df2 < 1 or rss_f <= 0:
        return {"error": "Not enough matched listings for a valid ANOVA (need repeated prices within pharmacies)."}

    f_stat = ((rss_r - rss_f) / df1) / (rss_f / df2)
    p_value = float(f_dist.sf(f_stat, df1, df2))
    base = float(np.mean(beta_f[:len(keys)]))
    adjusted = {pharmacies[0]: base}
    for i, name in enumerate(pharmacies[1:]):
        adjusted[name] = base + float(beta_f[len(keys) + i])

    return {
        "groups": {p: round(adjusted[p], 3) for p in pharmacies},
        "raw_means": {p: round(float(df.loc[df["pharmacy"] == p, "price_per_unit"].mean()), 3) for p in pharmacies},
        "sample_sizes": {p: int((df["pharmacy"] == p).sum()) for p in pharmacies},
        "medicines": keys,
        "f_stat": round(float(f_stat), 4),
        "p_value": p_value,
        "df": (df1, df2),
        "significant": bool(p_value < 0.05),
        "note": "Same composition, strength and dosage form only; price per tablet/capsule removes pack-size differences.",
    }

def _corr_row(label, x, y, note=""):
    """Pearson + Spearman correlation for two numeric series (NaNs dropped)."""
    d = pd.DataFrame({"x": pd.to_numeric(x, errors="coerce"), "y": pd.to_numeric(y, errors="coerce")}).dropna()
    row = {"label": label, "n": len(d), "note": note}
    if len(d) < 3 or d["x"].nunique() < 2 or d["y"].nunique() < 2:
        row["error"] = "Not enough varied data points."
        return row, d
    r, p_r = pearsonr(d["x"], d["y"])
    rho, p_s = spearmanr(d["x"], d["y"])
    strength = abs(r)
    row.update({
        "pearson": round(float(r), 3), "pearson_p": float(p_r),
        "spearman": round(float(rho), 3), "spearman_p": float(p_s),
        "significant": bool(p_r < 0.05),
        "strength": "very strong" if strength >= 0.8 else "strong" if strength >= 0.6
                    else "moderate" if strength >= 0.4 else "weak" if strength >= 0.2 else "very weak",
        "direction": "positive" if r > 0 else "negative",
    })
    return row, d


def calculate_correlations(data):
    """Correlation analyses on the cleaned dataset; returns table rows + scatter points."""
    if data.empty:
        return {"rows": [], "scatter": {}}
    df = data.copy()
    rows = []

    # 1. Does the discounted selling price track the MRP? Retail pharmacies only:
    # Jan Aushadhi has no separate selling price (price = MRP), which would give a fake r = 1.
    retail = df[df["pharmacy"].isin(["Tata 1mg", "Apollo Pharmacy"])]
    row, _ = _corr_row("MRP vs listed price (Tata 1mg + Apollo)", retail["mrp"], retail["selling_price"],
                       "Retail pharmacy listings only")
    rows.append(row)

    # 2. Does a higher strength cost more per tablet/capsule? (single-strength solid forms)
    strength_mg = df["strength"].astype(str).str.extract(
        r"^\s*(\d+(?:\.\d+)?)\s*mg\s*$", flags=re.I)[0]
    solid = df["dosage_form"].astype(str).str.casefold().isin(["tablet", "capsule"])
    sub = df[solid & strength_mg.notna()].copy()
    sub["strength_mg"] = pd.to_numeric(strength_mg[sub.index])
    sub["price_per_unit"] = pd.to_numeric(sub["price_per_unit"], errors="coerce")
    sub = sub[sub["price_per_unit"] > 0]
    row, pts = _corr_row("Strength (mg) vs price per tablet/capsule",
                         sub["strength_mg"], sub["price_per_unit"],
                         "Single-strength tablets and capsules")
    rows.append(row)

    # 3. Same relationship for paracetamol only (comparable across all three sources)
    para = sub[sub["composition"].astype(str).str.contains("paracetamol", case=False)
               & ~sub["composition"].astype(str).str.contains(r"\+|\band\b", case=False, regex=True)]
    row, _ = _corr_row("Strength vs price per unit (paracetamol only)",
                       para["strength_mg"], para["price_per_unit"],
                       "Single-ingredient paracetamol, all pharmacies")
    rows.append(row)

    # 4. Discount vs MRP for retail pharmacies (do pricier medicines get bigger discounts?)
    row, _ = _corr_row("MRP vs discount % (Tata 1mg + Apollo)", retail["mrp"], retail["discount_percent"],
                       "Retail pharmacy listings only")
    rows.append(row)

    scatter = {}
    for ph, g in sub.groupby("pharmacy"):
        scatter[ph] = [{"x": float(x), "y": round(float(y), 3)} for x, y in zip(g["strength_mg"], g["price_per_unit"])]
    return {"rows": rows, "scatter": scatter}

def filtered_data(data):

    if data.empty:

        return data

    search = request.args.get("search", "").strip()

    result = data.copy()

    if search:

        result = result[result["product_name"].astype(str).str.contains(search, case=False, na=False)]

    return result

def find_equivalent_medicines(data, search):

    """Return a primary search result and strictly matched alternatives."""

    if data.empty or not search:

        return None, pd.DataFrame(), ""

    names = data["product_name"].fillna("").astype(str)

    brands = data["brand"].fillna("").astype(str)

    ingredients = data.get("active_ingredient", pd.Series("", index=data.index)).fillna("").astype(str)

    compositions = data.get("composition", pd.Series("", index=data.index)).fillna("").astype(str)

    search_terms = data.get("search_composition", pd.Series("", index=data.index)).fillna("").astype(str)

    query = search.casefold()

    matches = data[

        names.str.contains(search, case=False, na=False)

        | brands.str.contains(search, case=False, na=False)

        | ingredients.str.contains(search, case=False, na=False)

        | compositions.str.contains(search, case=False, na=False)

        | search_terms.str.contains(search, case=False, na=False)

    ].copy()

    if matches.empty:

        return None, pd.DataFrame(), "No medicine matched your search."

    # Prefer a single-ingredient, fully described pharmacy record for the primary result.

    # This avoids selecting a combination medicine merely because it contains the query salt.

    exact = matches[

        names.loc[matches.index].str.strip().str.casefold().eq(query)

        | brands.loc[matches.index].str.strip().str.casefold().eq(query)

        | ingredients.loc[matches.index].str.strip().str.casefold().eq(query)

        | search_terms.loc[matches.index].str.strip().str.casefold().eq(query)

    ]

    complete = matches[

        matches["composition"].fillna("").astype(str).str.strip().ne("")

        & matches["strength"].fillna("").astype(str).str.strip().ne("")

        & matches["dosage_form"].fillna("").astype(str).str.strip().ne("")

    ]

    pure = complete[~complete["composition"].fillna("").astype(str).str.contains(r"\+|\band\b", case=False, regex=True)]

    preferred = pure if not pure.empty else (complete if not complete.empty else matches)

    primary = (exact.loc[exact.index.intersection(preferred.index)].iloc[0]

               if not exact.empty and len(exact.index.intersection(preferred.index))

               else preferred.iloc[0])

    key = str(primary.get("equivalent_key", "") or "").strip()

    if not key:

        return primary.to_dict(), pd.DataFrame(), (

            "Equivalent alternatives cannot be confirmed because composition, strength, "

            "or dosage-form information is missing for this record."

        )

    alternatives = data[

        data["equivalent_key"].fillna("").astype(str).eq(key)

        & data.index.to_series().ne(primary.name)

    ].copy()

    if not alternatives.empty:

        alternatives = alternatives.drop_duplicates(subset=["product_name", "brand", "pharmacy", "pack_size"])

    return primary.to_dict(), alternatives, ""

@app.route("/", methods=["GET"])

def home():

    jan, hub, all_data = load_data()

    data = filtered_data(all_data)

    # Keep ANOVA in sync with the category selected in Category Analysis.
    category_filter = request.args.get("category", "").strip()
    anova_data = all_data.copy()
    if category_filter and not anova_data.empty:
        anova_data = anova_data[
            anova_data["category"].astype(str).str.casefold() == category_filter.casefold()
        ]
    matched_anova = calculate_matched_anova(all_data)
    correlation = calculate_correlations(all_data)
    categories = sorted(
        {str(x).strip() for x in all_data["category"].dropna().astype(str) if str(x).strip()},
        key=str.casefold
    ) if not all_data.empty else []

    is_phase_one = hub.empty

    loaded_files = []
    for folder, frame in (("data/raw/janaushadhi/", jan), ("data/raw/hubscrape/", hub)):
        if not frame.empty:
            for source, count in frame.groupby("data_source_file").size().items():
                loaded_files.append({"name": str(source).split(": ", 1)[-1], "folder": folder, "records": int(count)})

    stats = {"records": 0, "categories": 0, "average": None, "highest": None, "valid_prices": 0, "missing_prices": 0}

    chart_categories, chart_prices = [], []

    medicine_rows = []

    category_stats, source_rows = [], []

    comparison_rows = []

    selected = request.args.get("medicine", "")

    medicine_search = request.args.get("search", "").strip()

    primary_medicine, equivalent_rows, equivalent_note = find_equivalent_medicines(all_data, medicine_search)

    if primary_medicine:
        primary_medicine = {k: ("" if not isinstance(v, (list, dict)) and pd.isna(v) else v) for k, v in primary_medicine.items()}

    equivalent_rows = equivalent_rows.fillna("").to_dict("records") if not equivalent_rows.empty else []

    if not data.empty:

        prices = pd.to_numeric(data["selling_price"], errors="coerce")

        valid = data[prices.notna() & (prices > 0)].copy()

        stats = {

            "records": len(data),

            "categories": data["category"].nunique(),

            "average": float(valid["selling_price"].mean()) if not valid.empty else None,

            "highest": float(valid["selling_price"].max()) if not valid.empty else None,

            "valid_prices": len(valid),

            "missing_prices": len(data) - len(valid),

        }

        counts = data.groupby("category").size().sort_values(ascending=False)

        chart_categories = [{"name": str(k), "count": int(v)} for k, v in counts.items()]

        chart_prices = [float(x) for x in valid["selling_price"].dropna().tolist() if float(x) > 0]

        cols = ["product_id", "drug_code", "product_name", "pack_size", "category", "pharmacy", "mrp", "selling_price"]

        explorer_data = data.copy()

        category_filter = request.args.get("category", "").strip()

        if category_filter:

            explorer_data = explorer_data[explorer_data["category"].astype(str).str.casefold().eq(category_filter.casefold())]

        medicine_rows = explorer_data[cols].sort_values("product_name").fillna("").to_dict("records")

        cat = data.groupby("category").agg(

            medicine_records=("product_name", "count"),

            average_listed_price=("selling_price", "mean"),

            minimum_listed_price=("selling_price", "min"),

            maximum_listed_price=("selling_price", "max")

        ).reset_index().sort_values("medicine_records", ascending=False).fillna("")

        category_stats = cat.to_dict("records")

        source_rows = data.groupby("data_source_file").size().reset_index(name="records").to_dict("records")

        if not selected and len(medicine_rows):

            selected = str(medicine_rows[0]["product_name"])

        selected_rows = data[data["product_name"].astype(str) == selected][cols].fillna("").to_dict("records")

    else:

        selected_rows = []

    comparison = price_comparison(data)

    if not comparison.empty:

        comparison_rows = comparison.sort_values("potential_saving", ascending=False).fillna("").to_dict("records")

    return render_template(

        "index.html", data_loaded=not all_data.empty, stats=stats, categories=categories, search=request.args.get("search", ""),

        chart_categories=chart_categories, chart_prices=chart_prices, medicine_rows=medicine_rows, selected=selected,

        selected_category=request.args.get("category", "").strip(),

        selected_rows=selected_rows, category_stats=category_stats,

        source_rows=source_rows, is_phase_one=is_phase_one, comparison_rows=comparison_rows,

        hub_loaded=not hub.empty, primary_medicine=primary_medicine,

        equivalent_rows=equivalent_rows, equivalent_note=equivalent_note, matched_anova=matched_anova,correlation=correlation,
        
        loaded_files=loaded_files

    )

@app.route("/download/cleaned.csv")

def download_cleaned():

    _, _, all_data = load_data()

    data = filtered_data(all_data)

    if data.empty:

        flash("There is no data to download yet.", "error")

        return redirect(url_for("home"))

    response = make_response("\ufeff" + data.to_csv(index=False))

    response.headers["Content-Disposition"] = "attachment; filename=medicine_catalogue_cleaned.csv"

    response.headers["Content-Type"] = "text/csv; charset=utf-8"

    return response

@app.route("/download/comparison.csv")

def download_comparison():

    _, _, all_data = load_data()

    data = filtered_data(all_data)

    comparison = price_comparison(data)

    if comparison.empty:

        # Explicitly explain why records could not be compared; never fabricate a price match.

        comparison = data[["product_name", "brand", "composition", "strength", "dosage_form", "pharmacy", "selling_price", "data_source_file"]].copy()

        comparison["comparison_status"] = "Not compared: no verified matching record from a second pharmacy source with complete composition, strength and dosage-form data."

        comparison["potential_saving"] = pd.NA

    response = make_response("\ufeff" + comparison.to_csv(index=False))

    response.headers["Content-Disposition"] = "attachment; filename=medicine_price_comparison.csv"

    response.headers["Content-Type"] = "text/csv; charset=utf-8"

    return response

if __name__ == "__main__":

    app.run(host="127.0.0.1", port=5000, debug=True)
