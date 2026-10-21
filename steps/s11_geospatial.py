"""Interactive HMDA census-tract map.

Uses all eligible prepared records for the selected year.

Browser optimizations:
- Includes only tracts with observations.
- Simplifies boundaries for display only.
- Includes only necessary tooltip properties.
- Uses compact GeoJSON and canvas rendering.

Outputs:
    tract_approval_map_YEAR.html
    tract_summary_YEAR.csv
    unmatched_tracts_YEAR.csv
    map_audit_YEAR.json

Rates are unweighted sample descriptions, not population estimates,
adjusted disparities, or findings of discrimination.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import config
import folium
import geopandas as gpd
import pandas as pd
from branca.colormap import LinearColormap
from branca.element import Element


def normalize_geoid(values):
    """Normalize numeric tract identifiers."""
    values = (
        values.astype("string")
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    valid = values.str.fullmatch(r"\d{1,11}").fillna(False)
    return values.where(valid).str.zfill(11)


def save_json(data, path):
    path.write_text(
        json.dumps(data, indent=2),
        encoding="utf-8",
    )


def save_interactive_map(
    mapped, path, year, min_applications, simplify_meters
):
    """Create an interactive map using observed tracts only."""
    columns = [
        "GEOID",
        "state_fips",
        "applications",
        "approvals",
        "denials",
        "approval_rate",
        "geometry",
    ]

    # Remove boundaries without observations from the browser layer.
    web = mapped.loc[
        mapped["applications"].notna(), columns
    ].copy()

    if web.empty:
        raise ValueError("No matched tracts available for mapping")

    print(f"Preparing {len(web):,} observed tract boundaries...")

    # Work in meters when simplifying. Original boundaries and
    # statistical results are not modified.
    if simplify_meters > 0:
        web = web.to_crs("EPSG:2163")
        original_geometry = web.geometry.copy()

        simplified = web.geometry.simplify(
            tolerance=simplify_meters,
            preserve_topology=True,
        )

        invalid = (
            simplified.isna()
            | simplified.is_empty
            | ~simplified.is_valid
        )
        simplified.loc[invalid] = original_geometry.loc[invalid]
        web = web.set_geometry(simplified)

    web = web.to_crs("EPSG:4326")

    eligible = web["applications"].ge(min_applications)
    web["approval_pct"] = (
        100 * web["approval_rate"]
    ).round(1).where(eligible)

    for column in ["applications", "approvals", "denials"]:
        web[column] = web[column].astype(int)

    def status(count):
        if count < min_applications:
            return "Below display threshold"
        if count < 30:
            return "Small sample: fewer than 30 applications"
        return "Displayed"

    web["status"] = web["applications"].map(status)
    web["rate_text"] = web["approval_pct"].map(
        lambda value: (
            "Not displayed"
            if pd.isna(value)
            else f"{value:.1f}%"
        )
    )

    state_codes = set(web["state_fips"])
    location = [39, -98]
    zoom = 4

    if state_codes == {"02"}:
        location = [64, -152]
    elif state_codes == {"15"}:
        location = [20.5, -157.5]
        zoom = 7

    m = folium.Map(
        location=location,
        zoom_start=zoom,
        tiles=None,
        prefer_canvas=True,
        control_scale=True,
    )

    colors = LinearColormap(
        colors=[
            "#ffffd9",
            "#c7e9b4",
            "#41b6c4",
            "#225ea8",
            "#081d58",
        ],
        vmin=0,
        vmax=100,
        caption="Observed approval rate (%)",
    )

    # Round color values to reduce the number of generated styles.
    web["fill_color"] = web["approval_pct"].map(
        lambda value: (
            "#bdbdbd"
            if pd.isna(value)
            else colors(round(float(value)))
        )
    )

    fields = [
        "GEOID",
        "applications",
        "approvals",
        "denials",
        "rate_text",
        "status",
    ]
    aliases = [
        "Census tract:",
        "Applications:",
        "Approvals:",
        "Denials:",
        "Approval rate:",
        "Data status:",
    ]

    display = web[
        ["geometry", "fill_color"] + fields
    ].copy()

    print("Building interactive map...")
    geojson = json.loads(
        display.to_json(
            na="null",
            drop_id=True,
            separators=(",", ":"),
        )
    )

    folium.GeoJson(
        data=geojson,
        name=f"Observed tracts — {year}",
        style_function=lambda feature: {
            "fillColor": feature["properties"]["fill_color"],
            "fillOpacity": 0.75,
            "color": "#666666",
            "weight": 0.25,
        },
        highlight_function=lambda feature: {
            "weight": 2,
            "color": "#111111",
            "fillOpacity": 0.9,
        },
        tooltip=folium.GeoJsonTooltip(
            fields=fields,
            aliases=aliases,
            sticky=False,
            labels=True,
            localize=True,
        ),
        zoom_on_click=True,
    ).add_to(m)

    colors.add_to(m)
    folium.LayerControl().add_to(m)

    # Fit a single-state selection without spanning the date line.
    if len(state_codes) == 1:
        west, south, east, north = web.total_bounds
        if east - west < 180:
            m.fit_bounds([[south, west], [north, east]])

    note = f"""
    <div style="
        position:fixed;
        bottom:25px;
        left:15px;
        z-index:9999;
        background:white;
        padding:12px;
        border:1px solid #aaa;
        border-radius:5px;
        max-width:300px;
        font-size:13px;">
        <strong>Mortgage approval rates — {year}</strong><br>
        Hover for tract details. Click to zoom.<br>
        Minimum applications per tract: {min_applications}.<br>
        Gray: observed tract below the display threshold.<br>
        Tracts without observations are not drawn.<br><br>
        Unweighted sample rates. Small samples can produce
        unstable percentages. These results do not establish
        discrimination.
    </div>
    """
    m.get_root().html.add_child(Element(note))

    # Compact the embedded GeoJSON without changing its contents.
    html = m.get_root().render()
    decoder = json.JSONDecoder()
    marker = ".addData("
    start = html.find(marker)

    if start != -1:
        data_start = start + len(marker)
        while data_start < len(html) and html[data_start].isspace():
            data_start += 1

        try:
            embedded, length = decoder.raw_decode(html[data_start:])
            compact = json.dumps(
                embedded,
                separators=(",", ":"),
                ensure_ascii=True,
            )
            html = (
                html[:data_start]
                + compact
                + html[data_start + length:]
            )
        except (ValueError, json.JSONDecodeError):
            # Rendering remains valid if the Folium template differs.
            pass

    path.write_text(html, encoding="utf-8")

    return {
        "interactive_tracts": len(web),
        "simplify_meters": simplify_meters,
        "html_size_mb": round(
            path.stat().st_size / (1024 * 1024), 2
        ),
    }


def run(args):
    if not args.input.exists():
        raise FileNotFoundError(
            f"Prepared data not found: {args.input}"
        )

    if not args.boundaries.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {args.boundaries}"
        )

    if args.min_applications < 1:
        raise ValueError(
            "--min-applications must be at least 1"
        )

    if args.simplify_meters < 0:
        raise ValueError(
            "--simplify-meters cannot be negative"
        )

    states = None
    if args.states:
        if any(
            not state.isdigit() or len(state) > 2
            for state in args.states
        ):
            raise ValueError(
                "--states requires state FIPS codes, e.g. 06 13"
            )
        states = sorted({
            state.zfill(2) for state in args.states
        })

    out = args.output
    out.mkdir(parents=True, exist_ok=True)

    print(f"Reading prepared data: {args.input}")

    # Read only the columns needed for this analysis.
    df = pd.read_parquet(
        args.input,
        columns=["activity_year", "census_tract", "approved"],
    )
    total_rows = len(df)

    years = pd.to_numeric(
        df["activity_year"], errors="coerce"
    )
    df = df.loc[years.eq(args.year)].copy()

    if df.empty:
        raise ValueError(
            f"No prepared records found for {args.year}"
        )

    year_rows = len(df)
    df["GEOID"] = normalize_geoid(df["census_tract"])

    invalid_tract_rows = int(
        df["GEOID"].isna().sum()
    )
    df = df.loc[df["GEOID"].notna()].copy()

    if states:
        df = df.loc[
            df["GEOID"].str[:2].isin(states)
        ].copy()

    if df.empty:
        raise ValueError(
            "No records remain for the selected geography"
        )

    df["approved"] = pd.to_numeric(
        df["approved"], errors="coerce"
    )

    invalid_outcomes = ~df["approved"].isin([0, 1])
    if invalid_outcomes.any():
        raise ValueError(
            f"{int(invalid_outcomes.sum())} records have missing "
            "or nonbinary approved values. "
            "Use Step 4's decided cohort."
        )

    tract = (
        df.groupby("GEOID", as_index=False)
        .agg(
            applications=("approved", "size"),
            approvals=("approved", "sum"),
            approval_rate=("approved", "mean"),
        )
    )
    tract["approvals"] = tract["approvals"].astype(int)
    tract["denials"] = (
        tract["applications"] - tract["approvals"]
    )

    print(f"Reading boundaries: {args.boundaries}")
    boundaries = gpd.read_file(args.boundaries)

    if "GEOID" not in boundaries.columns:
        raise ValueError(
            "Boundary file must contain a GEOID column"
        )

    if boundaries.crs is None:
        raise ValueError(
            "Boundary file has no coordinate system"
        )

    boundaries = boundaries[
        ["GEOID", "geometry"]
    ].copy()

    boundaries["GEOID"] = normalize_geoid(
        boundaries["GEOID"]
    )

    if boundaries["GEOID"].isna().any():
        raise ValueError(
            "Boundary file contains invalid tract identifiers"
        )

    if boundaries["GEOID"].duplicated().any():
        raise ValueError(
            "Boundary file contains duplicate tract identifiers"
        )

    boundaries["state_fips"] = (
        boundaries["GEOID"].str[:2]
    )

    if states:
        boundaries = boundaries.loc[
            boundaries["state_fips"].isin(states)
        ].copy()

    if boundaries.empty:
        raise ValueError(
            "No boundaries found for the selected geography"
        )

    tract["boundary_matched"] = tract["GEOID"].isin(
        boundaries["GEOID"]
    )
    tract["meets_display_threshold"] = (
        tract["applications"].ge(args.min_applications)
    )

    unmatched = tract.loc[
        ~tract["boundary_matched"]
    ].copy()

    # Inner join: do not send empty-data tracts to the browser.
    mapped = boundaries.merge(
        tract,
        on="GEOID",
        how="inner",
        validate="one_to_one",
    )

    bad_geometry = (
        mapped.geometry.isna()
        | mapped.geometry.is_empty
        | ~mapped.geometry.is_valid
    )
    if bad_geometry.any():
        raise ValueError(
            f"{int(bad_geometry.sum())} matched boundary geometries "
            "are missing, empty, or invalid"
        )

    eligible = mapped["applications"].ge(
        args.min_applications
    )

    suffix = str(args.year)
    tract.to_csv(
        out / f"tract_summary_{suffix}.csv",
        index=False,
    )
    unmatched.to_csv(
        out / f"unmatched_tracts_{suffix}.csv",
        index=False,
    )

    audit = {
        "year": args.year,
        "states": states,
        "input_file": str(args.input.resolve()),
        "boundary_file": str(args.boundaries.resolve()),
        "boundary_crs": str(boundaries.crs),
        "total_prepared_rows": total_rows,
        "records_in_selected_year": year_rows,
        "invalid_tract_rows_in_selected_year": invalid_tract_rows,
        "applications_analyzed": len(df),
        "minimum_applications_per_tract": args.min_applications,
        "observed_tracts": len(tract),
        "matched_tracts": len(mapped),
        "unmatched_tracts": len(unmatched),
        "unmatched_applications": int(
            unmatched["applications"].sum()
        ),
        "tracts_displayed_with_rates": int(eligible.sum()),
        "observed_tracts_below_30_applications": int(
            tract["applications"].lt(30).sum()
        ),
        "map_created": False,
        "limitations": [
            "Unweighted descriptions of the prepared sample.",
            "No adjustment for applicant or lender characteristics.",
            "Small tract counts can produce unstable approval rates.",
            "Boundary vintage must match the HMDA tract geography.",
            "Display boundaries are simplified, not analysis inputs.",
            "Unobserved tracts are omitted from the interactive layer.",
            "No spatial hotspot or causal discrimination testing.",
        ],
    }

    audit_path = out / f"map_audit_{suffix}.json"
    save_json(audit, audit_path)

    if not eligible.any():
        raise ValueError(
            "No matched tracts meet the display threshold. "
            "Review the saved audit and unmatched-tract table."
        )

    html_path = out / f"tract_approval_map_{suffix}.html"

    map_info = save_interactive_map(
        mapped,
        path=html_path,
        year=args.year,
        min_applications=args.min_applications,
        simplify_meters=args.simplify_meters,
    )

    audit.update(map_info)
    audit["map_created"] = True
    audit["interactive_map"] = str(html_path.resolve())
    save_json(audit, audit_path)

    print(f"Total prepared rows: {total_rows:,}")
    print(f"Records for {args.year}: {year_rows:,}")
    print(f"Applications analyzed: {len(df):,}")
    print(f"Observed tracts: {len(tract):,}")
    print(f"Matched tracts: {len(mapped):,}")
    print(f"Unmatched tracts: {len(unmatched):,}")
    print(f"Tracts colored: {int(eligible.sum()):,}")
    print(f"HTML size: {map_info['html_size_mb']:.2f} MB")
    print(f"Open in your browser: {html_path.resolve()}")


def main():
    p = argparse.ArgumentParser(description=__doc__)

    p.add_argument(
        "--input",
        type=Path,
        default=config.OUT / "s04_prepare/prepared.parquet",
    )
    p.add_argument(
        "--boundaries",
        type=Path,
        default=ROOT / "data/geo/cb_2023_us_tract_500k.shp",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=config.OUT / "s11_geospatial",
    )
    p.add_argument(
        "--year",
        type=int,
        default=2023,
    )
    p.add_argument(
        "--min-applications",
        type=int,
        default=1,
    )
    p.add_argument(
        "--simplify-meters",
        type=float,
        default=100,
        help="Boundary simplification for display; 0 disables it",
    )
    p.add_argument(
        "--states",
        nargs="+",
        help="Optional state FIPS codes, e.g. 06 for California",
    )

    run(p.parse_args())


if __name__ == "__main__":
    main()
