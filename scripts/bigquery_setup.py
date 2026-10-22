"""Create the NYC taxi warehouse the BigQuery workspace and golden set use.

    python scripts/bigquery_setup.py --project <gcp-project>

The public `tlc_yellow_trips_2022` table is not partitioned or clustered,
so every query scans each referenced column in full and "filter on the
partition column" saves nothing. A team would not store a 36M-row fact
table that way, so this copies 2022 into the project as

  nyc_taxi.trips_2022   PARTITION BY DATE(pickup_datetime)
                        CLUSTER BY pickup_location_id
  nyc_taxi.zones        zone_id, zone_name, borough (260 zones, no geometry)

One-time cost: the copy scans ~7 GiB (~$0.04 on demand, inside the free
tier); storage is ~7 GiB (~$0.15/month). Idempotent: existing tables are
left alone unless named in --replace (e.g. `--replace zones`).
"""

from __future__ import annotations

import argparse
import sys

SOURCE = "bigquery-public-data.new_york_taxi_trips"

TRIPS_SQL = """
CREATE {replace} TABLE `{project}.nyc_taxi.trips_2022`
PARTITION BY DATE(pickup_datetime)
CLUSTER BY pickup_location_id
OPTIONS (description = "{trips_description}")
AS
SELECT vendor_id, pickup_datetime, dropoff_datetime, passenger_count, trip_distance,
       rate_code, store_and_fwd_flag, payment_type, fare_amount, extra, mta_tax,
       tip_amount, tolls_amount, imp_surcharge, airport_fee, total_amount,
       pickup_location_id, dropoff_location_id
FROM `{source}.tlc_yellow_trips_2022`
WHERE pickup_datetime >= '2022-01-01' AND pickup_datetime < '2023-01-01'
"""

ZONES_SQL = """
CREATE {replace} TABLE `{project}.nyc_taxi.zones`
OPTIONS (description = "NYC taxi zones: join trips.pickup_location_id / dropoff_location_id to zone_id.")
AS
-- The geometry table repeats zones 56 and 103 (multi-polygon rows); without
-- DISTINCT, joins would double- and triple-count their trips.
SELECT DISTINCT zone_id, zone_name, borough FROM `{source}.taxi_zone_geom`
"""

# The source has January-November only (December: 57 stray rows on Dec 1),
# which the Coder should know before it promises a December answer.
TRIPS_DESCRIPTION = ("NYC yellow taxi trips, one row per trip, pickups from 2022-01-01 to 2022-11-30 "
                     "(the source has no December data). Copied from " + SOURCE + ".tlc_yellow_trips_2022.")

# Column descriptions the Coder sees (BigQueryEngine reads them from table metadata).
DESCRIPTIONS = {
    "trips_2022": {
        "rate_code": "stored as text: '1.0' = standard, '2.0' = JFK, '3.0' = Newark, '4.0' = Nassau/Westchester, '5.0' = negotiated, '6.0' = group ride, '99.0' = unknown",
        "payment_type": "stored as text: '1' = credit card, '2' = cash, '3' = no charge, '4' = dispute, '5' = unknown, '0' = not recorded",
        "vendor_id": "1 = Creative Mobile Technologies, 2 = VeriFone",
        "trip_distance": "miles",
        "pickup_location_id": "taxi zone id; join to zones.zone_id",
        "dropoff_location_id": "taxi zone id; join to zones.zone_id",
        "total_amount": "total charged to passengers, USD (excludes cash tips)",
        "tip_amount": "USD; recorded for credit card payments only",
    },
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project", required=True)
    parser.add_argument("--location", default="US")
    parser.add_argument("--replace", nargs="*", default=[], choices=["trips_2022", "zones"],
                        help="tables to rebuild")
    args = parser.parse_args(argv)

    from google.cloud import bigquery
    client = bigquery.Client(project=args.project)
    ds = bigquery.Dataset(f"{args.project}.nyc_taxi")
    ds.location = args.location
    ds.description = "NYC TLC yellow taxi trips (2022), partitioned copy for DataAgent."
    client.create_dataset(ds, exists_ok=True)

    existing = {t.table_id for t in client.list_tables(ds)}
    for name, sql in (("trips_2022", TRIPS_SQL), ("zones", ZONES_SQL)):
        replace = "OR REPLACE" if name in args.replace else ""
        if name in existing and not replace:
            print(f"exists  nyc_taxi.{name}")
            continue
        job = client.query(sql.format(replace=replace, project=args.project, source=SOURCE,
                                      trips_description=TRIPS_DESCRIPTION))
        job.result()
        print(f"created nyc_taxi.{name}  (scanned {job.total_bytes_processed / 2**30:.2f} GiB)")

    trips = client.get_table(f"{args.project}.nyc_taxi.trips_2022")
    trips.description = TRIPS_DESCRIPTION
    client.update_table(trips, ["description"])

    for name, notes in DESCRIPTIONS.items():
        table = client.get_table(f"{args.project}.nyc_taxi.{name}")
        table.schema = [
            bigquery.SchemaField(f.name, f.field_type, mode=f.mode, description=notes.get(f.name, f.description))
            for f in table.schema
        ]
        client.update_table(table, ["schema"])

    for name in ("trips_2022", "zones"):
        t = client.get_table(f"{args.project}.nyc_taxi.{name}")
        print(f"{name}: {t.num_rows:,} rows, {t.num_bytes / 2**30:.2f} GiB, "
              f"partitioned on {getattr(t.time_partitioning, 'field', None)}, clustered by {t.clustering_fields}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
