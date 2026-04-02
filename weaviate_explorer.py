import argparse
import json
import os
from collections import Counter

import pandas as pd
import weaviate


RCA_FIELDS = [
    "anomaly_id",
    "stage",
    "timestamp",
    "phase",
    "guilty_feature",
    "guilty_feature_score",
    "anomalous_sensors",
    "anomalous_sensor_scores",
    "root_causes",
    "propagation_paths",
    "confidence",
]


def get_weaviate_client(url: str):
    client = weaviate.Client(url)
    if not client.is_ready():
        raise RuntimeError(f"Weaviate not ready at {url}")
    return client


def safe_json_loads(value, default):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return parsed
        except json.JSONDecodeError:
            return default
    return default


def get_available_query_fields(client):
    schema = client.schema.get()
    classes = schema.get("classes", [])
    rca_class = next((c for c in classes if c.get("class") == "RCAResult"), None)
    if not rca_class:
        return []

    existing_properties = {p.get("name") for p in rca_class.get("properties", [])}
    return [field for field in RCA_FIELDS if field in existing_properties]


def build_where_filter(stage=None, root_cause=None, timestamp=None):
    conditions = []

    if stage:
        conditions.append(
            {
                "path": ["stage"],
                "operator": "Equal",
                "valueText": stage,
            }
        )

    if root_cause:
        conditions.append(
            {
                "path": ["root_causes"],
                "operator": "Like",
                "valueString": f"*{root_cause}*",
            }
        )

    if timestamp:
        conditions.append(
            {
                "path": ["timestamp"],
                "operator": "Equal",
                "valueText": timestamp,
            }
        )

    if not conditions:
        return None

    if len(conditions) == 1:
        return conditions[0]

    return {
        "operator": "And",
        "operands": conditions,
    }


def parse_rca_row(row):
    anomalous_sensors = safe_json_loads(row.get("anomalous_sensors"), [])
    anomalous_sensor_scores = safe_json_loads(row.get("anomalous_sensor_scores"), {})
    root_causes = safe_json_loads(row.get("root_causes"), [])
    propagation_paths = safe_json_loads(row.get("propagation_paths"), [])

    return {
        "anomaly_id": row.get("anomaly_id"),
        "stage": row.get("stage"),
        "timestamp": row.get("timestamp"),
        "phase": row.get("phase"),
        "guilty_feature": row.get("guilty_feature"),
        "guilty_feature_score": row.get("guilty_feature_score"),
        "anomalous_sensors": anomalous_sensors,
        "anomalous_sensor_scores": anomalous_sensor_scores,
        "root_causes": root_causes,
        "propagation_paths": propagation_paths,
        "confidence": row.get("confidence"),
    }


def query_rca_results(client, query_fields, stage=None, root_cause=None, timestamp=None, limit=10, offset=0):
    if not query_fields:
        return []

    where_filter = build_where_filter(stage=stage, root_cause=root_cause, timestamp=timestamp)
    query = client.query.get("RCAResult", query_fields).with_limit(limit).with_offset(offset)

    if where_filter is not None:
        query = query.with_where(where_filter)

    response = query.do()
    rows = response.get("data", {}).get("Get", {}).get("RCAResult", [])
    if rows is None:
        return []
    if not isinstance(rows, list):
        return []
    return [parse_rca_row(row) for row in rows]


def export_rca_results_to_csv(client, query_fields, output_csv, stage=None, root_cause=None, timestamp=None, page_size=1000):
    offset = 0
    all_rows = []

    while True:
        batch = query_rca_results(
            client,
            query_fields=query_fields,
            stage=stage,
            root_cause=root_cause,
            timestamp=timestamp,
            limit=page_size,
            offset=offset,
        )
        if not batch:
            break
        all_rows.extend(batch)
        offset += page_size

    df = pd.DataFrame(all_rows)
    if not df.empty:
        for col in ["anomalous_sensors", "anomalous_sensor_scores", "root_causes", "propagation_paths"]:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    return len(df)


def export_all_rca_results_to_csv_cursor(client, output_csv, page_size=1000):
    after = None
    all_rows = []

    while True:
        response = client.data_object.get(
            class_name="RCAResult",
            limit=page_size,
            after=after,
        )
        objects = (response or {}).get("objects", [])
        if not objects:
            break

        for obj in objects:
            props = obj.get("properties", {}) or {}
            parsed = parse_rca_row(props)
            parsed["uuid"] = obj.get("id")
            all_rows.append(parsed)

        after = objects[-1].get("id")

    df = pd.DataFrame(all_rows)
    if not df.empty:
        for col in ["anomalous_sensors", "anomalous_sensor_scores", "root_causes", "propagation_paths"]:
            if col in df.columns:
                df[col] = df[col].apply(json.dumps)

    os.makedirs(os.path.dirname(output_csv) or ".", exist_ok=True)
    df.to_csv(output_csv, index=False)
    return len(df)


def print_one_result(item, index):
    print(f"\n--- RCA Result #{index} ---")
    print(f"anomaly_id: {item.get('anomaly_id')}")
    print(f"stage: {item.get('stage')}")
    print(f"timestamp: {item.get('timestamp')}")
    print(f"phase: {item.get('phase')}")
    print(f"guilty_feature: {item.get('guilty_feature')}")
    print(f"guilty_feature_score: {item.get('guilty_feature_score')}")
    print(f"confidence: {item.get('confidence')}")

    print("anomalous_sensors:")
    sensors = item.get("anomalous_sensors", [])
    sensor_scores = item.get("anomalous_sensor_scores", {})
    if sensors:
        for sensor in sensors:
            score = sensor_scores.get(sensor)
            if score is None:
                print(f"  - {sensor}")
            else:
                print(f"  - {sensor}: {score}")
    else:
        print("  (none)")

    print("root_cause_candidates (ranked):")
    root_causes = item.get("root_causes", [])
    if root_causes:
        for rank, root in enumerate(root_causes, start=1):
            if isinstance(root, list) and len(root) >= 2:
                print(f"  {rank}. {root[0]} (score={root[1]})")
            else:
                print(f"  {rank}. {root}")
    else:
        print("  (none)")

    print("propagation_paths:")
    paths = item.get("propagation_paths", [])
    if paths:
        for path in paths:
            if isinstance(path, list):
                print("  - " + " -> ".join(path))
            else:
                print(f"  - {path}")
    else:
        print("  (none)")


def print_results(items, start_index=1):
    if not items:
        print("No RCA results matched the filters.")
        return

    stage_counts = Counter(item.get("stage") for item in items)
    print("\nBatch stage distribution:")
    for stage, count in sorted(stage_counts.items()):
        print(f"  {stage}: {count}")

    for idx, item in enumerate(items, start=start_index):
        print_one_result(item, idx)


def interactive_explorer(client, query_fields, stage=None, root_cause=None, timestamp=None, limit=5):
    current_stage = stage
    current_root = root_cause
    current_timestamp = timestamp
    current_limit = limit
    current_offset = 0

    print("\nInteractive Weaviate RCA Explorer")
    print("Commands: show, next, prev, stage <P1..P6>, root <sensor>, timestamp <yyyy-mm-dd HH:MM:SS>, clear, limit <n>, quit")

    while True:
        command = input("\nexplorer> ").strip()
        if not command:
            command = "show"

        if command in {"quit", "exit", "q"}:
            print("Exiting explorer.")
            break

        if command.startswith("stage "):
            current_stage = command.split(" ", 1)[1].strip() or None
            current_offset = 0
            print(f"Stage filter set to: {current_stage}")
            continue

        if command.startswith("root "):
            current_root = command.split(" ", 1)[1].strip() or None
            current_offset = 0
            print(f"Root-cause filter set to: {current_root}")
            continue

        if command.startswith("timestamp "):
            current_timestamp = command.split(" ", 1)[1].strip() or None
            current_offset = 0
            print(f"Timestamp filter set to: {current_timestamp}")
            continue

        if command.startswith("limit "):
            try:
                current_limit = max(1, int(command.split(" ", 1)[1].strip()))
                current_offset = 0
                print(f"Limit set to: {current_limit}")
            except ValueError:
                print("Invalid limit. Use an integer.")
            continue

        if command == "clear":
            current_stage = None
            current_root = None
            current_timestamp = None
            current_offset = 0
            print("Filters cleared.")
            continue

        if command == "next":
            current_offset += current_limit
        elif command == "prev":
            current_offset = max(0, current_offset - current_limit)
        elif command != "show":
            print("Unknown command.")
            continue

        print(
            f"\nQuerying with stage={current_stage}, root={current_root}, timestamp={current_timestamp}, "
            f"limit={current_limit}, offset={current_offset}"
        )
        items = query_rca_results(
            client,
            query_fields=query_fields,
            stage=current_stage,
            root_cause=current_root,
            timestamp=current_timestamp,
            limit=current_limit,
            offset=current_offset,
        )
        print_results(items, start_index=current_offset + 1)


def main():
    parser = argparse.ArgumentParser(description="Interactive explorer for Weaviate RCAResult objects")
    parser.add_argument("--url", default=os.getenv("WEAVIATE_URL", "http://localhost:8080"), help="Weaviate URL")
    parser.add_argument("--stage", default=None, help="Filter by stage (e.g., P3)")
    parser.add_argument("--root-cause", default=None, help="Filter by root-cause sensor name")
    parser.add_argument("--sensor", default=None, help="Alias for --root-cause")
    parser.add_argument("--timestamp", default=None, help="Filter by exact timestamp string")
    parser.add_argument("--limit", type=int, default=5, help="Number of records per query")
    parser.add_argument("--offset", type=int, default=0, help="Offset for non-interactive query")
    parser.add_argument("--no-interactive", action="store_true", help="Run one query and exit")
    parser.add_argument("--export-csv", default=None, help="Export matching RCA results to CSV and exit")
    parser.add_argument("--page-size", type=int, default=1000, help="Pagination size for --export-csv")

    args = parser.parse_args()
    client = get_weaviate_client(args.url)
    query_fields = get_available_query_fields(client)

    root_cause_filter = args.root_cause or args.sensor

    if not query_fields:
        raise RuntimeError("RCAResult class not found or has no queryable properties.")

    aggregate = client.query.aggregate("RCAResult").with_meta_count().do()
    total_count = aggregate.get("data", {}).get("Aggregate", {}).get("RCAResult", [{}])[0].get("meta", {}).get("count", 0)
    print(f"Connected to {args.url}")
    print(f"Total RCAResult objects: {total_count}")
    print(f"Query fields: {query_fields}")

    if args.export_csv:
        has_filters = any([args.stage, root_cause_filter, args.timestamp])
        if has_filters:
            exported_rows = export_rca_results_to_csv(
                client,
                query_fields=query_fields,
                output_csv=args.export_csv,
                stage=args.stage,
                root_cause=root_cause_filter,
                timestamp=args.timestamp,
                page_size=max(1, args.page_size),
            )
        else:
            exported_rows = export_all_rca_results_to_csv_cursor(
                client,
                output_csv=args.export_csv,
                page_size=max(1, args.page_size),
            )
        print(f"Exported {exported_rows} rows to {args.export_csv}")
        return

    if args.no_interactive:
        items = query_rca_results(
            client,
            query_fields=query_fields,
            stage=args.stage,
            root_cause=root_cause_filter,
            timestamp=args.timestamp,
            limit=args.limit,
            offset=args.offset,
        )
        print_results(items, start_index=args.offset + 1)
        return

    interactive_explorer(
        client,
        query_fields=query_fields,
        stage=args.stage,
        root_cause=root_cause_filter,
        timestamp=args.timestamp,
        limit=args.limit,
    )


if __name__ == "__main__":
    main()
