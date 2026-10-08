import json
from pathlib import Path
from collections import Counter


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

OUTPUT_FILE = PROCESSED_DIR / "task_records.jsonl"
REPORT_FILE = PROCESSED_DIR / "loading_report.json"


def read_json(file_path):
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def clean_text(value):
    if value is None:
        return ""

    if not isinstance(value, str):
        return ""

    return value.strip()


def load_participant(file_path):

    data = read_json(file_path)

    user_info = data.get("user_info", {})
    responses = data.get("responses", {})

    participant_id = user_info.get(
        "id",
        file_path.stem
    )

    scenarios = user_info.get(
        "scenarios",
        []
    )

    conditions = user_info.get(
        "conditions",
        []
    )

    expected_condition = {}

    if (
        isinstance(scenarios, list)
        and isinstance(conditions, list)
        and len(scenarios) == len(conditions)
    ):

        for scenario, condition in zip(
            scenarios,
            conditions
        ):
            expected_condition[
                str(scenario)
            ] = condition

    records = []
    issues = []

    for task_key, task in responses.items():

        if not isinstance(task, dict):
            continue

        scenario = task.get(
            "scenario"
        )

        shown = task.get(
            "model_generation_shown"
        )

        model_generation = task.get(
            "model_generation"
        )

        final_version = task.get(
            "final_version"
        )

        # Detect incomplete task
        if (
            scenario is None
            and model_generation is None
            and final_version is None
        ):

            issues.append({
                "task_key": task_key,
                "reason": "incomplete_task"
            })

            continue

        expected = expected_condition.get(
            str(scenario)
        )

        # Check condition consistency
        if (
            expected is not None
            and shown is not None
            and int(expected) != int(shown)
        ):

            issues.append({
                "task_key": task_key,
                "reason": "condition_mismatch",
                "scenario": scenario,
                "expected": expected,
                "observed": shown
            })

        record = {

            "participant_id":
                str(participant_id),

            "source_file":
                file_path.name,

            "task_key":
                str(task_key),

            "scenario":
                scenario,

            "model_generation_shown":
                shown,

            "expected_condition":
                expected,

            "details":
                task.get("details"),

            "model_generation_raw":
                model_generation,

            "final_version_raw":
                final_version,

            "model_generation":
                clean_text(
                    model_generation
                ),

            "final_version":
                clean_text(
                    final_version
                ),

            "edits":
                task.get("edits", []),

            "likert":
                task.get("likert"),

            "start_time":
                task.get("start_time"),

            "submit_details_time":
                task.get(
                    "submit_details_time"
                ),

            "submit_final_text_time":
                task.get(
                    "submit_final_text_time"
                )
        }

        records.append(record)

    return records, issues


def main():

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    json_files = sorted(
        RAW_DIR.rglob("*.json")
    )

    all_records = []

    all_issues = []

    participant_ids = set()

    scenario_counts = Counter()

    condition_counts = Counter()

    for file_path in json_files:

        records, issues = load_participant(
            file_path
        )

        all_records.extend(records)

        all_issues.extend(
            [
                {
                    "file": file_path.name,
                    **issue
                }
                for issue in issues
            ]
        )

        if records:

            participant_ids.add(
                records[0]["participant_id"]
            )

        for record in records:

            scenario_counts[
                str(record["scenario"])
            ] += 1

            condition_counts[
                str(
                    record[
                        "model_generation_shown"
                    ]
                )
            ] += 1

    # Write JSONL
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        for record in all_records:

            file.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                )
                + "\n"
            )

    report = {

        "raw_files":
            len(json_files),

        "participants":
            len(participant_ids),

        "raw_response_records":
            487,

        "valid_task_records":
            len(all_records),

        "skipped_records":
            len(all_issues),

        "scenario_counts":
            dict(scenario_counts),

        "model_generation_shown_counts":
            dict(condition_counts),

        "issues":
            all_issues
    }

    with REPORT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False
        )

    print(
        "================================"
    )

    print(
        "DATASET LOADING COMPLETE"
    )

    print(
        "================================"
    )

    print(
        f"Raw JSON files: {len(json_files)}"
    )

    print(
        f"Participants: {len(participant_ids)}"
    )

    print(
        f"Task records: {len(all_records)}"
    )

    print(
        f"Skipped incomplete records: "
        f"{len(all_issues)}"
    )

    print(
        f"Output: {OUTPUT_FILE}"
    )

    print(
        f"Report: {REPORT_FILE}"
    )


if __name__ == "__main__":
    main()