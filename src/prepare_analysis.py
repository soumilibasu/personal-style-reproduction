import json
from pathlib import Path
from collections import defaultdict


INPUT_FILE = Path("data/processed/task_records.jsonl")
OUTPUT_FILE = Path("data/processed/analysis_ready.jsonl")


def main():

    if not INPUT_FILE.exists():
        print(f"ERROR: File not found: {INPUT_FILE}")
        return

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if line:
                records.append(json.loads(line))

    print("=" * 50)
    print("PREPARING ANALYSIS DATASET")
    print("=" * 50)

    analysis_records = []

    for record in records:

        condition = record["model_generation_shown"]

        if condition == 0:
            condition_name = "control"
        elif condition == 1:
            condition_name = "treatment"
        else:
            raise ValueError(
                f"Unexpected condition: {condition}"
            )

        analysis_record = {
            "participant_id": record["participant_id"],
            "task_key": record["task_key"],
            "scenario": record["scenario"],

            "condition": condition,
            "condition_name": condition_name,

            "model_generation_shown":
                record["model_generation_shown"],

            "details": record["details"],

            "model_generation":
                record["model_generation"],

            "final_version":
                record["final_version"],
        }

        analysis_records.append(analysis_record)

    # --------------------------------------------------
    # Save analysis-ready JSONL
    # --------------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open("w", encoding="utf-8") as f:

        for record in analysis_records:
            f.write(
                json.dumps(
                    record,
                    ensure_ascii=False
                ) + "\n"
            )

    # --------------------------------------------------
    # Summary
    # --------------------------------------------------

    participants = defaultdict(list)

    for record in analysis_records:
        participants[
            record["participant_id"]
        ].append(record)

    control_count = sum(
        r["condition"] == 0
        for r in analysis_records
    )

    treatment_count = sum(
        r["condition"] == 1
        for r in analysis_records
    )

    print("\n--- Dataset summary ---")
    print(f"Total records: {len(analysis_records)}")
    print(f"Participants: {len(participants)}")
    print(f"Control records: {control_count}")
    print(f"Treatment records: {treatment_count}")

    print("\n--- Per-participant check ---")

    problems = []

    for participant_id, tasks in participants.items():

        control = sum(
            task["condition"] == 0
            for task in tasks
        )

        treatment = sum(
            task["condition"] == 1
            for task in tasks
        )

        if len(tasks) != 6 or control != 2 or treatment != 4:
            problems.append(
                (
                    participant_id,
                    len(tasks),
                    control,
                    treatment
                )
            )

    if problems:
        print(
            f"Problems found: {len(problems)}"
        )

        for problem in problems:
            print(problem)
    else:
        print(
            "All participants have "
            "6 tasks = 2 control + 4 treatment."
        )

    print("\n--- Output ---")
    print(f"Saved to: {OUTPUT_FILE}")

    print("\n" + "=" * 50)
    print("ANALYSIS DATASET READY")
    print("=" * 50)


if __name__ == "__main__":
    main()