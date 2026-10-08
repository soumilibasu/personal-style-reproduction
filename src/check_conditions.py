import json
from pathlib import Path
from collections import Counter, defaultdict


INPUT_FILE = Path("data/processed/task_records.jsonl")


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
    print("CONDITION VALIDATION")
    print("=" * 50)

    print(f"Total task records: {len(records)}")

    # --------------------------------------------------
    # 1. Compare expected_condition with model_generation_shown
    # --------------------------------------------------

    mismatches = []

    for record in records:
        expected = record.get("expected_condition")
        shown = record.get("model_generation_shown")

        if expected != shown:
            mismatches.append(record)

    print(f"\nCondition mismatches: {len(mismatches)}")

    if mismatches:
        print("\n--- MISMATCHES ---")

        for record in mismatches:
            print(
                f"Participant: {record['participant_id']} | "
                f"Task: {record['task_key']} | "
                f"Scenario: {record['scenario']} | "
                f"Expected: {record['expected_condition']} | "
                f"Shown: {record['model_generation_shown']}"
            )
    else:
        print("All records have matching expected and observed conditions.")

    # --------------------------------------------------
    # 2. Check condition distribution
    # --------------------------------------------------

    print("\n--- Overall condition distribution ---")

    condition_counts = Counter(
        record["model_generation_shown"]
        for record in records
    )

    for condition, count in sorted(condition_counts.items()):
        print(f"Condition {condition}: {count} tasks")

    # --------------------------------------------------
    # 3. Check each participant
    # --------------------------------------------------

    participants = defaultdict(list)

    for record in records:
        participants[record["participant_id"]].append(record)

    print("\n--- Per-participant validation ---")

    participant_problems = []

    for participant_id, tasks in participants.items():

        counts = Counter(
            task["model_generation_shown"]
            for task in tasks
        )

        # Expected design: 2 control + 4 treatment
        if counts.get(0, 0) != 2 or counts.get(1, 0) != 4:
            participant_problems.append(
                {
                    "participant_id": participant_id,
                    "total_tasks": len(tasks),
                    "condition_counts": dict(counts),
                }
            )

    print(f"Participants checked: {len(participants)}")

    if participant_problems:
        print(
            f"Participants with unexpected condition distribution: "
            f"{len(participant_problems)}"
        )

        for problem in participant_problems:
            print(problem)
    else:
        print(
            "All participants have the expected "
            "2 control + 4 treatment tasks."
        )

    # --------------------------------------------------
    # 4. Check scenarios
    # --------------------------------------------------

    print("\n--- Scenario validation ---")

    scenario_problems = []

    for participant_id, tasks in participants.items():

        scenarios = [task["scenario"] for task in tasks]

        if len(scenarios) != len(set(scenarios)):
            scenario_problems.append(
                {
                    "participant_id": participant_id,
                    "scenarios": scenarios,
                }
            )

    if scenario_problems:
        print(
            f"Participants with duplicate scenarios: "
            f"{len(scenario_problems)}"
        )

        for problem in scenario_problems:
            print(problem)
    else:
        print("No participant has duplicate scenarios.")

    # --------------------------------------------------
    # 5. Show examples
    # --------------------------------------------------

    print("\n--- Example participants ---")

    shown_participants = list(participants.items())[:5]

    for participant_id, tasks in shown_participants:

        print(f"\nParticipant: {participant_id}")

        for task in sorted(tasks, key=lambda x: int(x["task_key"])):

            print(
                f"  Task {task['task_key']}: "
                f"{task['scenario']} | "
                f"shown={task['model_generation_shown']} | "
                f"expected={task['expected_condition']}"
            )

    print("\n" + "=" * 50)
    print("VALIDATION COMPLETE")
    print("=" * 50)


if __name__ == "__main__":
    main()