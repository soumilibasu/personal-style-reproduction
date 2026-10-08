
import json
from pathlib import Path
from collections import Counter

RAW_DIR = Path("data/raw")


def load_json_file(file_path):
    """Load one JSON file and return its contents."""
    with file_path.open("r", encoding="utf-8") as file:
        return json.load(file)


def describe_value(value):
    """Return a short description of a JSON value."""
    if isinstance(value, dict):
        return f"dict with {len(value)} keys"
    if isinstance(value, list):
        return f"list with {len(value)} items"
    if isinstance(value, str):
        return f"string with {len(value)} characters"
    return type(value).__name__


def main():
    json_files = sorted(RAW_DIR.rglob("*.json"))

    print(f"Raw data directory: {RAW_DIR.resolve()}")
    print(f"JSON files found: {len(json_files)}")

    if not json_files:
        print("No .json files found. Check the folder and file extensions.")
        return

    file_sizes = [file.stat().st_size for file in json_files]
    print(f"Smallest file: {min(file_sizes)} bytes")
    print(f"Largest file: {max(file_sizes)} bytes")

    valid_files = 0
    invalid_files = []
    top_level_keys = Counter()
    participant_ids = set()
    task_counts = Counter()
    scenario_counts = Counter()
    shown_counts = Counter()
    missing_fields = Counter()

    for file_path in json_files:
        try:
            data = load_json_file(file_path)
            valid_files += 1
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as error:
            invalid_files.append((str(file_path), str(error)))
            continue

        if not isinstance(data, dict):
            print(f"Unexpected top-level type in {file_path}: {type(data).__name__}")
            continue

        top_level_keys.update(data.keys())

        user_info = data.get("user_info", {})
        if isinstance(user_info, dict):
            participant_id = user_info.get("id")
            if participant_id is not None:
                participant_ids.add(str(participant_id))

        responses = data.get("responses", {})
        if not isinstance(responses, dict):
            missing_fields["responses_not_dict"] += 1
            continue

        task_counts[len(responses)] += 1

        for response_key, task in responses.items():
            if not isinstance(task, dict):
                missing_fields["task_not_dict"] += 1
                continue

            scenario = task.get("scenario")
            if scenario is not None:
                scenario_counts[str(scenario)] += 1

            shown = task.get("model_generation_shown")
            shown_counts[str(shown)] += 1

            for field in ("scenario", "model_generation_shown",
                          "model_generation", "final_version"):
                if field not in task or task[field] is None:
                    missing_fields[field] += 1

    print("\n--- File validation ---")
    print(f"Valid JSON files: {valid_files}")
    print(f"Invalid JSON files: {len(invalid_files)}")

    for path, error in invalid_files:
        print(f"INVALID: {path} | {error}")

    print("\n--- Top-level keys ---")
    for key, count in top_level_keys.most_common():
        print(f"{key}: {count}")

    print("\n--- Participant IDs ---")
    print(f"Unique IDs found: {len(participant_ids)}")

    print("\n--- Tasks per participant ---")
    for count, participants in sorted(task_counts.items()):
        print(f"{count} tasks: {participants} files")

    print("\n--- Scenarios ---")
    for scenario, count in scenario_counts.most_common():
        print(f"{scenario}: {count} task records")

    print("\n--- model_generation_shown values ---")
    for value, count in shown_counts.most_common():
        print(f"{value}: {count} task records")

    print("\n--- Missing or unexpected fields ---")
    if missing_fields:
        for field, count in missing_fields.most_common():
            print(f"{field}: {count}")
    else:
        print("No missing fields detected by these checks.")

    if json_files:
        print("\n--- Example file structure ---")
        example = next(
            (load_json_file(path) for path in json_files
             if path not in [Path(p) for p, _ in invalid_files]),
            None
        )

        if isinstance(example, dict):
            for key, value in example.items():
                print(f"{key}: {describe_value(value)}")

            responses = example.get("responses", {})
            if isinstance(responses, dict) and responses:
                first_task_key = next(iter(responses))
                first_task = responses[first_task_key]
                print(f"\nExample response key: {first_task_key}")
                if isinstance(first_task, dict):
                    for key, value in first_task.items():
                        print(f"  {key}: {describe_value(value)}")


if __name__ == "__main__":
    main()