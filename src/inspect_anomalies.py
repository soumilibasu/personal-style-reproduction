
import json
from pathlib import Path

RAW_DIR = Path("data/raw")

for file_path in sorted(RAW_DIR.rglob("*.json")):
    with file_path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    user_info = data.get("user_info", {})
    participant_id = user_info.get("id", file_path.stem)
    responses = data.get("responses", {})

    if not isinstance(responses, dict):
        print(f"\nInvalid responses structure: {file_path}")
        continue

    if len(responses) != 6:
        print("\n--- Participant with unusual task count ---")
        print("File:", file_path.name)
        print("Participant ID:", participant_id)
        print("Number of tasks:", len(responses))
        print("Task keys:", list(responses.keys()))

    for task_key, task in responses.items():
        if not isinstance(task, dict):
            print("\n--- Unexpected task structure ---")
            print("File:", file_path.name)
            print("Participant ID:", participant_id)
            print("Task key:", task_key)
            print("Task type:", type(task).__name__)
            continue

        required_fields = [
            "scenario",
            "model_generation_shown",
            "model_generation",
            "final_version",
        ]

        missing = [
            field for field in required_fields
            if field not in task or task[field] is None
        ]

        if missing:
            print("\n--- Task with missing fields ---")
            print("File:", file_path.name)
            print("Participant ID:", participant_id)
            print("Task key:", task_key)
            print("Missing fields:", missing)
            print("Available fields:", list(task.keys()))
            print("Scenario:", task.get("scenario"))
            print("Shown:", task.get("model_generation_shown"))
            print("Model generation type:",
                  type(task.get("model_generation")).__name__)
            print("Final version type:",
                  type(task.get("final_version")).__name__)

    if isinstance(user_info, dict) and (
        len(responses) != 6
        or any(
            isinstance(task, dict) and any(
                field not in task or task[field] is None
                for field in [
                    "scenario",
                    "model_generation_shown",
                    "model_generation",
                    "final_version",
                ]
            )
            for task in responses.values()
        )
    ):
        print("\nUser info scenarios:", user_info.get("scenarios"))
        print("User info conditions:", user_info.get("conditions"))