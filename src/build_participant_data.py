import json
from pathlib import Path
from collections import defaultdict


INPUT_FILE = Path("data/processed/analysis_ready.jsonl")
OUTPUT_FILE = Path("data/processed/participant_data.json")


def main():

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    participants = defaultdict(
        lambda: {
            "control": [],
            "treatment": []
        }
    )

    # ---------------------------------------------
    # Organize tasks by participant and condition
    # ---------------------------------------------

    for record in records:

        participant_id = record["participant_id"]

        task = {
            "task_key": record["task_key"],
            "scenario": record["scenario"],
            "details": record["details"],
            "model_generation": record["model_generation"],
            "final_version": record["final_version"]
        }

        if record["condition"] == 0:
            participants[participant_id]["control"].append(task)

        elif record["condition"] == 1:
            participants[participant_id]["treatment"].append(task)

    # ---------------------------------------------
    # Sort tasks by task key
    # ---------------------------------------------

    for participant_id in participants:

        participants[participant_id]["control"].sort(
            key=lambda x: int(x["task_key"])
        )

        participants[participant_id]["treatment"].sort(
            key=lambda x: int(x["task_key"])
        )

    # ---------------------------------------------
    # Build participant-level representation
    # ---------------------------------------------

    output = {}

    for participant_id, data in participants.items():

        control_texts = [
            task["final_version"]
            for task in data["control"]
        ]

        treatment_llm_texts = [
            task["model_generation"]
            for task in data["treatment"]
        ]

        treatment_postedit_texts = [
            task["final_version"]
            for task in data["treatment"]
        ]

        output[participant_id] = {
            "control": data["control"],
            "treatment": data["treatment"],

            "control_texts": control_texts,

            "treatment_llm_texts":
                treatment_llm_texts,

            "treatment_postedit_texts":
                treatment_postedit_texts,

            "control_concatenated":
                "\n\n".join(control_texts)
        }

    # ---------------------------------------------
    # Save
    # ---------------------------------------------

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            output,
            f,
            ensure_ascii=False,
            indent=2
        )

    # ---------------------------------------------
    # Validation
    # ---------------------------------------------

    print("=" * 60)
    print("PARTICIPANT-LEVEL DATA")
    print("=" * 60)

    print(f"Participants: {len(output)}")

    problems = []

    for participant_id, data in output.items():

        controls = len(data["control"])
        treatments = len(data["treatment"])

        if controls != 2 or treatments != 4:
            problems.append(
                (
                    participant_id,
                    controls,
                    treatments
                )
            )

    if problems:
        print(
            f"Participants with problems: {len(problems)}"
        )

        for problem in problems:
            print(problem)

    else:
        print(
            "All participants have "
            "2 control + 4 treatment tasks."
        )

    # ---------------------------------------------
    # Show one participant
    # ---------------------------------------------

    first_id = next(iter(output))

    first = output[first_id]

    print("\n--- Example participant ---")
    print(f"Participant: {first_id}")

    print("\nControl scenarios:")

    for task in first["control"]:
        print(
            f"  {task['scenario']} "
            f"(task {task['task_key']})"
        )

    print("\nTreatment scenarios:")

    for task in first["treatment"]:
        print(
            f"  {task['scenario']} "
            f"(task {task['task_key']})"
        )

    print("\nControl concatenated text length:",
          len(first["control_concatenated"]))

    print("\n" + "=" * 60)
    print("PARTICIPANT DATA READY")
    print("=" * 60)


if __name__ == "__main__":
    main()