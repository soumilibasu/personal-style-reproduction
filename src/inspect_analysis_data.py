import json
from pathlib import Path


INPUT_FILE = Path("data/processed/analysis_ready.jsonl")


def main():

    records = []

    with INPUT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))

    print("=" * 60)
    print("ANALYSIS DATA INSPECTION")
    print("=" * 60)

    # Show one control and one treatment example
    control = next(
        r for r in records
        if r["condition"] == 0
    )

    treatment = next(
        r for r in records
        if r["condition"] == 1
    )

    print("\n--- CONTROL EXAMPLE ---")
    print(f"Participant: {control['participant_id']}")
    print(f"Scenario: {control['scenario']}")
    print(f"Condition: {control['condition_name']}")

    print("\nDetails:")
    print(control["details"][:300])

    print("\nModel generation:")
    print(control["model_generation"][:300])

    print("\nFinal version:")
    print(control["final_version"][:500])

    print("\n\n--- TREATMENT EXAMPLE ---")
    print(f"Participant: {treatment['participant_id']}")
    print(f"Scenario: {treatment['scenario']}")
    print(f"Condition: {treatment['condition_name']}")

    print("\nDetails:")
    print(treatment["details"][:300])

    print("\nModel generation:")
    print(treatment["model_generation"][:500])

    print("\nFinal version:")
    print(treatment["final_version"][:500])

    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()