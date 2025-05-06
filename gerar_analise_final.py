from pathlib import Path
import json
import pandas as pd
from collections import Counter
from tqdm import tqdm
from scipy.stats import entropy


def extract_crime_category(json_path):
    parts = Path(json_path).parts
    for i, part in enumerate(parts):
        if "resultados" in part.lower() and i + 1 < len(parts):
            return parts[i + 1].split("_")[0]
    return "Unknown"


def get_person_demographics(person_data):
    faces = person_data.get("faces_detected", [])
    if not faces:
        return None

    races, genders, ages, age_ranges = [], [], [], []

    for face in faces:
        demo = face.get("demographics", {})
        if not demo:
            continue
        races.append(demo.get("dominant_race"))
        genders.append(demo.get("gender"))
        ages.append(demo.get("age"))
        age_ranges.append(demo.get("age_range"))

    if not races:
        return None

    return {
        "race": Counter(races).most_common(1)[0][0],
        "gender": Counter(genders).most_common(1)[0][0],
        "age": round(sum(ages) / len(ages)),
        "age_range": Counter(age_ranges).most_common(1)[0][0]
    }


def process_all_analysis_files(base_dir="resultados"):
    base_path = Path(base_dir)
    all_json_files = list(base_path.rglob("analysis/*.json"))
    resultados = []

    for json_path in tqdm(all_json_files, desc="Processando arquivos"):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        persons = data.get("persons", {})
        summary_rows = []

        for person_data in persons.values():
            person_demo = get_person_demographics(person_data)
            if person_demo:
                summary_rows.append(person_demo)

        if summary_rows:
            df = pd.DataFrame(summary_rows)
            category = extract_crime_category(json_path)

            race_counts = df["race"].value_counts().to_dict()
            gender_counts = df["gender"].value_counts().to_dict()
            age_counts = df["age_range"].value_counts().to_dict()
            race_entropy = entropy(list(race_counts.values())) if len(race_counts) > 1 else 0.0

            resultados.append({
                "arquivo": str(json_path),
                "crime_category": category,
                "total_pessoas": len(persons),
                "total_faces": len(df),
                "raça_dominante": df["race"].mode().iloc[0] if not df["race"].mode().empty else None,
                "gênero_dominante": df["gender"].mode().iloc[0] if not df["gender"].mode().empty else None,
                "faixa_etária_dominante": df["age_range"].mode().iloc[0] if not df["age_range"].mode().empty else None,
                "entropia_racial": race_entropy,
                "distribuição_racial": race_counts,
                "distribuição_gênero": gender_counts,
                "distribuição_idade": age_counts,
            })

    return pd.DataFrame(resultados)


df = process_all_analysis_files("resultados")
df.to_csv("resumo_demografico.csv", index=False)