import requests
import io
import pandas as pd
import argparse
from pathlib import Path
from sklearn.model_selection import train_test_split

URL = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe-tertiaire/full"


def download_data(base_dir):
    output_dir = Path(base_dir) / "data"

    # Create directory if not exist
    output_dir.mkdir(parents=True, exist_ok=True)

    # Request URL
    response = requests.get(URL)
    data = pd.read_csv(io.StringIO(response.text), low_memory=False, index_col=0)

    # Remove target not in [A, B, C, D, E, F, G]
    data = data[
        (data["classe_consommation_energie"].isin(["A", "B", "C", "D", "E", "F", "G"]))
        & (data["classe_estimation_ges"].isin(["A", "B", "C", "D", "E", "F", "G"]))
    ]

    # Remove useless features
    data = data.drop(
        columns=[
            "numero_dpe",
            "usr_diagnostiqueur_id",
            "usr_logiciel_id",
            "tr001_modele_dpe_id",
            "nom_methode_dpe",
            "version_methode_dpe",
            "nom_methode_etude_thermique",
            "version_methode_etude_thermique",
            "adresse_organisme_certificateur",
            "date_visite_diagnostiqueur",
            "date_etablissement_dpe",
            "date_arrete_tarifs_energies",
            "commentaires_ameliorations_recommandations",
            "explication_personnalisee",
            "organisme_certificateur",
            "tr001_modele_dpe_code",
            "tr001_modele_dpe_type_id",
            "tr001_modele_dpe_modele",
            "tr001_modele_dpe_description",
            "tr001_modele_dpe_fichier_vierge",
            "tr001_modele_dpe_est_efface",
            "tr001_modele_dpe_type",
            "tr001_modele_dpe_type_libelle",
            "tr001_modele_dpe_type_ordre",
        ]
    )

    # Remove target as values
    data = data.drop(columns=["consommation_energie", "estimation_ges"], axis=0)

    # Remove ~40k rows of data without latitude and longitude
    data = data.dropna(subset=["latitude", "longitude"], how="any")

    # Remove the rows having weird construction date ~ 35k rows
    data = data[(data['annee_construction'] < 2025) & (data['annee_construction'] > 500) ]

    # Create X and Y
    Y = data[["classe_consommation_energie", "classe_estimation_ges"]]
    X = data.drop(
        columns=["classe_consommation_energie", "classe_estimation_ges"], axis=0
    )

    # Train test split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.3, random_state=42
    )

    # Export X and Y
    X_train.to_parquet(output_dir / "data_train.parquet")
    X_test.to_parquet(output_dir / "data_test.parquet")
    Y_train.to_csv(output_dir / "labels_train.csv")
    Y_test.to_csv(output_dir / "labels_test.csv")

    print("Downloaded data to {}/".format(output_dir))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=".", type=str)
    args = parser.parse_args()

    download_data(Path(args.output))
