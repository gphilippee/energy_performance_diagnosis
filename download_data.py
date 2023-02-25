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

    # Clean the data

    # Remove useless columns 
    # useless columns
    data = data.drop(columns=[
        "numero_dpe",
        "usr_diagnostiqueur_id",
        "usr_logiciel_id",
        "tr001_modele_dpe_id",
        "nom_methode_dpe",
        "version_methode_dpe",
        "nom_methode_etude_thermique",
        "version_methode_etude_thermique",
        "date_visite_diagnostiqueur",
        "date_etablissement_dpe",
        "date_arrete_tarifs_energies",
        "commentaires_ameliorations_recommandations",
        "explication_personnalisee",
        "commune",
        "arrondissement",
        "type_voie",
        "nom_rue",
        "numero_rue",
        "code_postal",
        "code_insee_commune",
        "code_insee_commune_actualise",
        "numero_lot",
        "nom_centre_commercial",
        "tr002_type_batiment_id",
        "secteur_activite",
        "tr012_categorie_erp_id",
        "tr013_type_erp_id",
        "tv016_departement_id",
        "batiment",
        "escalier",
        "etage",
        "porte",
        "quote_part",
        "portee_dpe_batiment",
        "partie_batiment",
        "en_souterrain",
        "en_surface",
        "nombre_circulations_verticales", # faible variance
        "nombre_boutiques", # faible variance"
        "presence_verriere",
        "type_vitrage_verriere", # faible variance
        "etat_avancement",
        "adresse_organisme_certificateur",
        "dpe_vierge", # faible variance
        "est_efface", # faible variance
        "date_reception_dpe", # date
        "geo_score",
        "geo_type",
        "geo_adresse",
        "geo_id",
        "geo_l4",
        "geo_l5",
        "organisme_certificateur",
        'tr001_modele_dpe_code', 'tr001_modele_dpe_type_id',
        'tr001_modele_dpe_modele', 'tr001_modele_dpe_description',
        'tr001_modele_dpe_fichier_vierge', 'tr001_modele_dpe_est_efface',
        'tr001_modele_dpe_type', 'tr001_modele_dpe_type_libelle',
        'tr001_modele_dpe_type_ordre', 'tr002_type_batiment_code',
        'tr002_type_batiment_description', 'tr002_type_batiment_libelle',
        'tr002_type_batiment_est_efface', 'tr002_type_batiment_ordre',
        'tr002_type_batiment_simulateur', 'tr012_categorie_erp_code',
        'tr012_categorie_erp_categorie', 'tr012_categorie_erp_groupe',
        'tr012_categorie_erp_est_efface', 'tr013_type_erp_code',
        'tr013_type_erp_type', 'tr013_type_erp_categorie_id',
        'tr013_type_erp_est_efface', 'tr013_type_erp_categorie',
        'tv016_departement_code', 'tv016_departement_departement',
        'tv017_zone_hiver_id', 'tv018_zone_ete_id', 'tv016_departement_altmin',
        'tv016_departement_altmax', 'tv016_departement_nref',
        'tv016_departement_dhref',
        'tv016_departement_pref', 'tv016_departement_c2',
        'tv016_departement_c3', 'tv016_departement_c4',
        'tv016_departement_t_ext_basse', 'tv016_departement_e',
        'tv016_departement_fch', 'tv016_departement_fecs_ancienne_m_i',
        'tv016_departement_fecs_recente_m_i',
        'tv016_departement_fecs_solaire_m_i',
        'tv016_departement_fecs_ancienne_i_c',
        'tv016_departement_fecs_recente_i_c', 'tv017_zone_hiver_code',
        'tv017_zone_hiver_t_ext_moyen', 'tv017_zone_hiver_peta_cw',
        'tv017_zone_hiver_dh14', 'tv017_zone_hiver_prs1', 'tv018_zone_ete_code',
        'tv018_zone_ete_sclim_inf_150', 'tv018_zone_ete_sclim_sup_150',
        'tv018_zone_ete_rclim_autres_etages',
        'tv018_zone_ete_rclim_dernier_etage',
        ]
    )
    
    # list of columns to convert to float which is replacing a "\N" to "0.00" among possible values
    N_to_clean = ['surface_parois_verticales_opaques_deperditives', 'surface_planchers_bas_deperditifs','surface_planchers_hauts_deperditifs','surface_baies_orientees_sud','surface_baies_orientees_est_ouest',
              'surface_baies_orientees_nord','nombre_entrees_sans_sas','nombre_entrees_avec_sas','surface_verriere','nombre_niveaux','surface_utile','shon','surface_commerciale_contractuelle',
              'surface_thermique_lot','surface_habitable','surface_thermique_parties_communes']
    for col in N_to_clean :
        data[col] = data[col].replace(r"\N",'0.00') 
        data[col] = data[col].replace(',','') # clean the decimals
        data[col] = pd.to_numeric(data[col] ,errors='coerce')

    # Remove the rows having weird construction date ~ 35k rows
    data = data[~data['annee_construction'].isin([0,1,2,3,4,5,6,7,8,9,10,11,12])]

    # Replace NaN by 0.0 for 'surface_commerciale_contractuelle'
    data['surface_commerciale_contractuelle'] = data['surface_commerciale_contractuelle'].fillna(0.0)

    # Remove target as values
    data = data.drop(columns=["consommation_energie", "estimation_ges"], axis=0)

    # Remove ~40k rows of data without latitude and longitude
    data = data.dropna(subset=['latitude', 'longitude'], how='any')

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
