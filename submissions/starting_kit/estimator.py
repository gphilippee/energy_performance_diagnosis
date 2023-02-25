from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.base import BaseEstimator
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np
import pandas as pd


class Preprocessor(BaseEstimator):
    def fit(self, X, y):
        return self

    def transform(self, X):
        # Remove useless columns
        X = X.drop(
            columns=[
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
                "nombre_circulations_verticales",  # faible variance
                "nombre_boutiques",  # faible variance"
                "presence_verriere",
                "type_vitrage_verriere",  # faible variance
                "etat_avancement",
                "adresse_organisme_certificateur",
                "dpe_vierge",  # faible variance
                "est_efface",  # faible variance
                "date_reception_dpe",  # date
                "geo_score",
                "geo_type",
                "geo_adresse",
                "geo_id",
                "geo_l4",
                "geo_l5",
                "tr002_type_batiment_code",
                "tr002_type_batiment_description",
                "tr002_type_batiment_libelle",
                "tr002_type_batiment_est_efface",
                "tr002_type_batiment_ordre",
                "tr002_type_batiment_simulateur",
                "tr012_categorie_erp_code",
                "tr012_categorie_erp_categorie",
                "tr012_categorie_erp_groupe",
                "tr012_categorie_erp_est_efface",
                "tr013_type_erp_code",
                "tr013_type_erp_type",
                "tr013_type_erp_categorie_id",
                "tr013_type_erp_est_efface",
                "tr013_type_erp_categorie",
                "tv016_departement_code",
                "tv016_departement_departement",
                "tv017_zone_hiver_id",
                "tv018_zone_ete_id",
                "tv016_departement_altmin",
                "tv016_departement_altmax",
                "tv016_departement_nref",
                "tv016_departement_dhref",
                "tv016_departement_pref",
                "tv016_departement_c2",
                "tv016_departement_c3",
                "tv016_departement_c4",
                "tv016_departement_t_ext_basse",
                "tv016_departement_e",
                "tv016_departement_fch",
                "tv016_departement_fecs_ancienne_m_i",
                "tv016_departement_fecs_recente_m_i",
                "tv016_departement_fecs_solaire_m_i",
                "tv016_departement_fecs_ancienne_i_c",
                "tv016_departement_fecs_recente_i_c",
                "tv017_zone_hiver_code",
                "tv017_zone_hiver_t_ext_moyen",
                "tv017_zone_hiver_peta_cw",
                "tv017_zone_hiver_dh14",
                "tv017_zone_hiver_prs1",
                "tv018_zone_ete_code",
                "tv018_zone_ete_sclim_inf_150",
                "tv018_zone_ete_sclim_sup_150",
                "tv018_zone_ete_rclim_autres_etages",
                "tv018_zone_ete_rclim_dernier_etage",
            ]
        )

        # list of columns to convert to float which is replacing a "\N" to "0.00" among possible values
        nan_to_clean = [
            "surface_parois_verticales_opaques_deperditives",
            "surface_planchers_bas_deperditifs",
            "surface_planchers_hauts_deperditifs",
            "surface_baies_orientees_sud",
            "surface_baies_orientees_est_ouest",
            "surface_baies_orientees_nord",
            "nombre_entrees_sans_sas",
            "nombre_entrees_avec_sas",
            "surface_verriere",
            "nombre_niveaux",
            "surface_utile",
            "shon",
            "surface_commerciale_contractuelle",
            "surface_thermique_lot",
            "surface_habitable",
            "surface_thermique_parties_communes",
        ]
        for col in nan_to_clean:
            X[col] = X[col].replace(r"\N", "0.00")
            X[col] = X[col].replace(",", "")  # clean the decimals
            X[col] = pd.to_numeric(X[col], errors="coerce")

        # Replace NaN by mean for 'surface_commerciale_contractuelle'
        X["surface_commerciale_contractuelle"] = X[
            "surface_commerciale_contractuelle"
        ].fillna(X["surface_commerciale_contractuelle"].mean())
        return X


class Classifier(BaseEstimator):
    def __init__(self):
        self.model1 = LogisticRegression(max_iter=10_000)
        self.model2 = LogisticRegression(max_iter=10_000)

    def fit(self, X, Y):
        # Y are pd.DataFrame here
        self.model1.fit(X, Y.iloc[:, 0])
        self.model2.fit(X, Y.iloc[:, 1])

    def predict(self, X):
        y1 = self.model1.predict_proba(X)
        y2 = self.model2.predict_proba(X)
        # Y_pred are nd.ndarray here
        Y_pred = np.concatenate([y1, y2], axis=1)
        # 2 discrete probability distributions
        assert Y_pred.shape[1] == 14
        return Y_pred


def get_estimator():
    pipe = make_pipeline(Preprocessor(), StandardScaler(), Classifier())
    return pipe
