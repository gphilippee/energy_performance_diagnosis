import os
import requests
import io
import pandas as pd

if not os.path.exists('data'):
    os.makedirs('data/')

# Binary Alphadigits 

#if not os.path.exists('data/dpe-tertiaire.csv'):
url = "https://data.ademe.fr/data-fair/api/v1/datasets/dpe-tertiaire/full"

# Envoi de la requête
response = requests.get(url)
pd.read_csv(io.StringIO(response.text)).to_csv('data/data.csv')
