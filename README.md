# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.

## Workflow de modélisation

Les scripts s'exécutent depuis la racine du dépôt dans l'ordre suivant :

```bash
python src/data/split_data.py
python src/data/normalize_data.py
python src/models/grid_search.py
python src/models/train_model.py
python src/models/evaluate_model.py
```

Le split retire la colonne `date` des variables explicatives et utilise
`silica_concentrate` comme cible. Les jeux de données intermédiaires sont
enregistrés dans `data/processed_data`, le scaler et les modèles dans `models`,
les prédictions dans `data/processed_data/predictions.csv` et les métriques dans
`metrics/scores.json`.

## Versioning DVC et DagsHub

Le dépôt DVC utilise le remote DagsHub configuré dans `.dvc/config` :

```text
https://dagshub.com/fabper01-hub/examen-dvc.dvc
```

Après avoir configuré les identifiants DagsHub localement, publier les données
et les modèles avec :

```bash
venv/bin/dvc push
```

Pour conserver les identifiants hors du dépôt, les définir dans la configuration
locale DVC :

```bash
venv/bin/dvc remote modify --local dagshub auth basic
venv/bin/dvc remote modify --local dagshub user <identifiant-dagshub>
venv/bin/dvc remote modify --local dagshub password <token-dagshub>
```

Le fichier `data/raw_data/raw.csv.dvc` versionne le dataset source. Les sorties
de la pipeline, dont `models/gradient_boosting_model.pkl`, sont déclarées dans
`dvc.yaml` et verrouillées dans `dvc.lock`.
