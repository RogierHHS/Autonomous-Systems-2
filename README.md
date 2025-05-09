# Reinforcement Learning Project – ViZDoom

Dit project onderzoekt hoe verschillende Reinforcement Learning (RL) agents kunnen worden toegepast in een dynamische en visueel complexe spelomgeving: **ViZDoom**. Het doel is om de prestaties van drie benaderingen te vergelijken:

- Een baseline agent die willekeurige acties uitvoert,
- Een tabulaire Q-learning agent,
- Een Deep Q-learning agent met een neuraal netwerk.

## Projectstructuur

Alle logica is gescheiden in `.py` bestanden en aangestuurd via een Jupyter Notebook. De benodigde bestanden bevinden zich in de map `SRC/` en omvatten onder andere:

- `agent.py`: Q-learning agent
- `training.py`: Trainingslogica voor tabulaire Q-learning
- `deep_q_model.py`: Neuraal netwerkmodel en replay memory
- `deep_q_environment.py`: ViZDoom omgeving voor Deep Q-learning
- `deep_q_baseline.py`: Willekeurige baseline voor Deep Q-learning
- `environment.py`: Standaardomgeving voor Q-learning
- `utils.py` & `deep_qutils.py`: Hulpfuncties voor visualisatie en preprocessing

## Installatie

Maak een Python-omgeving aan (bijv. met venv of conda) en installeer de vereiste pakketten:

```bash
pip install -r requirements.txt
```

Maak vervolgens een map `SRC/` en plaats hierin alle `.py` bestanden uit dit project.

## Vereisten

Hieronder de inhoud van `requirements.txt` die je kunt gebruiken:

```
opencv-python
numpy
pandas
matplotlib
torch
tensorflow
scikit-image
vizdoom
```

> Tip: sommige systemen vereisen installatie van VizDoom via conda of het bouwen van binaries. Raadpleeg hiervoor: https://github.com/mwydmuch/ViZDoom

## Uitvoeren

Start het Jupyter Notebook en voer de cellen uit. Zorg dat alle `.py` bestanden correct in de `SRC/` map staan en dat je werkdirectory juist is ingesteld. Het notebook bevat:

- Trainingscode voor alle drie de agenten
- Vergelijkende evaluaties en reward-grafieken
- Reflectie op de prestaties van de modellen
