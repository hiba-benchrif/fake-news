# 🕵️‍♂️ Fake News Detector App

Une application web full-stack permettant de détecter si une information (texte ou titre) est vraie ou fausse grâce à un modèle d'intelligence artificielle (`scikit-learn`).

## 🚀 Fonctionnalités
- **Interface web claire** développée en pur HTML/CSS.
- **Analyse de texte** alimentée par un modèle de Machine Learning (NLP).
- **Backend rapide et léger** construit avec **Python (Flask)**.
- **Prêt pour la production** : projet entièrement conteneurisé avec **Docker** pour un déploiement très facile !

## 📂 Structure du projet
- `app/` : Contient l'application Flask (backend) et les templates HTML (frontend).
- `data/` : Le dataset utilisé pour l'entraînement du modèle.
- `notebook/` : Fichiers Jupyter d'exploration des données.
- `train_model.py` : Script de génération et d'entraînement du modèle IA.

## 🛠️ Installation et Lancement

### Avec Docker (Recommandé 🐳)
Assurez-vous d'avoir [Docker et Docker Compose](https://www.docker.com/) installés sur votre machine.
1. Clonez ce dépôt.
2. Lancez le projet avec la commande suivante :
```bash
docker-compose up --build -d
```
3. Ouvrez votre navigateur sur `http://localhost:5000`

### Sans Docker (Environnement Local 🐍)
1. Installez les dépendances requises :
```bash
pip install -r requirements.txt
```
2. Démarrez le serveur Flask :
```bash
python app/app.py
```
3. L'application sera disponible sur `http://localhost:5000`.

---
*Projet conçu pour être facilement hébergeable, modulaire et performant.*
