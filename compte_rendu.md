# COMPTE-RENDU — TP 4 : Assistant documentaire RAG avec évaluation et traçage

**Module :** Transformers & LLMs — ENSAM Casablanca  
**Étudiantes :** Oumayma Zerrik ,** Hala Belmoukhtar
**Filière :** Master Big Data & Internet of Things  
**Cas d'usage :** 1 — Assistant documentaire du cours Transformers & LLMs  
**Langue des documents et des questions de test :** anglais  
**LLM :** GPT-OSS 20B, via l'API Groq  
**Mode d'exécution :** API pour le LLM ; embeddings, recherche et index FAISS en local (CPU)  
**Embedding :** `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` (384 dimensions)  
**Chunking :** `RecursiveCharacterTextSplitter`, taille 800 caractères, chevauchement 150 caractères  
**Top-k :** 5 pour le RAG initial  
**Base vectorielle :** FAISS  
**Évaluation :** RAGAS 0.4.4

> **Remarque sur la langue :** le cas d'usage 1 est indiqué en français dans l'énoncé, mais les cinq supports de cours dont je disposais étaient en anglais. J'ai conservé la langue originale du corpus et rédigé les questions de test en anglais. Ce choix constitue un écart à faire valider par l'enseignant.

## 1. Présentation du travail réalisé

J'ai développé un assistant documentaire qui répond aux questions sur les Transformers et les LLMs à partir d'un corpus de cours. Le principe est de rechercher les passages les plus pertinents dans les documents, puis de les fournir à un modèle de langage pour générer une réponse accompagnée de citations de pages.

J'ai regroupé cinq supports en un PDF de **192 pages**. Le chargement du document avec `PyPDFLoader`, suivi du découpage, a produit **283 chunks**. J'ai transformé ces chunks en vecteurs de **384 dimensions** à l'aide de MiniLM, puis je les ai indexés dans **FAISS**. Pour chaque question, le système retrouve cinq passages avant d'envoyer le contexte à **GPT-OSS 20B via Groq**.

La chaîne utilisée est :

```text
PDF (192 pages)
     ↓ PyPDFLoader
Documents avec métadonnées (source, page)
     ↓ RecursiveCharacterTextSplitter
283 chunks
     ↓ MiniLM multilingue
Vecteurs de 384 dimensions
     ↓
Index FAISS

Question → Embedding → FAISS top-5 → Prompt avec sources → Groq / GPT-OSS 20B
                                                           ↓
                                               Réponse avec citations
```

J'ai ensuite préparé un **golden set de 15 questions/réponses de référence**, réalisé une évaluation RAGAS complète sur **3 questions**, et comparé deux stratégies de récupération sur les **15 questions** : FAISS seul et FAISS suivi d'un reranker.

## 2. Scores RAGAS obtenus

Les scores ci-dessous correspondent **uniquement aux 3 questions évaluées avec succès**, et non à l'ensemble des 15 questions du golden set.

| Métrique | Score | Interprétation |
|---|---:|---|
| Fidélité (*Faithfulness*) | **1,0000** | Les affirmations évaluées ont été jugées soutenues par les passages récupérés. |
| Pertinence de la réponse (*Answer Relevancy*) | **0,9925** | Les réponses évaluées sont très proches des questions posées. |
| Précision du contexte (*Context Precision*) | **0,9722** | Le classement des contextes a été jugé très pertinent sur cet échantillon. |
| Rappel du contexte (*Context Recall*) | **1,0000** | Les passages récupérés couvrent les éléments requis par les réponses de référence sur cet échantillon. |

**Fichier de résultats :** `ragas_scores_test3.csv`.

J'avais aussi effectué un premier test sur une seule question, avec une fidélité de **0,8889**, une pertinence de **1,0000**, une précision du contexte de **1,0000** et un rappel de **1,0000** (`ragas_scores_test.csv`). Les scores varient selon les questions évaluées.

L'évaluation des **15 questions avec RAGAS** n'a pas pu être achevée, car l'API Groq a atteint sa limite de tokens. Je ne présente donc pas les résultats sur trois questions comme des performances globales du système.

## 3. Exemple de question/réponse avec citation

**Question :** *What is the attention mechanism in Transformers?*

**Réponse obtenue (résumé fidèle) :** le mécanisme d'attention utilise des projections apprises **Query, Key et Value**. Il compare les requêtes aux clés, divise les scores par la racine carrée de la dimension des clés, applique `softmax`, puis utilise les poids obtenus pour agréger les vecteurs de valeurs. Le résultat représente chaque token en tenant compte de son contexte.

**Formule :**

\[
\operatorname{Attention}(Q,K,V)=\operatorname{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
\]

**Sources citées par le système :** pages **118, 70, 59 et 122** du PDF fusionné. La **page 59** contient notamment l'équation complète et l'explication de ses étapes.

Les numéros sont issus des métadonnées des chunks (`page + 1`). Les citations générées doivent néanmoins être vérifiées : une page récupérée ne justifie pas automatiquement chaque phrase de la réponse.

---

## 4. Réponses aux questions du TP

### Q1. Quelle stratégie de chunking ai-je choisie ? Pourquoi la structure compte-t-elle pour un texte juridique ?

J'ai utilisé `RecursiveCharacterTextSplitter` avec des chunks de **800 caractères maximum** et un **chevauchement de 150 caractères**. Le découpage privilégie les séparateurs de paragraphes, de lignes, puis les espaces. Comme j'ai chargé le PDF page par page, les chunks conservent les métadonnées de leur page d'origine. Cela facilite les citations, même si les chunks ne correspondent pas nécessairement à des sections complètes.

Pour un texte juridique, la structure est particulièrement importante : une règle dépend de son article, de ses exceptions et parfois des dispositions voisines. Un découpage arbitraire peut séparer une condition de son application et entraîner une réponse trompeuse. Dans un assistant juridique, je privilégierais donc un découpage **par article ou section**, avec le numéro de l'article dans les métadonnées. Dans mon cas, j'ai choisi un corpus pédagogique, donc mes sources sont citées **par page**, et non par article.

### Q2. Le modèle d'embedding est-il multilingue ? Pourquoi est-ce indispensable pour l'arabe ?

Oui. J'ai utilisé `paraphrase-multilingual-MiniLM-L12-v2`, un modèle d'embedding **multilingue** qui produit des vecteurs de **384 dimensions**. Il transforme les chunks et les questions en représentations numériques comparables.

Un modèle multilingue est indispensable si le corpus ou les questions sont en arabe, car la recherche doit pouvoir représenter le sens des phrases arabes. Un modèle principalement entraîné pour l'anglais risquerait de moins bien rapprocher une question arabe de son passage pertinent. Dans mon expérimentation, les documents et les questions étaient en anglais ; je n'ai donc pas mesuré les performances en arabe.

### Q3. Comment le système cite-t-il ses sources ? Pourquoi est-ce essentiel en juridique ?

Lors de l'ingestion, `PyPDFLoader` conserve le nom du fichier et le numéro de page. Après la récupération FAISS, je construis le contexte envoyé au LLM en ajoutant l'indication **`[Page X]`** devant chaque passage. Le prompt demande ensuite au LLM de mentionner les pages qui soutiennent sa réponse.

Les citations rendent la réponse **vérifiable et traçable**. C'est encore plus important dans le domaine juridique, où l'utilisateur doit pouvoir consulter précisément le texte applicable, son article et ses conditions. Dans une version juridique, j'ajouterais les **numéros d'articles**, la version du texte et la référence légale. Mon système actuel produit des citations de pages, mais je n'ai pas développé de vérification automatique de leur exactitude.

### Q4. Quels scores RAGAS ai-je obtenus et comment les interpréter ?

Pour les **3 questions évaluées avec succès**, j'ai obtenu :

- **Faithfulness : 1,0000** — les réponses évaluées sont bien étayées par les passages transmis au LLM, selon RAGAS.
- **Answer Relevancy : 0,9925** — elles répondent très directement aux questions.
- **Context Precision : 0,9722** — les passages pertinents sont globalement bien classés.
- **Context Recall : 1,0000** — le contexte récupéré couvre les informations attendues dans les réponses de référence.

Ces résultats sont encourageants, mais leur portée reste **limitée** : mon golden set contient 15 questions alors que les scores RAGAS exploitables portent sur 3 seulement. Je ne peux donc pas conclure que le RAG atteint ces scores sur l'ensemble du corpus.

### Q5. Si la fidélité est basse mais le rappel du contexte élevé, s'agit-il d'un problème de récupération ou de génération ?

Il s'agit **principalement d'un problème de génération**. Un rappel élevé indique que les informations nécessaires ont bien été récupérées. Si la fidélité est faible malgré cela, le LLM peut avoir ajouté des affirmations non justifiées, mal interprété le contexte ou mélangé des éléments.

Je commencerais donc par analyser le **prompt**, la réponse produite et les citations, puis par renforcer la consigne de ne répondre qu'à partir des passages fournis. Je vérifierais aussi la qualité réelle des chunks, car un contexte pertinent mais fragmenté peut être difficile à exploiter.

### Q6. Que fait le système lorsque le contexte est insuffisant ? Invente-t-il ou refuse-t-il honnêtement ?

J'ai testé la question : **« What is the exact population of Casablanca in 2026? »**. Cette information n'appartient pas au corpus de cours Transformers & LLMs.

Même si FAISS a renvoyé cinq passages, le LLM a répondu exactement :

> The provided documents do not contain enough information.

Pour ce test, mon assistant a donc **refusé de fournir une réponse non justifiée**, ce qui est le comportement attendu. Cependant, un seul test réussi ne garantit pas que le système évitera toutes les hallucinations.

### Q7. Quel gain ai-je mesuré avec le reranking ? Quelle étape est la plus lente selon le traçage ?

J'ai expérimenté un **reranking par Cross-Encoder**. Dans la configuration initiale, FAISS récupère directement les **5 premiers chunks**. Dans la seconde configuration, FAISS récupère **10 candidats**, puis `cross-encoder/ms-marco-MiniLM-L6-v2` les reclasse pour conserver les **5 premiers**.

J'ai comparé les deux méthodes sur **15 questions**, en utilisant les pages annotées du golden set comme références :

| Métrique | FAISS seul | FAISS + reranking | Différence |
|---|---:|---:|---:|
| Recall@5 | 0,7667 | 0,7000 | **−0,0667** |
| Hit@5 | 0,8000 | 0,7333 | **−0,0667** |
| MRR@5 | 0,5211 | 0,5444 | **+0,0233** |

Le reranking améliore légèrement **MRR@5**, c'est-à-dire le rang de la première page de référence trouvée, mais il réduit **Recall@5** et **Hit@5**. Il ne constitue donc **pas une amélioration globale** selon les mesures réalisées. J'ai conservé **FAISS top-5** comme configuration de référence.

**Fichier :** `retrieval_comparison.csv`.

Je n'ai pas implémenté le bonus **OpenTelemetry**. J'ai enregistré des durées d'exécution pour les requêtes RAG, mais je n'ai pas mesuré séparément les spans d'embedding, de recherche et de génération. Je ne peux donc pas affirmer expérimentalement quelle étape est la plus lente. L'ajout d'OpenTelemetry serait une amélioration future.

### Q8. API ou exécution locale pour un cabinet soumis à la confidentialité (CNDP) : que recommanderais-je ?

Pour un cabinet traitant des **données confidentielles ou personnelles**, je privilégierais une **exécution locale ou dans une infrastructure privée contrôlée**, si les ressources disponibles le permettent. Cela permet de mieux maîtriser l'emplacement des documents, les accès, la journalisation et la transmission des données.

Une API externe peut simplifier le déploiement et réduire les besoins matériels, comme dans mon TP avec Groq. En contrepartie, une partie du contenu documentaire peut être transmise au fournisseur. Avant d'utiliser une API pour des dossiers sensibles, il faudrait examiner les engagements contractuels, les conditions de traitement et de conservation des données, les transferts éventuels et la conformité avec la **loi marocaine 09-08** ainsi que les exigences applicables de la **CNDP**.

Pour mon TP, les données utilisées sont des supports pédagogiques. Le choix de l'API était donc surtout pratique : éviter de faire tourner un LLM lourd sur mon PC. Cette décision ne doit pas être transposée automatiquement à un cabinet juridique.

---

## 5. Décision : API vs local et confidentialité

Dans mon prototype, j'ai retenu une architecture **hybride** : extraction du PDF, embeddings et recherche FAISS **en local**, génération du texte **via l'API Groq**. Cette architecture a permis de développer le RAG sans GPU local et de limiter la charge sur mon ordinateur.

Pour un usage réel avec des documents confidentiels, je recommanderais une étude préalable des risques et, si nécessaire, un déploiement **entièrement local ou privé**. Je veillerais aussi à ne pas transmettre de données personnelles à un fournisseur externe sans garanties appropriées.

## 6. Limites et pistes d'amélioration

Les principales limites que j'ai identifiées sont les suivantes :

1. **Évaluation RAGAS partielle :** 3 questions évaluées avec succès sur les 15 préparées ; les autres tentatives ont été interrompues par les limites de tokens Groq.
2. **Chunking non structurel :** découpage par taille de texte et frontières de pages, sans prise en compte systématique des sections ou des figures.
3. **Classement imparfait :** certains passages généraux peuvent être mieux classés que des explications directement utiles.
4. **Reranking non concluant :** amélioration du MRR@5, mais baisse du Recall@5 et du Hit@5.
5. **Citations non vérifiées automatiquement :** les pages affichées doivent être contrôlées par rapport au texte source.
6. **Pas de traçage OpenTelemetry :** je ne dispose pas d'une mesure détaillée des durées par composant.
7. **Corpus en anglais :** je n'ai pas testé un corpus français ou arabe, contrairement aux langues prévues dans l'énoncé.

Pour la suite, je pourrais tester une **recherche hybride FAISS + BM25**, adopter un **chunking par section**, comparer plusieurs modèles d'embedding et de reranking, ajouter une **validation des citations**, terminer RAGAS sur les 15 questions et instrumenter les étapes avec **OpenTelemetry**.

## 7. Conclusion personnelle

Ce TP m'a permis de comprendre concrètement la différence entre **recherche documentaire** et **génération de texte**. J'ai appris à extraire un corpus PDF, le découper, créer des embeddings, indexer les passages dans FAISS et construire un prompt enrichi pour un LLM via LangChain.

J'ai aussi compris qu'une réponse qui semble correcte ne suffit pas : il faut mesurer la **fidélité**, la **pertinence** et la **qualité des contextes récupérés**. Les expériences de reranking m'ont particulièrement montré qu'une technique considérée comme plus avancée n'améliore pas forcément tous les indicateurs. L'essentiel est de **comparer les résultats et de justifier les décisions à partir des mesures**.

Le système final fonctionne pour le question-réponse documentaire, avec des résultats d'évaluation préliminaires prometteurs. Les limites identifiées constituent des axes d'amélioration pour une version plus robuste et reproductible.

---

## 8. Fichiers du projet

- `create_vectorstore.py` — ingestion, chunking, embeddings et création de l'index FAISS.
- `rag.py` — génération des réponses avec citations.
- `golden_set.json` — 15 questions et réponses de référence.
- `evaluate_rag.py` et `rag_results.json` — génération et sauvegarde des réponses évaluables.
- `ragas_evaluation.py` — calcul des métriques RAGAS.
- `ragas_scores_test3.csv` — résultats RAGAS sur les 3 questions réussies.
- `test_reranking.py` et `compare_retrieval.py` — expériences de classement.
- `retrieval_comparison.csv` — résultats de la comparaison sur 15 questions.

**Dépôt GitHub :** https://github.com/oumaymazerrik/RAG-Document-Assistant
