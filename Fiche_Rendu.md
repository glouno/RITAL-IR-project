# Fiche de Rendu - HiRAG

## 1. Dataset & Metriques

Nous avons evalue HiRAG sur deux sous-ensembles d'UltraDomain Benchmark : `Agriculture` et `Mix`. Les deux autres domaines du papier, `CS` et `Legal`, n'ont pas ete relances dans la phase finale car le cout de reconstruction complete des graphes etait plus eleve. Notre objectif etait d'abord de valider proprement le pipeline complet sur deux domaines contrastes.

Graphes reconstruits avec la stack OpenAI :

| Dataset | Docs | Questions | Chunks | Nodes | Edges | Community reports |
|---|---:|---:|---:|---:|---:|---:|
| Agriculture | 12 | 100 | 1 756 | 23 224 | 48 428 | 4 639 |
| Mix | 61 | 130 | 579 | 16 356 | 25 960 | 2 695 |

Nous avons utilise `gpt-5.4-mini` pour la generation des reponses et le LLM-as-a-judge, et `text-embedding-3-small` pour les embeddings. Les comparaisons principales portent sur `naive`, `hi` (baseline HiRAG de notre fork), `hi_weighted`, `hi_minmax`, `hi_minmax_budgeted`, `hi_rerank_weighted` et `hi_mcts`.

Les metriques sont partiellement comparables au papier HiRAG. Comme dans l'article, nous utilisons une evaluation pairwise par juge LLM et des win rates. En revanche, ce n'est pas une replication exacte : les modeles ne sont pas ceux du papier original, le baseline `hi` correspond a notre fork avec query-aware source snippets, et nous ne couvrons pas les quatre domaines finaux.

Metriques utilisees :

- `LLM-as-a-judge win rate` : proportion de comparaisons ou une variante est preferee a HiRAG.
- `Strict swapped-order win rate` : win rate plus conservateur, garde seulement les paires ou le juge reste coherent quand l'ordre A/B est inverse.
- `Human win rate` : preference humaine pairwise contre HiRAG classique, ties comptes comme 0.5.
- `Answer-question score` : proportion de reponses jugees comme repondant a la question par les annotateurs humains.
- `Span density` : proportion de la reponse surlignee comme utile par les annotateurs.
- `Retrieval metrics` : temps de retrieval, nombre d'edges de bridge, taille des chemins, tokens de contexte, composition du prompt final.

Resultats principaux :

| Variante | Human win rate vs HiRAG | LLM win rate strict | Span density |
|---|---:|---:|---:|
| Naive | 65.0% | 78.6% | 32.0% |
| Minmax budgeted | 45.0% | 61.5% | 32.1% |
| MCTS | 52.5% | 38.5% | 30.4% |

Lecture principale : nos variantes graphe, surtout MCTS et weighted/minmax selon le domaine, montrent qu'il est possible d'ameliorer le parcours dans le graphe par rapport a HiRAG classique. Cependant, `naive` reste tres fort en evaluation finale, car il conserve beaucoup plus d'evidence source exacte dans le prompt. Dans nos traces, les modes graphe donnent environ 20-21% d'evidence source directe en Agriculture et environ 34% en Mix, contre presque 100% pour naive RAG.

## 2. Difficultes d'Implementation & Solutions

La premiere difficulte a ete la reconstruction complete du pipeline HiRAG. Le systeme ne consiste pas seulement a faire du retrieval : il faut extraire les entites, extraire les relations, construire un graphe, calculer des embeddings, creer une hierarchie de communautes, generer des community reports, materialiser les vector stores, puis seulement ensuite generer et juger les reponses.

Solution : nous avons construit un pipeline OpenAI Batch complet pour les etapes couteuses, avec manifests, retries, couts reels, et verification des erreurs. Cela nous a permis de reconstruire Agriculture et Mix proprement, sans dependre des limites de temps d'inference du vLLM local.

La deuxieme difficulte concernait les limites de tokens et la stabilite des sorties LLM. Certaines extractions ou generations pouvaient etre tronquees, mal parsees, ou depasser les budgets prevus.

Solution : nous avons utilise des plafonds de `max_completion_tokens` plus hauts, des retries cibles sur les lignes tronquees ou mal parsees, et des fichiers de revue manuelle pour les cas residuels. Nous avons aussi corrige un probleme important de `max_token_for_community_report`, qui supprimait parfois des community reports entiers du contexte final.

La troisieme difficulte etait le cout et le stockage des embeddings, notamment pour les edge embeddings necessaires aux variantes query-aware.

Solution : nous avons ajoute de l'outillage pour batcher les embeddings d'edges avec OpenAI, les importer dans le cache runtime, puis reutiliser ce cache pendant les evaluations. Cela rend les variantes weighted, minimax et MCTS executables sans recalculer les embeddings a chaque requete.

La quatrieme difficulte etait algorithmique. Le bridge original de HiRAG est surtout topologique : il relie des entites locales et des communautes globales, mais le chemin n'est pas assez conditionne par la requete.

Solution : nous avons implemente plusieurs variantes query-aware : Dijkstra pondere, Minimax, Minimax budgeted, reranking ColBERT des entites locales, et MCTS pour explorer dynamiquement des chemins dans un sous-graphe candidat.

La derniere difficulte etait l'evaluation. Le LLM-as-a-judge est utile mais bruite, et il peut etre sensible a l'ordre des reponses.

Solution : nous avons juge les paires dans les deux ordres, mesure les desaccords d'ordre, puis ajoute une annotation humaine sur un sous-ensemble de 60 comparaisons. Les humains annotaient la preference, si chaque reponse repondait a la question, et les spans utiles.

## 3. Limites de Resolution (L'Impossible)

Une replication strictement fidele du papier est difficile, meme avec une implementation rigoureuse. D'abord, les modeles exacts du papier, les prompts internes, les parametres de generation et certains details de pipeline ne sont pas tous disponibles ou pas parfaitement reproductibles. Nos resultats sont donc comparables dans l'esprit, mais pas identiques au protocole original.

Ensuite, la construction complete des graphes sur tous les domaines est couteuse. Meme Agriculture, qui ne contient que 12 documents, produit 1 756 chunks et plus de 48 000 edges. Refaire Agriculture, Mix, CS et Legal avec extraction, relations, hierarchy, community reports, embeddings et evaluation complete demanderait un budget et un temps beaucoup plus importants.

Une autre limite est que le graphe est une representation compressee. Les entites, relations et community reports structurent l'information, mais ils ne remplacent pas toujours les passages sources exacts. Pour des taches QA, le LLM a souvent besoin de citations ou details textuels precis. C'est une limite de resolution importante : un meilleur graphe ne garantit pas une meilleure reponse si le prompt final ne preserve pas assez d'evidence.

Enfin, l'evaluation par LLM-as-a-judge reste imparfaite. Nous avons observe des desaccords lorsque l'ordre des reponses est inverse. L'annotation humaine confirme aussi que "repondre a la question" et "etre prefere" ne mesurent pas exactement la meme chose : MCTS repond tres souvent correctement, mais n'est pas toujours prefere a HiRAG.

Notre travail doit donc etre presente comme une reproduction experimentale et une analyse critique de HiRAG, pas comme une replication exacte et definitive du papier.

## 4. Amelioration Majeure (Next Step)

L'amelioration prioritaire serait de construire un HiRAG `evidence-aware` : utiliser le graphe pour trouver les bons chemins, mais reinjecter explicitement les meilleurs extraits sources exacts dans le prompt final.

Nos resultats montrent que les variantes query-aware changent vraiment le retrieval. MCTS, par exemple, explore des chemins pertinents et obtient un score humain legerement positif contre HiRAG classique. Mais cela ne suffit pas toujours, car le contexte final reste trop abstrait : entites, relations, summaries et community reports remplacent parfois les passages sources utiles.

Le next step logique serait donc :

> Combiner traversal dynamique du graphe avec selection controlee de snippets sources exacts.

Concretement, le systeme devrait :

- partir des entites locales et communautes recuperees par HiRAG ;
- utiliser weighted/minimax/MCTS pour trouver des chemins query-aware ;
- recuperer les chunks sources associes aux entites, relations et communautes de ces chemins ;
- selectionner plusieurs snippets courts mais precis, pas seulement une fenetre trop agressive ;
- optimiser un budget de contexte qui equilibre graph context et evidence textuelle ;
- evaluer separement la qualite du retrieval, la composition du prompt, et la qualite de la reponse finale.

Conclusion courte :

> HiRAG ne doit pas seulement mieux raisonner sur un graphe ; il doit mieux transformer ce graphe en preuves textuelles utiles pour le LLM. Notre contribution montre que le graph traversal est ameliorable, mais que le vrai goulot d'etranglement est l'assemblage final du contexte.
