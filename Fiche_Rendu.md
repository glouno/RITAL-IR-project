# Fiche de Rendu - HiRAG

## 1. Dataset & Metriques

Nous avons travaille principalement sur deux sous-ensembles UltraDomain :
`Agriculture` et `Mix`. Ces datasets font partie des quatre domaines utilises
par le papier HiRAG, avec `CS` et `Legal`. Nous n'avons pas finalise `CS` et
`Legal`, car le cout de construction des graphes etait trop eleve, et les deux
datasets deja traites suffisaient pour evaluer notre hypothese principale.

Par rapport au papier HiRAG, nos metriques sont partiellement comparables, mais
pas parfaitement identiques. Le papier utilise une evaluation par juge LLM sur
des criteres qualitatifs comme la completude, la diversite et l'utilite de la
reponse, ainsi que des comparaisons de type win-rate. Nous avons reproduit ce
principe avec un juge LLM via OpenAI Batch, en comparant les reponses generees
par le HiRAG baseline `hi` contre nos variantes.

Metriques utilisees :

- `Raw win rate` : proportion de jugements ou une variante bat le baseline `hi`.
- `Swapped-order agreement` : stabilite du juge quand l'ordre des reponses est inverse.
- `Strict variant win rate` : taux de victoire conservateur, comptant seulement les requetes ou les deux ordres de jugement donnent la meme variante gagnante.
- `Mean context tokens` : taille moyenne du contexte donne au generateur.
- `Mean bridge edges` et `mean path edges` : taille du contexte de pont dans le graphe.
- `Actual total tokens` : cout reel en tokens des reponses.
- Metriques proxy de retrieval sur Agriculture q50 : query overlap, local recall, global recall, latence.

Resultat principal :

| Dataset | Variante vs `hi` | Raw Win Rate | Agreement | Strict Win Rate |
|---|---:|---:|---:|---:|
| Agriculture | `hi_minmax_budgeted` | 50.0% | 40.0% | 20.0% |
| Agriculture | `hi_rerank_weighted` | 58.3% | 76.7% | 46.7% |
| Mix | `hi_minmax_budgeted` | 28.3% | 46.7% | 6.7% |
| Mix | `hi_rerank_weighted` | 55.0% | 70.0% | 40.0% |

La metrique la plus proche du papier est donc le win-rate par juge LLM. En
revanche, notre protocole n'est pas une reproduction complete, car nous
n'utilisons pas exactement les memes modeles, pas tous les domaines, et Mix
utilise des community reports extractifs plutot que generes par LLM.

## 2. Difficultes d'Implementation & Solutions

La premiere difficulte majeure a ete le cout de construction des graphes. HiRAG
necessite extraction d'entites, extraction de relations, clustering
hierarchique, generation de resumes de communautes et indexation vectorielle.
Meme sur Agriculture, seulement 12 documents longs produisent 1 756 chunks. Un
run complet peut durer plusieurs heures avec un backend local.

Solution : nous avons d'abord reduit le perimetre experimental, puis ajoute des
regimes de prompts plus concis (`lean`, `ultra_lean`) et des outils de batch
OpenAI pour deporter certaines etapes couteuses.

La deuxieme difficulte concernait la stabilite du pipeline. Les etapes de
clustering GMM peuvent echouer quand les embeddings sont trop proches ou
degeneres. Certaines requetes depassaient aussi les limites de contexte des
modeles.

Solution : nous avons ajoute des protections sur les budgets de tokens, des
marges de securite, des limites de taille pour les prompts, et du durcissement
cote GMM.

La troisieme difficulte etait l'evaluation. Le papier utilise une evaluation
qualitative par LLM, mais cela coute cher et introduit du bruit.

Solution : nous avons utilise un protocole pairwise avec ordre inverse des
reponses. Cela permet de mesurer non seulement le win-rate, mais aussi la
stabilite du juge. C'est important, car certaines variantes semblent bonnes en
raw win-rate mais deviennent moins convaincantes avec le strict agreement.

La quatrieme difficulte etait theorique : le bridge retrieval original est
partiellement query-aware, mais le chemin dans le graphe reste surtout
topologique.

Solution : nous avons implemente des variantes de retrieval query-aware :
chemins ponderes par similarite avec la requete, strategie minmax, budget de
pont, et reranking local.

## 3. Limites de Resolution

Certains aspects du papier sont difficiles, voire impossibles a reproduire
fidelement dans notre contexte.

D'abord, les modeles exacts ne sont pas tous disponibles ou pratiques a
utiliser. Le papier mentionne DeepSeek-V3, GLM-4-Plus embeddings et GPT-4o
comme juge. Nous avons utilise un melange de vLLM/local, FastEmbed et OpenAI
Batch. Cela rend les resultats comparables dans l'esprit, mais pas strictement
equivalents.

Ensuite, la construction complete sur les quatre datasets est tres couteuse.
`Legal` et `CS` ont beaucoup de chunks, et refaire extraction, relations,
hierarchie et community reports serait long et cher.

Autre limite : la generation des community reports. Dans Mix, le graphe batch a
ete materialise avec des community reports extractifs, pas des resumes LLM
complets. Cela reduit la fidelite au papier.

Enfin, l'evaluation par juge LLM reste bruitee. Meme avec l'ordre inverse, on
observe des desaccords. C'est pourquoi nous rapportons aussi le strict win rate.
Ce bruit rend impossible une conclusion absolue du type "la variante X est
toujours meilleure".

Donc notre travail doit etre presente comme une reproduction partielle et une
analyse experimentale ciblee, pas comme une replication complete du papier.

## 4. Amelioration Majeure / Next Step

L'amelioration prioritaire serait de rendre HiRetrieval plus query-aware tout en
controlant strictement le budget de contexte.

Notre resultat principal montre que `hi_rerank_weighted` est la variante la plus
robuste. Elle combine :

- reranking local des entites candidates ;
- ponderation des chemins de bridge par rapport a la requete ;
- contexte plus compact que le baseline ;
- meilleur win-rate sur Agriculture et Mix.

A l'inverse, `hi_minmax_budgeted` confirme que les chemins query-aware peuvent
ameliorer la pertinence du bridge, mais il ajoute trop de contexte et devient
instable au niveau des reponses finales.

Le next step logique serait donc :

> Developper un module de retrieval adaptatif qui choisit dynamiquement combien
> de contexte local, global et bridge inclure selon la requete.

Concretement :

- garder `hi_rerank_weighted` comme base ;
- ajouter un budget dynamique pour le bridge ;
- penaliser les hubs generiques dans le graphe ;
- selectionner les aretes de pont selon un score combinant pertinence semantique, cout en tokens et diversite ;
- evaluer avec win-rate, agreement et cout token.

Conclusion possible pour le rapport :

> Notre contribution principale n'est pas de remplacer HiRAG, mais d'identifier
> une faiblesse precise de HiRetrieval : le bridge retrieval est trop dependant
> de la topologie du graphe. Une amelioration query-aware, surtout combinee avec
> du reranking local et un controle du budget, donne un signal plus robuste et
> plus economique que le baseline.
