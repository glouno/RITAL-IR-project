# Script 15 Minutes - Presentation HiRAG V3

Objectif narratif : montrer que HiRAG apporte une structure de retrieval plus riche que naive RAG, que nos variantes améliorent effectivement le parcours du graphe, mais que l'experience finale depend aussi beaucoup de l'assemblage du contexte donne au LLM.

These centrale :

> HiRAG ameliore la structure du retrieval, mais nos experiences montrent qu'un meilleur graph traversal seul ne suffit pas. Pour des taches QA, le contexte final doit conserver assez d'evidence source exacte ; sinon naive chunk retrieval peut gagner.

## Timing Global

- Slides 1-4 : probleme et motivation, 2 min.
- Slides 5-16 : ameliorations implementees, 5 min.
- Slides 17-25 : evaluation et resultats, 6 min.
- Slides 26-27 : conclusion, 2 min.

Recommandation : ne pas expliquer toutes les formules en detail. Les slides algorithmiques servent a montrer que le travail est serieux ; la narration doit rester orientee evaluation.

## Slide 1 - Titre

Temps : 15 s.

Dire :

> Nous presentons notre reproduction et extension de HiRAG, un systeme de retrieval augmente par graphe hierarchique. Notre objectif n'etait pas seulement de refaire le pipeline, mais aussi de comprendre dans quels cas le graphe aide vraiment la generation finale.

Point a mettre en avant :

- Projet en deux dimensions : reproduction + ameliorations.
- Comparaison finale contre naive RAG, HiRAG classique, Minmax, MCTS.

## Slide 2 - Plan

Temps : 20 s.

Dire :

> On commence par rappeler l'idee de HiRAG, puis les problemes que nous avons identifies dans l'implementation. Ensuite on presente nos ameliorations, puis l'evaluation : LLM-as-a-judge, annotations humaines, et analyse de composition du contexte.

Transition :

> Le point important est que nous separons retrieval structure et prompt final : ce n'est pas exactement la meme chose.

## Slide 3 - Concept de HiRAG

Temps : 45 s.

Dire :

> Dans un RAG standard, la requete est transformee en embedding, on recupere des chunks proches, puis on les donne au LLM. GraphRAG ajoute une structure entites-relations, mais reste souvent a une granularite fixe. HiRAG ajoute une hierarchie : local knowledge, global knowledge et un knowledge bridge entre les deux.

Mettre en avant :

- Standard RAG : simple et fort pour retrouver du texte exact.
- Graph RAG : structure entites-relations.
- HiRAG : local + global + bridge.

Phrase cle :

> La promesse de HiRAG est de mieux gerer les questions qui demandent de relier plusieurs informations, pas seulement de retrouver un passage proche.

## Slide 4 - Problemes de HiRAG

Temps : 50 s.

Dire :

> En regardant le papier et le code, nous avons identifie trois points faibles pratiques. D'abord, l'implementation peut remettre des chunks sources entiers dans le prompt final, ce qui peut saturer le contexte. Ensuite, les entites locales sont choisies par recherche dense seule, sans reranking fin. Enfin, le bridge entre entites locales et communaute globale n'est pas vraiment conditionne par la requete : le plus court chemin n'est pas forcement le plus pertinent.

Point important :

- Ce sont des problemes d'implementation et d'assemblage du contexte, pas seulement des problemes de graphe.

Transition :

> Nos ameliorations ciblent donc trois endroits : compacter les sources, mieux choisir les entites, et rendre le bridge query-aware.

## Slide 5 - Saturation du contexte

Temps : 40 s.

Dire :

> Premier correctif : au lieu de mettre un chunk entier dans le prompt, on selectionne une fenetre plus courte dans le chunk. On decoupe en phrases ou fenetres, on normalise les tokens, puis on choisit la fenetre qui chevauche le plus la requete.

Attention a dire :

> C'est utile pour reduire le bruit, mais nos resultats montrent que cette strategie peut aussi etre trop agressive : parfois on garde trop peu d'evidence source.

Stats a citer plus tard :

- Dans nos traces, HiRAG garde environ 20-21% d'evidence source directe en Agriculture, et 34% en Mix, contre quasiment 100% pour naive.

## Slide 6 - Reranker ColBERT

Temps : 30 s.

Dire :

> Deuxieme amelioration : apres la recherche dense des entites locales, on applique un reranker ColBERT. L'idee est de ne pas se limiter a un score embedding global, mais de permettre une interaction plus fine entre les tokens de la requete et les tokens des entites.

Mettre en avant :

- Dense retrieval top-100.
- Reranking ColBERT.
- Selection top-20.

Nuance :

> Dans cette premiere evaluation, le reranking n'a pas suffi a battre naive RAG, mais il nous donne une brique utile pour mieux filtrer le graphe.

## Slide 7 - Query-aware bridge

Temps : 35 s.

Dire :

> Le bridge original de HiRAG relie les entites locales aux communautes globales avec des chemins relativement statiques. Nous avons voulu rendre ce bridge sensible a la requete : si la question parle d'un mecanisme specifique, le meilleur chemin n'est pas necessairement le plus court.

Phrase cle :

> On remplace donc "chemin court" par "chemin utile pour cette requete".

## Slide 8 - Variantes de bridge

Temps : 35 s.

Dire :

> Nous avons teste quatre familles : Dijkstra pondere, Minimax, Minimax budgeted, et MCTS. Dijkstra minimise le cout total. Minimax evite les chemins avec une tres mauvaise arete. La version budgeted impose aussi une contrainte de taille. MCTS explore des decisions de bridge avec une recompense long terme.

Conseil :

- Ne pas detailler ici ; annoncer que les slides suivantes donnent l'intuition.

## Slide 9 - Cout d'une arete

Temps : 45 s.

Dire :

> Pour rendre les chemins query-aware, on donne un embedding a chaque relation enrichie par les descriptions des entites source et cible. Ensuite, on compare cet embedding avec la requete. Une bonne arete est semantiquement proche de la requete, forte dans le graphe, et evite les noeuds trop generiques.

Mettre en avant :

- Embedding de relation contextualisee.
- Cout = pertinence semantique + penalite generique + force relationnelle.
- Meme modele d'embedding OpenAI : `text-embedding-3-small`.

Phrase cle :

> C'est cette metrique qui permet aux algorithmes de graphe d'etre orientes par la question.

## Slide 10 - Dijkstra pondere

Temps : 30 s.

Dire :

> Dijkstra pondere est une premiere correction naturelle : au lieu de chercher le chemin le plus court en nombre d'aretes, on cherche le chemin de cout total minimal.

Limite a citer :

> Mais le cout total peut masquer une tres mauvaise arete si les autres aretes sont tres bonnes. C'est exactement ce que Minimax essaie d'eviter.

## Slide 11 - Minimax

Temps : 40 s.

Dire :

> Minimax cherche le chemin dont la pire arete est la moins mauvaise possible. Cela correspond mieux a notre intuition : dans une chaine de raisonnement, une seule transition non pertinente peut casser tout le contexte.

Version budgeted :

> La version budgeted ajoute une contrainte de taille pour eviter d'obtenir un chemin trop long ou trop couteux en tokens.

## Slides 12-16 - MCTS

Temps total : 2 min.

Strategie : presenter MCTS comme une contribution interessante sans se perdre dans les equations.

### Slide 12 - Idee MCTS

Dire :

> MCTS est notre variante la plus "agentique" : au lieu de calculer un chemin une fois pour toutes, on simule plusieurs choix de parcours dans le graphe et on favorise les chemins qui atteignent la cible, restent courts, et gardent des aretes pertinentes.

Point cle :

- Objectif : meilleur compromis pertinence, longueur, budget token.

### Slide 13 - Sous-graphe

Dire :

> Comme le graphe complet est trop grand, on construit d'abord un sous-graphe candidat autour de la source et de la cible. On augmente le rayon si necessaire, puis on prune les voisins les moins utiles.

Pourquoi c'est important :

> Sans cette restriction, MCTS aurait trop de branches et serait trop lent.

### Slides 14-16 - Selection, expansion, rollout, reward

Dire :

> Le MCTS suit les etapes classiques : selection avec UCT, expansion de quelques actions prometteuses, rollout limite, puis backpropagation de la recompense. La recompense combine utilite, succes d'arrivee a la cible, penalite de longueur et penalite de tokens.

Stats a citer plus tard :

- Agriculture : MCTS prend environ 2.04 s de retrieval contre 0.22 s pour HiRAG classique.
- Mix : MCTS prend environ 0.96 s contre 0.16 s pour HiRAG classique.
- MCTS explore effectivement des chemins : environ 15 bridge edges en Agriculture et 12 en Mix.

Transition :

> Donc nos variantes changent vraiment le retrieval. La question suivante est : est-ce que cela ameliore la reponse finale ?

## Slide 17 - Benchmarks

Temps : 55 s.

Dire :

> Nous avons reconstruit les graphes sur deux datasets d'UltraDomain Benchmark : Agriculture et Mix. Tout le pipeline final utilise OpenAI : `gpt-5.4-mini` pour generation et juge, `text-embedding-3-small` pour embeddings. Cela evite les limites de notre backend vLLM local.

Stats a citer :

- Agriculture : 12 documents, 100 questions, 1 756 chunks, 23 224 nodes, 48 428 edges, 4 639 community reports.
- Mix : 61 documents, 130 questions, 579 chunks, 16 356 nodes, 25 960 edges, 2 695 community reports.

Point important :

> Meme si Agriculture a seulement 12 documents, ils sont tres longs : le nombre de chunks est beaucoup plus grand que le nombre de documents.

## Slide 18 - LLMaJ Agriculture

Temps : 55 s.

Dire :

> Sur Agriculture, les variantes graphe battent parfois HiRAG classique, ce qui est encourageant. Weighted et MCTS sont autour de 53-55% de win rate contre HiRAG. Mais naive RAG gagne tres clairement : 88% de win rate dans cette evaluation.

Stats a mettre en avant :

- `hi_weighted` : 55% vs HiRAG.
- `hi_mcts` : 53% vs HiRAG.
- `hi_min_budget` : 53% vs HiRAG.
- `naive` : 88% vs HiRAG.
- Latence : naive 0.51 s ; MCTS 2.04 s ; Minmax budgeted 5.02 s.
- Tokens : naive 6016 ; MCTS 4990.

Interpretation :

> Nos graph variants ameliorent le choix de chemins par rapport a HiRAG classique, mais naive garde plus d'evidence brute et gagne au niveau reponse finale.

## Slide 19 - LLMaJ Mix

Temps : 50 s.

Dire :

> Sur Mix, le resultat est similaire mais un peu moins extreme. Naive reste devant avec 67% de win rate contre HiRAG. MCTS est proche du baseline, autour de 48%, et Minmax budgeted tombe a 38%.

Stats a mettre en avant :

- `naive` : 67% vs HiRAG.
- `hi_mcts` : 48% vs HiRAG.
- `hi_min_budget` : 38% vs HiRAG.
- Latence : naive 0.41 s ; MCTS 0.96 s ; Minmax budgeted 2.43 s.
- Tokens : naive 6614 ; MCTS environ 4573 dans cette table.

Interpretation :

> Le graphe donne une structure, mais cette structure ne garantit pas que le prompt final contienne les passages les plus utiles.

## Slide 20 - Annotation humaine

Temps : 45 s.

Dire :

> Pour ne pas dependre seulement du LLM-as-a-judge, nous avons aussi fait une annotation humaine pairwise. Les annotateurs comparent deux reponses, disent laquelle ils preferent, si chaque reponse repond a la question, et surlignent le span utile.

Stats a citer :

- Naive vs HiRAG : 65% de win rate humain.
- Minmax budgeted vs HiRAG : 45%.
- MCTS vs HiRAG : 52.5%.
- Densite du span utile : HiRAG 28.7%, naive 32.0%, Minmax 32.1%, MCTS 30.4%.

Interpretation :

> MCTS est interessant : il est legerement prefere a HiRAG, mais pas assez fortement pour conclure qu'il gagne clairement.

## Slide 21 - Repond a la question + preferences humain/LLM

Temps : 50 s.

Dire :

> Cette slide est importante parce qu'elle montre une nuance. Les humains jugent que MCTS repond tres souvent a la question : 100%. HiRAG classique est aussi tres haut : 98.3%. Naive est un peu plus bas, 92.5%, mais naive est plus souvent preferee en head-to-head.

Comment expliquer :

> "Repond a la question" mesure surtout la validite minimale. La preference head-to-head capture aussi la precision, la completude, le style, et la presence d'evidence concrete. Une reponse peut repondre correctement, mais etre moins satisfaisante qu'une reponse naive plus detaillee ou plus directement ancree dans le texte source.

Point cle :

> MCTS semble robuste pour trouver une reponse acceptable, mais cela ne veut pas dire que son contexte final donne toujours la meilleure reponse preferee.

## Slide 22 - Annotation humaine + LLMaJ

Temps : 45 s.

Dire :

> Ici on compare humains et LLM-as-a-judge. Ils sont d'accord sur le fait que naive est fort, mais pas toujours sur les variantes graphe. Le LLM juge Minmax budgeted plus favorablement que les humains, alors que les humains donnent a MCTS un leger avantage mais le LLM est plus severe.

Stats a citer :

- Naive : humain 65%, LLMaJ strict 78.6%.
- Minmax budgeted : humain 45%, LLMaJ strict 61.5%.
- MCTS : humain 52.5%, LLMaJ strict 38.5%.

Message :

> LLM-as-a-judge est utile pour scaler l'evaluation, mais l'annotation humaine revele des preferences plus nuancees.

## Slide 23 - Decision breakdown

Temps : 35 s.

Dire :

> Ce graphique montre que les decisions ne sont pas seulement des victoires nettes. Il y a beaucoup de ties et de desaccords dans le juge LLM avec ordre inverse. C'est important : les differences entre variantes graphe sont souvent faibles, donc l'evaluation doit etre interpretee prudemment.

Si besoin d'expliquer "swapped-order disagree" :

> On juge chaque paire deux fois, en inversant l'ordre A/B. Si le LLM ne choisit pas le meme systeme apres inversion, on marque un desaccord d'ordre.

## Slide 24 - Span density vs human preference

Temps : 40 s.

Dire :

> Ce graphique relie la densite de span utile et la preference humaine. Naive a une densite de span utile plus haute et gagne nettement. MCTS est proche de HiRAG : il repond bien, mais ne donne pas toujours une reponse suffisamment plus utile ou plus dense pour etre preferee.

Interpretation :

> Ce n'est pas seulement la quantite de contexte qui compte, mais la proportion de contexte vraiment utile pour la reponse.

## Slide 25 - Composition des prompts

Temps : 1 min 10.

Dire :

> C'est probablement la slide la plus importante pour expliquer nos resultats. Naive donne presque uniquement du texte source direct au LLM. Les modes graphe donnent une grande part de contexte abstrait : entites, relations, chemins, community reports. C'est utile pour structurer, mais cela peut remplacer l'evidence textuelle exacte.

Stats a citer :

- Agriculture : naive = environ 100% source evidence ; HiRAG/MCTS/Minmax = environ 20-21% source evidence.
- Mix : naive = environ 100% source evidence ; HiRAG/MCTS/Minmax = environ 34% source evidence.
- Agriculture : naive utilise environ 6016 tokens ; HiRAG environ 5040 ; MCTS environ 4990.
- Mix : naive environ 6614 tokens ; HiRAG environ 3174.

These :

> Nos variantes ameliorent le parcours dans le graphe, mais le prompt final ne preserve pas assez l'evidence source. Pour des questions QA, le LLM a besoin de passages exacts, pas seulement de resumes ou de descriptions d'entites.

Phrase forte :

> Le bottleneck n'est pas seulement retrieval ; c'est retrieval-to-prompt assembly.

## Slide 26 - Conclusion propre

Temps : 1 min.

Dire :

> Notre conclusion est en trois points. Premierement, notre strategie de snippets query-aware reduit le bruit, mais elle peut etre trop agressive : selectionner une seule fenetre par chunk peut enlever de l'evidence utile. Deuxiemement, nos methodes de traversal, surtout MCTS et weighted, battent parfois HiRAG classique, donc l'idee de query-aware bridge est prometteuse. Troisiemement, un graphe seul ne suffit pas a repondre parfaitement a la question, parce que le graphe contient surtout des representations resumees.

Phrase finale :

> La prochaine etape serait un HiRAG evidence-aware : utiliser le graphe pour trouver les bons chemins, puis reinjecter les meilleurs extraits sources exacts associes a ces chemins.

## Slide 27 - Conclusion detaillee / backup

Temps : 30 s si gardee, sinon backup.

Avis : cette slide est trop dense et visuellement difficile a lire. Pour une presentation de 15 minutes, elle devrait plutot etre une backup slide ou etre remplacee par une conclusion plus simple.

Si vous la gardez, dire seulement :

> MCTS peut trouver un chemin plus pertinent, mais si le prompt final donne surtout du contexte abstrait et pas les preuves textuelles exactes, la reponse finale ne s'ameliore pas forcement. C'est pour cela que naive RAG peut sembler meilleur : il donne moins de structure, mais plus de preuves.

Ne pas lire tout le texte.

## Statistiques A Memoriser

- Agriculture graph : 23 224 nodes, 48 428 edges, 4 639 community reports.
- Mix graph : 16 356 nodes, 25 960 edges, 2 695 community reports.
- LLMaJ Agriculture : naive 88%, weighted 55%, MCTS 53%, Minmax budgeted 53%.
- LLMaJ Mix : naive 67%, MCTS 48%, weighted 40%, Minmax budgeted 38%.
- Human annotation : naive 65%, MCTS 52.5%, Minmax budgeted 45%.
- "Repond a la question" humain : MCTS 100%, HiRAG 98.3%, Minmax 97.5%, naive 92.5%.
- Source evidence share : Agriculture graph modes 20-21% vs naive 100%; Mix graph modes ~34% vs naive 100%.
- Retrieval latency : MCTS plus cher que HiRAG classique, mais encore raisonnable ; Minmax budgeted est le plus lent dans nos variantes principales.

## Histoire En Une Minute

> HiRAG propose de remplacer le retrieval de chunks par une recherche structuree dans un graphe hierarchique. Nous avons reproduit le pipeline avec OpenAI, puis attaque trois limites : le bruit dans les chunks sources, le manque de reranking des entites, et le bridge non query-aware. Nous avons implemente weighted Dijkstra, Minmax, Minmax budgeted et MCTS. Les resultats montrent que ces variantes changent vraiment la recherche dans le graphe et peuvent battre HiRAG classique, surtout weighted et MCTS. Mais naive RAG reste tres fort, notamment parce qu'il conserve beaucoup plus d'evidence source exacte dans le prompt final. Notre conclusion est donc que le futur de HiRAG n'est pas seulement un meilleur graph traversal : c'est un meilleur assemblage entre structure graphe et snippets sources exacts.

## Questions Probables Et Reponses Courtes

### Pourquoi naive gagne ?

Parce qu'il donne plus d'evidence textuelle exacte au LLM. Les graph modes structurent mieux, mais compressent beaucoup l'information en entites, relations et summaries.

### Est-ce que MCTS est un echec ?

Non. MCTS est encourageant : il repond tres souvent a la question et obtient 52.5% de preference humaine contre HiRAG. Mais il ne suffit pas seul ; il faut ameliorer la composition du contexte final.

### Pourquoi "repond a la question" peut etre haut mais preference basse ?

Parce qu'une reponse peut etre correcte mais moins complete, moins precise, ou moins bien justifiee. La preference compare la qualite relative, pas seulement la validite minimale.

### Est-ce comparable au papier original ?

C'est une reproduction homogene OpenAI, pas exactement les memes modeles que le papier. Mais les comparaisons internes sont fair : meme generation model, meme judge model, memes datasets, seules les strategies de retrieval changent.

### Quelle est la prochaine amelioration ?

Un HiRAG evidence-aware : traversal dynamique du graphe + selection explicite de snippets sources exacts lies aux entites, chemins et communautes recuperes.
