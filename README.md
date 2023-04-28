# Frozen_lake

La mise à jour de ses valeurs internes tient compte de valeur `Q(s', a')`, `Q(s, a)` et de la récompense `r`. On dit que SARSA est un algorithme **on-policy** car la mise à jour tient compte en particulier de `Q(s', a')`, où `a'` a été choisi par la politique en cours d'apprentissage.

## Le taux d'apprentissage

`Q[s, a] = (1 - alpha) * Q[s, a] + alpha * (reward + gamma * Q[s_, a_])`

Dans l'expression précédente, α est le facteur d'apprentissage qui exprime un compromis (une combinaison linéaire) entre :

- **(α = 0)** rien apprendre du tout, i.e. `Q[s, a] = Q[s, a]`
- **(α = 1)** apprendre en oubliant ce que l'on a déjà appris, i.e. `Q[s, a] = r + γQ(s', a')`
L'expression `r + γQ(s', a')` représente la valeur de récompense courante r, à laquelle on additionne la valeur d'être en s' et d'exécuter a', pondérée par γ.

Après avoir effectué un mouvement pendant l'apprentissage, la valeur Q pour un état et une action donnés est remplacée par la nouvelle valeur.

La nouvelle valeur est une somme de deux parties. La première partie est (1-taux d'apprentissage)*ancienne valeur. C'est la part de l'ancienne valeur que nous conservons. Un taux d'apprentissage de 0 signifie que rien de nouveau ne sera appris. Un taux d'apprentissage de 1 signifie que l'ancienne valeur sera complètement supprimée.

La deuxième partie est le taux d'apprentissage * (récompense immédiate pour l'action + estimation actualisée de la valeur future optimale). Le taux d'apprentissage, comme expliqué ci-dessus, détermine la quantité de la nouvelle valeur apprise qui sera utilisée. La valeur apprise est la somme de la récompense immédiate et de l'estimation actualisée de la valeur future optimale. Le facteur de remise détermine l'importance des récompenses futures. Lorsqu'il est défini sur 0, nous ne prendrons en compte que les récompenses immédiates et 1 obligera l'algorithme à les prendre en charge dans leur intégralité.

## Le facteur epsilon

Il y a aussi la notion d'exploration. Peut-être que lors des premiers essais, l'algorithme trouve une action particulière pour un état donné gratifiante. S'il continue de sélectionner l'action de récompense maximale tout le temps, il continuera à effectuer la même action et n'essaiera rien d'autre et peut-être qu'une autre action non essayée aura une meilleure récompense que celle-ci.

Pour équilibrer exploration vs exploitation, nous varierons epsilon tout au long de l'entraînement. Nous allons commencer avec epsilon=1 (exploration pure) et décliner epsilon à chaque épisode pour passer progressivement de l'exploration pure à l'exploitation.

## Attention : 

Notez bien que nous avons des cas dans lesquels le joueur se déplace dans une direction différente de celle choisie par l'agent. Ce comportement est tout à fait normal dans l'environnement de Frozen Lake car il simule une surface glissante. De plus, ce comportement représente une caractéristique importante des environnements réels : les transitions d'un état à un autre, pour une action donnée, sont probabilistes. Par exemple, si nous tirons avec un arc et des flèches, il y a une chance de toucher la cible aussi bien que de la manquer. La répartition entre ces deux possibilités dépendra de notre habileté et d'autres facteurs, comme la direction du vent, par exemple. En raison de cette nature probabiliste, le résultat final d'une transition d'état ne dépend pas entièrement de l'action entreprise.

Par défaut, l'environnement Frozen Lake fourni dans Gym a des transitions probabilistes entre les états. En d'autres termes, même lorsque notre agent choisit de se déplacer dans une direction, l'environnement peut exécuter un mouvement dans une autre direction.

Cependant, l'environnement peut également utilisé en mode déterministe en modidfiant la propriété is_slippery pour false pendant la création de l'environnement. Ainsi la surface n'est plus glissante et les actions sont correctement exécutées.

## Pré-requis :

Pour exécuter les scripts sur le dépôt, vous devez avoir installé :

- **Python** : [Cliquez ici](https://www.python.org/downloads/)

- [**Gymnasium**](https://gymnasium.farama.org/) : `python3 -m pip install gymnasium`

- [**Toy_text**](https://gymnasium.farama.org/environments/toy_text/frozen_lake/) : `python3 -m pip install "gymnasium\[toy_text\]"`

## Exécution

Après avoir cloné le dépôt, vous obtenez 6 fichiers :

- **frozen.py** : Un fichier python contenant le code dans une version où l'environnement n'a pas une surface glissante : (*is_slippery = False*) avec is_slippery = False;

- **save_frozen_lake.pkl** : Un fichier contenant des données binaires correspondant à la Q_table obtenue après un entraînement sur *500000 épisodes*;

- **frozen_slip.py** : Un fichier python contenant le code dans une version où l'environnement a une surface glissante : (*is_slippery = True*);

- **save_frozen_lake_slip.pkl** : Un fichier contenant des données binaires correspondant à la Q_table obtenue après un entraînement sur *500000 épisodes* avec is_slippery = True;

- **frozen_lake_1: Un fichier contenant des données binaires correspondant à la Q_table obtenue après un entraînement sur *800000 épisodes* avec is_slippery = True;

- **main.py : Un fichier python contenant le code daans une version où l'environnement a une surface non glissante : (*is_slippery = False*) mais utilisation de fonctions 

Ensuite vous ouvrez un terminal dans le dossier contenant les fichiers puis vous changez les droit d'exécution des scripts en faisant : 

**chmod u+x frozen_slip.py** 

**chmod u+x frozen.py**

**chmod u+x main.py**

Enfin, vous pouvez lançer les scripts directement dans le terminal.
