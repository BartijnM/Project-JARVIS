# Voortgang

## J01 — NumPy-bigrammodel

Ik heb de tokenizer, de gedeelde functies voor tekstsplits en tokenparen, en een NumPy-bigrammodel op CPU gebouwd. Ik heb de volgende controles uitgevoerd.

| Onderdeel | Resultaat |
| --- | --- |
| J01.1 — Tokenizer | Met `baba` als trainingstekst gaf `ab!a` de ID’s `[1, 2, 0, 1]`. Ik kreeg bekende tekens exact terug en zag `�` op de plek van het onbekende teken. |
| J01.2 — Tokenparen | Op `abacbc` met de grens op index 4 kreeg ik voor training `([1, 2, 1], [2, 1, 3])` en voor validatie `([2], [3])`. Het grensoverschrijdende paar `3 → 2` kwam in geen van beide voor. |
| J01.3 — Logits en loss | Ik kreeg voor invoer `[1, 2, 1]` de verwachte drie rijen in een array met vorm `(3, 3)`. De cross-entropy was `0.895880` voor het handmatig doorgerekende voorbeeld. |
| J01.4 — Gradiënt | Voor twee voorbeelden met invoer-ID `1` kreeg ik voor rij 1 ongeveer `[0.33333333, -0.16666667, -0.16666667]`. Voor gewicht `[1, 1]` gaven eindige verschillen `-0.166667` op zes decimalen. |
| J01.4 — Training | Op `abab` daalde de trainingsloss na 30 stappen met leersnelheid `0.5` van `1.098612` naar `0.103528`. De voorspelde doel-ID’s waren `[2, 1, 2]`. |
| J01.4 — Generatie | Vanaf `a` genereerde ik met seed `42` tweemaal acht nieuwe tekens. Beide keren kreeg ik inclusief het startteken `abababbab`. |

## J02 — PyTorch-bigrammodel

Ik heb `bigram_logits` en `cross_entropy` in `src/jarvis/bigram_torch.py` toegevoegd. Ik heb de volgende controles uitgevoerd en de uitkomsten bevestigd.

| Onderdeel | Resultaat |
| --- | --- |
| J02.1 — Omgeving | Ik kon `jarvis`, PyTorch en NumPy importeren met Python 3.12.10. Mijn omgeving gebruikte PyTorch 2.15.0+cpu en NumPy 2.5.5. De tensor `[1, 2, 1]` had vorm `(3,)` en stond op CPU. |
| J02.1 — Logits | Voor invoer `[1, 2, 1]` en de vaste gewichtentabel kreeg ik `[[0, 1, 2], [0, 3, 4], [0, 1, 2]]`, met vorm `(3, 3)`. De uitvoer was exact gelijk aan NumPy. |
| J02.2 — Cross-entropy | Voor logits `[[0, ln(2), 0], [0, 0, 0]]` en doelen `[1, 2]` waren de NumPy-loss, PyTorch-loss en handberekening alle drie `0.895880` op zes decimalen. De vergelijkingen slaagden met absolute tolerantie `1e-12`. De PyTorch-loss was een scalaire tensor met vorm `()`. |

## Datasetkeuze — Shakespeare

Ik wil de modellen trainen om Shakespeare-achtige tekst te genereren. Ik heb daarvoor `Shakespear.txt` aangeleverd vanuit Downloads. Ik kies `data/shakespeare.txt` als vaste locatie voor het gedeelde corpus. De kopie naar die locatie, het inlezen, de tekstselectie, de splitgrens en training op dit corpus zijn nog niet door mij uitgevoerd en bevestigd. De oorspronkelijke herkomst van het bestand is nog niet bevestigd.

## Volgende stap

Bij **J02.3** bereken ik de gradiënten met autograd en vergelijk ik ze met de handmatige NumPy-implementatie. Daarna volgen training en generatie op het kleine controlevoorbeeld. Bij **J02.5** bereid ik het gedeelde Shakespeare-corpus voor en train ik beide bigramvarianten daarop. J02 is nog niet afgerond.

De theorie en het overzicht van de modelonderdelen staan in [Theorie van het taalmodel](theorie.md).
