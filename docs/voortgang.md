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

## Volgende stap

Bij **J02** bouw ik hetzelfde bigrammodel met PyTorch en autograd. Ik vergelijk eerst de logits en loss met NumPy op dezelfde invoer en gewichten. Daarna train ik op dezelfde tekst.

De theorie en het overzicht van de modelonderdelen staan in [Theorie van het taalmodel](theorie.md).
