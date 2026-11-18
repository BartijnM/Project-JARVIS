# Roadmap

Ik vink een taak af zodra ik de bijbehorende controle heb uitgevoerd en het resultaat in [voortgang.md](voortgang.md) heb vastgelegd.

## J01 — NumPy-bigrammodel

- [x] **J01.1** Maak een tokenizer op tekenniveau. Controleer of bekende tekens via encode en decode terugkomen en of een onbekend teken de afgesproken token-ID krijgt.
- [x] **J01.2** Maak trainings- en validatieparen. Controleer op een korte tekst of ieder doel één teken opschuift en ieder paar binnen zijn eigen split blijft.
- [x] **J01.3** Bouw het NumPy-bigrammodel en bereken cross-entropy. Controleer de uitvoervorm en loss met een klein handmatig voorbeeld.
- [x] **J01.4** Schrijf de gradiënt en trainingslus. Vergelijk de gradiënt met eindige verschillen, controleer dat de trainingsloss daalt op een klein vast voorbeeld en genereer tekst met een vaste seed.

## J02 — PyTorch-bigrammodel

- [x] **J02.1** Controleer de PyTorch-omgeving en vergelijk de tabelindexering en logits met NumPy op identieke invoer en gewichten.
- [x] **J02.2** Bereken cross-entropy met PyTorch en vergelijk de loss met NumPy en een handberekening.
- [ ] **J02.3** Bereken de gradiënten met autograd en vergelijk ze met de handmatige NumPy-gradiënten, ook bij herhaalde invoer-ID’s.
- [ ] **J02.4** Bouw training en generatie. Vergelijk de training op hetzelfde kleine voorbeeld als bij J01.
- [ ] **J02.5** Bereid het gedeelde Shakespeare-corpus voor: kopieer het naar `data/shakespeare.txt`, controleer UTF-8-invoer, leg een eventueel CPU-proefdeel en de splitgrens vast en bouw de tokenizer op de trainingssplit. Train beide bigramvarianten op dezelfde tekst en leg trainingsloss, validatieloss en voorbeeldgeneraties vast.

## J03 — NumPy-Transformer

- [ ] Bouw token- en positie-embeddings. Controleer de uitvoervormen op een kleine batch.
- [ ] Bouw causale aandacht. Controleer dat het wijzigen van een toekomstig token de voorspellingen op eerdere posities niet verandert.
- [ ] Voeg de feed-forwardlaag, normalisatie en restverbindingen toe. Controleer de vormen en vergelijk gradiënten met eindige verschillen.
- [ ] Train het volledige model op CPU. Leg trainingsloss, validatieloss en een voorbeeldgeneratie vast.

## J04 — PyTorch-Transformer

- [ ] Bouw dezelfde Transformer met PyTorch. Vergelijk uitvoer en loss op vaste invoer en gewichten. Train het model daarna op dezelfde dataset.

## J05 — Vergelijking

- [ ] Leg per variant de instellingen, CPU-trainingsduur, validatieloss en voorbeeldgeneraties vast.

Mijn gedeelde corpus is Shakespeare-tekst uit `data/shakespeare.txt`. Het doel is Shakespeare-achtige tekst te genereren. Ik vergelijk modellen met dezelfde tekstselectie, tokenizer, datasplits, contextlengte en meetmethode. Ik leg seeds vast zodat ik runs per variant kan herhalen.
