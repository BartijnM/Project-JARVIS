# Roadmap

Ik vink een taak af zodra ik de bijbehorende controle heb uitgevoerd en het resultaat in [voortgang.md](voortgang.md) heb vastgelegd.

## J01 — NumPy-bigrammodel

- [x] **J01.1** Maak een tokenizer op tekenniveau. Controleer of bekende tekens via encode en decode terugkomen en of een onbekend teken de afgesproken token-ID krijgt.
- [x] **J01.2** Maak trainings- en validatieparen. Controleer op een korte tekst of ieder doel één teken opschuift en ieder paar binnen zijn eigen split blijft.
- [x] **J01.3** Bouw het NumPy-bigrammodel en bereken cross-entropy. Controleer de uitvoervorm en loss met een klein handmatig voorbeeld.
- [x] **J01.4** Schrijf de gradiënt en trainingslus. Vergelijk de gradiënt met eindige verschillen, controleer dat de trainingsloss daalt op een klein vast voorbeeld en genereer tekst met een vaste seed.

## J02 — PyTorch-bigrammodel

- [ ] Bouw hetzelfde bigrammodel met autograd. Vergelijk de logits en loss op dezelfde invoer en gewichten met NumPy. Train daarna op dezelfde tekst.

## J03 — NumPy-Transformer

- [ ] Bouw token- en positie-embeddings. Controleer de uitvoervormen op een kleine batch.
- [ ] Bouw causale aandacht. Controleer dat het wijzigen van een toekomstig token de voorspellingen op eerdere posities niet verandert.
- [ ] Voeg de feed-forwardlaag, normalisatie en restverbindingen toe. Controleer de vormen en vergelijk gradiënten met eindige verschillen.
- [ ] Train het volledige model op CPU. Leg trainingsloss, validatieloss en een voorbeeldgeneratie vast.

## J04 — PyTorch-Transformer

- [ ] Bouw dezelfde Transformer met PyTorch. Vergelijk uitvoer en loss op vaste invoer en gewichten. Train het model daarna op dezelfde dataset.

## J05 — Vergelijking

- [ ] Leg per variant de instellingen, CPU-trainingsduur, validatieloss en voorbeeldgeneraties vast.

Ik vergelijk modellen met dezelfde tokenizer, datasplits, contextlengte en meetmethode. Ik leg seeds vast zodat ik runs per variant kan herhalen.
