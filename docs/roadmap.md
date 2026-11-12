# Roadmap

| Stap | Resultaat | Controle |
| --- | --- | --- |
| J01.1 | Gedeelde tokenizer op tekenniveau | Een tekst met bekende tekens doorloopt encode en decode. Een onbekend teken krijgt de afgesproken token-ID. |
| J01.2 | Trainings- en validatieparen | Een klein voorbeeld laat zien dat ieder doel precies één teken opschuift en dat paren de splitgrens niet kruisen. |
| J01.3 | NumPy-bigram en cross-entropy | Controleer uitvoervormen en loss met een klein handmatig voorbeeld. |
| J01.4 | NumPy-training en generatie | Controleer de gradiënt, train op CPU en laat de loss dalen op een klein vast trainingsvoorbeeld. Genereer daarna tekst met een vaste seed. |
| J02 | PyTorch-bigram | Train op dezelfde tekst. Vergelijk voor vaste gewichten en invoer de logits en loss met NumPy. |
| J03 | NumPy-Transformer | Voeg de bouwstenen één voor één toe. Controleer vormen, het causale masker en gradiënten vóór training. |
| J04 | PyTorch-Transformer | Gebruik dezelfde modelarchitectuur. Controleer de uitvoer op vaste voorbeelden en train het model. |
| J05 | Vergelijking | Leg per variant instellingen, CPU-trainingsduur, validatieloss en voorbeeldgeneraties vast. |

Ik vergelijk modellen met dezelfde tokenizer, datasplits, contextlengte en meetmethode. Ik leg seeds vast zodat ik runs per variant kan herhalen.
