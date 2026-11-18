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

J02 is afgerond. Ik heb `bigram_logits`, `cross_entropy`, `train_bigram` en `generate_bigram` in `src/jarvis/bigram_torch.py` gebouwd. Ik heb de volgende controles uitgevoerd en de uitkomsten bevestigd.

| Onderdeel | Resultaat |
| --- | --- |
| J02.1 — Omgeving | Ik kon `jarvis`, PyTorch en NumPy importeren met Python 3.12.10. Mijn omgeving gebruikte PyTorch 2.15.0+cpu en NumPy 2.5.5. De tensor `[1, 2, 1]` had vorm `(3,)` en stond op CPU. |
| J02.1 — Logits | Voor invoer `[1, 2, 1]` en de vaste gewichtentabel kreeg ik `[[0, 1, 2], [0, 3, 4], [0, 1, 2]]`, met vorm `(3, 3)`. De uitvoer was exact gelijk aan NumPy. |
| J02.2 — Cross-entropy | Voor logits `[[0, ln(2), 0], [0, 0, 0]]` en doelen `[1, 2]` waren de NumPy-loss, PyTorch-loss en handberekening alle drie `0.895880` op zes decimalen. De vergelijkingen slaagden met absolute tolerantie `1e-12`. De PyTorch-loss was een scalaire tensor met vorm `()`. |
| J02.3 — Autograd | Met nulgewichten, invoer `[1, 1]` en doelen `[1, 2]` kreeg ik loss `1.098612` en rijgradiënt `[1/3, -1/6, -1/6]`. De gradiënten kwamen overeen met de handberekening en NumPy; de gewichten bleven nul. |
| J02.4 — Training | Op `abab`, met leersnelheid `0.5` en 30 stappen, daalde de loss van `1.098612` naar `0.103528`. De voorspelde ID’s waren `[2, 1, 2]`. De gewichten kwamen overeen met NumPy binnen absolute tolerantie `1e-12`; de begintabel bleef nul. |
| J02.4 — Generatie | Vanaf `a`, met seed `42` en acht nieuwe tokens, kreeg ik tweemaal `ab�bababa`. De lengte, geldige ID’s, nul nieuwe tokens en het intact blijven van de gewichten zijn gecontroleerd. `�` is het getrokken onbekende token met ID `0`. |

## Datasetkeuze — Shakespeare

Ik gebruik het aangeleverde bestand `Shakespear.txt` als `data/shakespeare.txt`. Ik heb bevestigd dat het UTF-8-bestand leesbaar is, begint met `First Citizen:` en 1.115.394 tekens en 65 unieke tekens bevat. De oorspronkelijke herkomst van het aangeleverde bestand is nog niet bevestigd.

Voor mijn eerste CPU-proef heb ik de eerste 10.000 tekens geselecteerd en gesplitst na 9.000 tekens. Ik kreeg 8.999 trainingsparen en 999 validatieparen. De tokenizer is uitsluitend op de trainingstekst gebouwd en heeft 57 tokens inclusief ID `0`. De letter `J` komt eenmaal in validatie voor en ontbreekt in training; die wordt ID `0`. Alle splitcontroles slaagden.

### J02.5 — Training en generatie op Shakespeare

Ik heb beide varianten getraind met nulgewichten, leersnelheid `5.0`, 100 stappen en alle 8.999 trainingsparen per stap. Ik gebruikte `float64` en CPU.

| Meting | NumPy | PyTorch |
| --- | --- | --- |
| Beginloss bij gelijke tokenkansen | `4.043051` | `4.043051` (dezelfde nul-logits) |
| Trainingsloss na 100 stappen | `2.884268` | `2.884268` |
| Validatieloss na 100 stappen | `2.991439` | `2.991439` |

Ik heb bevestigd dat de geleerde gewichten en beide losses overeenkomen binnen absolute tolerantie `1e-12`, en dat de trainingsloss daalt.

De controle staat in `tests/test_shakespeare_bigram.py`. Ik heb die test uitgevoerd met `unittest`: één test eindigde met `OK`. De getoonde duur was 1,381 seconde voor de hele test, inclusief beide trainingen, evaluatie en generatie; dit is geen afzonderlijke trainingsbenchmark.

Ik genereerde per variant 200 nieuwe tokens vanaf `F`, met seed `42`. Beide reeksen bevatten inclusief start 201 tokens en waren per variant herhaalbaar. Ook geldige ID’s, nul generatiestappen en het intact blijven van de gewichten zijn gecontroleerd. De teksten verschillen tussen de varianten doordat hun toevalsgeneratoren anders trekken. Dit zijn fragmenten van mijn uitvoer:

```text
NumPy:   FmWrg!you, fverYCc'renRypoFV ShevSMTA.PBherhatiick.,
PyTorch: FRIEjaBhERurPkdRdCojbHElyi;dd qIWEpere cIW pecOcoNxE'
```

De tekst bevat korte herkenbare patronen, maar is grotendeels onsamenhangend. Het bigrammodel gebruikt alleen het laatste teken als context. Een eventueel `�` in de generatie betekent dat token-ID `0` is getrokken.

## J03 — NumPy-Transformer in opbouw

Mijn bouwstenen staan in `src/jarvis/transformer_numpy.py`.

| Onderdeel | Door mij bevestigde stand |
| --- | --- |
| J03.1 — Token-embeddings | De test met twee reeksen van drie tokens en vectorlengte twee slaagde. De uitvoer had vorm `(2, 3, 2)` en alle geselecteerde vectoren klopten. |
| J03.1 — Positie-embeddings | Beide tests in `tests/test_embeddings_numpy.py` eindigden met `OK`. De juiste positierijen werden bij iedere reeks opgeteld; de extra vierde positierij werd niet gebruikt en de oorspronkelijke tokenvectoren bleven intact. |
| J03.2 — Q/K/V-projecties | Ik heb bevestigd dat de projectietest werkt. Voor invoer `[[1, 2], [3, 4]]` kreeg ik queries `[[1, 2], [3, 4]]`, keys `[[2, 1], [4, 3]]` en values `[[2, 6], [6, 12]]`, ieder met batchvorm `(1, 2, 2)`. |
| J03.3 — Aandachtsscores | `attention_scores` en `test_scaled_query_key_scores` staan in de broncode. Ik heb de testuitkomst nog niet bevestigd; dit onderdeel blijft open. |

## Stoppunt en volgende stap

J01 en J02 zijn afgerond. Bij J03 zijn embeddings en Q/K/V-projecties bevestigd. Ik hervat bij **J03.3** door vanuit de projecthoofdmap deze controle uit te voeren:

```powershell
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -p "test_attention_numpy.py" -v
```

Ik verwacht twee tests en `OK`, maar leg dit pas als geslaagd vast nadat ik die uitvoer heb bevestigd. Daarna volgt **J03.4**: het causale masker, softmax over aandachtsscores en het mengen van values. De volledige Transformer en zijn gradiënten zijn nog niet gebouwd.

Ik voeg nieuwe implementatie- en testcode zelf toe. Ik werk met kleine stappen: eerst theorie en een handvoorbeeld, dan code met de exacte invoegplek en een gerichte controle. Ik ga verder zodra ik de uitkomst heb bevestigd. Bestanden laat ik alleen wijzigen wanneer ik daar expliciet om vraag.

Mijn actuele branch, commits en GitHub-status zijn nog niet bevestigd. De voorgestelde Git-commando’s zijn geen bewijs dat ik ze heb uitgevoerd.

De theorie en het overzicht van de modelonderdelen staan in [Theorie van het taalmodel](theorie.md).
