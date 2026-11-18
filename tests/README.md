# Tests

Ik bewaar nieuwe controles in testbestanden, zodat ik dezelfde controle opnieuw kan uitvoeren zonder steeds een lang terminalblok te plakken. Ik gebruik `unittest` uit de standaardbibliotheek. Mijn PyTorch-omgeving bevat ook NumPy en kan daarom alle huidige testbestanden importeren.

Ik voer de volgende opdrachten uit vanuit de projecthoofdmap.

## J02.5 — Shakespeare-bigrammodellen

```powershell
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -p "test_shakespeare_bigram.py" -v
```

Ik heb deze test bevestigd met `OK`. Hij leest `data/shakespeare.txt`, selecteert 10.000 tekens, splitst na 9.000 tekens en traint beide bigrammodellen vanaf nulgewichten met leersnelheid `5.0` en 100 stappen. De test vergelijkt gewichten en losses binnen absolute tolerantie `1e-12`, controleert dat de trainingsloss daalt en de begintabellen intact blijven, en toont de losses en twee tekstgeneraties. Per variant controleert hij herhaalbaarheid met seed `42`, starttoken, lengte, geldige ID’s, nul generatiestappen en het intact blijven van de gewichten. Bij een ontbrekend databestand moet ik eerst het corpus op de afgesproken locatie zetten.

## J03.1 — Embeddings

```powershell
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -p "test_embeddings_numpy.py" -v
```

Ik heb beide tests bevestigd met `OK`: tokenrijen worden correct opgehaald en de juiste positievectoren worden per reeks opgeteld. De tests controleren ook de uitvoervorm en het intact blijven van de tokenvectoren.

## J03.2 en J03.3 — Aandacht in opbouw

```powershell
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -p "test_attention_numpy.py" -v
```

Ik heb de Q/K/V-projectietest bevestigd. Inmiddels bevat dit bestand ook `test_scaled_query_key_scores`. De eerstvolgende controle is beide tests draaien en hun uitkomst bevestigen. De scoretest verwacht de matrix `[[4, 10], [10, 24]] / sqrt(2)`, met batchvorm `(1, 2, 2)`. Het causale masker is nog niet gebouwd of getest.

## Alle aanwezige tests

```powershell
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -v
```

De huidige bestanden bevatten samen vijf testmethoden. Ik heb deze volledige gecombineerde run nog niet bevestigd. Een afzonderlijke NumPy-omgeving kan ik voor de NumPy-tests gebruiken door gericht `test_embeddings_numpy.py` of `test_attention_numpy.py` te selecteren; de Shakespeare-vergelijking importeert beide bibliotheken.

De eerdere kleine J01-controles en eerste J02-controles zijn door mij via terminalfragmenten bevestigd. Ze staan nog niet allemaal als afzonderlijke tests in `tests/`.
