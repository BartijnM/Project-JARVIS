# Project JARVIS

In Project JARVIS bouw ik vanaf nul een klein taalmodel dat tekst leert aanvullen. Ik maak eerst een NumPy-variant en daarna een PyTorch-variant. Mijn doel is Shakespeare-achtige tekst te genereren. Beide varianten gebruiken dezelfde Shakespeare-tekst, tokenizer en evaluatie, zodat ik de resultaten kan vergelijken.

| Variant | Wat ik zelf bouw |
| --- | --- |
| NumPy | De modelberekeningen, de gradiënten en de updates van de gewichten. NumPy is de enige externe runtime-afhankelijkheid. |
| PyTorch | Dezelfde modelarchitectuur. PyTorch berekent de gradiënten met autograd. |

## Lokaal starten

Ik gebruik Python 3.12 voor beide varianten. Ik voer de commando’s uit vanuit de hoofdmap van de repo en maak voor iedere variant een aparte virtuele omgeving. Mijn bevestigde PyTorch-omgeving gebruikt Python 3.12.10, PyTorch 2.15.0+cpu en NumPy 2.5.5.

Als Python 3.12 nog ontbreekt, installeer ik die met `py install 3.12`.

**NumPy (PowerShell)**

```powershell
py -3.12 -m venv .venv-numpy
.\.venv-numpy\Scripts\python.exe -m pip install -e '.[numpy]'
.\.venv-numpy\Scripts\python.exe -c "import jarvis, numpy"
```

**PyTorch (PowerShell)**

```powershell
py -3.12 -m venv .venv-torch
.\.venv-torch\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv-torch\Scripts\python.exe -m pip install -e '.[numpy,torch]'
.\.venv-torch\Scripts\python.exe -c "import jarvis, torch, numpy"
```

De importcontrole controleert de installatie. De eerste modelmijlpaal is een bigrammodel op CPU. Het leert welk teken waarschijnlijk volgt op het vorige teken en genereert daarna korte tekst. J01 en J02 zijn afgerond: beide bigrammodellen trainen en genereren tekst op CPU. Op hetzelfde Shakespeare-proefdeel heb ik gelijke gewichten, trainingsloss en validatieloss bevestigd. Bij J03 heb ik token- en positie-embeddings en de query-, key- en valueprojecties gecontroleerd. De aandachtsscores en hun test staan in de broncode; de testuitkomst heb ik nog niet bevestigd. Mijn volgende stap is die test draaien, daarna volgt het causale masker.

## Trainingsdata

Ik gebruik het aangeleverde bestand `Shakespear.txt` als Shakespeare-corpus. Het bestand staat als UTF-8-tekst in `data/shakespeare.txt`. Ik heb het inlezen bevestigd: 1.115.394 tekens en 65 unieke tekens. Bij het lezen worden Windows-regelafbrekingen naar `\n` omgezet.

Als het corpus bij een nieuwe lokale installatie nog ontbreekt, kan ik het vanuit Downloads kopiëren. Ik gebruik binnen het project de bestandsnaam `shakespeare.txt`:

```powershell
New-Item -ItemType Directory -Path .\data -Force | Out-Null
Copy-Item -LiteralPath "$env:USERPROFILE\Downloads\Shakespear.txt" -Destination .\data\shakespeare.txt
```

Ik gebruik één bronbestand voor NumPy en PyTorch. Ik bewaar hoofdletters, leestekens en regelafbrekingen. Ik splits de tekst vóór tokenisatie in training en validatie en bouw de tokenizer uitsluitend op de trainingssplit. Voor mijn bevestigde bigramproef gebruik ik de eerste 10.000 tekens: 9.000 voor training en 1.000 voor validatie. Dat geeft 8.999 trainingsparen en 999 validatieparen. Mijn tokenizer heeft 57 tokens inclusief ID `0`. De letter `J` komt eenmaal uitsluitend in validatie voor en krijgt daar ID `0`. Beide varianten krijgen dezelfde selectie en splits.

De kleine voorbeelden zoals `abab` blijven mijn handmatige controles. Met nulgewichten, leersnelheid `5.0` en 100 stappen daalde de trainingsloss van `4.043051` naar `2.884268`; de validatieloss was `2.991439` in beide varianten. Ik heb per variant herhaalbare generatie vanaf `F` met seed `42` en 200 nieuwe tokens bevestigd. De map `data/` is al uitgesloten in `.gitignore`; het corpus wordt daardoor niet standaard opgenomen in versiebeheer. De oorspronkelijke herkomst van het aangeleverde bestand is nog niet bevestigd.

## Controles opnieuw uitvoeren

Ik bewaar nieuwe controles in testbestanden en voer ze vanuit de projecthoofdmap uit met mijn PyTorch-omgeving, die ook NumPy bevat:

```powershell
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -p "test_shakespeare_bigram.py" -v
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -p "test_embeddings_numpy.py" -v
.\.venv-torch\Scripts\python.exe -m unittest discover -s tests -p "test_attention_numpy.py" -v
```

De Shakespeare-test en beide embeddingtests heb ik bevestigd. Bij aandacht heb ik de projectietest bevestigd; de toegevoegde scoretest is mijn eerstvolgende controle. Meer uitleg staat in [tests/README.md](tests/README.md).

## Projectdocumentatie

- [Architectuur](docs/architectuur.md)
- [Roadmap](docs/roadmap.md)
- [Voortgang](docs/voortgang.md)
- [Theorie van het taalmodel](docs/theorie.md)

De twee video’s zijn startreferenties voor de NumPy-route. [Video 1](https://www.youtube.com/watch?v=s9w3gtgvNSU) · [Video 2](https://www.youtube.com/watch?v=vKdKtxpfvIc)
