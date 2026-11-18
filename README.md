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

De importcontrole controleert de installatie. De eerste modelmijlpaal is een bigrammodel op CPU. Het leert welk teken waarschijnlijk volgt op het vorige teken en genereert daarna korte tekst. Het NumPy-bigrammodel is gebouwd. Bij J02 heb ik de PyTorch-logits en cross-entropy met NumPy vergeleken en de controles bevestigd. De volgende stap is autograd; daarna volgen training en generatie.

## Trainingsdata

Ik gebruik het aangeleverde bestand `Shakespear.txt` als Shakespeare-corpus. De vaste locatie binnen mijn project wordt `data/shakespeare.txt` (UTF-8). Het bestand staat bij het vastleggen van deze keuze nog in Downloads; de kopie naar de projectmap en het inlezen zijn nog niet bevestigd.

Vanuit de projecthoofdmap kan ik het bestand als volgt kopiëren:

```powershell
New-Item -ItemType Directory -Path .\data -Force | Out-Null
Copy-Item -LiteralPath "$env:USERPROFILE\Downloads\Shakespear.txt" -Destination .\data\shakespeare.txt
```

Ik gebruik één bronbestand voor NumPy en PyTorch. Ik bewaar hoofdletters, leestekens en regelafbrekingen. Ik splits de tekst vóór tokenisatie in training en validatie en bouw de tokenizer uitsluitend op de trainingssplit. De exacte splitgrens en een eventueel kleiner CPU-proefdeel leg ik vast voordat ik ga trainen. Beide varianten krijgen hetzelfde geselecteerde deel en dezelfde splits.

De kleine voorbeelden zoals `abab` blijven mijn handmatige controles. Training op Shakespeare is nog niet uitgevoerd. De map `data/` is al uitgesloten in `.gitignore`; het corpus wordt daardoor niet standaard opgenomen in versiebeheer. De oorspronkelijke herkomst van het aangeleverde bestand is nog niet bevestigd.

## Projectdocumentatie

- [Architectuur](docs/architectuur.md)
- [Roadmap](docs/roadmap.md)
- [Voortgang](docs/voortgang.md)
- [Theorie van het taalmodel](docs/theorie.md)

De twee video’s zijn startreferenties voor de NumPy-route. [Video 1](https://www.youtube.com/watch?v=s9w3gtgvNSU) · [Video 2](https://www.youtube.com/watch?v=vKdKtxpfvIc)
