# Project JARVIS

In Project JARVIS bouw ik vanaf nul een klein taalmodel dat tekst leert aanvullen. Ik maak eerst een NumPy-variant en daarna een PyTorch-variant. Beide gebruiken dezelfde voorbeeldtekst, tokenizer en evaluatie, zodat ik de resultaten kan vergelijken.

| Variant | Wat ik zelf bouw |
| --- | --- |
| NumPy | De modelberekeningen, de gradiënten en de updates van de gewichten. NumPy is de enige externe runtime-afhankelijkheid. |
| PyTorch | Dezelfde modelarchitectuur. PyTorch berekent de gradiënten met autograd. |

## Lokaal starten

Gebruik Python 3.11 of nieuwer. Voer de commando’s uit vanuit de hoofdmap van de repo. Maak voor iedere variant een aparte virtuele omgeving.

**NumPy (PowerShell)**

```powershell
py -m venv .venv-numpy
.\.venv-numpy\Scripts\python.exe -m pip install -e '.[numpy]'
.\.venv-numpy\Scripts\python.exe -c "import jarvis, numpy"
```

**PyTorch (PowerShell)**

```powershell
py -m venv .venv-torch
.\.venv-torch\Scripts\python.exe -m pip install -e '.[torch]'
.\.venv-torch\Scripts\python.exe -c "import jarvis, torch"
```

De importcontrole controleert de installatie. De eerste modelmijlpaal is een bigrammodel op CPU. Het leert welk teken waarschijnlijk volgt op het vorige teken en genereert daarna korte tekst. Het NumPy-bigrammodel is gebouwd. Bij J02 bouw ik de PyTorch-variant en vergelijk ik de logits en loss op dezelfde invoer en gewichten.

## Projectdocumentatie

- [Architectuur](docs/architectuur.md)
- [Roadmap](docs/roadmap.md)
- [Voortgang](docs/voortgang.md)
- [Theorie van het taalmodel](docs/theorie.md)

De twee video’s zijn startreferenties voor de NumPy-route. [Video 1](https://www.youtube.com/watch?v=s9w3gtgvNSU) · [Video 2](https://www.youtube.com/watch?v=vKdKtxpfvIc)
