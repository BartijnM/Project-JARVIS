# Tests

Ik voeg tests toe bij de bouwstap die ze controleren. De eerste test gebruikt een korte tekst om de tokenizer te controleren. Ik gebruik Python’s standaardmodule `unittest` zodat de NumPy-omgeving geen extra testafhankelijkheid nodig heeft.

Draai de tests vanuit de projecthoofdmap met de Python-interpreter van de variant die je test.

```powershell
.\.venv-numpy\Scripts\python.exe -m unittest discover -s tests
```
