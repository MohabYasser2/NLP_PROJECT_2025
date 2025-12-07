import sys
sys.path.insert(0, 'c:/Users/mohab/Desktop/Uni/Courses/Grad Project/NLP_PROJECT_2025')
from src.config import ID_TO_DIACRITIC, DIACRITIC_TO_ID

print('Label 0:', repr(ID_TO_DIACRITIC.get(0, 'NOT_FOUND')))
print('\nAll label mappings:')
for k, v in sorted(ID_TO_DIACRITIC.items()):
    print(f'  {k}: {repr(v)}')
