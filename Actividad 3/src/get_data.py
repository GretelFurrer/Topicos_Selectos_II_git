import random, datetime, os, shutil, math
from pathlib import Path

train_dir = Path("/Users/gretelfurrer/MAESTRIA_CIENCIA_DE_DATOS/3rd_sem_MCD/GRANDES_BASES_DE_DATOS_II/Topicos_Selectos_II_git/Actividad 3/data/train")
test_dir = Path("/Users/gretelfurrer/MAESTRIA_CIENCIA_DE_DATOS/3rd_sem_MCD/GRANDES_BASES_DE_DATOS_II/Topicos_Selectos_II_git/Actividad 3/data/test")

def prep_test_data(pokemon, train_dir, test_dir):
    pop = os.listdir(train_dir+'/'+pokemon)
    test_data=random.sample(pop, 15)
    print(test_data)
    for f in test_data:
        shutil.copy(train_dir+'/'+pokemon+'/'+f, test_dir+'/'+pokemon+'/')

for poke in os.listdir(train_dir):

    if poke.startswith("."):
        continue
    else:
        os.makedirs('../data/test/'+poke, exist_ok=True)
        prep_test_data(poke, train_dir, test_dir)
print('test folder complete!!')