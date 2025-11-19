import random, datetime, os, shutil, math
from pathlib import Path

train_dir = Path("/Users/gretelfurrer/MAESTRIA_CIENCIA_DE_DATOS/3rd_sem_MCD/GRANDES_BASES_DE_DATOS_II/Topicos_Selectos_II_git/Actividad 3/data/train")
test_dir  = Path("/Users/gretelfurrer/MAESTRIA_CIENCIA_DE_DATOS/3rd_sem_MCD/GRANDES_BASES_DE_DATOS_II/Topicos_Selectos_II_git/Actividad 3/data/test")

def prep_test_data(pokemon, train_dir, test_dir, n=15):
    src_dir = train_dir / pokemon
    dst_dir = test_dir / pokemon
    dst_dir.mkdir(parents=True, exist_ok=True)

    # list only files (ignore hidden and subdirs)
    pop = [f for f in os.listdir(src_dir) if not f.startswith('.') and (src_dir / f).is_file()]
    if not pop:
        print(f"No files found in {src_dir}, skipping {pokemon}")
        return

    # avoid ValueError if there are fewer than n files
    k = min(n, len(pop))
    if k < n:
        print(f"Warning: requested {n} files but only {len(pop)} available for '{pokemon}'. Copying {k} files.")
    test_data = random.sample(pop, k)

    for f in test_data:
        shutil.copy2(src_dir / f, dst_dir / f)


for poke in sorted(os.listdir(train_dir)):
    if poke.startswith("."):
        continue
    prep_test_data(poke, train_dir, test_dir)

print('test folder complete!!')