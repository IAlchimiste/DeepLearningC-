import os

# Fonction pour supprimer tous les fichiers dans un répertoire
def clear_directory(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                clear_directory(file_path)
                os.rmdir(file_path)
        except Exception as e:
            print(f"Erreur lors de la suppression de {file_path}: {e}")

# Supprimer les éléments présents dans les dossiers inputs et outputs
clear_directory('data/inputs')
clear_directory('data/outputs')

print('Les éléments présents dans les dossiers data/inputs et data/outputs ont été supprimés avec succès.')

# Créer les dossiers inputs et outputs
os.makedirs('data/inputs', exist_ok=True)
os.makedirs('data/outputs', exist_ok=True)

# Lire le fichier
with open('data/human_chat.txt', 'r') as file:
    lines = file.readlines()
# Créer les fichiers
for i in range(0, len(lines), 2):
    # Vérifier si l'indice suivant est valide
    if i + 1 < len(lines):
        # Écrire dans les fichiers d'entrée et de sortie
        with open(f'data/inputs/{i // 2}.txt', 'w') as file:
            file.write(lines[i])
        with open(f'data/outputs/{i // 2}.txt', 'w') as file:
            file.write(lines[i + 1])
    else:
        print(f"Fin de la création du dataset !")

# Lire le fichier
with open('data/human_chat.txt', 'r') as file:
    lines = file.readlines()

# Trouver la longueur maximale parmi les lignes
max_length = max(len(line) for line in lines)

print(f"La ligne avec le plus de caractères a une longueur de : {max_length}")


print('Done')