# {
#     echo "Category 1: $(grep -E '^(1- Negativo|-)' classes.csv | grep '1. Paciente com PE' | wc -l)"
#     echo "Category 2: $(grep -E '^3- Borderline' classes.csv | grep '1. Paciente com PE' | wc -l)"
#     echo "Category 3: $(grep -E '^2- Positivo' classes.csv | grep '1. Paciente com PE' | wc -l)"
#     echo "Category 4: $(grep -E '^(1- Negativo|-)' classes.csv | grep '2. Paciente sem PE' | wc -l)"
#     echo "Category 5: $(grep -E '^3- Borderline' classes.csv | grep '2. Paciente sem PE' | wc -l)"
#     echo "Category 6: $(grep -E '^2- Positivo' classes.csv | grep '2. Paciente sem PE' | wc -l)"
# } > category_counts.txt