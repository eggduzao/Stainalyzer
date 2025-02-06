#!/bin/bash

# Removendo os arquivos .DS_Store recursivamente.
find . -name ".DS_Store" -type f -delete

# Removendo qualquer outro arquivo de metadados do MAC (arquivos que começam com "._") recursivamente.
find . -name "._*" -type f -delete

# Removendo .Trashes and outros que, às vezes, o MAC gera, quando quer.
find . -name ".Trashes" -type d -exec rm -rf {} +

# (CUIDADO) Removendo todos os arquivos escondidos (i.e. TUDO que começa com ".")
# find . -name ".*" -type f ! -name ".git*" -delete

echo "All tidy mam'!"

