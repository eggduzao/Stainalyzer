#!/usr/bin/env bash

# Paths
DAB1="/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle"
DAB1C="/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle_Clean"
DAB2="/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento"
DAB2C="/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento_Clean"

# Trees
# tree "$DAB1" > dab1.txt
# tree "$DAB1C" > dab1c.txt
# tree "$DAB2" > dab2.txt
# tree "$DAB2C" > dab2c.txt

# List DAB_Grupo_Controle:
find "$DAB1" -type f \( -iname "*.jpg" \) > $DAB1".txt"

# List DAB_IMIP_Tratamento:
find "$DAB2" -type f \( -iname "*.jpg" \) > $DAB2".txt"

# List DAB_Grupo_Controle_Clean:
find "$DAB1C" -type f \( -iname "*.tiff" \) > $DAB1C".txt"

# List DAB_IMIP_Tratamento_Clean:
find "$DAB2C" -type f \( -iname "*.tiff" \) > $DAB2C".txt"

# Count the number of elements
wc -l $DAB1".txt"
wc -l $DAB1C".txt"
wc -l $DAB2".txt"
wc -l $DAB2C".txt"

# Query uniquely file types
# find $DAB1 -type f | sed -n 's/.*\.\([a-zA-Z0-9]*\)$/\1/p' | sort | uniq
# find $DAB1C -type f | sed -n 's/.*\.\([a-zA-Z0-9]*\)$/\1/p' | sort | uniq
# find $DAB2 -type f | sed -n 's/.*\.\([a-zA-Z0-9]*\)$/\1/p' | sort | uniq
# find $DAB2C -type f | sed -n 's/.*\.\([a-zA-Z0-9]*\)$/\1/p' | sort | uniq

# Replace
IL1="/Users/egg/Projects/Stainalyzer/data/DAB_Grupo_Controle_Clean/DAB_Grupo_Controle/GRUPO_CONTROLE_PE_ACE2/"
IL2="/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento_Clean/DAB_IMIP_Tratamento/IMIP_HLAG5/"
IL3="/Users/egg/Projects/Stainalyzer/data/DAB_IMIP_Tratamento_Clean/DAB_IMIP_Tratamento/IMIP_ACE2/"

mv $IL1"ACE2_REPETIÇÃO_20_03/" $IL1"ACE2_REPEAT_20_03/"
mv $IL2"6520_21F_REPETIÇÃO" $IL2"6520_21F_REPEAT"
mv $IL2"4541_21_REPETIÇÃO" $IL2"4541_21_REPEAT"
mv $IL2"3830_21_REPETIÇÃO" $IL2"3830_21_REPEAT"
mv $IL2"5293_21F_REPETIÇÃO" $IL2"5293_21F_REPEAT"
mv $IL2"4578_21E_REPETIÇÃO" $IL2"4578_21E_REPEAT"
mv $IL2"4579_21E_REPETIÇÃO" $IL2"4579_21E_REPEAT"
mv $IL2"6519_21F_REPETIÇÃO" $IL2"6519_21F_REPEAT"
mv $IL2"3835_21_REPETIÇÃO" $IL2"3835_21_REPEAT"
mv $IL2"5164_21F_REPETIÇÃO" $IL2"5164_21F_REPEAT"
mv $IL3"ACE2_REPETIÇÃO" $IL3"ACE2_REPEAT"

