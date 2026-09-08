#!/bin/bash
set -e
cd /home/abshe/MyCodes/alchemy
for i in 00 01 02 03 04 05 06 07 08 09; do
  in="CNCC_reactions_radicals_2_sampled/g1_chunks/chunk_${i}.csv"
  out="CNCC_reactions_radicals_2_sampled/g1_chunks/chunk_${i}_with_energy.csv"
  if [ -f "$out" ]; then
    echo "SKIP chunk ${i} (already done)"
    continue
  fi
  echo "START chunk ${i}"
  alchemy reaction predict-reaction-energies -i "$in" -o "$out" --model-path /home/abshe/MyCodes/alchemy/model.script
  echo "DONE chunk ${i}"
done
echo "ALL_CHUNKS_DONE"
