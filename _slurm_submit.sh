#!/usr/bin/env bash

# Sync GitHub repository
echo -e "\e[91mSyncing GitHub...\e[0m"
git pull

echo -e "\e[1m\e[91mSubmitting script to SLURM...\e[0m"

sbatch slurm_submit.sh

echo -e "\e[5m\e[1m\e[91mJob submitted.\e[0m"
squeue -u dc-alta2