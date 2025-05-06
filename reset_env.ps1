# Desativa qualquer ambiente Conda ativo
conda deactivate

# Remove o ambiente 'tcc' se ele existir
conda env remove -n tc1 -y

# Cria o ambiente 'tcc' baseado no arquivo environment.yml
conda env create -f environment.yml

# Ativa o ambiente 'tcc'
conda activate tc1
 

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124