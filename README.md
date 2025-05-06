# Sistema de Detecção e Análise de Faces em Vídeos

Este sistema utiliza YOLO para detecção de pessoas e DeepFace para análise facial, permitindo processar vídeos ou conjuntos de dados inteiros para extrair informações sobre pessoas presentes nos vídeos.

## Requisitos

```
opencv-python>=4.8.0
ultralytics>=8.0.0
tqdm>=4.65.0
deepface>=0.0.79
numpy>=1.23.0
```

Instale as dependências com:

```bash
pip install -r requirements.txt
```

## Como Usar

O sistema pode ser executado em dois modos:

### 1. Processar um Único Vídeo

```bash
python main.py video PATH_DO_VIDEO --output-dir DIRETORIO_SAIDA --category CATEGORIA [--debug]
```

Exemplo:

```bash
python main.py video exemplos/roubo1.mp4 --output-dir resultados --category roubo
```

### 2. Processar um Conjunto de Dados (Dataset)

```bash
python main.py dataset --dataset_dir DIRETORIO_DATASET --output-dir DIRETORIO_SAIDA [--force-reprocess] [--debug]
```

Exemplo:

```bash
python main.py dataset --dataset_dir dataset --output-dir resultados
```

O diretório `dataset` deve ter a seguinte estrutura:

```
dataset/
  ├── categoria1/
  │   ├── video1.mp4
  │   └── video2.mp4
  └── categoria2/
      ├── video3.mp4
      └── video4.mp4
```

## Modo Debug

O sistema inclui um modo de debug que pode ser ativado com a opção `--debug`. Quando ativado, o vídeo de saída incluirá:

- **Retângulos de detecção para pessoas** com seus IDs e valores de confiança
- **Retângulos de detecção para rostos** com valores de confiança
- **Histórico de trajetória** mostrando o caminho percorrido por cada pessoa
- **Informações demográficas** (idade, gênero, etnia) das pessoas detectadas
- **Contadores e timestamps** mostrando informações de execução

Para ativar o modo debug:

```bash
# Para vídeo único
python main.py video meu_video.mp4 --debug

# Para dataset
python main.py dataset --dataset_dir dataset --debug
```

O modo debug é útil para:

- Visualizar como o sistema está detectando e rastreando pessoas
- Verificar a qualidade da detecção facial
- Depurar problemas no rastreamento ou na detecção
- Avaliar visualmente os resultados do processamento

## Sistema de Checkpoints

O sistema implementa um mecanismo de checkpoints para permitir a retomada do processamento em caso de falhas ou interrupções. Isso é especialmente útil ao processar grandes conjuntos de dados que podem levar muito tempo.

### Como Funciona

1. A cada vídeo processado com sucesso, o sistema atualiza um arquivo de checkpoint (`checkpoint.json`) no diretório de saída.
2. Se o programa for interrompido e reiniciado, ele carregará automaticamente o arquivo de checkpoint e pulará os vídeos já processados.
3. O progresso é exibido durante a execução, mostrando quantos vídeos já foram processados e quantos faltam.

### Forçar Reprocessamento

Para ignorar o checkpoint e reprocessar todos os vídeos, use a opção `--force-reprocess`:

```bash
python main.py dataset --dataset_dir dataset --output-dir resultados --force-reprocess
```

### Identificando Problemas

- Se um vídeo falhar durante o processamento, ele não será marcado como processado no checkpoint e será tentado novamente em uma próxima execução.
- O arquivo de checkpoint (`checkpoint.json`) pode ser inspecionado manualmente para verificar quais vídeos foram processados com sucesso.

## Saída

Para cada vídeo processado, o sistema gera:

1. Um vídeo anotado com as detecções (`nome_do_video_out.mp4`)
2. Imagens das faces detectadas (na pasta `faces`)
3. Dados de análise em formato JSON com informações sobre:
   - Pessoas detectadas
   - Faces detectadas
   - Informações demográficas (idade, gênero, etnia)
   - Estatísticas do vídeo
