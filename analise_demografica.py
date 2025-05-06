import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import numpy as np

def carregar_dados_json(diretorio):
    dados_videos = []
    for root, _, files in os.walk(diretorio):
        for file in files:
            if file.endswith('_analysis.json'):
                caminho = os.path.join(root, file)
                with open(caminho, 'r') as f:
                    dados = json.load(f)
                    dados_videos.append(dados)
    return dados_videos

def extrair_dados_demograficos(dados_videos):
    dados_agregados = {
        'raças': defaultdict(int),
        'gêneros': defaultdict(int),
        'faixas_etarias': defaultdict(int),
        'videos_por_categoria': defaultdict(int),
        'pessoas_por_video': [],
        'faces_por_video': [],
        'confianca_media_face': [],
        'confianca_media_pessoa': [],
        'resolucoes': defaultdict(int),
        'condicoes_luz': defaultdict(int)
    }
    
    for video in dados_videos:
        # Estatísticas gerais do vídeo
        stats = video['video_stats']
        dados_agregados['videos_por_categoria'][stats['crime_category']] += 1
        dados_agregados['pessoas_por_video'].append(stats['total_persons_detected'])
        dados_agregados['faces_por_video'].append(stats['total_faces_detected'])
        dados_agregados['resolucoes'][stats['resolution']] += 1
        dados_agregados['condicoes_luz'][stats['lighting_condition']] += 1
        
        # Análise por pessoa
        for person_id, person_data in video['persons'].items():
            if person_data['faces_detected']:
                dados_agregados['confianca_media_pessoa'].append(person_data['avg_confidence'])
                dados_agregados['confianca_media_face'].append(person_data['avg_face_confidence'])
                
                for face in person_data['faces_detected']:
                    if 'demographics' in face:
                        dem = face['demographics']
                        dados_agregados['raças'][dem['dominant_race']] += 1
                        dados_agregados['gêneros'][dem['gender']] += 1
                        dados_agregados['faixas_etarias'][dem['age_range']] += 1
    
    return dados_agregados

def criar_visualizacoes(dados_agregados):
    # Configuração do estilo
    sns.set_theme()
    sns.set_palette("husl")
    
    # Criar diretório para salvar os gráficos
    os.makedirs('analise_graficos', exist_ok=True)
    
    # 1. Distribuição de Raças
    plt.figure(figsize=(12, 6))
    sns.barplot(x=list(dados_agregados['raças'].keys()), 
                y=list(dados_agregados['raças'].values()))
    plt.title('Distribuição de Raças Detectadas')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analise_graficos/distribuicao_racas.png')
    plt.close()
    
    # 2. Distribuição de Gêneros
    plt.figure(figsize=(8, 6))
    plt.pie(dados_agregados['gêneros'].values(), 
            labels=dados_agregados['gêneros'].keys(),
            autopct='%1.1f%%')
    plt.title('Distribuição de Gêneros')
    plt.savefig('analise_graficos/distribuicao_generos.png')
    plt.close()
    
    # 3. Distribuição de Faixas Etárias
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(dados_agregados['faixas_etarias'].keys()), 
                y=list(dados_agregados['faixas_etarias'].values()))
    plt.title('Distribuição de Faixas Etárias')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analise_graficos/distribuicao_idades.png')
    plt.close()
    
    # 4. Distribuição de Pessoas por Vídeo
    plt.figure(figsize=(10, 6))
    sns.histplot(dados_agregados['pessoas_por_video'], bins=20)
    plt.title('Distribuição do Número de Pessoas por Vídeo')
    plt.xlabel('Número de Pessoas')
    plt.ylabel('Frequência')
    plt.savefig('analise_graficos/distribuicao_pessoas_por_video.png')
    plt.close()
    
    # 5. Boxplot de Confiança
    plt.figure(figsize=(10, 6))
    dados_confianca = pd.DataFrame({
        'Confiança Face': dados_agregados['confianca_media_face'],
        'Confiança Pessoa': dados_agregados['confianca_media_pessoa']
    })
    sns.boxplot(data=dados_confianca)
    plt.title('Distribuição de Confiança nas Detecções')
    plt.savefig('analise_graficos/distribuicao_confianca.png')
    plt.close()
    
    # 6. Distribuição de Resoluções
    plt.figure(figsize=(10, 6))
    sns.barplot(x=list(dados_agregados['resolucoes'].keys()), 
                y=list(dados_agregados['resolucoes'].values()))
    plt.title('Distribuição de Resoluções dos Vídeos')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('analise_graficos/distribuicao_resolucoes.png')
    plt.close()
    
    # 7. Condições de Iluminação
    plt.figure(figsize=(8, 6))
    plt.pie(dados_agregados['condicoes_luz'].values(), 
            labels=dados_agregados['condicoes_luz'].keys(),
            autopct='%1.1f%%')
    plt.title('Distribuição de Condições de Iluminação')
    plt.savefig('analise_graficos/distribuicao_iluminacao.png')
    plt.close()
    
    # 8. Relação entre Número de Pessoas e Faces
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=dados_agregados['pessoas_por_video'], 
                   y=dados_agregados['faces_por_video'])
    plt.title('Relação entre Número de Pessoas e Faces Detectadas')
    plt.xlabel('Número de Pessoas')
    plt.ylabel('Número de Faces')
    plt.savefig('analise_graficos/relacao_pessoas_faces.png')
    plt.close()

def converter_para_serializavel(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: converter_para_serializavel(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [converter_para_serializavel(item) for item in obj]
    return obj

def gerar_relatorio_estatistico(dados_agregados):
    relatorio = {
        'Estatísticas Gerais': {
            'Total de Vídeos Analisados': sum(dados_agregados['videos_por_categoria'].values()),
            'Categorias de Crime': dict(dados_agregados['videos_por_categoria']),
            'Média de Pessoas por Vídeo': float(np.mean(dados_agregados['pessoas_por_video'])),
            'Média de Faces por Vídeo': float(np.mean(dados_agregados['faces_por_video'])),
            'Média de Confiança Face': float(np.mean(dados_agregados['confianca_media_face'])),
            'Média de Confiança Pessoa': float(np.mean(dados_agregados['confianca_media_pessoa'])),
            'Resoluções dos Vídeos': dict(dados_agregados['resolucoes']),
            'Condições de Iluminação': dict(dados_agregados['condicoes_luz'])
        },
        'Distribuição Demográfica': {
            'Raças': dict(dados_agregados['raças']),
            'Gêneros': dict(dados_agregados['gêneros']),
            'Faixas Etárias': dict(dados_agregados['faixas_etarias'])
        },
        'Estatísticas Descritivas': {
            'Pessoas por Vídeo': {
                'Média': float(np.mean(dados_agregados['pessoas_por_video'])),
                'Mediana': float(np.median(dados_agregados['pessoas_por_video'])),
                'Desvio Padrão': float(np.std(dados_agregados['pessoas_por_video'])),
                'Mínimo': int(np.min(dados_agregados['pessoas_por_video'])),
                'Máximo': int(np.max(dados_agregados['pessoas_por_video']))
            },
            'Faces por Vídeo': {
                'Média': float(np.mean(dados_agregados['faces_por_video'])),
                'Mediana': float(np.median(dados_agregados['faces_por_video'])),
                'Desvio Padrão': float(np.std(dados_agregados['faces_por_video'])),
                'Mínimo': int(np.min(dados_agregados['faces_por_video'])),
                'Máximo': int(np.max(dados_agregados['faces_por_video']))
            }
        }
    }
    
    # Converter valores numpy para tipos Python nativos
    relatorio = converter_para_serializavel(relatorio)
    
    with open('analise_graficos/relatorio_estatistico.json', 'w') as f:
        json.dump(relatorio, f, indent=4)

def main():
    # Carregar dados
    dados_videos = carregar_dados_json('resultados')
    
    # Extrair e agregar dados
    dados_agregados = extrair_dados_demograficos(dados_videos)
    
    # Criar visualizações
    criar_visualizacoes(dados_agregados)
    
    # Gerar relatório estatístico
    gerar_relatorio_estatistico(dados_agregados)
    
    print("Análise concluída! Os gráficos e relatório foram salvos no diretório 'analise_graficos'")

if __name__ == "__main__":
    main() 