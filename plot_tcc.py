import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("resumo_demografico.csv")

plt.figure(figsize=(8,5))
sns.histplot(df['entropia_racial'], bins=10, kde=True, color='skyblue')
plt.title("Histograma da Entropia Racial dos Vídeos")
plt.xlabel("Entropia (Diversidade Racial)")
plt.ylabel("Número de Vídeos")
plt.tight_layout()
plt.savefig("figuras/hist_entropia_racial.png")

plt.figure(figsize=(10,6))
sns.boxplot(x='crime_category', y='total_faces', data=df, palette='pastel')
plt.xticks(rotation=45)
plt.title("Distribuição do Número de Faces por Categoria de Crime")
plt.ylabel("Número de Faces Detectadas")
plt.xlabel("Categoria")
plt.tight_layout()
plt.savefig("figuras/boxplot_faces_categoria.png")


def plot_heatmap(df, feature, title):
    counts = df.groupby(['crime_category', feature]).size().unstack(fill_value=0)

    proportions = counts.div(counts.sum(axis=1), axis=0)

    plt.figure(figsize=(12, 6))
    sns.heatmap(proportions, annot=True, cmap="YlOrRd", fmt=".2f")
    plt.title(title)
    plt.ylabel('Categoria do crime')
    plt.xlabel(feature.replace("_", " ").capitalize())
    plt.tight_layout()
    plt.savefig("figuras/heat_map_{}.png".format(feature))


plot_heatmap(df, 'raça_dominante', 'Raça dominante x classe do crime')
