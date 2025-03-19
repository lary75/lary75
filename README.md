# Bloco 1 Código para importação dos dados
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Configuração do estilo dos gráficos
sns.set(style="whitegrid")

# Carregar os datasets
ratings = pd.read_csv('ratings.csv')
movies = pd.read_csv('movies.csv')

# Visualizar as primeiras linhas
print("Ratings:")
print(ratings.head())
print("\nMovies:")
print(movies.head()

# Bloco 2 Limpeza dos Dados

# Verificação de valores ausentes
print(ratings.isnull().sum())
print(movies.isnull().sum())

# Remover registros com valores ausentes (se aplicável)
ratings.dropna(inplace=True)
movies.dropna(inplace=True)
Análise Exploratória de Dados (EDA)
print(ratings['rating'].describe())
Distribuição das Avaliações
plt.figure(figsize=(10,6))
sns.histplot(ratings['rating'], bins=20, kde=True, color='blue')
plt.title("Distribuição das Avaliações dos Filmes")
plt.xlabel("Avaliação")
plt.ylabel("Frequência")
plt.show()

# Bloco 3 Média de Avaliação por Filme

# Média de avaliação por filme
avg_rating = ratings.groupby('movieId')['rating'].mean().reset_index()

# Mesclando com o dataset de filmes
movie_ratings = pd.merge(avg_rating, movies, on='movieId')

# Exibindo os 10 melhores filmes
top_movies = movie_ratings.sort_values(by='rating', ascending=False).head(10)
print("Top 10 Filmes (por média de avaliação):")
print(top_movies[['title', 'rating']])
Relação entre Número de Avaliações e Média
# Contar o número de avaliações por filme
rating_counts = ratings.groupby('movieId')['rating'].count().reset_index()
rating_counts.columns = ['movieId', 'rating_count']

# Unir com a média das avaliações
movie_stats = pd.merge(avg_rating, rating_counts, on='movieId')

# Visualizar a relação
plt.figure(figsize=(10,6))
sns.scatterplot(x='rating_count', y='rating', data=movie_stats, color='green')
plt.xlabel("Número de Avaliações")
plt.ylabel("Média das Avaliações")
plt.show()

# Bloco 4 Desenvolvimento de um Sistema de Recomendação Simples

# Criação da matriz usuário-filme
user_movie_matrix = ratings.pivot(index='userId', columns='movieId', values='rating').fillna(0)
print("Matriz usuário-filme:")
print(user_movie_matrix.head())
Implementação do KNN
from sklearn.neighbors import NearestNeighbors

# Transpor a matriz para ter filmes nas linhas
movie_matrix = user_movie_matrix.T

# Criação do modelo KNN
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=5, n_jobs=-1)
model_knn.fit(movie_matrix)

# Função para recomendar filmes semelhantes
def get_similar_movies(movie_id, n_neighbors=5):
    try:
        movie_index = movie_matrix.index.get_loc(movie_id)
    except KeyError:
        print("MovieId não encontrado.")
        return []
    
    distances, indices = model_knn.kneighbors(movie_matrix.iloc[movie_index].values.reshape(1, -1), n_neighbors=n_neighbors+1)
    # Ignorando o próprio filme (primeiro índice)
    similar_movie_ids = movie_matrix.index[indices.flatten()[1:]]
    return similar_movie_ids.tolist()

# Exemplo de recomendação para o filme com movieId = 1
similar_movies = get_similar_movies(1)
print("Filmes recomendados para movieId 1:", similar_movies)
