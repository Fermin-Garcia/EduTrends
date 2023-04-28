from sklearn.cluster import KMeans



e.best_kmeans(data=df[['final_score']],k_max=10)
final_grade_cluster = e.apply_kmeans(data=train[['final_score']],k=3)
train['cluster_pattern'] = final_grade_cluster['k_means_3']