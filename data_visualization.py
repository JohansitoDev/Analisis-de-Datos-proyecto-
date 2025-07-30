import matplotlib.pyplot as plt
import seaborn as sns

def plot_pairplot(df, hue_column='species'):
    print("\n--- Generando Pair Plot (distribución y relación entre características) ---")
    sns.pairplot(df, hue=hue_column)
    plt.suptitle('Pair Plot del Dataset Iris', y=1.02) 
    plt.show()

def plot_histograms(df, features):
    print("\n--- Generando Histogramas de Características ---")
    df[features].hist(bins=15, figsize=(10, 8))
    plt.suptitle('Histogramas de las Características de Iris', y=1.02)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) 
    plt.show()

def plot_boxplots(df, features, hue_column='species'):
    print("\n--- Generando Box Plots por Especie ---")
    plt.figure(figsize=(12, 8))
    for i, feature in enumerate(features):
        plt.subplot(2, 2, i + 1)
        sns.boxplot(x=hue_column, y=feature, data=df)
        plt.title(f'Box Plot de {feature} por Especie')
    plt.tight_layout()
    plt.show()