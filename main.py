import os
from data_processing import load_iris_data, basic_data_info, check_missing_values, preprocess_data
from data_visualization import plot_pairplot, plot_histograms, plot_boxplots

def main():

    if not os.path.exists('data'):
        os.makedirs('data')

 
    df = load_iris_data()
    basic_data_info(df)
    check_missing_values(df)


    df_processed = preprocess_data(df.copy()) 


    features = [col for col in df_processed.columns if col not in ['species']]
    
    plot_pairplot(df_processed)
    plot_histograms(df_processed, features)
    plot_boxplots(df_processed, features)

    print("\n--- Análisis de datos completado ---")

if __name__ == "__main__":
    main()