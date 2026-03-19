"""
Clustering stability and meta-analysis across k-means runs.

This module provides :class:`ClusterMetaAnalysis` for loading clustering
results across the full parameter grid, computing Hungarian-matched
centroid instability, and identifying the most stable solution for each
(k, reduction_percent) combination.  :class:`ClusterPlotter` visualises
quality and instability metrics.
"""
import pandas as pd
import os
import json
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.optimize import linear_sum_assignment
import numpy as np
from math import pi

class ClusterMetaAnalysis:
    """Analyzes clustering metadata and centroids for stability and patterns.

    This class is designed to load clustering metadata and centroid information
    from JSON files, calculate stability metrics across clustering attempts,
    and visualize results.

    Attributes:
        directory (str): Directory containing JSON files with clustering data.
        df (pd.DataFrame): DataFrame holding loaded clustering metadata.
    """
    def __init__(self, directory):
        """Initializes the ClusterMetaAnalysis with a specific directory.

        Parameters
        ----------
        directory : str
            The path to the directory containing clustering metadata JSON files.
            """
            self.directory = directory
            self.df = None

    def load_data(self):
        """Loads clustering metadata from JSON files within the specified directory.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the clustering metadata.
            """
            # Initialize an empty list to store the data
            data = []

        # Walk through the directory
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if file.endswith(".json"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, 'r') as f:
                            # Load the JSON content
                            content = json.load(f)

                            # Infer the k-number from the centroids field
                            content['k_number'] = len(content['centroids'])

                            # Exclude the 'centroids' field
                            content.pop('centroids', None)

                            # Add the file path to the content
                            content['file_path'] = file_path

                            # Append the modified content to the data list
                            data.append(content)
                    except:
                        print(f'file not loadable: {file_path}')

        # Convert the list of data to a DataFrame
        df = pd.DataFrame(data)

        return df
    
    def load_centroids_for_analysis(self, reduction_percent, k_number):
        """
        Loads centroids for a specific reduction_percent and k_number.

        Parameters:
        - reduction_percent: The reduction percentage of interest.
        - k_number: The number of clusters (k) of interest.

        Returns
        -------
            - List of centroids arrays for the specified reduction_percent and k_number.
            """
            centroids_list = []
            # Filter the DataFrame for the specific reduction_percent and k_number
            filtered_df = self.df[(self.df['reduction_percent'] == reduction_percent) &
            (self.df['k_number'] == k_number)]

        # Iterate over the filtered DataFrame and load the centroids
        for _, row in filtered_df.iterrows():
            file_path = row['file_path']
            with open(file_path, 'r') as f:
                content = json.load(f)
                centroids = np.array(content['centroids'])
                centroids_list.append(centroids)

        return centroids_list



    @staticmethod
    def calculate_instability(centroids_list):
        """
        Calculates the instability measure for a list of centroids arrays.

        Parameters:
        - centroids_list: List of numpy arrays, each containing the centroids of one clustering attempt.

        Returns
        -------
            - Instability measure for each clustering attempt.
            """

        # Number of clustering attempts
        num_cas = len(centroids_list)
        
        # Initialize a matrix to store the sum of distances between all pairs of CAs
        distances = np.zeros((num_cas, num_cas))

        # Compare each pair of CAs
        for i in range(num_cas):
            for j in range(i + 1, num_cas):
                cost_matrix = np.linalg.norm(centroids_list[i][:, np.newaxis, :] - centroids_list[j], axis=2)
                row_ind, col_ind = linear_sum_assignment(cost_matrix)
                total_cost = cost_matrix[row_ind, col_ind].sum()
                # Record the total cost in the distances matrix
                distances[i, j] = total_cost
                distances[j, i] = total_cost

        # Calculate instability for each CA
        instability = distances.sum(axis=0)
        min_instability_index = np.argmin(instability)

        return distances[:,min_instability_index]

    def calculate_and_assign_instability(self):
        """Calculates and assigns instability values to the DataFrame for each clustering attempt."""
        unique_combinations = self.df[['k_number', 'reduction_percent']].drop_duplicates()

        for _, row in unique_combinations.iterrows():
            k_number = row['k_number']
            reduction_percent = row['reduction_percent']
            centroids_list = self.load_centroids_for_analysis(reduction_percent, k_number)
            instability_values = self.calculate_instability(centroids_list)

            # Assign instability values back to the DataFrame
            condition = (self.df['k_number'] == k_number) & (self.df['reduction_percent'] == reduction_percent)
            for i, (index, _) in enumerate(self.df[condition].iterrows()):
                self.df.at[index, 'instability'] = instability_values[i]


    def analyze(self):
        """Performs the analysis by loading data, calculating instability, and optionally visualizing results."""
        # Load the data into a DataFrame
        self.df = self.load_data()

        self.calculate_and_assign_instability()
        # Perform any analysis or manipulation on the DataFrame here
        # For now, just print the DataFrame
        print(self.df)
    
    def save_df(self, save_path):
        """Saves the DataFrame to the specified path."""
        self.df.to_csv(save_path, index=False)
        print(f"DataFrame saved to {save_path}")

    def load_df(self, load_path):
        """Loads the DataFrame from a CSV file at the specified path.

        Parameters
        ----------
        load_path : str
            The path from where the DataFrame should be loaded.

        Returns
        -------
        pd.DataFrame
            The loaded DataFrame.
            """
            df = pd.read_csv(load_path)
            print(f"DataFrame loaded from {load_path}")
            return df

    def find_most_stable_centroids(self, k_number, reduction_percent):
        """
        Returns the most stable centroids for a given k_number and reduction_percent,
        identified by the minimal instability. If there's a tie, the first one is taken.

        Parameters:
        - k_number: The number of clusters.
        - reduction_percent: The reduction percentage.

        Returns
        -------
            - The centroids array of the most stable clustering attempt.
            """
            # Filter the DataFrame for the given k_number and reduction_percent
            filtered_df = self.df[(self.df['k_number'] == k_number) &
            (self.df['reduction_percent'] == reduction_percent)]

        # Find the entry with the minimal instability
        most_stable = filtered_df.loc[filtered_df['instability'].idxmin()]

        # Load and return the centroids from the corresponding JSON file
        with open(most_stable['file_path'], 'r') as file:
            content = json.load(file)
            centroids = np.array(content['centroids'])
        
        return centroids,most_stable['file_path']

    def de_zscore_centroids(self, centroids, mu, sigma):
        """
        Reverses the z-scoring process for centroids using the original mean (mu) and standard deviation (sigma).
        """
        centroids_de_zscored = np.zeros(shape=centroids.shape)
        for i in range(centroids.shape[1]):
            centroids_de_zscored[:,i] = centroids[:,i] * sigma[i] + mu[1]
        return centroids_de_zscored 


    def normalize_centroids(self, centroids):
        """
        Normalizes the centroids in each feature to the absolute max value of that feature across all centroids.

        Parameters:
        - centroids: A 2D NumPy array of centroids, shape (n_centroids, n_features)

        Returns
        -------
            - Normalized centroids: A 2D NumPy array, shape (n_centroids, n_features)
            """
            abs_max = np.max(np.abs(centroids), axis=0)
            normalized_centroids = centroids / abs_max
            return normalized_centroids

class ClusterPlotter:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def plot_metric(self, y_metric, log_scale=False):
        # Set seaborn theme with a light grid for contrast and colorblind-friendly palette
        sns.set_theme(context='talk', style="whitegrid", palette="colorblind")

        # Adjust DataFrame for plotting
        if y_metric == 'analysis_duration' and self.dataframe[y_metric].dtype == 'object':
            self.dataframe[y_metric] = pd.to_timedelta(self.dataframe[y_metric])
            self.dataframe[y_metric] = self.dataframe[y_metric].dt.total_seconds() / 60

        # Initialize the plot
        fig = plt.figure(figsize=(10, 6))
        
        # Define marker styles for each 'reduction_percent' to ensure distinct visualization
        markers = ["o", "v", "^", "<", ">", "s", "p", "P", "*", "+", "x"]
        
        # Creating a lineplot with markers for each line and using the 'colorblind' palette
        ax = sns.lineplot(data=self.dataframe, x='k_number', y=y_metric, hue='reduction_percent',
                        ci=95, estimator='median', style='reduction_percent',
                        markers=markers[:len(self.dataframe['reduction_percent'].unique())],
                        dashes=False, linewidth=2.5)

        # Optional: Set y-axis to log scale
        if log_scale:
            ax.set_yscale('log')

        # Customizing the grid lines for better visibility
        ax.grid(which='major', linestyle='-', linewidth='0.5', color='gray')
        ax.grid(which='minor', linestyle=':', linewidth='0.5', color='lightgray')

        # Improve legend readability
        legend = ax.legend(title='Reduction Percent', loc='upper right', bbox_to_anchor=(1.15, 1), borderaxespad=0.)
        frame = legend.get_frame()
        frame.set_color('white')
        frame.set_edgecolor('gray')

        # Title and labels
        plt.title(f'{y_metric.capitalize()} by K-Number with Different Reduction Percentages')
        plt.xlabel('K-Number')
        plt.ylabel(y_metric.capitalize().replace('_', ' '))

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_radar_charts(centroids, feature_labels,normalise = True):
        num_vars = len(feature_labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        centroids = np.concatenate((centroids, centroids[:, [0]]), axis=1)  # Complete the loop
        angles += angles[:1]  # Complete the loop

        # Calculate subplot grid size
        n_clusters = centroids.shape[0]
        ncols = 2  # You can adjust this
        nrows = (n_clusters + ncols - 1) // ncols

        fig, axs = plt.subplots(nrows=ncols, ncols=nrows, figsize=(ncols*6,6),
                                subplot_kw=dict(polar=True))
        axs = axs.flatten()  # Flatten in case of a single row

        for i in range(n_clusters):
            ax = axs[i]
            ax.plot(angles, centroids[i], linewidth=1, linestyle='solid', label=f'Cluster {i+1}')
            ax.fill(angles, centroids[i], alpha=0.1)
            
            # Set the x-ticks (for each feature/angle)
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(feature_labels, color='grey', size=12)
            
            if normalise == True:
                # Draw y-labels and custom rings at specified values
                ax.set_yticks([-1, -0.5, 0, 0.5, 1])  # Custom y-ticks
                ax.set_ylim(-1, 1)
                
                # Customizing the 0 ring to be more visible
                ax.plot(angles, [0]*len(angles), 'k-', linewidth=2)  # 'k-' for black solid line
                
                ax.set_title(f'Cluster {i+1}', size=16, color='blue', y=1.1)

        # Hide any remaining subplots not used
        for i in range(n_clusters, len(axs)):
            axs[i].axis('off')

        plt.tight_layout()
        return fig
