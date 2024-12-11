# Ireland-Housing-Prices
Ireland Hosuing Price Prediction












 
 
1.1	Dataset Description and Initial Exploration
NATURE OF THE DATASET
The given dataset contains real estate information for properties in Ireland, mainly focusing on housing prices. The dataset includes various attributes like property scope, location, availability, bedroom size, bath, balcony, price-per-sqft-$, etc. The main aim of the analysis is to predict the housing prices using these attributes.
1.2 Key Features
•	ID: Unique identifier for each property.
•	availability: Binary feature indicating property availability.
•	size: Number of bedrooms.
•	total_sqft: Total area in square feet.
•	bath: Number of bathrooms.
•	balcony: Number of balconies.
•	buying or not buying: Target variable for classification tasks.
•	price-per-sqft-$: Target variable for regression tasks.
•	BER: Building Energy Rating (ordinal).
•	location: Property location.

2.2	Appendix: Libraries Used
•	pandas: Data manipulation.
•	numpy: Numerical operations.
•	matplotlib, seaborn: Visualization.
•	scikit-learn: Machine learning models and preprocessing.

2.3 Dataset Exploration using Key Pandas Methods
When exploring a dataset for the first time, it is important to understand its structure and basic characteristics. This can be achieved using the pandas library.

Summary of Methods
Method	Description
df.head()	Displays the first 5 rows of the dataset (customizable).
df.tail()	Displays the last 5 rows of the dataset (customizable).
df.info()	 Provides a summary of the dataset, including non-null count, etc
df.shape	 Returns the dimensions of the dataset (rows, columns).
df.describe()	 Provides summary statistics for numerical columns (count, mean, std, min, max)
	
	
Correlation heatmap for numeric columns:
1.	Filter Numeric Columns: It selects only the columns with numeric data types from the dataset using select_dtypes.
2.	Compute Correlations: The corr() method calculates pairwise correlation coefficients between all numeric columns.
3.	Heatmap Visualization: Using Seaborn's heatmap function, it plots a visual representation of the correlation matrix. Each cell shows the correlation coefficient, with colors representing the strength and direction of the relationship (positive or negative)


Handling Missing Values: Updated Project Report Section
1. Overview of Missing Values

Feature	           Number of Missing Values	 
location	                              1	
size	                             16	
bath	                             73	
balcony	                            609	
price-per-sqft-$	                            246	
________________________________________
2. Description of Affected Features
•	location: A categorical feature showing the geographical area of the property. Only one missing value exists.
•	size: Showing the number of bedrooms. Missing data can indicate incomplete records or undefined room counts.
•	bath: Indicates the number of bathrooms. Missing data might occur due to incomplete entry or lack of complete information about the property.
•	balcony: Shows the number of balconies. The high percentage of missing values suggests missing data might indicate properties with no balconies or incomplete records.
•	price-per-sqft-$: The target variable for regression tasks. Missing data here could significantly affect model training and accuracy if not handled properly.

3. Justification for Actions
Dropped Rows for location, size, price-per-sqft-$, and balcony
•	Reasoning:
o	Null values for these columns was removed or addressed by dropping the particular rows.
o	Dropping these rows is appropriate when:
1.	The proportion of missing values is low relative to the dataset size.
o	Impact:
	Reduces the dataset size but ensures clean, errorless data for model training.
Imputed bath with Mean
•	Reasoning:
o	Mean imputation is used to handle missing values in the bath column.
o	Mean is suitable when the missing values are random, and the distribution of the feature is relatively normal.


Converting Categorical to Numerical Values: buying or not buying and Renovation needed Columns
Describes the process of converting two categorical columns into numerical values: buying or not buying and Renovation needed. This is a important preprocessing step for machine learning, as most algorithms need numerical data to make predictions.



1. buying or not buying Column
Column Overview:
•	Feature Name: buying or not buying
•	Original Values: The column contains categorical values:
o	"Yes": Property is being bought.
o	"No": Property is not being bought.
Conversion Process:
•	Mapping: We convert these categorical values into numerical values using the pandas map() function:
o	"Yes" is mapped to 1.
o	"No" is mapped to 0.
Why This Is Necessary:
•	Machine Learning Compatibility:

2. Renovation needed Column
Column Overview:
Feature Name: Renovation needed
•	Original Values: The column contains categorical values indicating whether the property requires renovation:
o	"Yes": Renovation is needed.
o	"No": Renovation is not needed.
o	"Maybe": It's uncertain whether renovation is needed.
Conversion Process:
•	Mapping: We map these categorical values to numerical values:
o	"Yes" is mapped to 1.
o	"No" is mapped to 0.
o	"Maybe" is mapped to 2.
•	Ensure Integer Data Type: After mapping, the data type of the column may still be object (string). To ensure it is treated as numeric, we explicitly convert it to an integer data type using astype().
Why This Is Necessary:
Machine Learning Compatibility: 
Handling Uncertainty: 


Converting Categorical Columns to Numerical Features Using One-Hot Encoding: property scope and location, and Handling availability

1. One-Hot Encoding for property scope Column
One-Hot Encoding Process:
•	One-Hot Encoding is a technique used to convert categorical values into a binary matrix (0 or 1) where each category has its own column. For each record, a 1 is placed in the column corresponding to the category and 0 is placed in all other columns.
Why One-Hot Encoding Is Used:
•	Machine Learning Compatibility
•	Interpretability
Implementation:
•	Concatenate Encoded Columns:  Used pd.concat() to combine the newly created One-Hot Encoded columns with the original dataframe.

2. Encoding for availability Column
Column Overview:
•	Feature Name: availability
•	Original Values: The availability column contains categorical data such as "Ready To Move" and other statuses indicating the availability of the property.
Conversion Process:
•	For this column, we apply a binary encoding where "Ready To Move" is mapped to 1 and any other value is mapped to 0.
•	Why Binary Encoding: This conversion is necessary because the column contains only two categories: properties that are ready to move (1) and those that are not (0). This makes it a binary classification problem.

3. One-Hot Encoding for location Column
Column Overview:
•	Feature Name: location
•	Original Values: The location column contains categorical values representing different locations of the property, such as "South Dublin", "North Dublin", etc.
One-Hot Encoding Process:
•	Similar to the property_scope, we apply One-Hot Encoding to the location column.
•	We use pd.get_dummies () to create a binary matrix for the locations.
•	drop_first=True is used to avoid the "dummy variable trap" by dropping the first category and avoids multicollinearity

Implementation:
•	Concatenate Encoded Columns: Combining the newly created One-Hot Encoded location columns with the main data frame.


Ordinal encoding for the BER column:
1.	Define Ordinal Mapping: A dictionary (ber_mapping) assigns numerical values to the categorical levels of the BER column. Lower BER levels (e.g., "A") receive smaller values, indicating better energy ratings.
2.	Apply Mapping: The .map() method transforms the BER column into a new numerical column, BER_encoded, by replacing each BER category with its corresponding numeric value.


One-hot encoding on the location column:
1.	One-Hot Encoding:
•	The pd.get_dummies() function creates binary columns for each unique location in the location column, excluding the first category (drop_first=True) to prevent multicollinearity.
•	These columns are prefixed with location_ for clear identification.

2.	Update Dataset:
•	The newly created dummy variables are concatenated to the original dataframe (df).
•	The original location column is dropped as it is now redundant.

3.	Type Conversion:
•	Initially, the new columns (location_Dun Laoghaire, location_Fingal, location_Other, location_South Dublin) are converted to boolean for clarity.
•	They are later converted to integers for computational efficiency and compatibility with numerical analysis.


Identifying and handling outliers:
1.	1. Capping Total Square Footage (Log-Transformed)
•	Thresholds: The 5th and 95th percentiles of the total_sqft_log column are calculated as lower_cap and upper_cap.
•	Capping: Values below the lower threshold are replaced with lower_cap, and values above the upper threshold are replaced with upper_cap.

2. IQR-Based Outlier Detection
•	Interquartile Range (IQR): IQR is calculated as the difference between the 75th percentile (Q3) and 25th percentile (Q1) for numeric columns like bath and bedroom.
•	Bounds:
Lower bound = Q1 - 1.5 * IQR
Upper bound = Q3 + 1.5 * IQR
•	Outliers: Data points falling outside these bounds are identified as outliers for the respective columns (bath and bedroom).

3. Extracting and Filtering Outliers
•	Outliers are extracted for further analysis or visualization using logical indexing.
•	Separate outliers for bath and bedroom columns are identified using the IQR method to ensure accurate anomaly detection.
.
Adding Feature Columns:
1.	space_utilization
•	Description: This column represents the ratio of the total number of rooms (bedrooms and bathrooms) to the total square footage.
•	Reason: It quantifies how efficiently the available space is utilized, which can be a critical factor for homebuyers valuing functional layouts.
2.	total_sqft_bedrooms_interaction
•	Description: This column is the product of the total square footage and the number of bedrooms.
•	Reason: It captures the combined effect of space and room count, indicating properties with larger bedrooms or more luxurious accommodations.
3.	bath_per_bedroom
•	Description: This column calculates the ratio of bathrooms to bedrooms.
•	Reason: It highlights the convenience of having adequate bathroom facilities relative to the number of bedrooms, a key feature for family homes.
4.	room_area_per_sqft
•	Description: This column represents the average area allocated per room (bedroom or bathroom) into the total square footage.
•	Reason: It provides insight into the spaciousness of each room, helping assess comfort levels and also room functionality.
5.	Bedroom_Cost
•	Description: This column represents the cost contribution of each bedroom by dividing the total price by the number of bedrooms.
•	Reason: It simplifies price comparisons for the buyers focusing on bedroom count, making it easier to them to evaluate the relative affordability of homes.
6.	total_sqft_log
•	Description: A log-transformed version of the total_sqft column to normalize skewed distributions.
•	Reason: It reduces the impact of the extreme square footage values, making the data more suitable for the regression models and also statistical analyses.
7.	total_sqft_capped
•	Description: This column caps extreme values in the total_sqft_log column at the 5th and 95th percentiles.
•	Reason: It mitigates the effect of extreme outliers in square footage while preserving most of the data's variability.
8.	total_price
•	Description: The overall price of the property, Calculated by multiplying the price-per-square-foot by the total square footage.
•	Reason: It provides a straightforward metric for total property valuation, aiding in the cost comparisons and affordability analyses.
9.	price_per_bathroom
•	Description: The contribution of each bathroom to the total property price, calculated by dividing the total price by the number of bathrooms.
•	Reason: It helps evaluate how the presence of bathrooms influences property costs, which can be a priority for buyers seeking convenience and luxury.
10.	price_efficiency
•	Description: This column evaluates the efficiency of the price based on the space utilization and also the number of rooms.
•	Reason: It serves as a composite metric to compare properties holistically, factoring in both cost and usability metrics.

Overall Rationale
These feature columns enrich the dataset by providing derived metrics that encapsulate critical aspects of the property value, usability, as well as functionality. They are designed to enhance interpretability for buyers, improve predictive modelling, and allow for more nuanced insights into the factors driving property prices.


Visualizations and Insights

1. Pairplot for Multivariate Analysis
The pairplot showcases relationships between key numerical features such as total_sqft, price-per-sqft, and total_price. It helps to identify trends, clusters, and correlations while factoring in the binary variable buying or not buying.

2. Violin Plot for Price Per Sqft by Renovation Needed
 It highlights how the price variability shifts between properties that need renovation versus those that do not, providing insights into renovation impact on pricing.

3. Bubble Plot: Total Sqft vs Total Price with Rooms Per Sqft
It offers a dynamic view of property size, price, and space utilization, helping to visualize trends in property efficiency.




4. FacetGrid for Price Per Sqft by Location
This visualization breaks down the distribution of price-per-sqft across various locations. By isolating each location into individual plots, it provides a clear understanding of region-specific pricing patterns and trends.

5. Heatmap of Space Utilization by Location
The heatmap highlights the average space_utilization across different locations. It offers an aggregated view of how efficiently properties are designed in specific areas, showcasing regional differences in space management.

6. Regression Line for Price Efficiency vs Total Sqft
The regression line overlay demonstrates the linear relationship between price_efficiency and total_sqft. It emphasizes how the space size impacts the pricing efficiency, offering insights into value optimization trends.

Data Preprocessing for Model 
1.	Feature and Target Splitting
•	The dataset is divided into features (X) and target (Y), where X contains all columns except the last, and Y contains only the last column.
•	Purpose: To separate the independent variables (features) from the dependent variable (target) for predictive modelling.

2.	Train-Test Split
•	The data is split into training and testing sets using an 80:20 ratio with a fixed random state for reproducibility.
•	Purpose: To train the model on one portion of the data and evaluate its performance on unseen data, ensuring generalization.
Output:
•	Training set: Contains 80% of the data for training the model.
•	Test set: Contains 20% of the data for model evaluation.

3.	Handling Non-Numeric Data
•	Non-numeric columns in X are identified and converted to numeric values using one-hot encoding.
•	Purpose: Many machine learning algorithms require numerical input, so this step ensures all features are usable.

4.	Standardization
•	The StandardScaler is applied to scale all feature values to have a mean of 0 and a standard deviation of 1.
•	Purpose: Standardization ensures that all features contribute equally to the model by eliminating scale differences, which is particularly important for distance-based or gradient-based algorithms.

5.	Final Preview
•	The first few rows of both X and Y are displayed to confirm the data preparation process.
•	The shapes of training and testing sets are printed to verify the split.
•	Purpose: This ensures the correctness of data preparation steps and provides an overview of the processed features.


Model Comparison and Evaluation
In this analysis, we chose to evaluate K-Nearest Neighbors (KNN), Gradient Boosting, Decision Trees, and Linear Regression due to their diversity in approach and widespread usage in regression tasks.
1.	K-Nearest Neighbors (KNN): KNN is a non-parametric model that makes predictions based on the average of the nearest data points. It's simple, interpretable, and effective for datasets with the non-linear relationship between features.
2.	Gradient Boosting: This is an ensemble technique which is known for its high performance in regression and classification problems. It builds multiple trees sequentially, correcting errors of previous models, and is particularly useful for complex datasets where interactions between features may exist.
3.	Decision Trees: Decision trees are interpretable and capable of capturing non-linear relationships without needing of feature scaling. They work well for datasets with distinct decision boundaries and are a fundamental model for comparison.
4.	Linear Regression: Linear regression provides a baseline model due to its simplicity and efficiency. It helps in understanding the linear relationships between features and serves as a comparison point for more complex models.

These models were selected for their ability to handle the different types of data patterns, providing a comprehensive evaluation of the performance across varying complexities.
Hyperparameter tuning and cross-validation were crucial in optimizing these models, ensuring that they performed well across different subsets of the data. Gradient Boosting demonstrated the most consistent and accurate results, making it the recommended model for this dataset.

Conclusion:
Gradient Boosting is the best model among the three, as it has the lowest MSE (indicating fewer prediction errors) and the highest R² (indicating the best fit to the data). It outperforms both KNN Regression and Decision Trees based on these metrics.

















References:
1. Dataset Description and Initial Exploration
•	Han, J., Pei, J., & Kamber, M. (2011). Data Mining: Concepts and Techniques. Morgan Kaufmann. 
2. Libraries and Tools Used
•	McKinney, W. (2010). Data Structures for Statistical Computing in Python. Proceedings of the 9th Python in Science Conference. (Reference for pandas library.)
•	Van Der Walt, S., Colbert, S. C., & Varoquaux, G. (2011). The NumPy Array: A Structure for Efficient Numerical Computation. Computing in Science & Engineering.
(Reference for NumPy.)
•	Hunter, J. D. (2007). Matplotlib: A 2D Graphics Environment. Computing in Science & Engineering. (Reference for Matplotlib.)
•	Waskom, M. L. (2021). Seaborn: Statistical Data Visualization. Journal of Open Source Software. (Reference for Seaborn.)
3. Feature Engineering
•	Kuhn, M., & Johnson, K. (2013). Applied Predictive Modeling. Springer. 
o	For methods in feature creation and selection.
4. Model Evaluation
•	Breiman, L., Friedman, J., Olshen, R., & Stone, C.  Classification and Regression Trees. Wadsworth International Group. 
o	For Decision Trees.
•	Cover, T., & Hart, P. (1967). Nearest Neighbor Pattern Classification. IEEE Transactions on Information Theory. 
o	For KNN methodology.
•	Montgomery, D. C., Peck, E. A., & Vining, G. G. (2012). Introduction to Linear Regression Analysis. Wiley. 
o	For Linear Regression.


