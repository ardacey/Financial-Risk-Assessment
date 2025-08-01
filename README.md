# Financial Risk Assessment using a Custom ID3 Decision Tree

This project involves building a Decision Tree classifier from scratch using the **ID3 algorithm** to predict financial risk based on a dataset of individuals' attributes. The primary objectives are to implement the core logic of a decision tree, handle missing data, and apply both pre-pruning and post-pruning techniques to improve the model's generalization and prevent overfitting.

## 📊 Dataset

The project utilizes the **Financial Risk Assessment Dataset**.
-   **Content**: The dataset contains information on 15,000 individuals, each described by 19 attributes.
-   **Features Used**: A subset of categorical and numerical features deemed relevant after initial analysis.
-   **Target Variable**: `Risk Rating` (Categorized as Low, Medium, or High).
-   **Data Split**: The data is split into **70% training, 15% validation, and 15% testing** sets.

## ⚙️ Project Workflow & Methodology

The project is structured into two main parts: implementing the ID3 algorithm and then applying pruning techniques to optimize the resulting tree.

### 1. Data Preparation and Preprocessing
-   **Handling Missing Data**: Missing numerical values were imputed using the **median** of their respective columns. This approach was chosen over mean imputation to be robust against outliers.
-   **Feature Encoding**: Categorical features were converted into numerical representations using **label encoding**.
-   **Data Normalization**: All numerical features were scaled to a [0, 1] range using **min-max scaling** to ensure that no single feature dominates the model due to its scale.
-   **Feature Selection**: Low-variance features were removed to reduce noise and dimensionality, simplifying the model.

### 2. Part 1: Implementing the ID3 Decision Tree from Scratch
A decision tree was built from scratch using the **ID3 (Iterative Dichotomiser 3)** algorithm.
-   **Core Logic**: The algorithm recursively partitions the data based on the feature that provides the highest **Information Gain**.
-   **Splitting Criterion**:
    -   **Entropy**: Used to measure the impurity or disorder of a set of labels.
    -   **Information Gain**: Calculated as the reduction in entropy achieved by splitting the data on a particular attribute. At each node, the attribute with the highest information gain is chosen as the splitting feature.
-   **Pre-pruning**: A `max_depth` parameter was implemented to control the growth of the tree, serving as a basic pre-pruning mechanism to prevent overfitting.

### 3. Part 2: Pruning the Decision Tree
To improve the model's performance on unseen data, both pre-pruning (by limiting depth) and post-pruning techniques were applied and evaluated.
-   **Pre-pruning (Max Depth)**: The tree was trained with a limited depth (`max_depth=7`), which was found to be the optimal depth before overfitting became significant. This prevents the tree from becoming overly complex during the building phase.
-   **Post-pruning (Reduced Error Pruning)**: A custom post-pruning algorithm was implemented. It works by:
    1.  Building a fully grown tree (or the best pre-pruned tree).
    2.  Iteratively identifying "twigs" (nodes where both children are leaf nodes) that have very low information gain (≤ 0.0001).
    3.  Temporarily converting the least informative twig into a leaf node (with the majority class of its children).
    4.  Evaluating the accuracy of this new, smaller tree on the **validation set**.
    5.  If the accuracy improves or stays the same, the prune is made permanent. If it decreases, the change is reverted, and the algorithm terminates.

## 📈 Results and Analysis

The project successfully demonstrated the trade-offs between model complexity, overfitting, and generalization.

### Pre-pruning vs. Unpruned Tree

-   An **unpruned tree** achieved high accuracy on the training data (**87%**) but performed poorly on the test data (**46%**), indicating severe overfitting.
-   The **best pre-pruned tree (max_depth=7)** showed much better generalization, with a test accuracy of **60%** and an F1-score of **0.47**. While its training accuracy was lower (**61%**), its ability to perform on unseen data was significantly improved.

### Post-pruning Results

-   The post-pruning algorithm was applied to the `max_depth=7` tree. It successfully identified and pruned numerous nodes with minimal information gain.
-   The final **post-pruned tree** achieved a test accuracy of **61%** and an F1-score of **0.47**. This represents a slight improvement in accuracy over the pre-pruned tree, suggesting that post-pruning successfully removed branches that were not contributing positively to the model's generalization power.

| Model                       | Train Accuracy | Test Accuracy | Test F1 Score |
| --------------------------- | :------------: | :-----------: | :-----------: |
| Unpruned ID3 Tree           |      87%       |      46%      |     0.47      |
| **Pre-pruned (max_depth=7)**|      61%       |      60%      |     0.47      |
| **Post-pruned Tree**        |      60%       |    **61%**    |   **0.47**    |

### Misclassification Analysis
An analysis of misclassified samples revealed that most errors occurred in complex cases requiring many decision steps or where feature values were near the decision boundaries, making them ambiguous.

## 🛠️ Technology Stack

-   **Core Libraries**: Python, NumPy, Pandas
-   **Data Visualization**: Matplotlib, Seaborn
-   **Data Splitting**: Scikit-learn (`train_test_split`)

## 🚀 How to Run

This project is designed to be run in a Google Colab environment or any standard Jupyter Notebook setup.

1.  **Get the Data**: Ensure you have the `financial_risk_assessment.csv` file.
2.  **Upload to Google Drive (for Colab)**:
    -   Upload the CSV file to your Google Drive, for example, in a folder named `Colab_Notebooks/Assignment_2/`.
    -   Adjust the file path in the notebook if you use a different location.
3.  **Open in Colab/Jupyter**: Open the `notebook.ipynb` notebook.
4.  **Run the Cells**:
    -   Execute the cells sequentially. The notebook will guide you through data loading, preprocessing, building the ID3 tree, and finally, applying and evaluating the pruning techniques.
