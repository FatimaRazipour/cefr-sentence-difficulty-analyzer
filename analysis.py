import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================
# 1. Dataset Exploration
# ============================================

# Load the CEFR-SP dataset from a local JSON file
df = pd.read_json("data/cefr-sp.json")

# Inspect the first and last rows to understand the dataset structure
print(df.head())
print(df.tail())

# Check the dataset dimensions: rows and columns
print(df.shape)

# Inspect the available columns
print(df.columns)

# Review column data types and non-null counts
print(df.info())

# Generate descriptive statistics for numerical columns
print(df.describe())

# Check the data type of each column
print(df.dtypes)

# Identify missing values in each column
print(df.isna().sum())

# Check for duplicate rows
print(df.duplicated().sum())

# Examine the distribution of sentences across CEFR levels
print(df["cefr_level"].value_counts())

# Display one representative sentence from each CEFR level
print(df.groupby("cefr_level")["text"].first())


# ============================================
# 2. Data Cleaning
# ============================================

# Identify duplicate rows before removing them
print(df[df.duplicated()])

# Remove duplicate rows from the dataset
df.drop_duplicates(inplace=True)

# Verify that no duplicate rows remain
print(df.duplicated().sum())

# Check the dataset dimensions after cleaning
print(df.shape)

# Verify the available CEFR levels
print(df["cefr_level"].unique())

# Check the number of sentences in each CEFR level after cleaning
print(df["cefr_level"].value_counts())

# Check for empty or whitespace-only text entries
print(df["text"].str.strip().eq("").sum())

# Check for missing text values
print(df["text"].isna().sum())


# ============================================
# 3. Text Feature Engineering
# ============================================

# Count whitespace-separated tokens in each sentence
df["word_count"] = df["text"].str.split().str.len()

print(df[["cefr_level", "text", "word_count"]].head())
print(df[["cefr_level", "text", "word_count"]].tail())


# Count all characters, including spaces and punctuation
df["character_count"] = df["text"].str.len()

print(
    df[["cefr_level", "text", "word_count", "character_count"]].head()
)
print(
    df[["cefr_level", "text", "word_count", "character_count"]].tail()
)


# Count unique whitespace-separated tokens in each sentence
df["unique_word_count"] = (
    df["text"].str.split().apply(lambda words: len(set(words)))
)

print(
    df[["cefr_level", "text", "word_count", "unique_word_count"]].head()
)
print(
    df[["cefr_level", "text", "word_count", "unique_word_count"]].tail()
)


# Calculate the average number of characters per whitespace-separated token
df["average_word_length"] = df["character_count"] / df["word_count"]

print(
    df[
        [
            "cefr_level",
            "text",
            "word_count",
            "character_count",
            "average_word_length"
        ]
    ].head()
)

print(
    df[
        [
            "cefr_level",
            "text",
            "word_count",
            "character_count",
            "average_word_length"
        ]
    ].tail()
)


# Measure lexical diversity as the ratio of unique tokens to total tokens
df["lexical_diversity"] = (
    df["unique_word_count"] / df["word_count"]
)

print(
    df[["cefr_level", "text", "lexical_diversity"]].head()
)
print(
    df[["cefr_level", "text", "lexical_diversity"]].tail()
)


# ============================================
# 4. Pandas Analysis
# ============================================

# Calculate the mean word count for each CEFR level
mean_word_count_by_cefr = (
    df.groupby("cefr_level")["word_count"].mean()
)
print(mean_word_count_by_cefr)


# Calculate the mean lexical diversity for each CEFR level
mean_lexical_diversity_by_cefr = (
    df.groupby("cefr_level")["lexical_diversity"].mean()
)
print(mean_lexical_diversity_by_cefr)


# Calculate the mean character count for each CEFR level
mean_character_count_by_cefr = (
    df.groupby("cefr_level")["character_count"].mean()
)
print(mean_character_count_by_cefr)


# Calculate the mean average token length for each CEFR level
mean_average_word_length_by_cefr = (
    df.groupby("cefr_level")["average_word_length"].mean()
)
print(mean_average_word_length_by_cefr)


# Calculate the mean number of unique tokens for each CEFR level
mean_unique_word_count_by_cefr = (
    df.groupby("cefr_level")["unique_word_count"].mean()
)
print(mean_unique_word_count_by_cefr)


# Combine the main linguistic features into a single CEFR-level summary
cefr_summary = df.groupby("cefr_level").agg({
    "word_count": "mean",
    "lexical_diversity": "mean",
    "character_count": "mean",
    "average_word_length": "mean",
    "unique_word_count": "mean"
})

print(cefr_summary)


# Sort CEFR levels by mean word count in descending order
print(
    cefr_summary.sort_values(
        by="word_count",
        ascending=False
    )
)

# Identify the CEFR level with the highest mean word count
print(cefr_summary["word_count"].idxmax())

# Identify the CEFR level with the lowest mean word count
print(cefr_summary["word_count"].idxmin())


# Examine the absolute and relative distribution of CEFR levels
print(df["cefr_level"].value_counts())
print(df["cefr_level"].value_counts(normalize=True))


# Inspect C2 sentences and their token counts
print(
    df.query('cefr_level == "C2"')[["text", "word_count"]].head()
)


# ============================================
# 5. Comparative Linguistic Analysis
# ============================================

# Calculate the change in mean word count between consecutive CEFR levels
print(cefr_summary["word_count"].diff())

# Calculate the percentage change in mean word count between levels
print(cefr_summary["word_count"].pct_change() * 100)

# Calculate the percentage change in mean character count between levels
print(cefr_summary["character_count"].pct_change() * 100)

# Calculate the percentage change in average token length between levels
print(cefr_summary["average_word_length"].pct_change() * 100)

# Calculate the percentage change in lexical diversity between levels
print(cefr_summary["lexical_diversity"].pct_change() * 100)

# Calculate the percentage change in unique token count between levels
print(cefr_summary["unique_word_count"].pct_change() * 100)


# Encode CEFR levels numerically according to their ordinal progression
cefr_order = {
    "A1": 1,
    "A2": 2,
    "B1": 3,
    "B2": 4,
    "C1": 5,
    "C2": 6
}

df["cefr_numeric"] = df["cefr_level"].map(cefr_order)

print(df[["cefr_level", "cefr_numeric"]].tail())


# Examine correlations between CEFR level and the engineered linguistic features
print(
    df[
        [
            "cefr_numeric",
            "word_count",
            "character_count",
            "average_word_length",
            "unique_word_count",
            "lexical_diversity"
        ]
    ].corr()
)


# Rank the features by their correlation with the ordinal CEFR encoding
print(
    df[
        [
            "cefr_numeric",
            "word_count",
            "character_count",
            "average_word_length",
            "unique_word_count",
            "lexical_diversity"
        ]
    ]
    .corr()["cefr_numeric"]
    .sort_values(ascending=False)
)

# ============================================
# 6. Visualization
# ============================================



# Visualize the mean number of tokens across CEFR levels
plt.bar(cefr_summary.index,
        cefr_summary["word_count"],
        color="#4fb887",
        edgecolor="#313333"
    )
plt.title("Word count by CEFR-Level")
plt.xlabel("CEFR Level")
plt.ylabel("Word Count")
plt.grid(axis="y")
plt.savefig("visualizations/word_count_by_cefr.png")
plt.show()

# Visualize mean lexical diversity across CEFR levels
plt.bar(
    cefr_summary.index,
    cefr_summary["lexical_diversity"],
    color="#4fb887",
    edgecolor="#313333"
)

plt.title("Mean Lexical Diversity by CEFR Level")
plt.xlabel("CEFR Level")
plt.ylabel("Mean Lexical Diversity")
plt.grid(axis="y")
plt.savefig("visualizations/lexical_diversity_by_cefr.png")
plt.show()

# Visualize the distribution of sentences across CEFR levels
plt.bar(
    df["cefr_level"].value_counts().index,
    df["cefr_level"].value_counts().values,
    color="#4fb887",
    edgecolor="#313333"
)

plt.xlabel("CEFR Level")
plt.ylabel("Number of Sentences")
plt.title("Sentence Distribution by CEFR Level")
plt.grid(axis="y")
plt.savefig("visualizations/cefr_distribution.png")
plt.show()

# Visualize the relationship between token count and character count
plt.scatter(
    df["word_count"],
    df["character_count"],
    color="#4fb887",
    alpha=0.4
)

plt.xlabel("Token Count")
plt.ylabel("Character Count")
plt.title("Token Count vs Character Count")
plt.grid()
plt.savefig("visualizations/token_vs_character_count.png")
plt.show()




# Calculate correlations between CEFR and linguistic features
corr = df[
    [
        "cefr_numeric",
        "word_count",
        "character_count",
        "average_word_length",
        "unique_word_count",
        "lexical_diversity"
    ]
].corr()

# Visualize the correlation matrix
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="coolwarm"
)

plt.title("Correlation Matrix of Linguistic Features")
plt.tight_layout()
plt.savefig("visualizations/correlation_heatmap.png")
plt.show()

# Visualize the distribution of token counts across CEFR levels
sns.boxplot(
    data=df,
    x="cefr_level",
    y="word_count",
    color="#4fb887"
)

plt.xlabel("CEFR Level")
plt.ylabel("Token Count")
plt.title("Token Count Distribution by CEFR Level")
plt.tight_layout()
plt.savefig("visualizations/token_count_boxplot.png")
plt.show()


# Visualize average token length across CEFR levels
plt.bar(cefr_summary.index,
    cefr_summary["average_word_length"],
    color="#4fb887",
    edgecolor="#313333"
)
    

plt.xlabel("CEFR Level")
plt.ylabel("Average Characters per Token")
plt.title("Average Token Length by CEFR Level")
plt.grid(axis="y")

plt.tight_layout()
plt.savefig("visualizations/average_token_length.png")
plt.show()


# ============================================
# 7. Final Insights
# ============================================

# Display the main CEFR-level linguistic summary
print("\nFinal Linguistic Summary:")
print(cefr_summary)